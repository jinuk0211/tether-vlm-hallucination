import argparse
import sys
from pathlib import Path

import torch

from tether.analysis import common as C
from tether.utils import gray_image

GROUPS = ("pre_image", "image", "prompt", "caption", "recent")
STEPS = (0, 8, 16, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 448)
DEPTHS = (0.25, 0.5, 0.75, 1.0)
KO_FROM = (0, 32, 64, 128)
RECENT = 16


class AttnTap:
    def __init__(self, module, text_attn):
        self.idx = {id(a): i for i, a in enumerate(text_attn)}
        self.ko = None
        self.record = None
        self.calls = 0
        self.module = module
        self.orig = module.eager_attention_forward
        module.eager_attention_forward = self.forward

    def forward(self, mod, query, key, value, attention_mask, *args, **kw):
        i = self.idx.get(id(mod))
        if i is None:
            return self.orig(mod, query, key, value, attention_mask, *args, **kw)
        self.calls += 1
        if self.ko is not None and i in self.ko[1]:
            Lq, Lk = query.shape[2], key.shape[2]
            assert Lq == Lk, "knockout needs a full (uncached) forward"
            if attention_mask is None:
                attention_mask = torch.zeros(1, 1, Lq, Lk, dtype=query.dtype, device=query.device)
            assert attention_mask.is_floating_point()
            attention_mask = attention_mask.clone(memory_format=torch.contiguous_format)
            attention_mask[:, :, self.ko[0] :, self.img] = torch.finfo(attention_mask.dtype).min
        out, w = self.orig(mod, query, key, value, attention_mask, *args, **kw)
        if self.record is not None:
            self.reduce(i, w, attention_mask)
        return out, w

    def reduce(self, i, w, mask):
        R = self.record
        A = w[0].float().mean(0)
        q = R["q"]
        rows = A[q]
        R["attn"][i] = torch.stack([(rows * g).sum(-1) for g in R["groups"]])
        R["key_recv"][i] = rows[:, : R["capstart"]].mean(0)
        if mask is not None:
            vis = mask[0, 0][q][:, self.img] > torch.finfo(mask.dtype).min / 2
            R["visible"][i] = vis.float().mean(-1)
        else:
            R["visible"][i] = torch.ones(len(q), device=A.device)
        n = A.shape[0]
        eye = torch.eye(n, device=A.device)
        Ap = 0.5 * A + 0.5 * eye
        Ap[self.img] = eye[self.img]
        R["roll"] = Ap @ R["roll"]


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def main(a):
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    gray = gray_image()
    allrows = C.captions(a.model)
    rows = list(enumerate(allrows))[a.start : a.end]
    tag = f"{a.model}-{a.start}-{a.end}"

    gpu = torch.cuda.is_available()
    adapter = C.model(a.model)
    m = adapter.model
    tok = adapter.processor.tokenizer
    for p in m.parameters():
        p.requires_grad_(False)
    layers = adapter.layers
    NL = len(layers)
    dev = adapter.device
    tap = AttnTap(sys.modules[type(layers[0].self_attn).__module__], [l.self_attn for l in layers])
    res = C.Resid(layers)
    norm = adapter.norm
    head = m.get_output_embeddings()
    softcap = getattr(m.config.get_text_config(), "final_logit_softcapping", None)
    probe_layers = [round(f * NL) for f in DEPTHS]
    conds = (
        [("swap", None, None)]
        + [(f"ko{k}", k, set(range(NL))) for k in KO_FROM]
        + [("ko0_early", 0, set(range(NL // 2))), ("ko0_late", 0, set(range(NL // 2, NL)))]
    )
    img_id = adapter.image_token_id
    scorer = C.scorer()
    rp = float(getattr(m.generation_config, "repetition_penalty", None) or 1.0)

    def lens(h, pos, tgt):
        r = []
        for c in range(0, len(pos), 128):
            lg = head(norm(h[pos[c : c + 128]])).float()
            if softcap:
                lg = torch.tanh(lg / softcap) * softcap
            r.append(lg.log_softmax(-1).gather(-1, tgt[c : c + 128, None])[:, 0])
        return torch.cat(r)

    records = []
    lens_err = 0.0
    greedy = []
    for n, (gi, r) in enumerate(rows):
        pred = r["prediction"]
        if not pred.strip():
            continue
        gt = scorer.imid_to_objects[int(r["metadata"]["image_id"])]
        other = allrows[(gi + len(allrows) // 2) % len(allrows)]["image"]
        br = adapter.prepare(r["image"], r["prompt"], answer=pred)
        bg = adapter.prepare(str(gray), r["prompt"], answer=pred)
        bs = adapter.prepare(other, r["prompt"], answer=pred)
        for b in (bg, bs):
            assert torch.equal(br["input_ids"], b["input_ids"])
            b.pop("labels")
        ids = br["input_ids"][0]
        lab = br.pop("labels")[0]
        capstart = int((lab != -100).nonzero()[0])
        L = len(ids)
        img = ids == img_id
        tap.img = img
        T = L - capstart
        q = torch.arange(capstart - 1, L - 1, device=dev)
        tgt = ids[capstart:]
        k = torch.arange(L, device=dev)[None]
        qq = q[:, None]
        i0 = int(img.nonzero().min())
        i1 = int(img.nonzero().max())
        assert bool(img[i0 : i1 + 1].all()), "image tokens must be contiguous"
        groups = [
            (k < i0).expand(T, L),
            img[None].expand(T, L),
            ((k > i1) & (k < capstart)).expand(T, L),
            (k >= capstart) & (k <= qq),
            (k >= capstart) & (k <= qq) & (k > qq - RECENT),
        ]
        groups = [g.float() for g in groups]
        with torch.no_grad():
            tap.record = {
                "q": q,
                "groups": groups,
                "capstart": capstart,
                "attn": [None] * NL,
                "key_recv": [None] * NL,
                "visible": [None] * NL,
                "roll": torch.eye(L, device=dev),
            }
            tap.calls = 0
            res.on = True
            o = m(**br, use_cache=False, return_dict=True)
            res.on = False
            rec = tap.record
            tap.record = None
            assert tap.calls == NL, (tap.calls, NL)
            Hr = list(res.h)
            lg = o.logits[0, q].float()
            lpb = lg.log_softmax(-1)
            del o
            if rp != 1.0:
                seen = torch.zeros_like(lg, dtype=torch.bool)
                seen[:, ids[:capstart]] = True
                for t in range(1, T):
                    seen[t:, ids[capstart + t - 1]] = True
                lg = torch.where(seen, torch.where(lg > 0, lg / rp, lg * rp), lg)
                del seen
            greedy_cfg = (lg.argmax(-1) == tgt).cpu()
            del lg
            roll = rec["roll"][q]
            rollout = torch.stack([(roll * g).sum(-1) for g in groups])
            prompt_norm = torch.stack([h[:capstart].float().norm(dim=-1) for h in Hr]).half().cpu()
            res.on = True
            o = m(**bg, use_cache=False, return_dict=True)
            res.on = False
            Hg = list(res.h)
            lens_r = torch.stack([lens(h, q, tgt) for h in Hr])
            lens_g = torch.stack([lens(h, q, tgt) for h in Hg])
            lens_err = max(lens_err, float((lens_r[-1] - lpb.gather(-1, tgt[:, None])[:, 0]).abs().max()))
            assert lens_err < 0.5, lens_err
            rep = []
            for hr_, hg_ in zip(Hr, Hg):
                hr = hr_[q].float()
                hg = hg_[q].float()
                v = hr_[img].float().mean(0)
                mu = hr_[~img].float().mean(0)
                rep.append(
                    torch.stack(
                        [
                            hr.norm(dim=-1),
                            (hr - hg).norm(dim=-1),
                            cos(hr, hg),
                            cos(hr, v[None]),
                            cos(hr - mu, (v - mu)[None]),
                        ]
                    )
                )
            rep = torch.stack(rep)
            steps = [t for t in STEPS if t < T]
            ps = q[steps]
            probe_r = torch.stack([Hr[l][ps] for l in probe_layers]).half().cpu()
            probe_g = torch.stack([Hg[l][ps] for l in probe_layers]).half().cpu()
            vis_mean = torch.stack([Hr[l][img].float().mean(0) for l in probe_layers]).half().cpu()
            del Hr, Hg, res.h[:]
            res.h = [None] * (NL + 1)

            def metrics(logits):
                lp = logits[0, q].float().log_softmax(-1)
                return {
                    "logp": lp.gather(-1, tgt[:, None])[:, 0].half().cpu(),
                    "kl": (lpb.exp() * (lpb - lp)).sum(-1).half().cpu(),
                    "agree": (lp.argmax(-1) == lpb.argmax(-1)).cpu(),
                }

            causal = {"gray": metrics(o.logits)}
            del o
            for name, k0, lay in conds:
                if name == "swap":
                    o = m(**bs, use_cache=False, return_dict=True)
                else:
                    tap.ko = (capstart - 1 + k0, lay)
                    try:
                        o = m(**br, use_cache=False, return_dict=True)
                    finally:
                        tap.ko = None
                causal[name] = metrics(o.logits)
                del o
            base_logp = lpb.gather(-1, tgt[:, None])[:, 0].half().cpu()
            base_greedy = (lpb.argmax(-1) == tgt).cpu()
            del lpb
        capids = tgt.tolist()
        greedy.append(float(greedy_cfg.float().mean()))
        ments = [
            (nd, w, t, "correct" if nd in gt else "hallucinated")
            for nd, w, t in C.mention_positions(tok, capids, scorer)
        ]
        records.append(
            {
                "id": r["id"],
                "index": gi,
                "image_id": r["metadata"]["image_id"],
                "caption": pred,
                "swap_image": other,
                "T": T,
                "L": L,
                "capstart": capstart,
                "n_image_tokens": int(img.sum()),
                "image_span": [i0, i1],
                "caption_ids": capids,
                "gt_objects": sorted(gt),
                "mentions": ments,
                "attn": torch.stack(rec["attn"]).half().cpu(),
                "visible": torch.stack(rec["visible"]).half().cpu(),
                "key_recv": torch.stack(rec["key_recv"]).half().cpu(),
                "prompt_norm": prompt_norm,
                "rollout": rollout.half().cpu(),
                "lens_real": lens_r.half().cpu(),
                "lens_gray": lens_g.half().cpu(),
                "repr": rep.half().cpu(),
                "probe_layers": probe_layers,
                "probe_steps": steps,
                "probe_real": probe_r,
                "probe_gray": probe_g,
                "vis_mean": vis_mean,
                "base_logp": base_logp,
                "base_greedy": base_greedy,
                "base_greedy_cfg": greedy_cfg,
                "causal": causal,
            }
        )
        del rec, roll, groups
        if gpu:
            torch.cuda.empty_cache()
        if n % 100 == 99:
            torch.save(records, out / f"{tag}.partial.pt")
    torch.save(records, out / f"{tag}.pt")
    (out / f"{tag}.partial.pt").unlink(missing_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["qwen", "gemma"], required=True)
    p.add_argument("--output", type=Path, default=C.DECAY)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=500)
    main(p.parse_args())
