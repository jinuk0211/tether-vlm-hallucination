import argparse
from pathlib import Path

import torch

from tether.analysis import common as C
from tether.utils import gray_image


class Capture:
    def __init__(self, layer):
        self.h = None
        self.on = False
        self.handle = layer.register_forward_pre_hook(self.hook, with_kwargs=True)

    def hook(self, mod, args, kwargs):
        if not self.on:
            return None
        if args:
            h = args[0].detach().requires_grad_(True)
            self.h = h
            return (h,) + tuple(args[1:]), kwargs
        h = kwargs["hidden_states"].detach().requires_grad_(True)
        self.h = h
        kwargs = dict(kwargs, hidden_states=h)
        return args, kwargs


def share(grad, h, img, upto):
    a = (grad * h).float().sum(-1).abs()[0, :upto]
    return float(a[img[:upto]].sum() / a.sum().clamp_min(1e-12)), float(
        grad.float()[0, :upto][img[:upto]].norm() / grad.float()[0, :upto].norm().clamp_min(1e-12)
    )


def main(a):
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    gray = gray_image()
    sents = {str(s["image_id"]): s for s in C.chair_sentences(a.model)}
    rows = C.captions(a.model)[a.start : a.end]
    tag = f"{a.model}-{a.start}-{a.end}"

    adapter = C.model(a.model)
    m = adapter.model
    tok = adapter.processor.tokenizer
    for p in m.parameters():
        p.requires_grad_(False)
    img_id = adapter.image_token_id
    scorer = C.scorer()
    cap = Capture(adapter.layers[0])
    from tether.analysis.selfcheck import collect as S

    variants = S.variants(adapter)
    records = []
    for n, r in enumerate(rows):
        pred = r["prediction"]
        if not pred.strip():
            continue
        s = sents[str(r["metadata"]["image_id"])]
        hall = {x[1] for x in s["mscoco_hallucinated_words"]}
        br = adapter.prepare(r["image"], r["prompt"], answer=pred)
        bb = adapter.prepare(str(gray), r["prompt"], answer=pred)
        assert torch.equal(br["input_ids"], bb["input_ids"])
        ids = br["input_ids"][0]
        lab = br.pop("labels")[0]
        bb.pop("labels")
        capstart = int((lab != -100).nonzero()[0])
        L = len(ids)
        img = ids == img_id
        capids = ids[capstart:].tolist()
        with torch.no_grad():
            o = m(**br, output_attentions=True, use_cache=False, return_dict=True)
            lp = torch.log_softmax(o.logits[0].float(), -1)
            attn = torch.stack([x[0, :, :, :].float()[:, :, img].sum(-1).mean(0) for x in o.attentions])
            del o
            ob = m(**bb, use_cache=False, return_dict=True)
            lpb = torch.log_softmax(ob.logits[0].float(), -1)
            del ob
        pos = list(range(capstart, L))
        lpr = torch.tensor([float(lp[p - 1, ids[p]]) for p in pos])
        lpbk = torch.tensor([float(lpb[p - 1, ids[p]]) for p in pos])
        at = attn[:, [p - 1 for p in pos]].half()
        ments = [(nd, w, t) for nd, w, t in C.mention_positions(tok, capids, scorer)]
        targets = {t: ("hallucinated" if nd in hall else "correct", nd, w) for nd, w, t in ments}
        for t in range(0, len(capids) - 1, 16):
            targets.setdefault(t, ("control", None, None))
        cap.on = True
        o = m(**br, use_cache=False, return_dict=True)
        h = cap.h
        cap.on = False
        jac = {}
        for t, (kind, nd, w) in sorted(targets.items()):
            p = capstart + t
            if p >= L:
                continue
            lpt = torch.log_softmax(o.logits[0, p - 1].float(), -1)[ids[p]]
            (g,) = torch.autograd.grad(lpt, h, retain_graph=True)
            jac[t] = {
                "kind": kind,
                "node": nd,
                "word": w,
                "img_share": share(g, h, img, p),
                "logp": float(lpt),
            }
        del o, h
        qs = {}
        for nd in sorted({nd for nd, _, _ in ments}):
            art = "an" if nd[0] in "aeiou" else "a"
            bq = adapter.prepare(r["image"], f"Is there {art} {nd} in the image?\nPlease answer yes or no.")
            qimg = bq["input_ids"][0] == img_id
            cap.on = True
            oq = m(**bq, use_cache=False, return_dict=True)
            hq = cap.h
            cap.on = False
            lg = oq.logits[0, -1].float()
            ly = torch.logsumexp(lg[variants["yes"]], 0)
            ln = torch.logsumexp(lg[variants["no"]], 0)
            ans = ly if ly > ln else ln
            (g,) = torch.autograd.grad(ans - torch.logsumexp(lg, 0), hq)
            qs[nd] = {
                "margin": float(ly - ln),
                "kind": "hallucinated" if nd in hall else "correct",
                "img_share": share(g, hq, qimg, hq.shape[1]),
            }
            del oq, hq
        records.append(
            {
                "id": r["id"],
                "image_id": r["metadata"]["image_id"],
                "n_caption_tokens": len(capids),
                "n_image_tokens": int(img.sum()),
                "logp_real": lpr.half(),
                "logp_gray": lpbk.half(),
                "attn_img": at,
                "jacobian": jac,
                "question": qs,
            }
        )
        torch.cuda.empty_cache()
    torch.save(records, out / f"{tag}.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["qwen", "gemma"], required=True)
    p.add_argument("--output", type=Path, default=C.RELIANCE)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=200)
    main(p.parse_args())
