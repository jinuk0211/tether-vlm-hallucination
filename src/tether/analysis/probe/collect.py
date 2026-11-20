import argparse
import random
from pathlib import Path

import torch

from tether.analysis import common as C
from tether.utils import gray_image

LABELS = C.SELFCHECK


def main(a):
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    gray = gray_image()
    labels = {
        (r["image_id"], r["node"]): r
        for r in C.read(LABELS / f"{a.model}-base-rows.jsonl")
        if r["kind"] != "missed"
    }
    pool = sorted(
        {i for (i, _), r in labels.items() if r["kind"] == "hallucinated"}
        if not a.all_images
        else {i for i, _ in labels}
    )
    pick = set(random.Random(a.seed).sample(pool, a.n)) if 0 < a.n < len(pool) else set(pool)
    rows = [r for r in C.captions(a.model) if r["metadata"]["image_id"] in pick]
    assert len(rows) == len(pick)
    tag = f"{a.model}-n{len(rows)}" + ("-all" if a.all_images else "")

    adapter = C.model(a.model)
    m = adapter.model
    tok = adapter.processor.tokenizer
    layers = adapter.layers
    NL = len(layers)
    res = C.Resid(layers)
    scorer = C.scorer()
    meta = []
    HR = []
    HG = []
    unmatched = []
    for n, r in enumerate(rows):
        iid = r["metadata"]["image_id"]
        pred = r["prediction"]
        br = adapter.prepare(r["image"], r["prompt"], answer=pred)
        bg = adapter.prepare(str(gray), r["prompt"], answer=pred)
        assert torch.equal(br["input_ids"], bg["input_ids"])
        bg.pop("labels")
        ids = br["input_ids"][0]
        lab = br.pop("labels")[0]
        capstart = int((lab != -100).nonzero()[0])
        L = len(ids)
        T = L - capstart
        capids = ids[capstart:].tolist()
        ments = [(nd, w, t) for nd, w, t in C.mention_positions(tok, capids, scorer) if t < T]
        order = {}
        for nd, _, _ in ments:
            order.setdefault(nd, len(order))
        seen = set()
        items = []
        for k, (nd, w, t) in enumerate(ments):
            if (iid, nd) not in labels:
                unmatched.append([iid, nd])
                continue
            items.append((nd, w, t, k, nd not in seen))
            seen.add(nd)
        missing = [nd for (i, nd) in labels if i == iid and nd not in seen]
        if missing:
            unmatched += [[iid, nd, "label-without-mention"] for nd in missing]
        if not items:
            continue
        pos = torch.tensor(
            [[capstart - 1 + t, capstart + t] for _, _, t, _, _ in items], device=adapter.device
        )
        tg = ids[pos[:, 1]]
        with torch.no_grad():
            st = {}
            for name, b in (("real", br), ("gray", bg)):
                res.on = True
                o = m(**b, use_cache=False, return_dict=True)
                res.on = False
                st[name] = torch.stack([h[pos] for h in res.h], 1).half().cpu()
                st[name + "_logp"] = (
                    o.logits[0, pos[:, 0]].float().log_softmax(-1).gather(-1, tg[:, None])[:, 0].cpu()
                )
                del o
                res.h = [None] * (NL + 1)
        HR.append(st["real"])
        HG.append(st["gray"])
        for j, (nd, w, t, k, first) in enumerate(items):
            lb = labels[(iid, nd)]
            meta.append(
                {
                    "image_id": iid,
                    "node": nd,
                    "word": w,
                    "kind": lb["kind"],
                    "margin": lb["margin"],
                    "first_mention": first,
                    "mention_index": k,
                    "object_order": order[nd],
                    "t": t,
                    "T": T,
                    "logp_real": float(st["real_logp"][j]),
                    "logp_gray": float(st["gray_logp"][j]),
                }
            )
        torch.cuda.empty_cache()
    torch.save(
        {
            "meta": meta,
            "h_real": torch.cat(HR),
            "h_gray": torch.cat(HG),
            "n_layers": NL,
            "positions": ["query predicting the mention", "mention token"],
            "unmatched": unmatched,
        },
        out / f"{tag}.pt",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["qwen", "gemma"], required=True)
    p.add_argument("--output", type=Path, default=C.PROBE)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--all-images",
        action="store_true",
    )
    main(p.parse_args())
