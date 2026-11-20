import argparse
import json
from pathlib import Path

import torch

from tether.analysis import common as C


def queries(model):
    rows = {str(r["metadata"]["image_id"]): r for r in C.captions(model)}
    out = []
    for s in C.chair_sentences(model):
        r = rows[str(s["image_id"])]
        hall = {n for _, n in s["mscoco_hallucinated_words"]}
        gen = s["mscoco_generated_words"]
        gt = set(s["mscoco_gt_words"])
        first = {}
        for n in gen:
            first.setdefault(n, len(first))
        for n in sorted(set(gen) | gt):
            kind = "hallucinated" if n in hall else "correct" if n in gen else "missed"
            out.append(
                {
                    "image_id": s["image_id"],
                    "image": r["image"],
                    "node": n,
                    "kind": kind,
                    "mention_count": gen.count(n),
                    "mention_order": first.get(n),
                    "n_nodes": len(first),
                    "caption_tokens": r.get("generated_tokens"),
                }
            )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["qwen", "gemma"], required=True)
    p.add_argument("--output", type=Path, default=C.SELFCHECK)
    a = p.parse_args()

    a.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    adapter = C.model(a.model)
    v = C.variants(adapter)
    vis = {}
    with open(a.output / f"{a.model}-base-rows.jsonl", "w") as f:
        for q in queries(a.model):
            prompt = f"Is there a {q['node']} in the image?\nPlease answer yes or no."
            logits, _, pooled, _ = C.run(adapter, q["image"], prompt)
            ly = float(torch.logsumexp(logits[v["yes"]], 0))
            ln = float(torch.logsumexp(logits[v["no"]], 0))
            vis.setdefault(q["image_id"], pooled)
            f.write(
                json.dumps({**q, "prompt": prompt, "margin": ly - ln, "logit_yes": ly, "logit_no": ln}) + "\n"
            )
    ids = sorted(vis)
    torch.save(
        {
            "groups": [f"coco:{i}" for i in ids],
            "image_ids": ids,
            "pooled": torch.stack([vis[i] for i in ids]),
            "pool": ["mean", "max"],
        },
        a.output / f"{a.model}-base-vision-states.pt",
    )


if __name__ == "__main__":
    main()
