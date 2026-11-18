import argparse
import json
from pathlib import Path

import torch

from tether.analysis import common as C
from tether.paths import SETS


def pope(adapter, out, model):
    v = C.variants(adapter)
    rows, lasts, vis = [], [], {}
    for r in C.read(SETS / "pope_adversarial.jsonl"):
        logits, last, pooled, n = C.run(adapter, r["image"], r["prompt"])
        ly = float(torch.logsumexp(logits[v["yes"]], 0))
        ln = float(torch.logsumexp(logits[v["no"]], 0))
        group = f"coco:{r['metadata']['image_id']}"
        vis.setdefault(group, pooled)
        top = int(logits.argmax())
        rows.append(
            {
                "id": r["id"],
                "group_id": group,
                "label": r["label"],
                "prompt": r["prompt"],
                "logit_yes": ly,
                "logit_no": ln,
                "margin": ly - ln,
                "top1": top,
                "top1_is_yes_or_no": top in v["yes"] + v["no"],
                "image_tokens": n,
            }
        )
        lasts.append(last)
    with open(out / f"{model}-pope-rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    torch.save(
        {"ids": [r["id"] for r in rows], "last": torch.stack(lasts)}, out / f"{model}-pope-answer-states.pt"
    )
    groups = sorted(vis)
    torch.save(
        {"groups": groups, "pooled": torch.stack([vis[g] for g in groups]), "pool": ["mean", "max"]},
        out / f"{model}-pope-vision-states.pt",
    )


def train_images(adapter, out, model):
    pooled, ids = [], []
    for r in C.read(SETS / "probe5000.jsonl"):
        pooled.append(C.run(adapter, r["image"], "Please describe this image in detail.")[2])
        ids.append(r["id"])
    torch.save(
        {"ids": ids, "pooled": torch.stack(pooled), "pool": ["mean", "max"]},
        out / f"{model}-train-vision-states.pt",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["qwen", "gemma"], required=True)
    p.add_argument("--mode", choices=["pope", "images"], required=True)
    p.add_argument("--output", type=Path, default=C.POPE)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    adapter = C.model(a.model)
    (pope if a.mode == "pope" else train_images)(adapter, a.output, a.model)


if __name__ == "__main__":
    main()
