import argparse
import json
import warnings

import numpy as np
import torch

from tether.paths import RUNS, SCHEDULE
from tether.utils import gray_image, read_jsonl

EDGES = [0, 16, 32, 64, 96, 128, 192, 256, 384, 513]


def reliability(model):
    s = json.loads((SCHEDULE / f"{model}.json").read_text())
    r = np.array(s["kl"]) / s["kl"][0]
    return lambda t: float(np.interp(t, s["centers"], r))


@torch.no_grad()
def gray_kl(adapter, ref):
    real = adapter.prepare(ref["image"], ref["prompt"], answer=ref["prediction"])
    gray = adapter.prepare(gray_image(), ref["prompt"], answer=ref["prediction"])
    labels = real.pop("labels")[0]
    gray.pop("labels")
    start = int((labels != -100).nonzero()[0])
    lp = adapter.forward(real).logits[0, start - 1 : -1].float().log_softmax(-1)
    lq = adapter.forward(gray).logits[0, start - 1 : -1].float().log_softmax(-1)
    return (lp.exp() * (lp - lq)).sum(-1).clamp_min(0).cpu().numpy()


def main():
    from tether.models import MODELS, load

    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    args = p.parse_args()

    adapter = load(args.model, attn="eager")
    rows = []
    for ref in read_jsonl(RUNS / "chair_test500" / args.model / "base-0-500.jsonl"):
        if ref["prediction"].strip():
            kl = gray_kl(adapter, ref)
            rows.append([kl[a:b].mean() if a < len(kl) else np.nan for a, b in zip(EDGES, EDGES[1:])])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        means = np.nanmean(np.array(rows), 0)
    keep = ~np.isnan(means)
    centers = [(a + b - 1) / 2 for a, b in zip(EDGES, EDGES[1:])]
    SCHEDULE.mkdir(parents=True, exist_ok=True)
    out = {"centers": [c for c, k in zip(centers, keep) if k], "kl": [float(m) for m in means[keep]]}
    (SCHEDULE / f"{args.model}.json").write_text(json.dumps(out, indent=1))
    print(args.model, [round(x / out["kl"][0], 2) for x in out["kl"]])


if __name__ == "__main__":
    main()
