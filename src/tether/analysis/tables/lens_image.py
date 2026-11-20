import json

import numpy as np
import torch

from tether.analysis import common as C
from tether.analysis.lens.logit import auroc

ROWS = {
    "present_vs_absent": (("correct", "missed"), ("absent",)),
    "unwritten_present_vs_absent": (("missed",), ("absent",)),
    "correct_vs_hallucinated": (("correct",), ("hall",)),
    "hallucinated_vs_absent": (("hall",), ("absent",)),
    "unwritten_present_vs_hallucinated": (("missed",), ("hall",)),
}


def scores(model):
    d = torch.load(C.LENS / f"{model}-image-0-500.pt", weights_only=False)
    S, cls, nodes = [], [], []
    for r in d["records"]:
        labels = r["labels"]
        gt, mentioned, hall = set(labels["gt"]), set(labels["mentioned"]), set(labels["hallucinated"])
        for j, n in enumerate(d["nodes"]):
            if n in hall:
                c = "hall"
            elif n in mentioned:
                c = "correct" if n in gt else None
            else:
                c = "missed" if n in gt else "absent"
            if c:
                S.append(r["max"][:, j].float().numpy())
                cls.append(c)
                nodes.append(n)
    return np.stack(S), np.array(cls), np.array(nodes)


def row_auroc(x, cls, pos, neg):
    keep = np.isin(cls, pos + neg)
    return auroc(x[keep], np.isin(cls[keep], pos))


def within_object(x, cls, nodes, pos, neg, min_pos=5):
    values = []
    for n in np.unique(nodes[np.isin(cls, pos)]):
        m = nodes == n
        if np.isin(cls[m], pos).sum() >= min_pos and np.isin(cls[m], neg).any():
            values.append(row_auroc(x[m], cls[m], pos, neg))
    return float(np.mean(values)), len(values)


def main():
    out = {}
    for model in ("qwen", "gemma"):
        S, cls, nodes = scores(model)
        table = {}
        for name, (pos, neg) in ROWS.items():
            by_layer = [row_auroc(S[:, l], cls, pos, neg) for l in range(S.shape[1])]
            best = int(np.nanargmax(by_layer))
            table[name] = {
                "best_layer": best,
                "best": by_layer[best],
                "last": by_layer[-1],
                "by_layer": by_layer,
            }
        table["hallucinated_vs_absent"]["last_within_object"] = within_object(
            S[:, -1], cls, nodes, ("hall",), ("absent",)
        )
        out[model] = table
        for name, v in table.items():
            print(model, f"{name:36s} best L{v['best_layer']} {v['best']:.3f}  last {v['last']:.3f}")
        print(
            model,
            "hallucinated vs absent within object, last layer",
            table["hallucinated_vs_absent"]["last_within_object"],
        )
    C.TABLES.mkdir(parents=True, exist_ok=True)
    (C.TABLES / "lens-image.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
