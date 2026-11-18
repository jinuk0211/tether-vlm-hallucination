import json
from collections import Counter

import numpy as np
import torch

from tether.analysis import common as C

O = str(C.LENS)
s = json.load(open(f"{O}/gemma-lens-summary.json"))
im = s["image_lens"]
print(im["class_counts"])
for t, v in im["tasks"].items():
    m = v["max"]
    print(
        t,
        "pooled",
        [round(x, 2) for x in m["pooled"]][::6],
        "last",
        round(m["pooled"][-1], 3),
        "within",
        [None if x != x else round(x, 2) for x in m["within_object"]][::6],
        "best",
        m["best_layer"],
        round(m["best_pooled"], 3),
        m["best_ci95"],
        "objs",
        m["objects_used_within"],
    )
sp = im["spearman_with_question_margin"]["max"]
print(
    "spearman",
    [round(x, 2) for x in sp][::6],
    "max",
    round(np.nanmax(sp), 3),
    int(np.nanargmax(sp)),
    "q margin",
    round(im["question_margin_auroc_hallucinated"], 3),
)
d = torch.load(f"{O}/gemma-image-0-500.pt", weights_only=False)
nodes = d["nodes"]


def au(s, y):
    s = np.asarray(s, float)
    y = np.asarray(y, bool)
    p = y.sum()
    n = len(y) - p
    _, i, c = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(c) - (c - 1) / 2)[i]
    return float((r[y].sum() - p * (p + 1) / 2) / (p * n))


S = []
c = []
nd = []
for r in d["records"]:
    L = r["labels"]
    GT, M, H = set(L["gt"]), set(L["mentioned"]), set(L["hallucinated"])
    for j, n in enumerate(nodes):
        S.append(r["max"][:, j].float().numpy())
        nd.append(n)
        c.append(
            "hall"
            if n in H
            else "correct" if n in M and n in GT else "missed" if n in GT else "absent" if n not in M else "o"
        )
S = np.stack(S)
c = np.array(c)
nd = np.array(nd)
for l in (21, 26, 32, 36, 40, 42):
    x = S[:, l]
    P = lambda a, b: round(au(x[(c == a) | (c == b)], c[(c == a) | (c == b)] == a), 3)
    print(
        f" L{l} median",
        {k: float(f"{np.median(x[c==k]):.1e}") for k in ("correct", "missed", "hall", "absent")},
        "correct>hall",
        P("correct", "hall"),
        "missed>absent",
        P("missed", "absent"),
        "hall>absent",
        P("hall", "absent"),
        "missed>hall",
        P("missed", "hall"),
    )
x = S[:, -1]
a = [
    au(
        x[((c == "hall") | (c == "absent")) & (nd == n)],
        c[((c == "hall") | (c == "absent")) & (nd == n)] == "hall",
    )
    for n in np.unique(nd[c == "hall"])
    if ((c == "hall") & (nd == n)).sum() >= 5
]
print(" hall vs absent within object, last layer", round(float(np.mean(a)), 3), len(a))
r = d["records"][0]
print(
    r["image_id"], r["labels"]["gt"], Counter(w.strip()[:9] or "_" for w in r["top1_str"][32]).most_common(14)
)
