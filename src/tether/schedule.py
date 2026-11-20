import json

import numpy as np

from tether.paths import DECAY

BINS = [(0, 15), (16, 31), (32, 63), (64, 95), (96, 127), (128, 191), (192, 255), (256, 383), (384, 512)]


def gray_kl_by_bin(model):
    summary = json.loads((DECAY / f"{model}-visual-decay-summary.json").read_text())
    kl = {r["bin"]: r["mean"] for r in summary["E3_causal"]["by_bin"]["gray"]["kl"]}
    keys = [f"{a}-{b}" for a, b in BINS if f"{a}-{b}" in kl]
    centers = [(int(k.split("-")[0]) + int(k.split("-")[1])) / 2 for k in keys]
    return centers, np.array([kl[k] for k in keys])


def reliability(model):
    centers, kl = gray_kl_by_bin(model)
    r = kl / kl[0]
    return lambda t: float(np.interp(t, centers, r))
