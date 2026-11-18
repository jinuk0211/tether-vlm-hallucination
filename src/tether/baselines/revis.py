from contextlib import contextmanager

import torch
import torch.nn.functional as F

from tether.baselines.common import capture, steer_last_token

ALPHA = 1.6
QUANTILE = 0.8
EPS = 1e-8
NEUTRAL = "I cannot determine this without an image."


def orthogonalize(visual, prior):
    projection = (visual * prior).sum(-1, keepdim=True) / prior.square().sum(-1, keepdim=True).clamp_min(EPS)
    return visual - projection * prior


def extract(adapter, fit, calibration):
    layers = range(len(adapter.layers))
    raw = {i: [] for i in layers}
    prior = {i: [] for i in layers}
    for r in fit:
        conditions = [
            (r["image"], r["chosen"]),
            (None, r["chosen"]),
            (None, r["rejected"]),
            (None, r.get("neutral", NEUTRAL)),
        ]
        s = [
            capture(adapter, adapter.prepare(image, r["prompt"], answer), layers)
            for image, answer in conditions
        ]
        for i in layers:
            raw[i].append(s[0][i] - s[1][i])
            prior[i].append(s[2][i] - s[3][i])
    vectors = {i: orthogonalize(torch.stack(raw[i]).mean(0), torch.stack(prior[i]).mean(0)) for i in layers}

    factual = {i: [] for i in layers}
    hallucinated = {i: [] for i in layers}
    for r in calibration:
        for answer, risks in ((r["chosen"], factual), (r["rejected"], hallucinated)):
            s = capture(adapter, adapter.prepare(r["image"], r["prompt"], answer), layers)
            for i in layers:
                risks[i].append(-F.cosine_similarity(s[i], vectors[i], dim=0))
    separated = [
        i
        for i in layers
        if float(torch.stack(hallucinated[i]).mean() - torch.stack(factual[i]).mean()) > 0
        and vectors[i].norm() > EPS
    ]
    layer = max(separated)
    threshold = float(torch.quantile(torch.stack(factual[layer]), QUANTILE))
    return {"vector": vectors[layer], "layer": layer, "threshold": threshold}


@contextmanager
def steer(adapter, direction):
    vector = direction["vector"].to(adapter.device)

    def update(h):
        risk = -F.cosine_similarity(h.float(), vector.to(h).float(), dim=-1)
        return h + ALPHA * (risk > direction["threshold"]).unsqueeze(-1).to(h) * vector.to(h)

    handle = steer_last_token(adapter.layers[direction["layer"]], update)
    try:
        yield
    finally:
        handle.remove()
