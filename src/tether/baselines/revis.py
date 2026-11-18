import json
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from tether.baselines import authors
from tether.baselines.common import capture, steer_last_token
from tether.paths import DATA, UPSTREAM

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


def official_vectors(adapter):
    _, calculator = authors.revis()

    def to_device(module, args, kwargs):
        return args, {k: v.to(adapter.device) if torch.is_tensor(v) else v for k, v in kwargs.items()}

    model = adapter.model
    handle = model.register_forward_pre_hook(to_device, with_kwargs=True)
    if not hasattr(model, "language_model") and getattr(model.config, "num_hidden_layers", None) is None:
        model.config.num_hidden_layers = len(adapter.layers)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    with torch.no_grad():
        vectors = calculator._calculate_das_vectors_internal(
            model,
            adapter.processor,
            str(UPSTREAM / "revis/data/OH_sampled_data_100.json"),
            str(DATA / "revis/images"),
            "Qwen",
        )
    handle.remove()
    return vectors


def official_calibrate(adapter, vectors, percentile=80.0, min_layer=10, n=100):
    from PIL import Image

    rows = [
        json.loads(l) for l in open(UPSTREAM / "revis/data/coco_pope_calibration_merged.jsonl") if l.strip()
    ]
    for layer in range(len(adapter.layers) - 1, min_layer - 1, -1):
        vector = vectors[layer + 1]["visual"]
        seen = [None]

        def hook(module, args, out):
            if seen[0] is None:
                h = out[0] if isinstance(out, tuple) else out
                x = F.normalize(h[:, -1, :], dim=-1)
                u = F.normalize(vector.to(device=h.device, dtype=h.dtype), dim=-1)
                seen[0] = -torch.matmul(x, u).item()
            return out

        handle = adapter.layers[layer].register_forward_hook(hook)
        factual, hallucinated = [], []
        for r in rows[:n]:
            seen[0] = None
            image = Image.open(DATA / "revis/images" / r["image"]).convert("RGB")
            text = adapter.chat(r["text"] + " Please answer yes or no.")
            inputs = adapter.processor(text=text, images=[image], padding=True, return_tensors="pt")
            inputs = inputs.to(adapter.model.device)
            with torch.no_grad():
                out = adapter.model.generate(**inputs, max_new_tokens=5, do_sample=False)
            answer = adapter.processor.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
            answer = answer.strip().lower()
            pred = "yes" if "yes" in answer else ("no" if "no" in answer else "unknown")
            label = r["label"].lower()
            if pred == label:
                factual.append(seen[0])
            elif label == "no" and pred == "yes":
                hallucinated.append(seen[0])
        handle.remove()
        if factual and hallucinated and np.mean(hallucinated) - np.mean(factual) > 0:
            tau = float(np.percentile(factual, percentile))
            print("layer", layer, "tau", tau, flush=True)
            return {"vectors": vectors[layer + 1], "layer": layer, "tau": tau}
    raise SystemExit("no layer separates factual from hallucinated answers")


@contextmanager
def official_steer(adapter, direction):
    layers, _ = authors.revis()
    hook = layers.MADRSteeringHook_visual_only(
        vectors=direction["vectors"],
        layer_idx=direction["layer"],
        alpha_vis=ALPHA,
        tau_low=direction["tau"],
        risk_gamma=1.0,
    )
    hook.attach(adapter.layers[direction["layer"]])
    try:
        yield
    finally:
        hook.detach()
