import json
import random
import types
import urllib.request
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from tether.baselines import authors
from tether.baselines.common import capture
from tether.paths import COCO, UPSTREAM

ALPHA = 0.5
MASKS = 8
MASK_FRACTION = 0.1
PATCH = 14
AUTHOR_ALPHA = 0.9
DEMOS = UPSTREAM / "vti/experiments/data/hallucination_vti_demos.jsonl"
EPS = 1e-8


def principal_direction(samples):
    samples = samples.detach().float().cpu()
    mean = samples.mean(0)
    matrices = (samples - mean).movedim(0, -2)
    _, singular, vh = torch.linalg.svd(matrices, full_matrices=False)
    direction = vh[..., 0, :]
    unit_mean = mean / mean.norm(dim=-1, keepdim=True).clamp_min(EPS)
    direction = torch.where((singular[..., 0] < EPS)[..., None], unit_mean, direction)
    sign = torch.where((direction * mean).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    return direction * sign


def masked_pixels(batch, generator):
    pixels = batch["pixel_values"]
    masked = pixels.clone()
    if pixels.ndim == 2:
        n = pixels.shape[0]
        chosen = torch.randperm(n, generator=generator)[: max(1, round(n * MASK_FRACTION))]
        masked[chosen.to(masked.device)] = 0
    elif pixels.ndim == 3:
        valid = (batch["image_position_ids"][0] != -1).all(dim=-1).nonzero().flatten().cpu()
        n = len(valid)
        chosen = valid[torch.randperm(n, generator=generator)[: max(1, round(n * MASK_FRACTION))]]
        masked[:, chosen.to(masked.device)] = 0
    else:
        height, width = pixels.shape[-2:]
        rows, cols = -(-height // PATCH), -(-width // PATCH)
        chosen = torch.randperm(rows * cols, generator=generator)[
            : max(1, round(rows * cols * MASK_FRACTION))
        ]
        for index in chosen.tolist():
            row, col = divmod(index, cols)
            masked[..., row * PATCH : (row + 1) * PATCH, col * PATCH : (col + 1) * PATCH] = 0
    return {**batch, "pixel_values": masked}


def extract(adapter, fit):
    text_layers = range(len(adapter.layers))
    vision_layers = range(len(adapter.vision_layers))
    generator = torch.Generator().manual_seed(0)
    visual = {i: [] for i in vision_layers}
    textual = {i: [] for i in text_layers}
    for r in fit:
        batch = adapter.prepare(r["image"], r["prompt"])
        clean = capture(adapter, batch, vision_layers, vision=True, positions=True)
        masked = [
            capture(adapter, masked_pixels(batch, generator), vision_layers, vision=True, positions=True)
            for _ in range(MASKS)
        ]
        for i in vision_layers:
            visual[i].append(torch.stack([m[i] for m in masked]).mean(0) - clean[i])
        chosen = capture(adapter, adapter.prepare(r["image"], r["prompt"], r["chosen"]), text_layers)
        rejected = capture(adapter, adapter.prepare(r["image"], r["prompt"], r["rejected"]), text_layers)
        for i in text_layers:
            textual[i].append(chosen[i] - rejected[i])
    return {
        "textual": {i: principal_direction(torch.stack(v)) for i, v in textual.items()},
        "visual": {i: principal_direction(torch.stack(v)) for i, v in visual.items()},
    }


def mix(h, d, alpha):
    x = h.float()
    y = F.normalize(F.normalize(x, dim=-1) + 0.1 * alpha * F.normalize(d.to(x), dim=-1), dim=-1)
    return (y * x.norm(dim=-1, keepdim=True)).to(h.dtype)


@contextmanager
def steer(adapter, direction):
    with ExitStack() as stack:
        for layers, vectors in (
            (adapter.layers, direction["textual"]),
            (adapter.vision_layers, direction["visual"]),
        ):
            for i, d in vectors.items():
                d = d.to(adapter.device)
                hook = layers[i].mlp.register_forward_hook(lambda m, a, out, d=d: mix(out, d, ALPHA))
                stack.callback(hook.remove)
        yield


class LayerList(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers


class Batched(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x.unsqueeze(0)).squeeze(0)


def install(layers, direction, alpha, vision=False):
    _, vti_layers = authors.vti()
    originals = [layer.mlp for layer in layers]
    device = next(layers[0].parameters(), direction).device
    vti_layers.add_vti_layers(LayerList(layers), direction[:, None].to(device), alpha=[alpha])
    if vision:
        for layer in layers:
            layer.mlp[1] = Batched(layer.mlp[1])
    return originals


@contextmanager
def steer_authors(adapter, direction, alpha=AUTHOR_ALPHA):
    vision = install(adapter.vision_layers, direction["visual"][1:], alpha, vision=True)
    text = install(adapter.layers, direction["textual"][1:], alpha)
    try:
        yield
    finally:
        for layers, originals in ((adapter.vision_layers, vision), (adapter.layers, text)):
            for layer, mlp in zip(layers, originals):
                layer.mlp = mlp


class ImageProcessor:
    def __init__(self, processor):
        self.mean = torch.tensor(processor.image_mean)[:, None, None]
        self.std = torch.tensor(processor.image_std)[:, None, None]

    def __call__(self, image):
        image = ImageOps.exif_transpose(image).convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
        x = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        return {"pixel_values": ((x - self.mean) / self.std)[None]}


class VisionStates:
    def __init__(self, adapter):
        self.adapter = adapter

    def __call__(self, pixels, **kwargs):
        a = self.adapter
        states = []
        first = a.vision_layers[0].register_forward_pre_hook(
            lambda m, args, kw: states.append(
                (args[0] if args else kw["hidden_states"]).detach().cpu()[None]
            ),
            with_kwargs=True,
        )
        rest = [
            l.register_forward_hook(lambda m, args, out: states.append(out.detach().cpu()[None]))
            for l in a.vision_layers
        ]
        try:
            batch = a.processor.image_processor(
                images=[pixels[0].float().cpu()], do_rescale=False, do_normalize=False, return_tensors="pt"
            )
            a.model.model.visual(
                batch["pixel_values"].to(a.device, dtype=torch.float16),
                grid_thw=batch["image_grid_thw"].to(a.device),
            )
        finally:
            for h in [first, *rest]:
                h.remove()
        return types.SimpleNamespace(hidden_states=tuple(states))


class TextStates:
    def __init__(self, adapter):
        self.adapter = adapter

    def __call__(self, **batch):
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        inputs.update(output_hidden_states=True, return_dict=True, use_cache=False)
        return self.adapter.model(**inputs)


def demo_pairs(adapter, args, model, tokenizer, demos, question, model_is_llaval=True):
    pairs = []
    prompt = adapter.chat(question)
    for d in demos:
        with Image.open(Path(args.data_file) / "train2014" / d["image"]) as f:
            image = ImageOps.exif_transpose(f).convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
        styles = []
        for key in ("h_value", "value"):
            b = adapter.processor(text=[prompt + d[key]], images=[image], return_tensors="pt")
            styles.append(
                {
                    k: v.to(adapter.device, torch.float16 if v.is_floating_point() else v.dtype)
                    for k, v in b.items()
                    if torch.is_tensor(v)
                }
            )
        pairs.append(tuple(styles))
    return tuple(pairs)


def whitespace_demos(path):
    rows = [json.loads(l) for l in open(DEMOS) if l.strip()]
    with open(path, "w") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {**r, "value": " ".join(r["value"].split()), "h_value": " ".join(r["h_value"].split())}
                )
                + "\n"
            )
    return path


def author_demos(adapter, demos, trials):
    utils, _ = authors.vti()
    args = types.SimpleNamespace(num_demos=70, num_trials=trials, mask_ratio=0.99, data_file=str(COCO))
    saved = utils.get_prompts
    utils.get_prompts = lambda *a, **k: demo_pairs(adapter, *a, **k)
    random.seed(42)
    try:
        return utils.get_demos(
            args, ImageProcessor(adapter.processor.image_processor), adapter.model, None, file_path=str(demos)
        )
    finally:
        utils.get_prompts = saved


def per_layer_pc1(states, plus, minus):
    return torch.stack(
        [
            principal_direction(torch.stack([s[plus][i] - s[minus][i] for s in states]))
            for i in range(len(states[0][0]))
        ]
    )


def extract_authors(adapter, workdir, whitespace):
    utils, _ = authors.vti()
    images, _ = author_demos(adapter, DEMOS, trials=50)
    states = utils.get_visual_hiddenstates(types.SimpleNamespace(vision_model=VisionStates(adapter)), images)
    visual = per_layer_pc1(states, 0, 1)
    del states
    originals = install(adapter.vision_layers, visual[1:], AUTHOR_ALPHA, vision=True)
    demos = whitespace_demos(Path(workdir) / "vti_demos_whitespace.jsonl") if whitespace else DEMOS
    _, inputs = author_demos(adapter, demos, trials=1)
    textual = per_layer_pc1(utils.get_hiddenstates(TextStates(adapter), inputs, None), 1, 0)
    for layer, mlp in zip(adapter.vision_layers, originals):
        layer.mlp = mlp
    return {"visual": visual, "textual": textual}


def download_demo_images():
    folder = COCO / "train2014"
    folder.mkdir(parents=True, exist_ok=True)
    for line in open(DEMOS):
        name = json.loads(line)["image"]
        if not (folder / name).exists():
            url = f"http://images.cocodataset.org/train2014/{name}"
            (folder / name).write_bytes(urllib.request.urlopen(url, timeout=60).read())
