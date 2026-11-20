import json
import random
import types
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from tether.baselines import authors
from tether.paths import COCO, UPSTREAM

ALPHA = 0.9
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


class LayerList(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers


class Bridge(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        if x.ndim == 2:
            return self.layer(x.unsqueeze(0)).squeeze(0).to(x.dtype)
        return self.layer(x).to(x.dtype)


def install(layers, direction, alpha):
    _, vti_layers = authors.vti()
    originals = [layer.mlp for layer in layers]
    device = next(layers[0].parameters(), direction).device
    vti_layers.add_vti_layers(LayerList(layers), direction[:, None].to(device), alpha=[alpha])
    for layer in layers:
        layer.mlp[1] = Bridge(layer.mlp[1])
    return originals


@contextmanager
def steer(adapter, direction, alpha=ALPHA):
    vision = install(adapter.vision_layers, direction["visual"][1:], alpha)
    text = install(adapter.layers, direction["textual"][1:], alpha)
    try:
        yield
    finally:
        for layers, originals in ((adapter.vision_layers, vision), (adapter.layers, text)):
            for layer, mlp in zip(layers, originals):
                layer.mlp = mlp


def gemma_image(processor, image):
    batch = processor(images=[image], return_tensors="pt")
    positions = batch["image_position_ids"][0]
    real = positions[:, 0] >= 0
    width, height = (int(positions[real, i].max()) + 1 for i in (0, 1))
    p = processor.patch_size
    patches = batch["pixel_values"][0, real].reshape(height, width, p, p, 3)
    return patches.permute(4, 0, 2, 1, 3).reshape(3, height * p, width * p)


class ImageProcessor:
    def __init__(self, adapter):
        self.processor = adapter.processor.image_processor
        self.gemma = adapter.model_type == "gemma4"
        self.dtype = adapter.dtype
        self.mean = torch.tensor(self.processor.image_mean)[:, None, None]
        self.std = torch.tensor(self.processor.image_std)[:, None, None]

    def __call__(self, image):
        image = ImageOps.exif_transpose(image).convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
        if self.gemma:
            return {"pixel_values": gemma_image(self.processor, image).to(self.dtype)[None]}
        x = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        return {"pixel_values": ((x - self.mean) / self.std)[None]}


class VisionStates:
    def __init__(self, adapter):
        self.adapter = adapter

    def __call__(self, pixels, **kwargs):
        a = self.adapter
        states = []

        def keep(h):
            h = h.detach().cpu()
            states.append(h if h.ndim == 3 else h[None])

        first = a.vision_layers[0].register_forward_pre_hook(
            lambda m, args, kw: keep(args[0] if args else kw["hidden_states"]), with_kwargs=True
        )
        rest = [l.register_forward_hook(lambda m, args, out: keep(out)) for l in a.vision_layers]
        image = pixels[0].float().cpu()
        try:
            if a.model_type == "gemma4":
                batch = a.processor.image_processor(
                    images=[image], do_resize=False, do_rescale=False, return_tensors="pt"
                )
                a.model.model.vision_tower(
                    pixel_values=batch["pixel_values"].to(a.device, a.dtype),
                    pixel_position_ids=batch["image_position_ids"].to(a.device),
                )
            else:
                batch = a.processor.image_processor(
                    images=[image], do_rescale=False, do_normalize=False, return_tensors="pt"
                )
                a.model.model.visual(
                    batch["pixel_values"].to(a.device, a.dtype), grid_thw=batch["image_grid_thw"].to(a.device)
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
                    k: v.to(adapter.device, adapter.dtype if v.is_floating_point() else v.dtype)
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
    patch = adapter.processor.image_processor.patch_size if adapter.model_type == "gemma4" else 14
    saved = utils.get_prompts
    utils.get_prompts = lambda *a, **k: demo_pairs(adapter, *a, **k)
    random.seed(42)
    try:
        return utils.get_demos(
            args, ImageProcessor(adapter), adapter.model, None, patch_size=patch, file_path=str(demos)
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


def extract(adapter, workdir, whitespace):
    utils, _ = authors.vti()
    images, _ = author_demos(adapter, DEMOS, trials=50)
    states = utils.get_visual_hiddenstates(types.SimpleNamespace(vision_model=VisionStates(adapter)), images)
    visual = per_layer_pc1(states, 0, 1)
    del states
    originals = install(adapter.vision_layers, visual[1:], ALPHA)
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
