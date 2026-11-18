from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image

from tether.baselines import authors
from tether.utils import seed_everything

LAYER = 27
ALPHA = 10.0
TOP_K = 50


class QwenModel:
    def __init__(self, model):
        self.native = model

    def __getattr__(self, name):
        return getattr(self.native, name)

    def eval(self):
        self.native.eval()
        return self

    def get_input_embeddings(self):
        return self.native.get_input_embeddings()

    def __call__(self, *args, **kwargs):
        out = self.native(*args, **kwargs)
        if kwargs.get("output_hidden_states") and out.hidden_states is not None:
            device = next(self.native.parameters()).device
            out.hidden_states = tuple(h.to(device) for h in out.hidden_states)
        return out


class GemmaModel:
    def __init__(self, model):
        self.native = model
        self.per_layer = []
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.native, name)

    def eval(self):
        self.native.eval()
        return self

    def get_input_embeddings(self):
        self.per_layer = []
        self.calls = 0

        def embed(ids):
            masks = self.native.model.get_placeholder_mask(ids, None)
            clean = torch.where(
                masks[0] | masks[1] | masks[2], self.native.config.text_config.pad_token_id, ids
            )
            self.per_layer.append(self.native.model.language_model.get_per_layer_inputs(clean, None).detach())
            return self.native.get_input_embeddings()(clean)

        return embed

    def __call__(self, *args, **kwargs):
        if kwargs.get("inputs_embeds") is not None:
            per_layer = self.per_layer[self.calls % 2]
            self.calls += 1
            embeds = kwargs["inputs_embeds"]
            kwargs["per_layer_inputs"] = per_layer.expand(embeds.shape[0], -1, -1, -1)
            kwargs["use_cache"] = False
        return self.native(*args, **kwargs)


class LayerView:
    def __init__(self, module):
        self.module = module

    def named_modules(self):
        return iter([(f"language_model.model.layers.{LAYER}", self.module)])


def load_image(path):
    with Image.open(path) as f:
        return f.convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)


def attribution(adapter, record):
    gemma = adapter.model_type == "gemma4"
    content = [{"type": "text", "text": record["prompt"]}, {"type": "image"}]
    if not gemma:
        content.reverse()
    prompt = adapter.processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    model = GemmaModel(adapter.model) if gemma else QwenModel(adapter.model)
    return authors.grains().ablation(
        model,
        adapter.processor,
        load_image(record["image"]),
        prompt,
        record["chosen"],
        record["rejected"],
        top_k=TOP_K,
        top="pos",
        attribution_method="integrated_gradients",
    )


def extract(adapter, records):
    for p in adapter.model.parameters():
        p.requires_grad_(False)
    hidden = {}
    for i, record in enumerate(records):
        seed_everything(42)
        for k, v in attribution(adapter, record).items():
            if k.startswith("hidden_"):
                hidden.setdefault(k, []).append(v)
        print(f"{i + 1}/{len(records)}", flush=True)
    seed_everything(42)
    arrays = {k: np.stack(v) for k, v in hidden.items()}
    vector = authors.grains().steering.compute_steering_vector(arrays, LAYER, "pca")
    return {"vector": vector.float(), "layer": LAYER, "alpha": ALPHA}


@contextmanager
def steer(adapter, direction):
    layer = adapter.layers[LAYER]
    vector = direction["vector"].to(next(layer.parameters()).device)
    handles = [layer.register_forward_hook(lambda m, a, out: (out,))]
    handles += authors.grains().steering.add_steering_hook(LayerView(layer), LAYER, vector, alpha=ALPHA)
    handles.append(layer.register_forward_hook(lambda m, a, out: out[0]))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
