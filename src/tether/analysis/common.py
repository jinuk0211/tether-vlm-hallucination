import json

import torch

from tether.eval.chair import mention_positions, scorer, sentences
from tether.models import load
from tether.paths import COCO, DECAY, OUT, RUNS, SETS
from tether.utils import read_jsonl as read

AUDIT = OUT / "audit"
SELFCHECK = AUDIT / "selfcheck"
POPE = AUDIT / "pope"
RELIANCE = AUDIT / "reliance"
PROBE = AUDIT / "probe"
ATTENTION = AUDIT / "attention"
LENS = AUDIT / "lens"
RECHECK = AUDIT / "recheck"
JACOBIAN = AUDIT / "jacobian"
__all__ = ["COCO", "DECAY", "mention_positions", "read", "scorer"]


def model(name):
    return load(name, attn="eager")


def captions_path(name):
    return RUNS / "chair_test500" / name / "base-0-500.jsonl"


def captions(name):
    return read(captions_path(name))


def chair_sentences(name):
    path = AUDIT / f"{name}-chair.json"
    if not path.exists():
        refs = read(SETS / "chair_test500.jsonl")
        rows = {r["id"]: r["prediction"] for r in captions(name)}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sentences(scorer(), refs, rows)))
    return json.loads(path.read_text())["sentences"]


@torch.no_grad()
def run(adapter, image, prompt):
    batch = adapter.prepare(image, prompt)
    out = adapter.model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
    pos = (batch["input_ids"][0] == adapter.image_token_id).nonzero().squeeze(1)
    vis = (
        torch.stack(
            [torch.stack([h[0, pos].float().mean(0), h[0, pos].float().amax(0)]) for h in out.hidden_states]
        )
        .half()
        .cpu()
    )
    last = torch.stack([h[0, -1] for h in out.hidden_states]).half().cpu()
    return (out.logits[0, -1].float(), last, vis, len(pos))


def variants(adapter):
    v = {}
    for key, words in (("yes", ("Yes", "yes", " Yes", " yes")), ("no", ("No", "no", " No", " no"))):
        ids = (adapter.tokenizer.encode(w, add_special_tokens=False) for w in words)
        v[key] = sorted({i[0] for i in ids if len(i) == 1})
    return v


class Resid:

    def __init__(self, layers):
        self.on = False
        self.h = [None] * (len(layers) + 1)
        layers[0].register_forward_pre_hook(self.pre, with_kwargs=True)
        for i, layer in enumerate(layers):
            layer.register_forward_hook(self.post(i + 1))

    def pre(self, module, args, kwargs):
        if self.on:
            self.h[0] = (args[0] if args else kwargs["hidden_states"])[0].detach()

    def post(self, i):

        def hook(module, args, out):
            if self.on:
                self.h[i] = (out[0] if isinstance(out, tuple) else out)[0].detach()

        return hook
