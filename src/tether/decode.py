import argparse
import json
import time

import torch
from transformers import LogitsProcessor

from tether import lockstep
from tether.models import MODELS, load
from tether.schedule import reliability
from tether.utils import add_run_args, run_rows, seed_everything
from tether.vocab import load_vobj

SITES = ("resid", "attn", "mlp", "ple", "heads", "q", "k", "v", "gate", "up", "act")


def first(module, *names):
    for name in names:
        if getattr(module, name, None) is not None:
            return getattr(module, name)
    return None


def target(adapter, layer, site):
    block = adapter.layers[layer - 1]
    attn, mlp = block.self_attn, block.mlp
    gemma = adapter.model_type == "gemma4"
    targets = {
        "resid": (block, "out", 1),
        "attn": (block.post_attention_layernorm if gemma else attn, "out", 1),
        "mlp": (block.post_feedforward_layernorm if gemma else mlp, "out", 1),
        "ple": (first(block, "post_per_layer_input_norm"), "out", 1),
        "heads": (attn.o_proj, "in", attn.head_dim),
        "q": (first(attn, "q_norm", "q_proj"), "out", attn.head_dim),
        "k": (first(attn, "k_norm", "k_proj"), "out", attn.head_dim),
        "v": (first(attn, "v_norm", "v_proj"), "out", attn.head_dim),
        "gate": (mlp.gate_proj, "out", 1),
        "up": (mlp.up_proj, "out", 1),
        "act": (mlp.down_proj, "in", 1),
    }
    module, kind, width = targets[site]
    if module is None:
        raise ValueError(f"{adapter.model_type} layer {layer} has no {site} site")
    return module, kind, width


class Steer:
    def __init__(self, targets, alpha, r, units=None, topk=0):
        self.alpha = alpha
        self.r = r
        self.units = units
        self.topk = topk
        self.t = 0
        self.active = False
        for module, kind, width in targets:
            if kind == "in":
                module.register_forward_pre_hook(lambda m, args, w=width: self.wrap(args, w))
            else:
                module.register_forward_hook(lambda m, args, out, w=width: self.wrap(out, w))

    def wrap(self, value, width):
        h = self.edit(value[0] if isinstance(value, tuple) else value, width)
        if h is None:
            return None
        return (h,) + tuple(value[1:]) if isinstance(value, tuple) else h

    def select(self, d, width):
        d = d.reshape(-1, width)
        keep = torch.ones(d.shape[0], dtype=torch.bool, device=d.device)
        if self.units is not None:
            keep = torch.zeros_like(keep)
            keep[self.units] = True
        if self.topk:
            score = d.norm(dim=1).masked_fill(~keep, -1)
            keep = torch.zeros_like(keep)
            keep[score.topk(self.topk).indices] = True
        return d * keep[:, None]

    def edit(self, h, width):
        if not self.active or h.shape[1] != 1:
            return None
        d = (h[0] - h[1]).float()
        if self.units is not None or self.topk:
            d = self.select(d, width).reshape(h[0].shape)
        h = h.clone()
        h[0] = h[0] + (self.alpha * (1 - self.r(self.t)) * d).to(h.dtype)
        return h


class Belief(LogitsProcessor):
    def __init__(self, vobj, r, prompt_len, lam, pen, clip, steer=None):
        self.vobj = vobj
        self.r = r
        self.prompt_len = prompt_len
        self.lam = lam
        self.pen = pen
        self.clip = clip
        self.steer = steer
        self.A = torch.zeros(len(vobj), device=vobj.device)
        self.R = 0.0

    def __call__(self, input_ids, scores):
        t = input_ids.shape[1] - self.prompt_len
        if self.steer is not None:
            self.steer.t = t + 1
        if self.lam or self.pen:
            r = self.r(t)
            w = 1 - r
            adj = torch.zeros_like(self.A)
            if self.lam:
                logp = scores.float().log_softmax(-1)[:, self.vobj]
                self.A += r * (logp[0] - logp[1])
                self.R += r
                adj += self.lam * w * (self.A / max(self.R, 1e-6)).clamp(-self.clip, self.clip)
            if self.pen:
                adj -= self.pen * w
            written = torch.isin(self.vobj, input_ids[0, self.prompt_len :])
            adj = torch.where(written, torch.zeros_like(adj), adj)
            scores[0, self.vobj] += adj.to(scores.dtype)
        scores[1] = scores[0]
        return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--tag", required=True)
    add_run_args(p)
    p.add_argument("--site", choices=SITES, default="resid")
    p.add_argument("--layer", type=int, nargs="+", default=[16])
    p.add_argument("--units", type=int, nargs="+")
    p.add_argument("--topk", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--pen", type=float, default=0.0)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--attn")
    args = p.parse_args()

    refs, path = run_rows(args.data, args.output, args.tag, args.start, args.end)
    seed_everything(42)
    adapter = load(args.model, attn=args.attn)
    vobj = torch.tensor(load_vobj(args.model), device=adapter.device)
    r = reliability(args.model) if args.alpha or args.lam or args.pen else None
    steer = None
    if args.alpha:
        targets = [target(adapter, layer, args.site) for layer in args.layer]
        steer = Steer(targets, args.alpha, r, args.units, args.topk)
    setting = {k: getattr(args, k) for k in ("site", "layer", "units", "topk", "alpha", "lam", "pen", "clip")}

    with open(path, "a") as f:
        for ref in refs:
            start = time.perf_counter()
            batch = lockstep.gray_pair(adapter, ref)
            belief = Belief(vobj, r, batch["input_ids"].shape[1], args.lam, args.pen, args.clip, steer)
            if steer:
                steer.active = True
            try:
                text, n = lockstep.decode(adapter, batch, belief, ref["max_new_tokens"])
            finally:
                if steer:
                    steer.active = False
            row = {
                **ref,
                "method": args.tag,
                "prediction": text,
                "generated_tokens": n,
                "elapsed_seconds": time.perf_counter() - start,
                "setting": setting,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
