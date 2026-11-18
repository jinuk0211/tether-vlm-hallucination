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


class Steer:
    def __init__(self, layer, alpha, r):
        self.alpha = alpha
        self.r = r
        self.t = 0
        self.active = False
        layer.register_forward_hook(self.hook)

    def hook(self, module, args, out):
        h = out[0] if isinstance(out, tuple) else out
        if not self.active or h.shape[1] != 1:
            return None
        h = h.clone()
        s = 1 - self.r(self.t)
        h[0] = h[0] + (self.alpha * s * (h[0] - h[1]).float()).to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


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
    p.add_argument("--layer", type=int, default=16)
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
    steer = Steer(adapter.layers[args.layer - 1], args.alpha, r) if args.alpha else None
    setting = {k: getattr(args, k) for k in ("layer", "alpha", "lam", "pen", "clip")}

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
