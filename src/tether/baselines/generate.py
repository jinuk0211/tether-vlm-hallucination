import argparse
import json
import time
from contextlib import ExitStack, nullcontext

import torch

from tether.baselines import grains, revis, rudder, vti
from tether.baselines.fit import direction_path
from tether.baselines.fp16 import FP16, MODULES, fp32_attention_scores
from tether.models import CONFIGS, MODELS, load
from tether.utils import add_run_args, run_rows, seed_everything

METHODS = ("base", "revis", "rudder", "grains", "vti-authors", "vti-released")
STEER = {
    "revis": revis.steer,
    "grains": grains.steer,
    "vti-authors": vti.steer,
    "vti-released": vti.steer,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    add_run_args(p)
    p.add_argument("--attn")
    args = p.parse_args()

    refs, path = run_rows(args.data, args.output, args.method, args.start, args.end)
    fp16 = args.model in FP16.get(args.method, ())
    family = CONFIGS[args.model]["family"]
    with ExitStack() as stack:
        if fp16 and family in MODULES:
            stack.enter_context(fp32_attention_scores(family))
        adapter = load(
            args.model, dtype="float16" if fp16 else "bfloat16", attn="eager" if fp16 else args.attn
        )
        if args.method in STEER:
            direction = torch.load(direction_path(args.model, args.method), map_location="cpu")
            stack.enter_context(STEER[args.method](adapter, direction))

        with open(path, "a") as f:
            for ref in refs:
                seed_everything(42)
                start = time.perf_counter()
                batch = adapter.prepare(ref["image"], ref["prompt"])
                with rudder.steer(adapter) if args.method == "rudder" else nullcontext():
                    text, n = adapter.generate(batch, ref["max_new_tokens"])
                row = {
                    **ref,
                    "method": args.method,
                    "prediction": text,
                    "generated_tokens": n,
                    "elapsed_seconds": time.perf_counter() - start,
                }
                f.write(json.dumps(row) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
