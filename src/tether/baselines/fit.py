import argparse

import torch

from tether.baselines import grains, revis, vti
from tether.baselines.fp16 import fp32_attention_scores
from tether.models import MODELS, load
from tether.paths import DIRECTIONS, SETS
from tether.utils import read_jsonl, seed_everything

METHODS = ("revis", "vti", "grains", "vti-authors", "revis-official")


def direction_path(model, method):
    return DIRECTIONS / f"{model}-{method}.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--device-map")
    args = p.parse_args()

    fit = read_jsonl(SETS / "dev_fit.jsonl")
    calibration = read_jsonl(SETS / "dev_calibration.jsonl")
    seed_everything(42)
    DIRECTIONS.mkdir(parents=True, exist_ok=True)

    if args.method == "grains":
        adapter = load(args.model, dtype="float32", attn="eager", device_map=args.device_map)
        direction = grains.extract(adapter, (fit + calibration)[:50])
    elif args.method == "vti-authors":
        vti.download_demo_images()
        with fp32_attention_scores("qwen2_5_vl"):
            adapter = load(args.model, dtype="float16", attn="eager")
            direction = vti.extract_authors(adapter, DIRECTIONS)
    elif args.method == "revis-official":
        adapter = load(args.model, attn="sdpa")
        direction = revis.official_calibrate(adapter, revis.official_vectors(adapter))
    elif args.method == "revis":
        adapter = load(args.model, attn="eager")
        direction = revis.extract(adapter, fit, calibration)
    else:
        adapter = load(args.model, attn="eager")
        direction = vti.extract(adapter, fit)
    torch.save(direction, direction_path(args.model, args.method))


if __name__ == "__main__":
    main()
