import json
import random
from pathlib import Path

import numpy as np

from tether.paths import GRAY


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1), encoding="utf-8")


def seed_everything(seed=42):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_rows(data, output, tag, start=0, end=None):
    refs = read_jsonl(data)
    end = len(refs) if end is None else end
    path = Path(output) / f"{tag}-{start}-{end}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = len(read_jsonl(path)) if path.exists() else 0
    return refs[start:end][done:], path


def add_run_args(parser):
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)


def gray_image():
    if not GRAY.exists():
        from PIL import Image

        GRAY.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (448, 448), (128, 128, 128)).save(GRAY)
    return str(GRAY)
