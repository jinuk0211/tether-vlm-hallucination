import argparse
import contextlib
import importlib.util
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

from tether.paths import COCO, OUT, UPSTREAM
from tether.utils import read_jsonl, write_json


def chair_module():
    if "chair" not in sys.modules:
        spec = importlib.util.spec_from_file_location("chair", UPSTREAM / "chair/chair.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["chair"] = module
    return sys.modules["chair"]


def scorer():
    module = chair_module()
    cache = OUT / "chair.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
        evaluator = module.CHAIR(str(COCO / "annotations"))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(evaluator))
    return evaluator


def sentences(evaluator, refs, captions):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([{"image_id": r["metadata"]["image_id"], "caption": captions[r["id"]]} for r in refs], f)
    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
        out = evaluator.compute_chair(f.name, "image_id", "caption")
    os.unlink(f.name)
    return out


def score(evaluator, refs, runs, resamples=10000):
    ids = [r["metadata"]["image_id"] for r in refs]
    results, per = {}, {}
    for name, rows in runs.items():
        d = sentences(evaluator, refs, {i: r["prediction"] for i, r in rows.items()})
        m = d["overall_metrics"]
        s = {x["image_id"]: x for x in d["sentences"]}
        per[name] = {
            "hal": np.array([len(s[i]["mscoco_hallucinated_words"]) > 0 for i in ids], float),
            "gt": np.array([len(set(s[i]["mscoco_gt_words"])) for i in ids], float),
            "hit": np.array(
                [len(set(s[i]["mscoco_gt_words"]) & set(s[i]["mscoco_generated_words"])) for i in ids], float
            ),
        }
        precision, recall = 1 - m["CHAIRi"], m["Recall"]
        results[name] = {
            "CHAIR_s": 100 * m["CHAIRs"],
            "CHAIR_i": 100 * m["CHAIRi"],
            "recall": 100 * recall,
            "F1": 100 * 2 * precision * recall / (precision + recall),
            "tokens": float(np.mean([r["generated_tokens"] for r in rows.values()])),
        }
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(ids), len(ids)) for _ in range(resamples)]
    b = per["base"]
    for name, x in per.items():
        if name == "base":
            continue
        ds = [100 * (x["hal"][k].mean() - b["hal"][k].mean()) for k in draws]
        dr = [
            100 * (x["hit"][k].sum() / x["gt"][k].sum() - b["hit"][k].sum() / b["gt"][k].sum()) for k in draws
        ]
        results[name]["ci"] = {
            "CHAIR_s": [float(np.quantile(ds, q)) for q in (0.025, 0.975)],
            "recall": [float(np.quantile(dr, q)) for q in (0.025, 0.975)],
        }
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("rundir", type=Path)
    p.add_argument("--base", default="base")
    p.add_argument("--resamples", type=int, default=10000)
    args = p.parse_args()

    refs = read_jsonl(args.data)
    n = len(refs)
    runs = {}
    for f in sorted(args.rundir.glob(f"*-0-{n}.jsonl")):
        rows = {r["id"]: r for r in read_jsonl(f)}
        if all(r["id"] in rows for r in refs):
            runs[f.name[: -len(f"-0-{n}.jsonl")]] = rows
    runs = {"base": runs.pop(args.base), **runs}
    results = score(scorer(), refs, runs, args.resamples)
    write_json(args.rundir / "score.json", results)
    for name, v in results.items():
        ci = " ".join(f"d{k}[{a:+.1f},{b:+.1f}]" for k, (a, b) in v.get("ci", {}).items())
        print(
            f"{name:24s} CHAIR_s {v['CHAIR_s']:5.1f} CHAIR_i {v['CHAIR_i']:5.2f} recall {v['recall']:5.1f} tok {v['tokens']:4.0f} {ci}"
        )


if __name__ == "__main__":
    main()
