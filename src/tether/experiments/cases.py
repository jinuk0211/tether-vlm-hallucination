import argparse
import json
from pathlib import Path

from tether.eval.chair import scorer, sentences
from tether.paths import RUNS, SETS
from tether.utils import read_jsonl

TAGS = ("base", "rudder", "grains", "vti-released", "tether")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rundir", type=Path, default=RUNS / "chair_test500/qwen")
    p.add_argument("--selfcheck", type=Path)
    p.add_argument("--n", type=int, default=4)
    args = p.parse_args()

    refs = read_jsonl(SETS / "chair_test500.jsonl")
    evaluator = scorer()
    found = {}
    for tag in TAGS:
        rows = {r["id"]: r for r in read_jsonl(args.rundir / f"{tag}-0-{len(refs)}.jsonl")}
        s = {
            x["image_id"]: x
            for x in sentences(evaluator, refs, {i: r["prediction"] for i, r in rows.items()})["sentences"]
        }
        found[tag] = (rows, s)
    margins = {}
    if args.selfcheck:
        for r in read_jsonl(args.selfcheck):
            margins[(r["image_id"], r["node"])] = round(r["margin"], 2)
    base = found["base"][1]
    picked = [r for r in refs if base[r["metadata"]["image_id"]]["mscoco_hallucinated_words"]][: args.n]
    for r in picked:
        image_id = r["metadata"]["image_id"]
        case = {
            "id": r["id"],
            "image": r["image"],
            "gt_objects": sorted(set(base[image_id]["mscoco_gt_words"])),
            "methods": {},
        }
        for tag, (rows, s) in found.items():
            x = s[image_id]
            hallucinated = sorted({h[1] for h in x["mscoco_hallucinated_words"]})
            case["methods"][tag] = {
                "caption": rows[r["id"]]["prediction"],
                "hallucinated": {h: margins.get((image_id, h)) for h in hallucinated},
                "correct": sorted(set(x["mscoco_gt_words"]) & set(x["mscoco_generated_words"])),
            }
        print(json.dumps(case, indent=1))


if __name__ == "__main__":
    main()
