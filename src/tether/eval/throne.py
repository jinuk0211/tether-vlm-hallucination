import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from tether.paths import COCO, JUDGES, UPSTREAM
from tether.utils import read_jsonl, write_json

THRONE = UPSTREAM / "throne/throne"
ANNOTATIONS = COCO / "annotations/instances_val2017.json"
PROMPT = "Describe this image in detail."
SETTINGS = {"main": (("base", "large"), 6), "xl": (("base", "large", "xl"), 9)}
METRICS = [(s, q) for s in ("ALL", "CLS") for q in ("P", "R", "F1", "F0.5")]


def judge_model(which):
    return str(JUDGES / f"flan-t5-{which}")


def judged(rundir, tag, which):
    return Path(rundir) / "throne" / tag / which / f"evaluations_{judge_model(which).replace('/', '_')}.json"


def judge(rundir, tag, which):
    rows = read_jsonl(Path(rundir) / f"{tag}-0-5000.jsonl")
    folder = Path(rundir) / "throne" / tag / which
    folder.mkdir(parents=True, exist_ok=True)
    responses = {
        "prompts": [PROMPT],
        "responses": [[0, r["metadata"]["image_id"], r["prediction"]] for r in rows],
    }
    (folder / "responses.json").write_text(json.dumps(responses))
    env = {**os.environ, "LOCAL_RANK": "0", "WORLD_SIZE": "1", "RANK": "0"}
    subprocess.run(
        [
            os.environ.get("THRONE_PYTHON", sys.executable),
            "throne_aqa_evaluation.py",
            "--coco_file",
            str(ANNOTATIONS),
            "--response_file",
            str(folder / "responses.json"),
            "--evaluator_model_path",
            judge_model(which),
            "--per_device_batch_size",
            "32",
            "--save_path",
            str(folder / "evaluations.json"),
        ],
        cwd=THRONE,
        env=env,
        check=True,
    )


class Scorer:
    def __init__(self, refs):
        sys.path.insert(0, str(THRONE))
        import pycocotools.coco
        import throne_score_aqa

        self.aqa = throne_score_aqa
        with contextlib.redirect_stdout(io.StringIO()):
            coco = pycocotools.coco.COCO(str(ANNOTATIONS))
        cats = sorted(coco.cats.values(), key=lambda c: c["id"])
        self.names = {i: c["name"] for i, c in enumerate(cats)}
        self.index = {c["name"]: i for i, c in enumerate(cats)}
        names = {c["id"]: c["name"] for c in cats}
        self.keys = [r["metadata"]["image_id"] for r in refs]
        self.gt = {
            k: {names[a["category_id"]] for a in coco.loadAnns(coco.getAnnIds(imgIds=[k]))} for k in self.keys
        }

    def matrices(self, rundir, tag, judges, threshold):
        preds = []
        for j in judges:
            d = json.loads(judged(rundir, tag, j).read_text())
            preds.append(
                {
                    int(k): self.aqa.process_aqa_instance_preds(
                        v, num_categories=len(self.names), sorted_categories_idx_to_name=self.names
                    )
                    for k, v in d.items()
                }
            )
        collapsed, disputed = self.aqa.majority_vote_collapsing(
            {k: [p[k] for p in preds] for k in preds[0]}, voting_threshold=threshold
        )
        P, C, G = (np.zeros((len(self.keys), len(self.names))) for _ in range(3))
        for i, k in enumerate(self.keys):
            skip = set(disputed[k])
            for g in self.gt[k] - skip:
                G[i, self.index[g]] = 1
            for p in set(collapsed[k]) - skip:
                P[i, self.index[p]] = 1
                C[i, self.index[p]] = p in self.gt[k]
        return P, C, G


def f_scores(p, r):
    with np.errstate(invalid="ignore", divide="ignore"):
        return 2 * p * r / (p + r), 1.25 * p * r / (0.25 * p + r)


def metrics(W, P, C, G):
    n_pred, n_hit, n_gt = P.sum(1), C.sum(1), G.sum(1)
    ok = (n_pred > 0) & (n_gt > 0)
    p = np.where(ok, n_hit / np.maximum(n_pred, 1), 0.0)
    r = np.where(ok, n_hit / np.maximum(n_gt, 1), 0.0)
    total = W.sum(1)
    p_all, r_all = (W @ p) / total, (W @ r) / total
    with np.errstate(invalid="ignore", divide="ignore"):
        p_cls = np.nanmean((W @ C) / (W @ P), 1)
        r_cls = np.nanmean((W @ C) / (W @ G), 1)
    out = []
    for prec, rec in ((p_all, r_all), (p_cls, r_cls)):
        f1, f05 = f_scores(prec, rec)
        out += [prec, rec, f1, f05]
    return [100 * x for x in out]


def table(rundirs, rows, resamples=10000):
    results = {}
    for model, rundir in rundirs.items():
        refs = read_jsonl(Path(rundir) / "base-0-5000.jsonl")
        scorer = Scorer(refs)
        n = len(refs)
        rng = np.random.default_rng(0)
        W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(resamples)]).astype(
            float
        )
        one = np.ones((1, n))
        for setting, (judges, threshold) in SETTINGS.items():
            m = {tag: scorer.matrices(rundir, tag, judges, threshold) for tag in rows}
            base = metrics(W, *m["base"])
            for tag in rows:
                v = [float(x[0]) for x in metrics(one, *m[tag])]
                rec = {"metrics": dict(zip([f"{s}_{q}" for s, q in METRICS], v))}
                if tag != "base":
                    ci = [
                        (float(np.quantile(a - b, 0.025)), float(np.quantile(a - b, 0.975)))
                        for a, b in zip(metrics(W, *m[tag]), base)
                    ]
                    rec["ci"] = dict(zip([f"{s}_{q}" for s, q in METRICS], ci))
                tokens = np.mean(
                    [r["generated_tokens"] for r in read_jsonl(Path(rundir) / f"{tag}-0-5000.jsonl")]
                )
                rec["tokens"] = float(tokens)
                results.setdefault(setting, {}).setdefault(model, {})[tag] = rec
                print(setting, model, f"{tag:10s}", " ".join(f"{x:5.1f}" for x in v), f"{tokens:4.0f}")
    return results


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    j = sub.add_parser("judge")
    j.add_argument("rundir", type=Path)
    j.add_argument("tag")
    j.add_argument("judge", choices=["base", "large", "xl"])
    t = sub.add_parser("table")
    t.add_argument("--qwen", type=Path, required=True)
    t.add_argument("--gemma", type=Path, required=True)
    t.add_argument("--tags", nargs="+", default=["base", "revis", "rudder", "grains", "tether"])
    t.add_argument("--resamples", type=int, default=10000)
    t.add_argument("--json", type=Path)
    args = p.parse_args()
    if args.command == "judge":
        judge(args.rundir, args.tag, args.judge)
    else:
        results = table({"qwen": args.qwen, "gemma": args.gemma}, args.tags, args.resamples)
        if args.json:
            write_json(args.json, results)


if __name__ == "__main__":
    main()
