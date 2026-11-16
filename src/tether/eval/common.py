import importlib.util
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def parse_yes_no(text):
    match = re.match(
        r"^\s*(?:answer\s*:\s*)?(yes|no)(?=\s|[.!?,;:]|$)", text, flags=re.IGNORECASE
    )
    return match.group(1).lower() if match else None


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _validate_predictions(records):
    if not records:
        raise ValueError("No predictions provided")
    keys = set()
    for record in records:
        missing = {"id", "group_id", "task", "prediction"} - record.keys()
        if missing:
            raise ValueError(f"Prediction is missing fields: {sorted(missing)}")
        key = (str(record.get("method", "base")), str(record["id"]))
        if key in keys:
            raise ValueError(f"Duplicate prediction: {key}")
        keys.add(key)
        if not isinstance(record["prediction"], str):
            raise ValueError(f"{record['id']}: prediction must be a string")


def evaluate_predictions(records):
    _validate_predictions(records)
    tasks = {row["task"] for row in records}
    methods = {row.get("method", "base") for row in records}
    if len(tasks) != 1 or len(methods) != 1:
        raise ValueError("evaluate_predictions requires one task and method")
    task = next(iter(tasks))
    metrics = {
        "n": len(records),
        "n_image_groups": len({row["group_id"] for row in records}),
        "mean_words": sum(len(row["prediction"].split()) for row in records) / len(records),
    }
    durations = [row.get("elapsed_seconds") for row in records]
    if all(isinstance(t, (int, float)) and math.isfinite(t) and t >= 0 for t in durations):
        metrics["mean_seconds"] = sum(durations) / len(durations)
        tokens = [row.get("generated_tokens") for row in records]
        if all(isinstance(n, int) and n >= 0 for n in tokens):
            metrics["tokens_per_second"] = _ratio(sum(tokens), sum(durations))
            metrics["mean_generated_tokens"] = sum(tokens) / len(tokens)
    if task == "caption":
        has_chair = ["chair" in row for row in records]
        if any(has_chair) and not all(has_chair):
            raise ValueError("Partial CHAIR coverage: every caption needs scorer output")
        if all(has_chair):
            chairs = [row["chair"] for row in records]
            metrics.update({
                "CHAIR_s": sum(row["hallucinated_mentions"] > 0 for row in chairs) / len(chairs),
                "CHAIR_i": _ratio(
                    sum(row["hallucinated_mentions"] for row in chairs),
                    sum(row["generated_mentions"] for row in chairs),
                ),
                "object_recall": _ratio(
                    sum(row["correct_distinct_objects"] for row in chairs),
                    sum(row["ground_truth_objects"] for row in chairs),
                ),
                "mean_correct_distinct_objects": sum(
                    row["correct_distinct_objects"] for row in chairs
                ) / len(chairs),
            })
        else:
            metrics["chair_status"] = "not_scored"
        return metrics
    if task == "mmhal":
        has_score = ["mmhal_score" in row for row in records]
        if any(has_score) and not all(has_score):
            raise ValueError("Partial MMHal coverage: every response needs a judge rating")
        if not all(has_score):
            metrics["mmhal_status"] = "not_scored"
            return metrics
        scores = [row["mmhal_score"] for row in records]
        if any(type(score) is not int or score not in range(7) for score in scores):
            raise ValueError("MMHal ratings must be integers from 0 through 6")
        by_type = defaultdict(list)
        for row, score in zip(records, scores):
            category = row.get("metadata", {}).get("question_type")
            if not isinstance(category, str) or not category:
                raise ValueError("MMHal records require metadata.question_type")
            by_type[category].append(score)
        metrics.update({
            "mmhal_status": "scored",
            "mmhal_average_score": sum(scores) / len(scores),
            "mmhal_hallucination_rate": sum(score < 3 for score in scores) / len(scores),
            "mmhal_type_scores": {
                category: sum(values) / len(values) for category, values in sorted(by_type.items())
            },
        })
        return metrics
    if any(row.get("label") not in {"yes", "no"} for row in records):
        raise ValueError("Binary-task evaluation requires yes/no ground truth")
    parsed = [parse_yes_no(row["prediction"]) for row in records]
    tp = sum(p == "yes" and r["label"] == "yes" for p, r in zip(parsed, records))
    fp = sum(p == "yes" and r["label"] == "no" for p, r in zip(parsed, records))
    positives = sum(r["label"] == "yes" for r in records)
    correct = [p == r["label"] for p, r in zip(parsed, records)]
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, positives)
    metrics.update({
        "accuracy": sum(correct) / len(records),
        "precision": precision,
        "recall": recall,
        "F1": _ratio(2 * tp, 2 * tp + fp + positives - tp),
        "yes_ratio": parsed.count("yes") / len(records),
        "invalid_count": parsed.count(None),
        "invalid_ratio": parsed.count(None) / len(records),
        "positive_accuracy": recall,
        "negative_accuracy": _ratio(
            sum(p == "no" and r["label"] == "no" for p, r in zip(parsed, records)),
            len(records) - positives,
        ),
    })
    if task == "mme":
        grouped = defaultdict(list)
        for row, is_correct in zip(records, correct):
            category = row.get("metadata", {}).get("category")
            if not category:
                raise ValueError("MME records require metadata.category")
            grouped[(category, row["group_id"])].append(is_correct)
        if any(len(values) != 2 for values in grouped.values()):
            raise ValueError("MME acc+ requires exactly two questions per category/image")
        category_rows = defaultdict(list)
        category_pairs = defaultdict(list)
        for (category, _), values in grouped.items():
            category_rows[category].extend(values)
            category_pairs[category].append(all(values))
        scores = {
            category: 100 * (
                sum(values) / len(values)
                + sum(category_pairs[category]) / len(category_pairs[category])
            )
            for category, values in category_rows.items()
        }
        metrics["mme_category_scores"] = scores
        metrics["mme_score_sum"] = sum(scores.values())
        metrics["mme_protocol"] = "accuracy + paired accuracy, strict leading yes/no parser"
    return metrics


def _load_external_module(path, name):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _caption_image_id(record):
    value = record.get("metadata", {}).get("coco_image_id")
    if value is None and str(record["group_id"]).startswith("coco:"):
        value = str(record["group_id"]).split(":", 1)[1]
    if value is None:
        raise ValueError(f"{record['id']}: missing metadata.coco_image_id")
    return int(value)


def attach_chair_scores(records, output):
    _validate_predictions(records)
    if isinstance(output, (str, Path)):
        output = json.loads(Path(output).read_text(encoding="utf-8"))
    sentences = output["sentences"]
    scored = {int(row["image_id"]): row for row in sentences}
    ids = [_caption_image_id(row) for row in records]
    if len(set(ids)) != len(ids) or len(scored) != len(sentences):
        raise ValueError("CHAIR input/output must contain one caption per image")
    if set(ids) != set(scored):
        raise ValueError("CHAIR image coverage does not match predictions")
    enriched = []
    for record in records:
        row = scored[_caption_image_id(record)]
        if record["prediction"] != row["caption"]:
            raise ValueError(f"{record['id']}: CHAIR caption differs from actual prediction")
        generated = row["mscoco_generated_words"]
        ground_truth = set(row["mscoco_gt_words"])
        hallucinated = row["mscoco_hallucinated_words"]
        enriched.append({
            **dict(record),
            "chair": {
                "hallucinated_mentions": len(hallucinated),
                "generated_mentions": len(generated),
                "correct_distinct_objects": len(set(generated) & ground_truth),
                "ground_truth_objects": len(ground_truth),
                "source": output.get("scorer_provenance", "external CHAIR per-sentence output"),
            },
        })
    return enriched
