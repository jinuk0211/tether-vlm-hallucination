import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

SPLITS = {"discovery", "fit", "tune", "test"}
TASKS = {"pope", "caption", "causal", "mme", "mmhal"}


def _read_json_records(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    if not text.strip():
        return []
    if text.lstrip().startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError(f"{path}: expected a JSON array or JSONL objects")
    return records


def read_manifest(path):
    path = Path(path).resolve()
    records = _read_json_records(path)
    for record in records:
        for key in ("image", "object_masked_image", "control_image"):
            if key in record:
                image = Path(record[key])
                record[key] = str(
                    (path.parent / image).resolve() if not image.is_absolute() else image
                )
    validate_disjoint(records)
    return records


def validate_disjoint(manifests, *, check_files=True):
    records = []
    for item in manifests:
        if isinstance(item, Mapping):
            records.append(item)
        elif isinstance(item, (str, Path)):
            records.extend(read_manifest(item))
        else:
            records.extend(item)
    seen_ids = set()
    groups = {}
    files = {}
    split_counts = defaultdict(int)
    for record in records:
        missing = {"id", "group_id", "image", "prompt", "task", "split"} - record.keys()
        if missing:
            raise ValueError(f"Record is missing fields: {sorted(missing)}")
        identifier, group, split = str(record["id"]), str(record["group_id"]), record["split"]
        if not identifier or not group:
            raise ValueError("id and group_id must be nonempty")
        if identifier in seen_ids:
            raise ValueError(f"Duplicate record id: {identifier}")
        seen_ids.add(identifier)
        if split not in SPLITS or record["task"] not in TASKS:
            raise ValueError(f"{identifier}: unsupported split/task")
        if not isinstance(record["prompt"], str) or not record["prompt"].strip():
            raise ValueError(f"{identifier}: prompt must be a nonempty string")
        if record.get("label") not in (None, "yes", "no"):
            raise ValueError(f"{identifier}: label must be yes/no or absent")
        if group in groups and groups[group] != split:
            raise ValueError(f"Image-group leakage: {group} in {groups[group]} and {split}")
        groups[group] = split
        for field in ("image", "object_masked_image", "control_image"):
            if field not in record:
                continue
            path = Path(record[field]).resolve()
            if check_files and not path.is_file():
                raise FileNotFoundError(f"{identifier}: missing {field}: {path}")
            key = str(path).casefold() if os.name == "nt" else str(path)
            if key in files and files[key] != (group, split):
                raise ValueError(f"Image file reused under different groups/splits: {path}")
            files[key] = (group, split)
        split_counts[split] += 1
    return dict(split_counts)


def _local_image(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path escapes image_root: {name}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def prepare_mmhal(input_path, image_root, split="test"):
    records = []
    for index, row in enumerate(_read_json_records(input_path)):
        required = {"image_id", "image_src", "image_content", "question",
                    "gt_answer", "question_type", "question_topic"}
        if required - row.keys():
            raise ValueError(f"MMHal entry {index} missing {sorted(required - row.keys())}")
        image_name = urlsplit(row["image_src"]).path.rsplit("/", 1)[-1]
        if not image_name:
            raise ValueError(f"MMHal entry {index} has no image filename")
        records.append({
            "id": f"mmhal:{index:03d}",
            "group_id": f"mmhal:{row['image_id']}",
            "image": _local_image(image_root, image_name),
            "prompt": row["question"],
            "task": "mmhal", "split": split,
            "metadata": {
                "mmhal_index": index,
                "image_id": row["image_id"], "image_src": row["image_src"],
                "image_content": row["image_content"], "gt_answer": row["gt_answer"],
                "question_type": row["question_type"],
                "question_topic": row["question_topic"],
            },
        })
    if not records:
        raise ValueError("MMHal template is empty")
    validate_disjoint(records)
    return records
