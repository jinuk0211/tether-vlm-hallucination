import argparse
import json
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tether.paths import COCO, DATA, SETS
from tether.utils import write_jsonl

COCO_URL = "http://images.cocodataset.org"
COCO128 = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
COCO128_NAMES = (
    "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco128.yaml"
)
CAPTION = "Please describe this image in detail."


def download(url, path):
    path = Path(path)
    if path.exists() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(path)
    return path


def download_images(split, names):
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda n: download(f"{COCO_URL}/{split}/{n}", COCO / split / n), names))


def annotations(year):
    folder = COCO / "annotations"
    if not (folder / f"instances_val{year}.json").exists():
        archive = download(
            f"{COCO_URL}/annotations/annotations_trainval{year}.zip", COCO / f"annotations_trainval{year}.zip"
        )
        with zipfile.ZipFile(archive) as z:
            for m in z.namelist():
                if m.startswith("annotations/") and ("instances" in m or "captions" in m):
                    (folder / Path(m).name).write_bytes(z.read(m))
    return folder


def dev_pairs():
    import yaml

    root = DATA / "coco128"
    if not (root / "coco128").exists():
        with zipfile.ZipFile(download(COCO128, root / "coco128.zip")) as z:
            z.extractall(root)
    names = yaml.safe_load(download(COCO128_NAMES, root / "coco128.yaml").read_text())["names"]
    present, areas = {}, {}
    images = root / "coco128/images/train2017"
    for path in sorted((root / "coco128/labels/train2017").glob("*.txt")):
        image_id = int(path.stem)
        for line in path.read_text().splitlines():
            if line.strip():
                c, _, _, w, h = map(float, line.split())
                present.setdefault(image_id, set()).add(int(c))
                areas[image_id, int(c)] = areas.get((image_id, int(c)), 0) + w * h
    rows = [
        int(p.stem)
        for p in sorted(images.glob("*.jpg"))
        if int(p.stem) in present and len(present[int(p.stem)]) < len(names)
    ]
    rng = random.Random(42)
    rng.shuffle(rows)
    fit, calibration = [], []
    for i, image_id in enumerate(rows[:64]):
        yes = i % 2 == 0
        if yes:
            c = max(present[image_id], key=lambda k: (areas[image_id, k], -k))
        else:
            c = rng.choice(sorted(set(names) - present[image_id]))
        (fit if i < 32 else calibration).append(
            {
                "id": f"coco-dev:{image_id}",
                "image": str(images / f"{image_id:012d}.jpg"),
                "prompt": f"Is there a {names[c]} in this image? Answer yes or no.",
                "label": "yes" if yes else "no",
                "chosen": "Yes." if yes else "No.",
                "rejected": "No." if yes else "Yes.",
                "neutral": "I cannot determine this without an image.",
                "metadata": {"image_id": image_id},
            }
        )
    write_jsonl(SETS / "dev_fit.jsonl", fit)
    write_jsonl(SETS / "dev_calibration.jsonl", calibration)
    return {r["metadata"]["image_id"] for r in fit + calibration}


def chair_test(dev_ids):
    val = json.loads((annotations(2014) / "instances_val2014.json").read_text())
    candidates = sorted((r for r in val["images"] if r["id"] not in dev_ids), key=lambda r: r["id"])
    chosen = sorted(random.Random(42).sample(candidates, 500), key=lambda r: r["id"])
    download_images("val2014", [r["file_name"] for r in chosen])
    write_jsonl(
        SETS / "chair_test500.jsonl",
        [
            {
                "id": f"chair:{r['id']}",
                "image": str(COCO / "val2014" / r["file_name"]),
                "prompt": CAPTION,
                "max_new_tokens": 512,
                "metadata": {"image_id": r["id"]},
            }
            for r in chosen
        ],
    )


def chair_dev():
    train = json.loads((annotations(2014) / "instances_train2014.json").read_text())
    images = sorted(train["images"], key=lambda r: r["id"])
    random.Random(42).shuffle(images)
    chosen = images[:100]
    download_images("train2014", [r["file_name"] for r in chosen])
    write_jsonl(
        SETS / "chair_dev100.jsonl",
        [
            {
                "id": f"chairdev:{r['id']}",
                "image": str(COCO / "train2014" / r["file_name"]),
                "prompt": CAPTION,
                "max_new_tokens": 512,
                "metadata": {"image_id": r["id"]},
            }
            for r in chosen
        ],
    )


def throne():
    val = json.loads((annotations(2017) / "instances_val2017.json").read_text())
    ids = sorted(r["id"] for r in val["images"])
    download_images("val2017", [f"{i:012d}.jpg" for i in ids])
    write_jsonl(
        SETS / "throne.jsonl",
        [
            {
                "id": f"throne:{i}",
                "image": str(COCO / "val2017" / f"{i:012d}.jpg"),
                "prompt": "Describe this image in detail.",
                "max_new_tokens": 512,
                "metadata": {"image_id": i},
            }
            for i in ids
        ],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sets", nargs="*", default=["dev", "chair", "chair_dev", "throne"])
    args = p.parse_args()
    dev_ids = dev_pairs()
    for name in args.sets:
        if name == "chair":
            chair_test(dev_ids)
        elif name != "dev":
            globals()[name]()


if __name__ == "__main__":
    main()
