import argparse
import pickle

import numpy as np
from PIL import Image

from tether.analysis.jacobian.common import CHAIR_STATE, IMAGE_SIZE, SPEC, load_chair
from tether.analysis.jacobian.probe import PILOT_IMAGES
from tether.paths import COCO, SETS
from tether.utils import read_jsonl, write_jsonl


def main():
    from pycocotools.coco import COCO as Annotations

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    args = p.parse_args()

    api = Annotations(str(COCO / "annotations/instances_val2014.json"))
    scorer = load_chair(COCO / "annotations", CHAIR_STATE)
    cell = IMAGE_SIZE // SPEC.grid
    rows, masks = [], {}
    for r in read_jsonl(SETS / "chair_test500.jsonl")[: args.n]:
        image_id = r["metadata"]["image_id"]
        union = {}
        for a in api.loadAnns(api.getAnnIds(imgIds=[image_id], iscrowd=None)):
            name = api.loadCats(a["category_id"])[0]["name"]
            m = Image.fromarray(api.annToMask(a) * 255).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST
            )
            union[name] = union.get(name, False) | (np.asarray(m) > 127)
        masks[image_id] = {
            n: m.reshape(SPEC.grid, cell, SPEC.grid, cell).mean((1, 3)) for n, m in union.items()
        }
        rows.append(
            {
                "image_id": image_id,
                "path": r["image"],
                "with_caption": True,
                "ground_truth": sorted(scorer.imid_to_objects[image_id]),
            }
        )
    PILOT_IMAGES.mkdir(parents=True, exist_ok=True)
    write_jsonl(PILOT_IMAGES / "manifest.jsonl", rows)
    (PILOT_IMAGES / "object_masks.pkl").write_bytes(pickle.dumps(masks))


if __name__ == "__main__":
    main()
