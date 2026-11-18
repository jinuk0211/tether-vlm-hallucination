import argparse
import json
import re
from collections import Counter

import numpy as np

from tether.analysis.jacobian.common import object_vocabulary
from tether.analysis.jacobian.probe import KINDS, OUT_DIR

COLORS = {
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
    "gray",
    "grey",
    "silver",
    "gold",
    "golden",
    "beige",
    "tan",
    "teal",
    "navy",
    "maroon",
    "dark",
    "light",
    "bright",
    "colorful",
    "colored",
}
FRAME = {
    "this",
    "image",
    "images",
    "photo",
    "photos",
    "photograph",
    "picture",
    "pictures",
    "shows",
    "show",
    "showing",
    "depicts",
    "depicting",
    "features",
    "featuring",
    "showcases",
    "captures",
    "capturing",
    "displays",
    "scene",
    "view",
    "here",
    "shot",
    "snapshot",
    "close",
    "detailed",
    "description",
    "describe",
}
FUNCTION = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "from",
    "by",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "its",
    "there",
    "that",
    "these",
    "those",
    "which",
    "who",
    "what",
    "while",
    "where",
    "when",
    "into",
    "onto",
    "over",
    "under",
    "near",
    "next",
    "behind",
    "above",
    "below",
    "between",
    "has",
    "have",
    "had",
    "not",
    "no",
    "some",
    "two",
    "one",
    "three",
    "several",
    "both",
    "each",
    "all",
    "very",
    "also",
}
ORDER = [
    "COCO object",
    "other object",
    "color",
    "caption frame",
    "function",
    "format",
    "non-Latin",
    "no-space",
    "other word",
]
LATIN = re.compile(r"^[A-Za-zÀ-ɏ'’\-]+$")


def kind_of(text, token, coco, physical):
    stripped = text.strip()
    word = stripped.lower()
    if token in coco:
        return "COCO object"
    if token in physical:
        return "other object"
    if not any(ch.isalnum() for ch in stripped):
        return "format"
    if word in COLORS:
        return "color"
    if word in FRAME:
        return "caption frame"
    if word in FUNCTION:
        return "function"
    if not LATIN.match(stripped) and not stripped.isdigit():
        return "non-Latin" if any(ord(ch) > 0x24F for ch in stripped) else "other word"
    if not text.startswith(" "):
        return "no-space"
    return "other word"


def shares(ids, kinds):
    counts = Counter(kinds[int(t)] for t in ids.ravel())
    total = ids.size
    return {k: counts.get(k, 0) / total for k in ORDER}


def layer_table(data, vocab, kinds, layers, place):
    final = data["final_top_ids"]
    rows = []
    for i, layer in enumerate(layers):
        entry = {"layer": layer}
        for k, kind in enumerate(KINDS):
            tops = data["top_ids"][:, i, k, place]
            gray = set(data["gray_top_ids"][i, k, place].tolist())
            nxt = data["top_ids"][:, i + 1, k, place] if i + 1 < len(layers) else None
            frequent = Counter(tops.ravel().tolist()).most_common(20)
            entry[kind] = {
                "kinds": shares(tops, kinds),
                "not_in_gray": float(np.mean([[t not in gray for t in row] for row in tops])),
                "overlap_with_final": float(
                    np.mean([len(set(a) & set(b)) / len(a) for a, b in zip(tops.tolist(), final.tolist())])
                ),
                "overlap_with_next_layer": (
                    float(
                        np.mean([len(set(a) & set(b)) / len(a) for a, b in zip(tops.tolist(), nxt.tolist())])
                    )
                    if nxt is not None
                    else None
                ),
                "frequent": [[vocab[str(t)], kinds[int(t)], round(c / len(tops), 3)] for t, c in frequent],
                "gray_top": [[vocab[str(t)], kinds[int(t)]] for t in data["gray_top_ids"][i, k, place][:20]],
            }
        rows.append(entry)
    return rows


def image_token_table(data, vocab, kinds, layers, present_tokens):
    rows = []
    for i, layer in enumerate(layers):
        entry = {"layer": layer}
        for k, kind in enumerate(KINDS):
            tops = data["span_top_ids"][:, i, k]
            top1 = tops[:, :, 0]
            own = np.mean([[int(t) in present_tokens[n] for t in top1[n]] for n in range(len(top1))])
            frequent = Counter(top1.ravel().tolist()).most_common(15)
            entry[kind] = {
                "kinds_top20": shares(tops, kinds),
                "kinds_top1": shares(top1, kinds),
                "top1_is_object_of_this_image": float(own),
                "frequent_top1": [
                    [vocab[str(t)], kinds[int(t)], round(c / top1.size, 4)] for t, c in frequent
                ],
            }
        rows.append(entry)
    return rows


def all_layers_example(data, vocab, kinds, layers, index):
    return {
        "layers": [
            {
                "layer": layer,
                **{
                    kind: [[vocab[str(t)], kinds[int(t)]] for t in data["top_ids"][index, i, k, 0]]
                    for k, kind in enumerate(KINDS)
                },
            }
            for i, layer in enumerate(layers)
        ],
        "model_output": [[vocab[str(t)], kinds[int(t)]] for t in data["final_top_ids"][index]],
    }


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    with np.load(OUT_DIR / "probe.npz") as loaded:
        data = {k: loaded[k] for k in loaded.files}
    meta = json.loads((OUT_DIR / "meta.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    physical = set(object_vocabulary(tokenizer, OUT_DIR / "vobj.json"))
    coco = {t for ids in meta["node_ids"].values() for t in ids}
    vocab = meta["vocab"]
    kinds = {int(t): kind_of(text, int(t), coco, physical) for t, text in vocab.items()}
    layers = meta["layers"]
    present_tokens = [
        {t for node in r["ground_truth"] for t in meta["node_ids"].get(node, [])} for r in meta["rows"]
    ]
    picked = next(n for n, r in enumerate(meta["rows"]) if len(r["ground_truth"]) >= 3)
    out = {
        "n_images": len(meta["rows"]),
        "layers": layers,
        "kinds": ORDER,
        "v_obj_size": len(physical),
        "answer": layer_table(data, vocab, kinds, layers, 0),
        "post_image": layer_table(data, vocab, kinds, layers, 1),
        "image_tokens": image_token_table(data, vocab, kinds, layers, present_tokens),
        "final_kinds": shares(data["final_top_ids"], kinds),
        "example": {
            "image_id": meta["rows"][picked]["image_id"],
            "objects": meta["rows"][picked]["ground_truth"],
            **all_layers_example(data, vocab, kinds, layers, picked),
        },
    }
    (OUT_DIR / "layers.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
