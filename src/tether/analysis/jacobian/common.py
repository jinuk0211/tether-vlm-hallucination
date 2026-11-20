import contextlib
import importlib.util
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from tether.paths import OUT as OUT_ROOT
from tether.paths import UPSTREAM
from tether.vocab import object_tokens

JACOBIAN = OUT_ROOT / "audit" / "jacobian"
CHAIR_STATE = JACOBIAN / "chair_state.pkl"
CHAIR_PY = UPSTREAM / "chair" / "chair.py"

PROMPT = "Please describe this image in detail."
IMAGE_SIZE = 448
GRAY_RGB = (128, 128, 128)
MAX_NEW_TOKENS = 512
GRID = 14


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_image(path):
    with Image.open(path) as source:
        return (
            ImageOps.exif_transpose(source)
            .convert("RGB")
            .resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
        )


def gray_image():
    return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), GRAY_RGB)


def fast_patch_embedding(model):
    conv = model.model.visual.patch_embed.proj

    def forward(pixels):
        bias = None if conv.bias is None else conv.bias.float()
        output = torch.nn.functional.conv3d(
            pixels.float(), conv.weight.float(), bias, conv.stride, conv.padding, conv.dilation, conv.groups
        )
        return output.to(pixels.dtype)

    conv.forward = forward


def object_vocabulary(tokenizer, cache):
    if cache.exists():
        return json.loads(cache.read_text())
    ids = object_tokens(tokenizer)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(ids))
    return ids


def load_chair(annotations, cache):
    spec = importlib.util.spec_from_file_location("chair_scorer", CHAIR_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if cache.exists():
        scorer = module.CHAIR.__new__(module.CHAIR)
        scorer.__dict__.update(pickle.loads(cache.read_bytes()))
        return scorer
    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
        scorer = module.CHAIR(str(annotations))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(scorer.__dict__))
    return scorer


def mention_token_positions(tokenizer, ids, scorer):
    import nltk

    text = tokenizer.decode(ids, skip_special_tokens=True)
    low = text.lower()
    words = nltk.word_tokenize(low)
    offsets, cursor = [], 0
    for word in words:
        found = low.find(word, cursor)
        if found < 0:
            found = cursor
        offsets.append(found)
        cursor = found + len(word)
    mention_words, nodes, indices, _ = scorer.caption_to_words(text)
    cumulative = [0]
    for i in range(1, len(ids) + 1):
        cumulative.append(len(tokenizer.decode(ids[:i], skip_special_tokens=True)))
    mentions = []
    for node, word, index in zip(nodes, mention_words, indices):
        char = offsets[index] if index < len(offsets) else len(text)
        t = next((i for i in range(len(ids)) if cumulative[i] <= char < cumulative[i + 1]), len(ids) - 1)
        mentions.append((node, word, t))
    return mentions


def label_caption(tokenizer, ids, scorer, image_id):
    from pattern.en import (
        singularize,
    )

    ground_truth = set(scorer.imid_to_objects[image_id])
    mentions = mention_token_positions(tokenizer, ids, scorer)
    first, misaligned = [], 0
    seen = set()
    for node, word, t in mentions:
        if node in seen:
            continue
        seen.add(node)
        token_text = singularize(tokenizer.decode([ids[t]]).strip().lower())
        if not token_text or not (word.startswith(token_text[:3]) or token_text.startswith(word[:3])):
            misaligned += 1
            continue
        first.append(
            {
                "node": node,
                "word": word,
                "t": t,
                "token": int(ids[t]),
                "order": len(seen),
                "hallucinated": node not in ground_truth,
            }
        )
    hallucinated = [node for node, _, _ in mentions if node not in ground_truth]
    return {
        "first_mentions": first,
        "misaligned": misaligned,
        "n_mentions": len(mentions),
        "n_hallucinated_mentions": len(hallucinated),
        "hallucinated_caption": bool(hallucinated),
        "ground_truth": sorted(ground_truth),
        "mentioned": sorted(seen),
        "n_correct_distinct": len(seen & ground_truth),
    }


def canonical_tokens(tokenizer, scorer, vocabulary):
    allowed = set(vocabulary)
    synonyms = {}
    for word, node in scorer.inverse_synonym_dict.items():
        if node:
            synonyms.setdefault(node, []).append(word)
    assigned = {}
    taken = set()
    for node in sorted(synonyms, key=lambda name: (" " in name, name)):
        candidates = [node.split()[-1]] + [w for w in synonyms[node] if " " not in w and w != node]
        for word in candidates:
            token = tokenizer.convert_tokens_to_ids("Ġ" + word)
            if token in allowed and token not in taken:
                assigned[node] = int(token)
                taken.add(int(token))
                break
    return assigned


def tracked_tokens(canonical, first_mentions):
    return np.array(sorted(set(canonical.values()) | {m["token"] for m in first_mentions}), dtype=np.int64)
