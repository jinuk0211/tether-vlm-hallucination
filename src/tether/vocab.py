import argparse
import json
from functools import lru_cache

from tether.paths import VOBJ

ROOTS = (
    "object.n.01",
    "matter.n.03",
    "thing.n.12",
    "body_of_water.n.01",
    "vegetation.n.01",
    "cloud.n.02",
    "furniture.n.01",
)

SCENE = set("""
image picture photo photograph scene view shot frame background foreground setting side top bottom front back left right
center centre middle edge corner part piece area region section portion spot distance space place location position
surroundings environment atmosphere vicinity point end half close-up closeup detail element item thing object individual
one rest line row set group variety number lot couple pair bunch array collection series kind type sort mix assortment
range amount light lighting color colour feature ambiance ambience presence layer shadow texture pattern design style tone
hue shade glow reflection mood vibe theme composition perspective angle sense contrast focus depth blur facing interior
exterior landscape surface structure material display arrangement decor decoration blue red green white black yellow brown
gray grey pink purple gold silver beige tan navy maroon colors colours safety marking home
""".split())

FUNCTION = set("""
a an the this that these those there here it its it's they them their theirs he him his she her hers we us our you your i
me my mine who whom whose which what where when why how and or but nor so yet for of at by in on to from with without
within into onto upon over under above below between among through during before after about around against along across
behind beyond near off out up down as than then also too very just only even still not no yes all any both each few more
most other some such own same can could will would shall should may might must do does did done doing be is am are was
were been being have has had having get gets got make makes made appear appears seem seems look looks one ones another
while if because although though whether either neither like unlike per via etc overall perhaps possibly likely
""".split())

COLOR_WORDS = {"orange", "cream", "creams", "ivory", "violet", "coral", "bronze", "copper"}


@lru_cache(maxsize=None)
def wordnet():
    from nltk.corpus import wordnet

    return wordnet


@lru_cache(maxsize=None)
def roots():
    return {wordnet().synset(name) for name in ROOTS}


def noun_dominant(word):
    key = word.replace(" ", "_")
    noun = other = 0
    for s in wordnet().synsets(key):
        count = sum(l.count() for l in s.lemmas() if l.name().lower() == key)
        if s.pos() == "n":
            noun += count
        else:
            other += count
    return noun >= other


@lru_cache(maxsize=None)
def lemma(word):
    if word in FUNCTION:
        return word
    return wordnet().morphy(word, wordnet().NOUN) or word


@lru_cache(maxsize=None)
def physical(word):
    if word in SCENE or word in FUNCTION or len(word) < 2:
        return False
    senses = wordnet().synsets(word.replace(" ", "_"), pos=wordnet().NOUN)[:2]
    return any(r in path for s in senses for path in s.hypernym_paths() for r in roots())


def noun_count(word):
    return sum(
        l.count()
        for s in wordnet().synsets(word, pos=wordnet().NOUN)
        for l in s.lemmas()
        if l.name().lower() == word
    )


def object_tokens(tokenizer):
    ids = []
    for i in range(len(tokenizer)):
        piece = tokenizer.convert_ids_to_tokens(i)
        if not piece or piece[0] not in ("Ġ", "▁"):
            continue
        word = piece[1:]
        if not (word.isalpha() and word.islower() and len(word) >= 3):
            continue
        base = lemma(word)
        if word in (base, base + "s", base + "es"):
            if physical(base) and noun_dominant(base) and noun_count(base) >= 1:
                ids.append(i)
    return ids


def vobj_path(model, colors=True):
    return VOBJ / (f"{model}.json" if colors else f"{model}-nocolor.json")


def load_vobj(model, colors=True):
    return json.loads(vobj_path(model, colors).read_text())


def main():
    from transformers import AutoProcessor

    from tether.models import MODELS, model_config

    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS))
    args = p.parse_args()
    VOBJ.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        config = model_config(model)
        tok = AutoProcessor.from_pretrained(config["checkpoint"], revision=config["revision"]).tokenizer
        ids = object_tokens(tok)
        kept = [i for i in ids if tok.decode([i]).strip().lower() not in COLOR_WORDS]
        vobj_path(model).write_text(json.dumps(ids))
        vobj_path(model, False).write_text(json.dumps(kept))
        print(model, len(ids), len(kept))


if __name__ == "__main__":
    main()
