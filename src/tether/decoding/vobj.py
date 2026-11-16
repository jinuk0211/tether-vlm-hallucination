import glob,json
from transformers import AutoProcessor
from tether.decoding.common import ROOT, build_vobj
C=ROOT/'outputs/rbd-20261030/collect';HUB='/home/project-b/LJW/.cache/huggingface/hub'
SNAP={'qwen':'Qwen--Qwen2.5-VL-7B-Instruct','qwen3':'Qwen--Qwen3-VL-8B-Instruct','gemma':'google--gemma-4-E4B-it'}
DROP={'orange','cream','creams','ivory','violet','coral','bronze','copper'}
LOOKED_AT='orange gold golden silver lime olive peach plum rose salmon cream coral ivory amber tan navy maroon violet lavender chocolate wine berry bronze copper mint lemon sand'.split()
C.mkdir(parents=True,exist_ok=True)
for model,pat in SNAP.items():
    tok=AutoProcessor.from_pretrained(glob.glob(f'{HUB}/models--{pat}/snapshots/*')[0]).tokenizer;p=C/f'{model}-vobj2.json'
    if not p.exists():p.write_text(json.dumps(build_vobj(tok)))
    v=json.loads(p.read_text());word=lambda i:tok.decode([i]).strip().lower()
    out=[i for i in v if word(i) not in DROP];gone=[(i,tok.decode([i])) for i in v if word(i) in DROP]
    (C/f'{model}-vobj2cf.json').write_text(json.dumps(out))
    print(f'{model}: vobj {len(v)} -> {len(out)}; removed {gone}')
    print(f'   colour-like words checked and kept: {sorted({word(i) for i in out if word(i) in LOOKED_AT or word(i).rstrip("s") in LOOKED_AT})}')
