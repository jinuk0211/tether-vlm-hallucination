import shutil
from pathlib import Path
R=Path('/home/project-b/LJW/vlm-grounding-steering');S=R/'outputs/rbd2-20261102/sets'
S.mkdir(parents=True,exist_ok=True)
shutil.copy(R/'data/hallucination-suite/chair.jsonl',S/'chair_test500.jsonl')
n=sum(1 for l in open(S/'chair_test500.jsonl') if l.strip());assert n==500;print('chair_test500',n)
