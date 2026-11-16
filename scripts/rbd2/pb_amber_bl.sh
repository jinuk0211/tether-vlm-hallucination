#!/bin/bash
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
export PYTHONPATH=$R/src${PYTHONPATH:+:$PYTHONPATH}
LOG=$O/logs/amber_full.log; Q=$O/queues/amber_full.txt; VC=$O/logs/vti_chair.log
alive(){ [ -n "$(ps -o pid= -p $(echo $@ | tr ' ' ,))" ]; }
echo "=== $(date +%T) pb_amber_bl.sh start" >> $LOG
.venv/bin/python - >> $LOG 2>&1 <<'PY' || { echo "=== $(date +%T) SET_FAILED" >> $LOG; exit 1; }
import json
from pathlib import Path
R=Path('/home/project-b/LJW/vlm-grounding-steering');S=R/'outputs/rbd2-20261102/sets'
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
t=read(S/'amber_test150.jsonl')[0];gen=sorted(read(R/'data/amber/amber_generative.jsonl'),key=lambda g:g['metadata']['amber_id'])
rows=[{**t,'id':g['id'],'group_id':g['group_id'],'image':g['image'],'image_sha256':g['image_sha256'],'metadata':g['metadata']} for g in gen]
by={r['id']:r for r in rows};a500=read(S/'amber_test500.jsonl')
assert len(rows)==1004==len(by) and all(by[r['id']]==r for r in a500)
(S/'amber_full1004.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
print('amber_full1004',len(rows),'ids',rows[0]['metadata']['amber_id'],'..',rows[-1]['metadata']['amber_id'],'| amber_test500 rows identical')
PY
while alive 1582071 && ! grep -q 'picked\|PICK_FAILED' $VC; do sleep 60; done
V=$(grep -o 'picked vtiws_[^ ]*' $VC | cut -d' ' -f2); [ -n "$V" ] || V=vtiws_v0.9_t0.9
echo "=== $(date +%T) VTI setting $V" >> $LOG
U(){ b=$1; e=$((b+100)); [ $e -gt 1004 ] && e=1004; echo "amber_full1004 amber_full1004 $b $e"; }
{ echo "# baselines on full AMBER generative; revis last."
  for b in $(seq 0 100 1000); do
    echo "base sdpa $(U $b)"; echo "rudder_clone sdpa $(U $b)"; echo "grains_clone eager $(U $b)"; echo "$V eager $(U $b)"; done
  for b in $(seq 0 100 1000); do echo "revis sdpa $(U $b)"; done; } > $Q
start(){ G=$1; N=$2; D=$3; shift 3
  ( while alive $@; do sleep 60; done; sleep $D; echo "=== $(date +%T) GPU $G worker $N joins" >> $LOG
    exec bash scripts/rbd2/pb_amber_pool.sh $G $N $Q ) & }
start 3 a3 0 1476645
start 5 a5a 0 1476643
start 5 a5b 150 1476643
start 4 a4a 0 692460 692827 1829740
start 4 a4b 150 692460 692827 1829740
start 2 a2 0 1582071
wait
echo "=== $(date +%T) decoded" >> $LOG
.venv/bin/python - >> $LOG 2>&1 <<'PY'
import json,glob
from pathlib import Path
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102');S=O/'runs/amber_full1004';D=O/'runs/amber_full1004_all'
D.mkdir(exist_ok=True);ids=[json.loads(l)['id'] for l in open(O/'sets/amber_full1004.jsonl')]
for tag in sorted({Path(f).name.rsplit('-',2)[0] for f in glob.glob(str(S/'*.jsonl'))}):
    rows=[]
    for b in range(0,1004,100):
        f=S/f'{tag}-{b}-{min(b+100,1004)}.jsonl';rows+=[json.loads(l) for l in open(f)] if f.exists() else []
    ok=[r['id'] for r in rows]==ids
    if ok:(D/f'{tag}-0-1004.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print('join',tag,len(rows),'complete' if ok else 'INCOMPLETE')
PY
export RBD2_SETS=$O/sets AMBER_REPO=$R/data/amber/repo NLTK_DATA=$L/.cache/nltk_data PYTHONDONTWRITEBYTECODE=1 RBD2_N=1004
{ echo "=== $(date +%T) amber_full1004 (paired bootstrap vs base)"
  $L/venvs/amber/bin/python -m tether.eval.sets amber_full1004 $O/runs/amber_full1004_all 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) AMBER_FULL_DONE"; } >> $LOG
