#!/bin/bash
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
export PYTHONPATH=$R/src${PYTHONPATH:+:$PYTHONPATH}
LOG=$O/logs/amber_ours1004.log; Q=$O/queues/amber_ours1004.txt
{ echo "# base + mainMed on AMBER ids 101-200, 601-1004"
  for b in $(seq 100 50 150) $(seq 600 50 1000); do e=$((b+50)); [ $e -gt 1004 ] && e=1004
    echo "amber_full1004 amber_full1004_ours mainMed $b $e --alpha 1 --layer 16 --lam 1 --pen 0.5 --norm median"
    echo "amber_full1004 amber_full1004_ours base $b $e"; done; } > $Q
echo "=== $(date +%T) start, $(grep -vc '^#' $Q) units" >> $LOG
bash scripts/rbd2/pb_pool.sh 2 m2 $Q & sleep 5
bash scripts/rbd2/pb_pool.sh 3 m3 $Q &
wait
echo "=== $(date +%T) decoded" >> $LOG
.venv/bin/python - >> $LOG 2>&1 <<'PY'
import json
from pathlib import Path
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102');S=O/'runs/amber_full1004_ours';D=O/'runs/amber_full1004_ours_all'
D.mkdir(exist_ok=True);ids=[json.loads(l)['id'] for l in open(O/'sets/amber_full1004.jsonl')]
rd=lambda p:[json.loads(l) for l in open(p) if l.strip()] if p.exists() else []
for t in ['base','mainMed']:
    by={r['id']:r for r in rd(O/f'runs/amber_test500_all/{t}-0-500.jsonl')}
    for b in [100,150]+list(range(600,1004,50)):by.update({r['id']:r for r in rd(S/f'{t}-{b}-{min(b+50,1004)}.jsonl')})
    ok=all(i in by for i in ids)
    if ok:(D/f'{t}-0-1004.jsonl').write_text(''.join(json.dumps(by[i])+'\n' for i in ids))
    print('join',t,sum(i in by for i in ids),'/ 1004','complete' if ok else 'INCOMPLETE')
PY
export RBD2_SETS=$O/sets AMBER_REPO=$R/data/amber/repo NLTK_DATA=$L/.cache/nltk_data PYTHONDONTWRITEBYTECODE=1 RBD2_N=1004
{ echo "=== $(date +%T) amber_full1004 ours (paired bootstrap vs vdd base)"
  $L/venvs/amber/bin/python -m tether.eval.sets amber_full1004 $O/runs/amber_full1004_ours_all 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) AMBER_OURS_DONE"; } >> $LOG
