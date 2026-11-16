#!/bin/bash
# Same join + scoring as the tail of pb_amber_bl.sh, for running by hand once all five methods are decoded on the
# 1004 images (pb_amber_bl.sh itself still waits on its GPU 5 workers).
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R; LOG=$O/logs/amber_full_score.log
export PYTHONPATH=$R/src${PYTHONPATH:+:$PYTHONPATH}
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
  echo "=== $(date +%T) AMBER_SCORE_DONE"; } >> $LOG
