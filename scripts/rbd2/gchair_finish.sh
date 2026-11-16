#!/bin/bash
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
export PYTHONPATH=$R/src${PYTHONPATH:+:$PYTHONPATH}
LOG=$O/logs/gchair.log; TD=$O/runs/gchair_test500; C=$O/claims_gchair; P=s24a2bl1p05
GT=$R/outputs/gemma-grains-fp16-v2-20261029-1793246507/chair
open(){ n=0; for t in base $P; do for b in $(seq 0 50 450); do f=$TD/$t-$b-$((b+50)).jsonl; U=gchair_test500-$t-$b-$((b+50))
  if [ -f $f ] && [ $(wc -l < $f) -ge 50 ]; then continue; fi
  [ $(cat $C/fail-$U 2>/dev/null | wc -l) -ge 2 ] && continue; n=$((n+1)); done; done; echo $n; }
echo "=== $(date +%T) gchair_finish.sh: dev stopped, $P fixed; test 500 (ours + vdd Base)" >> $LOG
while [ $(open) -gt 0 ]; do sleep 120; done
echo "=== $(date +%T) decoded" >> $LOG
TA=$O/runs/gchair_test500_all; TT=$O/runs/gchair_test500_tbl; mkdir -p $TA $TT
.venv/bin/python -m tether.experiments.gemma_chair_join base $P >> $LOG 2>&1
for m in base revis rudder_clone grains_clone vti; do cp $GT/$m.jsonl $TT/$m-0-500.jsonl; done
[ -f $TA/base-0-500.jsonl ] && cp $TA/base-0-500.jsonl $TT/vddbase-0-500.jsonl
[ -f $TA/$P-0-500.jsonl ] && cp $TA/$P-0-500.jsonl $TT/$P-0-500.jsonl
export RBD2_SETS=$O/sets RBD2_CHAIR_TEST_STATE=$R/data/hallucination-suite/chair-scorer-state.json NLTK_DATA=$L/.cache/nltk_data PYTHONDONTWRITEBYTECODE=1
{ echo "=== $(date +%T) chair_test500 Gemma: ours vs vdd Base (paired bootstrap)"
  RBD2_N=500 .venv/bin/python -m tether.eval.sets chair_test500 $TA 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) chair_test500 Gemma: table rows (base = table Base) + vddbase + ours"
  RBD2_N=500 .venv/bin/python -m tether.eval.sets chair_test500 $TT 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) GCHAIR_DONE"; } >> $LOG
