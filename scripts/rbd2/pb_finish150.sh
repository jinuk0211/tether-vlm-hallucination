#!/bin/bash
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
export PYTHONPATH=$R/src${PYTHONPATH:+:$PYTHONPATH}
export RBD2_SETS=$O/sets RBD2_CHAIR_TEST_STATE=$R/data/hallucination-suite/chair-scorer-state.json AMBER_REPO=$R/data/amber/repo \
       BENCH=$L/bench NLTK_DATA=$L/.cache/nltk_data HF_HOME=$L/.cache/huggingface PYTHONDONTWRITEBYTECODE=1 RBD2_N=150
LOG=$O/logs/score150.log
while [ -n "$(ps -o pid= -p $(paste -sd, $O/logs/pool150.pids))" ]; do sleep 120; done
echo "=== $(date +%T) join" >> $LOG
.venv/bin/python -m tether.experiments.join_150 >> $LOG 2>&1 || { echo "=== $(date +%T) JOIN_FAILED" >> $LOG; exit 1; }
{ echo "=== $(date +%T) chair_test150"; PYTHONPATH=src .venv/bin/python -m tether.eval.sets chair_test150 $O/runs/chair_test150_all 2>&1 | grep -v 'it/s'; } >> $LOG
{ echo "=== $(date +%T) amber_test150"; $L/venvs/amber/bin/python -m tether.eval.sets amber_test150 $O/runs/amber_test150_all 2>&1 | grep -v 'it/s'; } >> $LOG
export RBD2_THRONE_SET=throne_test150; TD=$O/runs/throne_test150_all
U(){ nvidia-smi --id=$1 --query-gpu=uuid --format=csv,noheader; }
TAGS=($(ls $TD | sed -n 's/-0-150\.jsonl$//p')); GPUS=(4 4 4 3)
for k in 0 1 2 3; do
  G=(); for ((i=k; i<${#TAGS[@]}; i+=4)); do G+=(${TAGS[i]}); done
  $L/venvs/throne/bin/python -m tether.eval.throne_judge $(U ${GPUS[k]}) $TD ${G[@]} > $O/logs/judge150-$k.log 2>&1 &
done
wait
{ echo "=== $(date +%T) throne_test150"; $L/venvs/throne/bin/python -m tether.eval.throne $(U 4) $TD 2>&1 | grep -v 'it/s' | tail -20
  echo "=== $(date +%T) SCORE150_DONE"; } >> $LOG
