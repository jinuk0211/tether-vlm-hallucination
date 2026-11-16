#!/bin/bash
N=$1; shift; SLOTS=(${@:-4 4 4 3 3})
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
export PYTHONPATH=$R/src${PYTHONPATH:+:$PYTHONPATH}
export RBD2_SETS=$O/sets BENCH=$L/bench HF_HOME=$L/.cache/huggingface PYTHONDONTWRITEBYTECODE=1 RBD2_N=$N RBD2_THRONE_SET=throne_full$N
LOG=$O/logs/score_throne_full.log; TD=$O/runs/throne_full$N
{ echo "=== $(date +%T) throne_full$N join"; .venv/bin/python -m tether.experiments.baselines_throne_join $N; } >> $LOG 2>&1 || { echo "=== JOIN_FAILED" >> $LOG; exit 1; }
U(){ nvidia-smi --id=$1 --query-gpu=uuid --format=csv,noheader; }
M=(base revis rudder_clone grains_clone vti_author)
for ((k=0; k<${#SLOTS[@]}; k++)); do
  G=(); for ((i=k; i<${#M[@]}; i+=${#SLOTS[@]})); do G+=(${M[i]}); done
  [ ${#G[@]} -gt 0 ] && $L/venvs/throne/bin/python -m tether.eval.throne_judge $(U ${SLOTS[k]}) $TD ${G[@]} > $O/logs/judge_full$N-$k.log 2>&1 &
done
wait
{ echo "=== $(date +%T) throne_full$N"; $L/venvs/throne/bin/python -m tether.eval.throne $(U ${SLOTS[0]}) $TD 2>&1 | grep -v 'it/s' | tail -8
  echo "=== $(date +%T) THRONE_FULL${N}_DONE"; } >> $LOG
