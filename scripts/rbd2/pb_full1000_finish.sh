#!/bin/bash
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
export PYTHONPATH=$R/src${PYTHONPATH:+:$PYTHONPATH}
export RBD2_SETS=$O/sets BENCH=$L/bench HF_HOME=$L/.cache/huggingface PYTHONDONTWRITEBYTECODE=1 RBD2_N=1000 RBD2_THRONE_SET=throne_full1000
TD=$O/runs/throne_full1000; U4=$(nvidia-smi --id=4 --query-gpu=uuid --format=csv,noheader)
$L/venvs/throne/bin/python -m tether.eval.throne_judge $U4 $TD grains_clone > $O/logs/judge_full1000-grains-retry.log 2>&1
while [ -n "$(ps -o pid= -p $2)" ]; do sleep 60; done
{ echo "=== $(date +%T) throne_full1000 (GrAInS xl judge re-run on GPU 4 after OOM)"
  $L/venvs/throne/bin/python -m tether.eval.throne $U4 $TD 2>&1 | grep -v 'it/s' | tail -8
  echo "=== $(date +%T) THRONE_FULL1000_DONE"; } >> $O/logs/score_throne_full.log
kill -9 $1
