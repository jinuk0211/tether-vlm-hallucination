#!/bin/bash
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
P=("$@")
bash scripts/rbd2/pb_pool.sh 3 p3 $O/queues/pool500.txt & P+=($!)
echo "$(date +%T) GPU 3 joins (pb_run500c.sh)" >> $O/logs/run500.log
while [ -n "$(ps -o pid= -p $(IFS=,; echo "${P[*]}"))" ]; do sleep 60; done
bash scripts/rbd2/pb_score500.sh
