#!/bin/bash
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
while [ -n "$(ps -o pid= -p $1)" ]; do sleep 60; done
echo "$(date +%T) full THRONE baselines start" >> $O/logs/bl_chain.log
Q=$O/queues/bl_throne_full.txt
bash scripts/rbd2/pb_bl_pool.sh 3 b3 $Q & sleep 5
bash scripts/rbd2/pb_bl_pool.sh 4 b4a $Q & sleep 5
bash scripts/rbd2/pb_bl_pool.sh 4 b4b $Q &
wait
echo "$(date +%T) full THRONE baselines decoded" >> $O/logs/bl_chain.log
bash scripts/rbd2/pb_bl_full_score.sh 1000 4 4 4 3 3
