#!/bin/bash
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/gemma_split.log
alive(){ [ -n "$(ps -o pid= -p $1)" ]; }
echo "=== $(date +%T) gemma_split.sh start" >> $LOG
bash scripts/rbd2/gchair_pool.sh 4 c4a 18879 & C4A=$!
echo "=== $(date +%T) CHAIR worker c4a on GPU 4 (pid $C4A)" >> $LOG
sleep 90
bash scripts/rbd2/gchair_pool.sh 4 c4b 18879 & C4B=$!
echo "=== $(date +%T) CHAIR worker c4b on GPU 4 (pid $C4B)" >> $LOG
hand(){ P=$1; G=$2; N=$3; D=$4
  ( while alive $P; do sleep 60; done; sleep $D; echo "=== $(date +%T) CHAIR worker pid $P done; THRONE worker $N joins GPU $G" >> $LOG
    exec bash scripts/rbd2/gemma_pool.sh $G $N 21257 ) & }
hand 125200 2 g2 0
hand $C4A 4 g4a 0
hand $C4B 4 g4b 60
wait
echo "=== $(date +%T) GEMMA_SPLIT_DONE" >> $LOG
