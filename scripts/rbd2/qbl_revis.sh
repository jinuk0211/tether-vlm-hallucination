#!/bin/bash
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/qbl.log; Q=$O/queues/qbl.txt; C=$1
export HF_HOME=$L/.cache/huggingface HF_HUB_CACHE=$L/.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$R/src
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=5 --query-gpu=uuid --format=csv,noheader)
until [ $(nvidia-smi --id=5 --query-gpu=memory.free --format=csv,noheader,nounits) -ge 18900 ]; do sleep 20; done
echo "=== $(date +%T) REVIS official: vectors and calibration on GPU 5" >> $LOG
if ! .venv/bin/python -u -m tether.baselines.revis_official >> $LOG 2>$O/logs/qbl_revis.err; then echo "=== $(date +%T) QBL_REVIS_FAILED (logs/qbl_revis.err)" >> $LOG; exit 1; fi
until grep -q grains_clone $Q 2>/dev/null; do [ -z "$(ps -o pid= -p $C)" ] && { echo "=== $(date +%T) QBL_REVIS_FAILED: chain $C ended before its queue was written" >> $LOG; exit 1; }; sleep 20; done
[ -n "$(ps -o pid= -p $C)" ] || { echo "=== $(date +%T) QBL_REVIS_FAILED: chain $C has ended" >> $LOG; exit 1; }
for b in $(seq 0 50 450); do echo "revis_official sdpa $b $((b+50))"; done >> $Q
echo "=== $(date +%T) REVIS official artifacts ready; 10 units added to the queue" >> $LOG
