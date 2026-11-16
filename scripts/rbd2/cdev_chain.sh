#!/bin/bash
L=${HAT_L:-/home/project-b/LJW}; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
export HAT_RUN=chair_dev100 TMPDIR=$L/.cache/tmp XDG_CACHE_HOME=$L/.cache PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=$L/.cache/nltk_data PYTHONPATH=$R/src
LOG=$O/logs/cdev.log; Q=$O/queues/chair_dev100.txt; PY=.venv/bin/python; mkdir -p $TMPDIR
CPU="nice -n 10 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="
log(){ echo "=== $(date +%T) $*" >> $LOG; }
stop(){ log "CDEV_STOPPED: $*"; exit 1; }
log "cdev_chain.sh start"
$CPU $PY -m tether.experiments.chair_dev prep >> $LOG 2>&1 || stop "prep failed"
finished(){ while read -r M K T S E A; do [ -z "$M" ] && continue; F=$O/runs/${HAT_RUN}_$M/$T-$S-$E.jsonl; [ -f $F ] && [ $(wc -l < $F) -ge $((E-S)) ] || return 1; done < $Q; return 0; }
after(){ ! kill -0 $1 2>/dev/null; }
joined(){ [ -f $O/hat385/join_gpu45 ]; }
worker(){ G=$1; W=$2; shift 2; until "$@" || finished; do sleep 60; done
  finished || { log "GPU $G worker $W starts"; bash scripts/rbd2/hat_pool.sh $G $W $Q; }; }
worker 2 cd2 after ${CD2_AFTER:-1196277} &
worker 4 cd4 joined &
worker 5 cd5 joined &
wait
log "decoded"
for M in qwen qwen3 gemma; do log "$M tables"; $CPU $PY -m tether.experiments.chair_dev score $M >> $LOG 2>&1 || log "$M scoring failed"; done
log CDEV_DONE
