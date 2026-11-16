#!/bin/bash
GPU=$1; S=$2; E=$3
R=/home/project-b/LJW/vlm-grounding-steering; cd $R
export PYTHONPATH=$R/src${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTHONPATH=$R/src \
       CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$GPU --query-gpu=uuid --format=csv,noheader)

OUT=$R/outputs/rbd2-20261102/runs/qchair_test500_tbl
SET=$R/outputs/rbd2-20261102/sets/chair_test500.jsonl

for M in rudder_clone revis vti; do
  echo "$(date +%T) Starting $M on GPU $GPU (range $S-$E)" >> $OUT/baselines_gpu$GPU.log
  .venv/bin/python -u -m tether.baselines.bench_qwen3 --method $M --set $SET --output $OUT --start $S --end $E --attn sdpa >> $OUT/baselines_gpu$GPU.log 2>&1
  echo "$(date +%T) Finished $M on GPU $GPU (range $S-$E)" >> $OUT/baselines_gpu$GPU.log
done
