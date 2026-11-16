#!/bin/bash
cd ${HAT_L:-/home/project-b/LJW}/vlm-grounding-steering
export WORKERS=3 PARSE_GPUS=3
HAT_REPRO_ONLY=1 bash scripts/rbd2/hat_chain.sh || { echo "$(date +%T) reproduction failed, stopped"; exit 1; }
echo "$(date +%T) reproduction done"
HAT_PILOT=1 bash scripts/rbd2/hat_chain.sh; echo "$(date +%T) pilot exit $?"
MODELS=qwen bash scripts/rbd2/hat_chain.sh; echo "$(date +%T) qwen exit $?"
MODELS=gemma bash scripts/rbd2/hat_chain.sh; echo "$(date +%T) gemma exit $?"
echo "$(date +%T) HAT_GPU3_DONE"
