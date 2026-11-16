#!/bin/bash
set -u
G=$1; Q=$(realpath "$2"); cd "$(dirname "$0")/.."
PY=${3:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python)}
export PYTHONPATH=$PWD/src${PYTHONPATH:+:$PYTHONPATH} CUDA_VISIBLE_DEVICES=$G
while read -r M A; do
  case "$M" in ''|\#*) continue;; esac
  echo "$(date +%T) start $M $A"
  if $PY -u -m "$M" $A < /dev/null; then echo "$(date +%T) done  $M"; else echo "$(date +%T) FAILED $M $A" >&2; fi
done < "$Q"
