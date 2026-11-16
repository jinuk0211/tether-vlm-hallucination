#!/bin/bash
G=$1; N=$2
while pgrep -f "[w]orker.sh $G " > /dev/null; do sleep 20; done
exec /workspace/rbd_pilot/worker.sh $G $N
