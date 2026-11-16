#!/bin/bash
G=$1; W=$2; MF=$3
R=/home/project-b/LJW/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; L=$O/logs; C=$O/claims_gchair; mkdir -p $L $C
QD=$O/queues/gchair_dev.txt; QO=$O/queues/gchair_ours.txt; QB=$O/queues/gchair_base.txt; PF=$O/runs/gchair_dev60/PICK_FAILED
export HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=/home/project-b/LJW/.cache/nltk_data PYTHONPATH=$R/src \
       VDD_GRAY=$O/gray448.png CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
cd $R
export PYTHONPATH=$R/src${PYTHONPATH:+:$PYTHONPATH}
free(){ nvidia-smi --id=$G --query-gpu=memory.free --format=csv,noheader,nounits; }
run_one(){
  while read -r D OD T S E A; do
    [ -z "$D" ] && continue; case $D in \#*) continue;; esac
    OUT=$O/runs/$OD; F=$OUT/$T-$S-$E.jsonl; U=$OD-$T-$S-$E
    if [ -f $F ] && [ $(wc -l < $F) -ge $((E-S)) ]; then continue; fi
    [ $(cat $C/fail-$U 2>/dev/null | wc -l) -ge 2 ] && continue
    OPEN=$((OPEN+1)); [ $(free) -ge $MF ] || continue
    mkdir $C/$U 2>/dev/null || continue
    echo $W > $C/$U/worker; mkdir -p $OUT; echo "$(date +%T) start $U" >> $L/gcpool-$W.log
    if .venv/bin/python -u -m tether.decoding.vdd --model gemma --attn eager --data $O/sets/$D.jsonl --tag $T \
         --output $OUT --start $S --end $E $A < /dev/null >> $L/gc-$OD-$T.log 2>&1
    then echo "$(date +%T) end $U" >> $L/gcpool-$W.log
    else echo "$(date +%T) FAILED $U" >> $L/gcpool-$W.log; echo $W >> $C/fail-$U; rm -r "${C:?}/${U:?}"; fi
    return 0
  done < $1
  return 1
}
while true; do
  OPEN=0
  run_one $QD && continue
  [ -f $QO ] && run_one $QO && continue
  run_one $QB && continue
  if [ $OPEN -eq 0 ] && { [ -f $QO ] || [ -f $PF ]; }; then break; fi
  sleep 60
done
echo "$(date +%T) POOL_DONE" >> $L/gcpool-$W.log
