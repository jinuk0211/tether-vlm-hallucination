#!/bin/bash
G=$1; W=$2; MF=${3:-20000}
R=/home/project-b/LJW/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; L=$O/logs; C=$O/claims_gemma; mkdir -p $L $C
QD=$O/queues/gemma_dev.txt; QO=$O/queues/gemma_ours.txt; QB=$O/queues/gemma_bl.txt
export HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=/home/project-b/LJW/.cache/nltk_data PYTHONPATH=$R/src \
       VDD_GRAY=$O/gray448.png CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
cd $R
export PYTHONPATH=$R/src${PYTHONPATH:+:$PYTHONPATH}
free(){ nvidia-smi --id=$G --query-gpu=memory.free --format=csv,noheader,nounits; }
run_one(){
  while read -r K D OD T S E A; do
    [ -z "$K" ] && continue; case $K in \#*) continue;; esac
    OUT=$O/runs/$OD; F=$OUT/$T-$S-$E.jsonl; U=$OD-$T-$S-$E
    if [ -f $F ] && [ $(wc -l < $F) -ge $((E-S)) ]; then continue; fi
    [ $(cat $C/fail-$U 2>/dev/null | wc -l) -ge 2 ] && continue
    OPEN=$((OPEN+1)); [ -d $C/$U ] && continue
    [ $(free) -ge $MF ] || continue; sleep $((RANDOM%20+10)); [ $(free) -ge $MF ] || continue
    mkdir $C/$U 2>/dev/null || continue
    echo $W > $C/$U/worker; mkdir -p $OUT; echo "$(date +%T) start $U" >> $L/gpool-$W.log
    if [ $K = vdd ]; then X=(-m tether.decoding.vdd --model gemma --attn eager --data $O/sets/$D.jsonl --tag $T)
    else X=(-m tether.baselines.bench_gemma --method $T --set $O/sets/$D.jsonl); fi
    UL=$L/gunit-$W.log
    .venv/bin/python -u "${X[@]}" --output $OUT --start $S --end $E $A < /dev/null > $UL 2>&1; RC=$?
    cat $UL >> $L/g-$OD-$T.log
    if [ $RC -eq 0 ]; then echo "$(date +%T) end $U" >> $L/gpool-$W.log; return 0; fi
    rm -r $C/$U
    if grep -q OutOfMemoryError $UL; then echo "$(date +%T) OOM $U (not counted)" >> $L/gpool-$W.log; sleep $((RANDOM%120+60))
    else echo "$(date +%T) FAILED $U" >> $L/gpool-$W.log; echo $W >> $C/fail-$U; fi
    return 0
  done < $1
  return 1
}
while true; do
  OPEN=0
  run_one $QD && continue
  [ -f $QO ] && run_one $QO && continue
  run_one $QB && continue
  if [ $OPEN -eq 0 ] && { [ -f $QO ] || [ -f $O/runs/gemma_throne_dev60/PICK_FAILED ]; }; then break; fi
  sleep 120
done
echo "$(date +%T) POOL_DONE" >> $L/gpool-$W.log
