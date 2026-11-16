#!/bin/bash
G=$1; W=$2; Q=$3
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; C=$O/claims_xl; mkdir -p $C $O/logs; cd $R
export PYTHONPATH=$R/src${PYTHONPATH:+:$PYTHONPATH}
export RBD2_SETS=$O/sets BENCH=$L/bench HF_HOME=$L/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
       RBD2_THRONE_SET=throne_full1000 RBD2_N=1000 TMPDIR=$L/.cache/tmp XDG_CACHE_HOME=$L/.cache
UU=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
free(){ nvidia-smi --id=$G --query-gpu=memory.free --format=csv,noheader,nounits; }
judged(){ ls $O/runs/$1/eval/$2/combined/*flan-t5-xl.json > /dev/null 2>&1; }
while true; do
  RAN=; LEFT=0
  while read -r D T; do
    [ -z "$D" ] && continue; U=$D-$T
    judged $D $T && continue
    [ $(cat $C/fail-$U 2>/dev/null | wc -l) -ge 2 ] && continue
    [ -d $C/$U ] && continue
    LEFT=$((LEFT+1))
    [ $(free) -ge 16000 ] || continue; sleep $((RANDOM%20+10)); [ $(free) -ge 16000 ] || continue
    mkdir $C/$U 2>/dev/null || continue
    echo $W > $C/$U/worker; echo "$(date +%T) start $U" >> $O/logs/xlpool-$W.log; UL=$O/logs/xlunit-$W.log
    $L/venvs/throne/bin/python -u -m tether.eval.throne_xl_judge $UU $O/runs/$D $T < /dev/null > $UL 2>&1; RC=$?
    cat $UL >> $O/logs/xl-$U.log
    if [ $RC -eq 0 ] && judged $D $T; then echo "$(date +%T) end $U" >> $O/logs/xlpool-$W.log
    else
      rm -r $C/$U
      if grep -q 'OutOfMemoryError\|CUDA out of memory' $UL; then echo "$(date +%T) OOM $U (not counted)" >> $O/logs/xlpool-$W.log; sleep $((RANDOM%120+60))
      else echo "$(date +%T) FAILED $U" >> $O/logs/xlpool-$W.log; echo $W >> $C/fail-$U; fi
    fi
    RAN=1; break
  done < $Q
  [ -n "$RAN" ] && continue
  [ $LEFT -eq 0 ] && break
  sleep 120
done
echo "$(date +%T) POOL_DONE" >> $O/logs/xlpool-$W.log
