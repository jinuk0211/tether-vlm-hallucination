#!/bin/bash
G=$1; W=$2; Q=$3
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; LG=$O/logs; C=$O/claims_t4k; mkdir -p $LG $C $L/.cache/tmp $O/t4k
export HF_HOME=$L/.cache/huggingface HF_HUB_CACHE=$L/.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
       NLTK_DATA=$L/.cache/nltk_data VDD_GRAY=$O/gray448.png TMPDIR=$L/.cache/tmp XDG_CACHE_HOME=$L/.cache \
       CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
cd $R
free(){ nvidia-smi --id=$G --query-gpu=memory.free --format=csv,noheader,nounits; }
P3=$R/src
while [ ! -f $O/t4k/STOP ]; do
  RAN=; LEFT=0
  while read -r K D OD T S E A; do
    [ -z "$K" ] && continue; case $K in \#*) continue;; esac
    OUT=$O/runs/$OD; F=$OUT/$T-$S-$E.jsonl; U=$OD-$T-$S-$E
    if [ -f $F ] && [ $(wc -l < $F) -ge $((E-S)) ]; then continue; fi
    [ $(cat $C/fail-$U 2>/dev/null | wc -l) -ge 2 ] && continue
    [ -d $C/$U ] && continue
    LEFT=$((LEFT+1))
    case $K in gb|gv|v4g) MF=21257;; *) MF=19500;; esac
    [ $(free) -ge $MF ] || continue; sleep $((RANDOM%20+10)); [ $(free) -ge $MF ] || continue
    mkdir $C/$U 2>/dev/null || continue
    echo $W > $C/$U/worker; mkdir -p $OUT; echo "$(date '+%m-%d %T') start $U" >> $LG/t4kpool-$W.log; UL=$LG/t4kunit-$W.log
    case $K in
      qb)  PP=$P3; X=(-m tether.baselines.bench_qwen --method $T --set $O/sets/$D.jsonl);;
      gb)  PP=$P3; X=(-m tether.baselines.bench_gemma --method $T --set $O/sets/$D.jsonl);;
      qv)  PP=$R/src; X=(-m tether.decoding.vdd --model qwen --data $O/sets/$D.jsonl --tag $T);;
      gv)  PP=$P3; X=(-m tether.decoding.vdd --model gemma --attn eager --data $O/sets/$D.jsonl --tag $T);;
      v4q) PP=$P3; X=(-m tether.decoding.vdd_vobj --model qwen --data $O/sets/$D.jsonl --tag $T);;
      v4g) PP=$P3; X=(-m tether.decoding.vdd_vobj --model gemma --data $O/sets/$D.jsonl --tag $T);;
      *)   echo "$(date '+%m-%d %T') unknown kind $K" >> $LG/t4kpool-$W.log; rm -r $C/$U; continue;;
    esac
    PYTHONPATH=$PP .venv/bin/python -u "${X[@]}" --output $OUT --start $S --end $E $A < /dev/null > $UL 2>&1; RC=$?
    cat $UL >> $LG/t4k-$OD-$T.log
    if [ $RC -eq 0 ] && [ -f $F ] && [ $(wc -l < $F) -ge $((E-S)) ]; then echo "$(date '+%m-%d %T') end $U" >> $LG/t4kpool-$W.log
    else
      rm -r $C/$U
      if grep -q 'OutOfMemoryError\|CUDA out of memory' $UL; then echo "$(date '+%m-%d %T') OOM $U (not counted)" >> $LG/t4kpool-$W.log; sleep $((RANDOM%120+60))
      else echo "$(date '+%m-%d %T') FAILED $U rc $RC" >> $LG/t4kpool-$W.log; echo $W >> $C/fail-$U; fi
    fi
    RAN=1; break
  done < $Q
  [ -n "$RAN" ] && continue
  [ $LEFT -eq 0 ] && break
  sleep 60
done
echo "$(date '+%m-%d %T') POOL_DONE $(basename $Q)" >> $LG/t4kpool-$W.log
