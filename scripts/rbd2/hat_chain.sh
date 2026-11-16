#!/bin/bash
L=${HAT_L:-/home/project-b/LJW}; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
export PYTHONPATH=$R/src${PYTHONPATH:+:$PYTHONPATH}
[ -n "$HAT_PILOT" ] && HAT_RUN=pilot_chair500
export HAT_RUN=${HAT_RUN:-hat385}
PARSER=${PARSER:-hf:Qwen/Qwen3-VL-8B-Instruct}; MODELS=${MODELS:-qwen gemma}; WORKERS=${WORKERS:-2 3 4 4 5 5}; PARSE_GPUS=${PARSE_GPUS:-5 4 3 2}
H=$O/$HAT_RUN; LOG=$O/logs/$HAT_RUN.log; SE=$O/logs/$HAT_RUN-scorer.log; PL=$O/logs/$HAT_RUN-parse.log; AP=$L/venvs/aloha/bin/python; PY=.venv/bin/python; mkdir -p $H $O/logs
APN="nice -n 10 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 $AP"
export TMPDIR=$L/.cache/tmp XDG_CACHE_HOME=$L/.cache; mkdir -p $TMPDIR
log(){ echo "=== $(date +%T) $*" >> $LOG; }
stop(){ log "HAT_STOPPED: $*"; exit 1; }
fre(){ nvidia-smi --id=$1 --query-gpu=memory.free --format=csv,noheader,nounits; }
uuid(){ nvidia-smi --id=$1 --query-gpu=uuid --format=csv,noheader; }
wait_after(){ for p in $HAT_AFTER; do while kill -0 $p 2>/dev/null; do sleep 60; done; done; }
log "hat_chain.sh start: parser $PARSER, models $MODELS, workers $WORKERS"
case $PARSER in
  hf:*) S=$L/.cache/huggingface/hub/models--$(echo ${PARSER#hf:} | sed 's#/#--#g'); [ -d $S ] || stop "no $S in the HF cache";;
  openai:*) [ -n "$OPENAI_API_KEY" ] || stop "PARSER=$PARSER needs OPENAI_API_KEY";;
esac
for f in src/tether/decoding/vdd_vobj.py src/tether/baselines/bench_qwen.py src/tether/baselines/bench_gemma.py src/tether/baselines/vti_ws.py \
         outputs/rbd-20261030/collect/qwen-vobj2cf.json outputs/rbd-20261030/collect/gemma-vobj2cf.json; do [ -e $f ] || stop "missing $f"; done
[ -x $AP ] || bash scripts/rbd2/hat_setup.sh >> $LOG 2>&1 || stop "setup failed"
export HF_HOME=$L/.cache/huggingface HF_HUB_CACHE=$L/.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=$L/.cache/nltk_data TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
detect(){ [ -s $H/detections.json ] || $APN -m tether.eval.hat_aloha detect 2>> $SE >> $LOG || stop "detections failed ($SE)"; }
parse(){ for try in 1 2 3 4 5; do
    case $PARSER in
      hf:*) wait_after; until G=$(for x in $PARSE_GPUS; do [ $(fre $x) -ge 19500 ] && echo $x && break; done); [ -n "$G" ]; do sleep 60; done
            CUDA_VISIBLE_DEVICES=$(uuid $G) $PY -u -m tether.eval.hat_parse $PARSER "$@";;
      *)    $APN -u -m tether.eval.hat_parse $PARSER "$@";;
    esac > $PL 2>&1 && return 0
    grep -q OutOfMemoryError $PL || break; log "parser out of memory, try $try"; sleep 120
  done; stop "parse failed ($PL)"; }
score(){ $APN -m tether.eval.hat_aloha score $PARSER $O/runs/${HAT_RUN}_$1 $2 $3 $H/score-$1-$2-$3 2>> $SE | tee -a $LOG; [ ${PIPESTATUS[0]} -eq 0 ]; }
phase(){ wait_after; PIDS=(); i=0; for G in $WORKERS; do i=$((i+1)); bash scripts/rbd2/hat_pool.sh $G w$i $1 & PIDS+=($!); sleep 60; done; wait "${PIDS[@]}"; }
if [ -n "$HAT_PILOT" ]; then
  $PY -m tether.experiments.hat pilot $MODELS >> $LOG 2>&1 || stop "pilot set failed"
  detect; N=$(wc -l < $O/sets/$HAT_RUN.jsonl)
  for M in $MODELS; do
    parse --run $O/runs/${HAT_RUN}_$M; log "$M pilot table: existing CHAIR captions, COCO references"
    score $M 0 $N > /dev/null || log "$M pilot table failed"
  done
  log PILOT_DONE; exit 0
fi
[ -s $O/sets/$HAT_RUN.jsonl ] || $PY -m tether.experiments.hat set >> $LOG 2>&1 || stop "set failed"
detect
log "HAT reproduction check"; parse --hat
$APN -m tether.eval.hat_aloha validate $PARSER 2>> $SE >> $LOG || stop "reproduction check failed ($SE)"
[ -n "$HAT_REPRO_ONLY" ] && { log REPRO_DONE; exit 0; }
$PY -m tether.experiments.hat queue_dev $MODELS >> $LOG 2>&1 || stop "queue failed"
log "selection decode, rows 0-59"; phase $O/queues/${HAT_RUN}_dev.txt
for M in $MODELS; do
  parse --run $O/runs/${HAT_RUN}_$M; log "$M selection table (rows 0-59)"
  score $M 0 60 > /dev/null && $PY -m tether.experiments.hat pick $M $H/score-$M-0-60.json >> $LOG 2>&1 || log "$M pick failed: RBD not continued (table above, logs/hat-$M-*.log)"
done
$PY -m tether.experiments.hat queue_test $MODELS >> $LOG 2>&1 || stop "test queue failed"
log "test decode"; phase $O/queues/${HAT_RUN}_test.txt
for M in $MODELS; do
  parse --run $O/runs/${HAT_RUN}_$M
  log "$M table, all 385 rows"; score $M 0 385 > /dev/null || log "$M all-rows table failed"
  log "$M table, rows 60-384 (not used for selection)"; score $M 60 385 > /dev/null || log "$M rows 60-384 table failed"
done
log HAT_DONE
