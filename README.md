# TETHER: 코드와 결과 (초안 3, 2026-11-09)

초안 3(`CVPR_VL_model_steering (3).pdf`)의 표, 그림, 본문 수치를 만든 코드와 결과 파일입니다.
아래 표는 논문의 표·그림 번호별로 코드와 결과 파일을 연결합니다.

## 구조

```
src/tether/
  models/       Hugging Face 모델 어댑터 (Qwen2.5-VL, Qwen3-VL, Gemma 4)
  decoding/     TETHER 디코더 (vdd, vdd_vobj, vdd_cf, sweep), r(t)·V_obj (common), 첫 번째 변형 (pilot/)
  baselines/    REVIS·VTI 자체 구현 (local), 원저자 RUDDER·GrAInS·VTI·REVIS 연결, 방향 추출, 생성 worker, bench_*
  eval/         CHAIR·POPE·AMBER·THRONE·HAT(ALOHa) 채점, paired bootstrap
  analysis/     3장·부록 A 분석 (visual_decay, reliance, read_gap, selfcheck, probe, logit_lens)
  data/         평가 세트와 이미지 준비
  experiments/  실험별 큐·선택·join (ablation, CHAIR dev, THRONE, HAT 등)
scripts/        실행 스크립트(.sh): GPU worker pool과 실험 chain, 큐 파일(scripts/rbd2/queues/)
configs/        모델 설정
outputs/        결과 (캡션, 채점, 판정, 로그)
```

설치는 `pip install -e .`이고, 모듈은 `python -m tether.decoding.vdd --model qwen --data SET.jsonl ...`처럼 실행합니다.
셸 스크립트는 `PYTHONPATH=$R/src`를 잡고 같은 방식으로 모듈을 부릅니다.
아래 표에서 `.py` 경로는 `src/tether/` 기준이고, `.sh`는 따로 적지 않으면 `scripts/rbd2/`에 있습니다.

- 경로는 서버(`/home/project-b/LJW/vlm-grounding-steering`)와 같습니다. 큐 파일만 서버의 `outputs/rbd2-20261102/queues/` 대신
  `scripts/rbd2/queues/`에 있습니다.
- 출력 폴더와 셸 스크립트 이름에는 방법의 예전 이름 RBD가 남아 있습니다(`rbd-20261030` = 첫 번째 변형, `rbd2-20261102` = TETHER).
- 결과는 텍스트 파일만 넣었습니다: 캡션(jsonl), 채점 결과(json), THRONE 판정 파일, 로그. 모델 가중치, 이미지, steering 벡터(`.pt`),
  활성값 덤프(`.npz` 등)와 50 MB가 넘는 파일은 없습니다.
- 원저자 코드(RUDDER, GrAInS, VTI, REVIS 공식 코드, CHAIR·AMBER·THRONE·ALOHa 채점기)는 넣지 않았습니다. 고정 버전은 맨 아래에 있습니다.
- 스크립트는 서버 경로, `.venv`, HF 캐시, GPU UUID를 전제로 하므로 다른 머신에서는 경로를 바꿔야 합니다.
- 캡션에 나오는 이메일 주소는 COCO 사진의 워터마크를 모델이 그대로 옮겨 적은 것입니다.

경로 약칭: `R/` = `outputs/rbd2-20261102/`. 설정 태그 `s{L}a{α}bl{λ}p{π}` = `--layer L --alpha α --lam λ --pen π`
(`p05` = 0.5), `cf_` = 색 단어를 뺀 V_obj(THRONE 최종 설정). 태그별 인자는 `scripts/rbd2/queues/*.txt`에 있습니다.

## 3장과 부록 A: 감사

| 표·그림·본문 | 코드 | 결과 |
|---|---|---|
| 표 1, 그림 1 a1 (자기 검증) | `analysis/selfcheck/collect.py` → `analysis/selfcheck/analyze.py`; 마지막 행(위치 맞춤 AUROC 0.932, 0.862)은 `outputs/audit-recheck-20261105/audit_recheck.py` | `outputs/caption-selfcheck-20261030/{qwen,gemma}-base-analysis.json`, 객체별 행 `*-base-rows.jsonl`; `audit-recheck-20261105/{qwen,gemma}.json`의 `ranking/question_margin/position` |
| 3.1절, 부록 A.4 POPE (0.940 / 0.916 vs probe 0.880 / 0.859, FN 13/379, 17/303) | `analysis/read_gap/collect.py` → `analysis/read_gap/analyze.py`, `analysis/read_gap/complement.py` | `outputs/read-gap-e0-20261030/analysis-*.json` |
| 표 2 (위치별 새 물체), 부록 A.6 | `outputs/audit-recheck-20261105/audit_recheck.py`; 보조 `scripts/audit_inline/hall_share_by_position.sh` | `outputs/audit-recheck-20261105/{qwen,gemma}.json` |
| 표 3, 표 A.1, 그림 1 a3, 부록 A.2 (gray KL, attention, key blocking 0.81·0.72, logit-lens 기여 0.92·1.65, Gemma sliding window 35.5%), 부록 A.10 rollout | `scripts/visual_decay_run.sh` (`analysis/visual_decay/collect.py` → `analyze.py` → `plot.py`) | `outputs/visual-decay-20261030/*-visual-decay-summary.json` (`rollout_absorbing_image`, `window_limited_layers`) |
| grad×input (표 3, 표 A.2 위, 부록 A.4의 0.753 vs 0.749, A.2의 0.70→0.18) | `analysis/reliance/collect.py` → `analysis/reliance/analyze.py` | `outputs/visual-reliance-20261030/qwen-reliance-summary.json` |
| 표 A.2 아래 (e_t 전체·첫·반복 언급), 부록 A.7 공출현 | `outputs/audit-recheck-20261105/audit_recheck2.py` | `{qwen,gemma}-2.json` |
| 그림 1 a2, 표 4 (위치·log p 맞춤 AUROC), 부록 A.8 순위 구간 | `audit_recheck.py`, `audit_recheck3.py`, lens 행은 `analysis/logit_lens.py` | `{qwen,gemma}.json`, `{qwen,gemma}-3.json`, `outputs/logit-lens-20261104/*-lens-summary.json` |
| 부록 A.3 attention (0.64 / 0.55, 모양 0.49–0.52, head 21–24) | `analysis/probe/attention.py` | `outputs/query-anchor-step0-20261030/*-attn-summary.json` |
| 표 A.3 probe, 기준선 0.74 / 0.69 | `outputs/audit-recheck-20261105/probe_foldwise.py` (probe 학습은 `analysis/probe/collect.py` → `analysis/probe/analyze.py`) | `{qwen,gemma}-probe-foldwise.json`, `outputs/verify-probe-20261030/*-probe-summary.json` |
| 표 3·표 A.4 개입, 부록 A.5 선택성 차이 | `analysis/probe/intervention.py` | `outputs/verify-probe-20261030/*-e1-summary.json` |
| 표 A.5 위쪽 (이미지 토큰 lens AUROC, 같은 물체 안, 공출현 맞춤) | `analysis/logit_lens.py image/analyze`; `scripts/audit_inline/lens_image_layers_qwen.sh`, `lens_gemma_image_caption.sh` | `outputs/logit-lens-20261104/{qwen,gemma}-lens-summary.json`, `inline/*.out.txt` |
| 표 A.5 "Word in top-5 of a patch" (마지막 층, L21) | `scripts/audit_inline/lens_word_top5.sh` (`analysis/logit_lens.py image`가 저장한 L7·14·21·28 top-5에서 계산) | `outputs/audit-recheck-20261105/inline/lens_word_top5.out.txt` |
| 표 A.6 (top-100, 거리 × 위치), 부록 A.8 고정 거리 차이 | `analysis/logit_lens.py topk/topk-analyze` (`outputs/logit-lens-20261104/run_topk.sh`); `scripts/audit_inline/lens_top100_distance_position.sh`; CI는 `audit_recheck3.py` | `*-topk-summary.json`, `inline/lens_top100_distance_position.out.txt`, `{qwen,gemma}-3.json` |
| 부록 A.8 "Writing step" (L26 −13.3, L27 −14.0, Gemma L34 −34.0, L38 −14.5) | `scripts/audit_inline/lens_writing_step.sh` | `inline/lens_writing_step.out.txt` |
| 부록 A.8 hidden band top-1 (보조) | `scripts/audit_inline/lens_hidden_band_top1.sh` | `inline/lens_hidden_band_top1.out.txt` |
| 부록 A.1 addCriterion 77개 제외 | `audit_recheck3.py`; `scripts/audit_inline/addcriterion_exclusion.sh` | `{qwen,gemma}-3.json`, `inline/addcriterion_exclusion.out.txt` |

`decoding/selfverify.py`(멘션 위치), `decoding/open_vocab.py`(WordNet 명사 필터), `analysis/read_gap/collect.py`(모델 로더, 프롬프트)는
위 모듈들이 같이 쓰는 부품입니다. `scripts/audit_inline/`의 8개는 11-05에 서버에서 바로 실행한 분석 명령이고, 출력은 그때 찍힌 그대로입니다.

## 4장과 부록 B: TETHER

| 내용 | 코드 | 결과 |
|---|---|---|
| r(t), 표 B.2, 4.2절 가중치 | `decoding/common.py`의 `r_curve`: 표 A.1 gray KL을 첫 구간 1로 정규화, 구간 중심 사이 선형 보간 | 표 A.1과 같은 요약 파일 |
| V_obj (부록 B.3) | `decoding/common.py`의 `build_vobj`, `decoding/pilot/collect.py` | `outputs/rbd-20261030/collect/*-vobj2.json`, 색 단어 제외판 `*-vobj2cf.json`은 `experiments/throne_cf_setup.py` |
| 디코더 (그림 2) | `decoding/vdd.py`; V_obj 파일을 바꿀 수 있는 판 `decoding/vdd_vobj.py` (THRONE 최종 설정, ablation) | — |
| 초기 탐색 (AMBER 101–160, CHAIR dev 0–59) | 클라우드 박스 `experiments/cloud_dev/` (`prep.py`, `sets.py`, `benchmarks.py`) → `scripts/rbd2/cloud_dev/worker.sh`·`vrun.sh`·`chain.sh` | `R/runs/chair_dev60/`, `R/runs/amber_dev60/`, `R/runs_dev_cloud/amber_dev60/` |
| 5.2절, 표 A.7 앞 두 행 (log p 0.667·0.694, 첫 변형 0.55·0.52) | `decoding/pilot/collect.py` → `decoding/pilot/offline.py` | `outputs/rbd-20261030/collect/{qwen,gemma}-offline.json` |
| 부록 B.4 dev 결과 (λ, measured vs constant r) | `data/probe_set.py` → `data/chair_dev.py`, `decoding/pilot/direction.py`, `decoding/pilot/decode.py`, `eval/dev.py` | `outputs/rbd-20261030/dev/` |
| 부록 B.4 파일럿 (CHAIR 테스트 앞 100장 30.0→18.0, uniform 18.0, recall 58.2 vs 53.7; AMBER 1–100 Cover −5.2~−7.6) | `decoding/pilot/decode.py`(λ 0.1), 클라우드 박스 2080 Ti | `results/rbd_20261101/round1_test_pilot/score-{chair-test100,amber100}.json`, 표는 `results/rbd_20261101/README.md` 1절 |

## 표 5: CHAIR 500 (왼쪽)

| 행 | 코드 | 결과 |
|---|---|---|
| Qwen2.5 REVIS†·RUDDER (BF16 eager), GrAInS (FP16) | 생성 `baselines/worker_qwen.py`(`qwen_loader.py`), 방향 `baselines/fit_qwen.py`, `baselines/grains_qwen.py`, 원저자 함수 연결 `baselines/author_bridge.py`·`qwen_compat.py`, FP16 attention `baselines/fp16_attention.py`; 채점 `eval/official.py`, `eval/chair_scorer.py` | `outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386/original-vti-gpu1-final/chair/*.chair.official.json`, `metrics.csv` |
| Qwen2.5 VTI (공백 정규화 데모) | `baselines/vti_fix.py`, `vti_chair.sh` → `baselines/vti_ws.py` | `R/runs/chair_test500_vtiws/`, 데모 `R/vti_fix/` |
| Qwen2.5 Base (TETHER 코드 경로)·TETHER `s16a2bl1p05` | 세트 `data/sets_500.py`; 0–149행 `pb_worker.sh`·`pb_ext.sh`·`pb_pool.sh` → `experiments/join_150.py`, 150–499행 `pb_run500*.sh` → `experiments/join_500.py` → `pb_score500.sh`/`eval/sets.py` | `R/runs/chair_test500_all/score.json`. 설정 선택(0–149행): `R/runs/chair_test60_sweep/`, `R/runs/chair_test150_all/` |
| Gemma Base·REVIS†·RUDDER·VTI† (BF16) | 생성 `baselines/worker_gemma.py`(`author_bridge.py`); REVIS·VTI 자체 구현 `baselines/local.py`; 채점 `eval/official.py` | `outputs/author-clone-rerun-20261027-165446/bf16-full500-20261029-125744/bf16_common/*.chair.official.json` |
| Gemma GrAInS (FP16) | `baselines/grains_gemma.py` → `baselines/worker_grains_fp16.py` | `outputs/gemma-grains-fp16-v2-20261029-1793246507/chair/` |
| Gemma TETHER `s24a1bl1p05` | `gchair3.sh` (0–59행 8개 후보 → `experiments/gemma_chair_pick.py` → 60–499행) + `gchair_pool.sh`, `experiments/gemma_chair_collect.py`, `experiments/gemma_chair_join.py`, `eval/sets.py` | 표의 모든 Gemma 행을 같은 이미지로 채점한 `R/runs/gchair_test500_tbl/score.json`, 선택 `R/runs/gchair_t60/` |
| Qwen3 Base·TETHER `s20a1bl1p05` | `qchair3.sh` (0–59행 9개 후보 → 자동 선택 → 60–499행) + `qchair_pool.sh`, `experiments/qwen3_chair_collect.py` | `R/runs/qchair_test500_all/score.json`, 선택 `R/runs/qchair_t60/` |
| Qwen3 RUDDER·REVIS(자체)·VTI | `baselines/bench_qwen3.py`, `qwen3_run_baselines.sh`, `run_vti_gpu23.sh`, 보고 `experiments/reports/qwen3_chair.py` | `R/runs/qchair_test500_tbl/` |
| Qwen3 GrAInS, 공식 REVIS, 최종 표 | `qbl_chain.sh`: `baselines/grains_qwen3.py`(FP32) → `baselines/bench_qwen3_official.py`(FP16 eager); `qbl_revis.sh` → `baselines/revis_official.py`; 보고 `experiments/reports/qwen3_official.py` | `R/runs/qchair_bl2/`, 표 `R/logs/qbl.log` (Qwen3 행 전부) |

## 표 5: CHAIR dev 100 (오른쪽)

`cdev_chain.sh` + `experiments/chair_dev.py`: Base, RUDDER, GrAInS, VTI, TETHER(각 백본의 표 설정)를 세 백본에서 실행, 채점 `eval/chair_subset.py`.
REVIS는 `after_v7.sh` 1단계(`hat_pool.sh`, 큐 `queues/chair_dev100_revis.txt`) → `experiments/chair_dev.py score`. Qwen2.5·Gemma는 자체 구현, Qwen3는 공식 코드.
결과 `R/runs/chair_dev100_{qwen,qwen3,gemma}_all/score.json`, TETHER와 Qwen2.5 VTI-ws의 별표는 TETHER 코드 경로 Base 대비 `R/runs/chair_dev100_*_vdd/score.json`.
표 셀 `R/logs/cdev_table_{qwen,qwen3,gemma}.tex`(논문 표와 모든 칸이 같음), 로그 `R/logs/cdev.log`.
Qwen2.5 REVIS는 THRONE에서 같은 GPU 종류의 Base와 1,000/1,000 같은 캡션이었고, dev에서는 Base(A5000)와 REVIS(A6000)의 GPU 수치 차이로 96/100이 다릅니다.

## 표 6: Ablation (Qwen2.5 CHAIR 500)

`abl_chain.sh` + `experiments/chair_ablation.py` (`decoding/vdd_vobj.py`, 큐 `queues/abl_chair.txt`): 전체 TETHER `s16a2bl1p05`에서 한 요소씩 빼거나 하나만 남긴 7개 변형.
결과 `R/runs/chair_abl500_vsfull/score.json`(전체 TETHER 대비 CI, 표 6의 CI), `R/runs/chair_abl500_all/score.json`(Base 대비), 로그 `R/logs/abl_chair.log`.
표 6에 없는 네 변형(belief only, penalty only, steering only, constant schedule)도 같은 파일에 있습니다.

## 표 7: THRONE

| 단계 | 코드 | 결과 |
|---|---|---|
| 세트·이미지 | `data/throne_full.py`, `data/throne_images.py` | `R/sets/throne_full1000.jsonl`, `throne_t200.jsonl`, `throne_t800.jsonl` |
| Qwen 베이스라인 | `pb_bl_chain.sh` → `pb_bl_pool.sh` → `baselines/bench_qwen.py` → `experiments/baselines_join.py`/`experiments/baselines_throne_join.py` → `pb_bl_full_score.sh` | `R/runs/throne_full1000/` (캡션, `eval/*/` 판정) |
| Qwen 첫 번째 설정 `s16a1bl2p1` | 0–499행 `throne_test500`, 500–999행 `pb_full1000_ours.sh` → `experiments/throne1000_join.py` → `pb_full1000_finish.sh` | `R/runs/throne_test500_all/`, `R/runs/throne_full1000_ours/`, 선택 150장 `R/runs/throne_test150_all/` |
| Gemma 베이스라인·첫 번째 설정 `s24a1bl2p1` | `gemma_chain.sh`(선택 `experiments/gemma_throne_pick.py`, throne_full 4940–4999행) → `gemma_pool.sh`, `baselines/bench_gemma.py`, `gemma_ours_fix.sh`, `gemma_split.sh`, `experiments/gemma_throne_join.py`, 판정 `gemma_bl_judge.sh` | `R/runs/gemma_throne1000/`, 선택 `R/runs/gemma_throne_dev60/` |
| TETHER `cf_a3bl2p1` | `experiments/throne_cf_setup.py`(V_obj) → `vdd4_chain.sh` → `experiments/throne_cf.py`·`vdd4_pool.sh`·`decoding/vdd_vobj.py`, `vdd4_qwen_resume.sh`, 채점 `eval/throne_metrics.py`. 0–199행에서 선택, 200–999행 보고 | 선택 `R/runs/{qwen,gemma}_cf_t200_all/`, `R/runs/*_cf_pick.txt`, 1,000장 `R/runs/{qwen,gemma}_cf_test_all/`, 표 `R/logs/v4_{qwen,gemma}_t800_table.txt` (표 7의 2-judge 열과 모든 칸이 같음) |
| 판정 (Flan-T5 base·large, 임계값 6) | `eval/throne_judge.py`(`RBD2_JUDGES`), `eval/throne.py`, `eval/throne_partial.py` | `R/diag/*_2judges/` |
| xl 열 (base·large·xl, 임계값 9) | `after_v7.sh` 2단계 `xl_pool.sh` → `eval/throne_xl_judge.py` → `eval/throne_xl.py` (rows 200–999) | `R/runs/*/eval/*/combined/*flan-t5-xl.json`, 표 `R/logs/throne_xl_table.txt` (표 7의 xl 열과 같음) |
| ALL·CLS 지표, orange 분석 | `scripts/rbd2/analysis/throne_all_metrics.sh`, `throne_official_scorer.sh`(공식 `throne_score_aqa.py`), `throne_classwise_ci.sh`, `throne_orange.sh`, `throne_without_orange.sh`, `throne_breakdown.sh` | 출력은 표에 직접 |
| **5,000장으로 확장 (진행 중)** | `t4k_chain.sh`: `experiments/throne_5k.py`(세트 `throne_r4000` = throne_full 1000–4999행, 큐) → `t4k_pool.sh`(디코딩, 먼저 저장된 A6000 행 재현 확인) → `t4k_jpool.sh`·`eval/throne_5k_judge.py`(블록별 판정) → `eval/throne_5k.py` | 서버 `R/logs/t4k.log`(T5K_DONE), `R/logs/t4k_table.txt`. 아직 결과 없음 |

`eval/throne_5k.py`는 rows 200–999(현재 표 확인용), 200–N(보고), 0–N, 1000–N을 출력합니다. `--blocks 0`으로 현재 표 7의 모든 숫자를 재현했습니다.
GPU 0, 5(A6000 두 장), 디코딩 약 374 GPU-시간 + 판정 약 126 시간으로 추정해 11-09 기준 4–5일 걸립니다.

## 부록 C, D

| 표 | 코드 | 결과 |
|---|---|---|
| 표 C.1 설정 | `baselines/local.py`(REVIS, Gemma VTI), `baselines/author_bridge.py`, `baselines/qwen_compat.py`, `baselines/vti_fix.py` | — |
| 표 C.2 (VTI 강도, 앞 50장) | α=0.9: `baselines/vti_author_prep.py` → `baselines/vti_author.py`; α=0.45·0.225: `baselines/vti_alpha_sweep.py` → `eval/chair_scorer.py` | `outputs/qwen-original-vti-1793260357/`, `outputs/qwen-vti-alpha-sweep-20261030/score-pilot-chair50/score.json` |
| 부록 C의 VTI 데모 줄바꿈 88/100 vs 19/100, 7점 grid | `baselines/vti_fix.py`, `vti_chair.sh` | `R/vti_fix/`, `R/runs/chair_dev60_vtiws/` |
| 표 D.1, 사례 1–4 | `scripts/rbd2/analysis/case_study_extract.sh` (같은 CHAIR scorer로 방법별 객체 추출), Yes-No margin은 `caption-selfcheck-20261030/qwen-base-rows.jsonl` | 위 CHAIR 500 캡션 |

## 초안 3에 없는 실험

| 실험 | 코드 | 결과 |
|---|---|---|
| HAT + ALOHa 385장 | `hat_setup.sh`, `hat_gpu3.sh`/`hat_chain.sh`, `hat_pool.sh`·`hat_join.sh`, `experiments/hat.py`, `eval/hat_parse.py`, `eval/hat_aloha.py`, `eval/hat_common.py`, `hat_smoke.sh` | `R/runs/hat385_{qwen,gemma}/`, `R/hat385/`, `R/hat_parse/`, `R/logs/hat385.log` |
| HAT 두 번째 선택 | `decoding/vdd_cf.py`, `hat_v7_chain.sh`, `experiments/hat_reselect.py` | `R/hat385/score-qwen-v7-*.json`, `R/logs/hat_v7.log` |
| ALOHa 집계 버그 수정 | `eval/hat_aloha.py`(`HAT_ALOHA_FIX=1`), `R/hat_fix.sh` | `R/hat385/scorefix-*.json`, `R/logs/hat_aloha_fixed_table.tex` |
| AMBER 생성형 1,004장 | `pb_amber_bl.sh` → `pb_amber_pool.sh` → `baselines/bench_qwen.py`, `amber_full_score.sh`; `pb_amber_ours1004.sh` | `R/runs/amber_full1004_all/`, `R/runs/amber_full1004_ours_all/`, `R/runs/amber_test150_all/` |
| Gemma CHAIR 500 ablation | `gabl_chain.sh` + `experiments/chair_ablation_gemma.py`, 이어서 `after_v7.sh` 3단계 | `R/runs/gabl_chair500_{all,vsfull}/`, `R/logs/gabl_chair_table.tex` |

`hat_pool.sh`는 CHAIR dev 100의 REVIS 행(표 5)에도 쓰였습니다.

## 넣지 않은 것

- 논문에 없는 실험: self-verification 디코딩, MMVP, MMHal-Bench, five-profile POPE 결과, AMBER vdd3 변형, Qwen3 탐색 rbd5·rbd6,
  Gemma CHAIR 재선택 `gchair2.sh`, VTI-ws THRONE 100장, 진단 스크립트, recall 분석, HAT 파일럿.
- 그림 1 a5 Jacobian-lens 파일럿(Qwen3.5-4B, 별도 GPU 박스에서 실행)의 코드와 결과.
- 베이스라인 실행 관리 스크립트(GPU 배치, 중단 후 이어 돌리기, 실행 기록 검사). 생성·추출·채점 코드는 `baselines/`, `eval/`에 있습니다.
- M3ID·VCD CHAIR 500: 코드는 11-09에 준비했고 아직 실행 전입니다.
- 중간 산출물: 부분 실행 폴더(`*_parts`, 행 구간별 폴더), 60·120장 중간 채점과 그 xl 판정, 스모크 테스트 결과, 판정 전 백업 `combined.bak-1108`, 블러 이미지 캐시 `R/cf/`.

## 원저자 코드와 채점기 (포함하지 않음)

| 이름 | 고정 버전 |
|---|---|
| RUDDER | Akko000/RUDDER-Residual-Update-Directed-DEcoding-Regulation- `bd149d8` |
| GrAInS | duykhuongnguyen/GrAInS `8ad78db` |
| VTI | shengliu66/VTI `bb5dfee` |
| REVIS (공식, Qwen3 행) | antgroup/Revis `75f4d29` |
| CHAIR | Python 3 포트, `4087a26` (서버 `upstream/chair/chair.py`) |
| THRONE | amazon-science/THRONE `45badbb7` (one-click.sh 2–3단계) |
| AMBER | 공식 생성형 평가 (`inference.py`, spaCy `en_core_web_lg`) |
| ALOHa | DavidMChan/aloha `7da6b90` |

모델 revision: Qwen2.5-VL-7B-Instruct `cc59489`, Gemma 4 E4B-it `ee0ef60`, Qwen3-VL-8B-Instruct `0c351dd` (`configs/*.json`).
