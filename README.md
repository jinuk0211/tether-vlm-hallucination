# TETHER

TETHER is a decoding-time method for object hallucination in VLM image captioning. The image's contribution to a
caption fades as the caption gets longer; TETHER measures that decay offline (r(t), from the KL between predictions
with the real and a gray image) and uses it to schedule three terms while decoding:

```
steering   h_real += α·s(t)·(h_real − h_gray)                at decoder layer L, decode steps, s(t) = 1 − r(t)
belief     A(o) += r(t)·(log p_real(o) − log p_gray(o))       object tokens o in V_obj
           logit(o) += λ·w(t)·clip(A(o) / Σr, −c, c)          objects not written yet, w(t) = 1 − r(t)
penalty    logit(o) −= π·w(t)                                 objects not written yet
```

Each caption is decoded as a batch of two in lockstep (real image and gray image, the gray row is always fed the real
row's token), so the evidence comes without an extra pass. α = λ = π = 0 is plain greedy decoding through the same path.

The repository also has the baselines we compare against (REVIS, RUDDER, GrAInS, VTI), the evaluation code (CHAIR,
POPE, THRONE) and the analyses of how the image's influence changes along a caption.

## Layout

```
src/tether/
  models/       Hugging Face adapters (Qwen2.5-VL, Qwen3-VL, Gemma 4)
  decoding/     TETHER decoders: vdd (main), vdd_vobj (V_obj read from a file); common (r(t) curve, rule for the object
                vocabulary V_obj); vobj (writes the V_obj files, also the one without colour words used on THRONE)
  baselines/    local (our REVIS and VTI), author_bridge / qwen_compat (calls into the authors' RUDDER and GrAInS),
                vti_author, vti_ws, revis_official, direction extraction (grains_*, fit_qwen), generation workers, bench_*
  eval/         CHAIR, POPE and THRONE scoring, paired image bootstrap
  analysis/     visual_decay, reliance, read_gap, selfcheck, probe, logit_lens
  data/         evaluation sets and images (CHAIR 500, CHAIR dev 100, THRONE 5,000, POPE)
  experiments/  queues and joins of the individual experiments (ablation, CHAIR dev, THRONE)
scripts/        run_queue.sh, visual_decay_run.sh, scorer environment setup
configs/        model configurations
```

## Setup

```
pip install -e .            # with the scorer dependencies: pip install -e ".[eval]"
bash scripts/setup_scorers.sh   # THRONE environment and judges (flan-t5)
python -m tether.upstream chair   # pinned checkout into upstream/chair (also: revis, rudder, grains, vti)
```

The authors' code and the official scorers are not included; the modules expect local checkouts of these versions
(paths at the top of each module):

| | pinned version |
|---|---|
| RUDDER | Akko000/RUDDER-Residual-Update-Directed-DEcoding-Regulation- `bd149d8` |
| GrAInS | duykhuongnguyen/GrAInS `8ad78db` |
| VTI | shengliu66/VTI `bb5dfee` |
| REVIS | antgroup/Revis `75f4d29` |
| CHAIR | Python 3 port, `4087a26` |
| THRONE | amazon-science/THRONE `45badbb7` |

Models: Qwen2.5-VL-7B-Instruct `cc59489`, Gemma 4 E4B-it `ee0ef60`, Qwen3-VL-8B-Instruct `0c351dd` (`configs/*.json`).
Paths (data, model folders, caches) follow our server layout and are set at the top of each module.

## Usage

Every module runs with `python -m`. A setting tag `s{L}a{α}bl{λ}p{π}` stands for `--layer L --alpha α --lam λ --pen π`
(`p05` = 0.5).

```
# evaluation sets: POPE and CHAIR 500, THRONE 5,000 (COCO val2017); object vocabulary V_obj
python -m tether.data.benchmarks && python -m tether.data.test_sets && python -m tether.data.throne_full
python -m tether.decoding.vobj

# TETHER and the Base of the same code path on a caption set (resumes an unfinished file)
python -m tether.decoding.vdd --model qwen --data SET.jsonl --output RUNDIR --tag s16a2bl1p05 \
    --alpha 2 --layer 16 --lam 1 --pen 0.5
python -m tether.decoding.vdd --model qwen --data SET.jsonl --output RUNDIR --tag base

# a baseline
python -m tether.baselines.bench_qwen --method rudder_clone --attn sdpa --set SET.jsonl --output RUNDIR

# CHAIR for every complete run in RUNDIR against its 'base' run (paired bootstrap)
python -m tether.eval.sets chair_test500 RUNDIR

# THRONE on all 5,000 images, in five blocks of 1,000: decode and judge queues, one judge unit
# (model, tag, block, flan-t5 judge), joined caption files, table
python -m tether.experiments.throne queues
python -m tether.eval.throne_judge 0 qwen cf_a3bl2p1 0 base
python -m tether.experiments.throne join
python -m tether.eval.throne

# a queue of such commands on one GPU (one "MODULE ARGS..." per line)
bash scripts/run_queue.sh 0 queue.txt
```

Every decoder and baseline runs the whole set by default. `--start A --end B` decodes rows A to B-1 into
`TAG-A-B.jsonl`, so a set can be split over several GPUs; `python -m tether.experiments.join SET PARTS OUT TAG...`
(run folder names) puts the pieces back into one `TAG-0-N.jsonl`, which is what the scorers read.

`--attn sdpa` is the default for the decoders: with eager attention, bf16 Qwen2.5-VL starts repeating the token
` addCriterion` in some captions.

The result files and the run scripts of each experiment are on the `results` branch.
