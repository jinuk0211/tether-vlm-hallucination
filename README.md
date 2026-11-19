# TETHER

Code for *TETHER: Steering Vision-Language Models Where the Image Is Read Less*.

TETHER decodes every caption together with a gray-image copy that receives the same tokens. At each step it

```
steering   h_L += α·s(t)·(h_L − h̄_L)                      decoder layer L, decoding steps, s(t) = 1 − r(t)
support    A(o) += r(t)·(log p(o) − log p̄(o))                object tokens o in V_obj
correction z(o) += w(t)·(λ·clip(A(o) / Σr, −c, c) − π)        objects not written yet, w(t) = 1 − r(t)
```

where `h̄`, `p̄` come from the gray row and `r(t)` is the gray-image KL along the caption, normalized to 1 at the
opening (measured once per model from teacher-forced Base captions). With α = λ = π = 0 the same code path decodes Base.

## Layout

```
src/tether/
  decode.py        TETHER (and Base) decoding, with the steering at any point of the decoder block
  lockstep.py      real/contrast pair decoding shared by TETHER, VCD and M3ID
  schedule.py      r(t): measured from the teacher-forced Base captions, read by decode.py
  vocab.py         object vocabulary V_obj
  models.py        Qwen2.5-VL, Qwen3-VL, Gemma 4
  baselines/       generate.py (captions with any baseline), fit.py (directions), revis, rudder, grains, vti,
                   contrastive (VCD, M3ID), authors (loads the released code), fp16
  eval/            CHAIR with paired bootstrap, THRONE judging and table
  data/            CHAIR test 500, CHAIR dev 100, THRONE 5,000, development pairs
  experiments/     queues and joins for CHAIR, the ablation, CHAIR dev and THRONE
configs/           models, and the TETHER setting of each model (tether.json)
scripts/           run_queue.sh, setup_scorers.sh
```

Paths are relative to the repository root (`TETHER_ROOT` to change it): `data/` for images and sets, `outputs/` for
everything a run writes, `upstream/` for the released code of the baselines and the scorers.

## Setup

```
pip install -e ".[eval,baselines]"
python -m tether.upstream                 # pinned checkouts of RUDDER, GrAInS, VTI, CHAIR, THRONE
bash scripts/setup_scorers.sh             # NLTK data, THRONE environment and Flan-T5 judges
python -m tether.data.sets                # images, annotations and all evaluation sets
python -m tether.vocab                    # V_obj for each model
```

Commands are run with `python -m MODULE`. A queue file holds one `MODULE ARGS` per line and runs on one GPU with
`bash scripts/run_queue.sh GPU QUEUE`. Decoders write `TAG-START-END.jsonl` and resume an unfinished file.

## Experiments

```
python -m tether.experiments.chair queue --table base --gpus 4    # Base captions of the CHAIR test images
bash scripts/run_queue.sh 0 outputs/runs/queues/chair_test500-base-0.txt
python -m tether.experiments.chair join --table base
python -m tether.schedule --model qwen                            # r(t) of each model (also qwen3, gemma)

python -m tether.baselines.fit --model qwen --method revis        # directions: revis, grains (all models),
                                                                  # vti-authors (Qwen2.5, Qwen3), vti (Gemma)
python -m tether.experiments.chair queue --table test --gpus 4    # CHAIR 500: REVIS, RUDDER, GrAInS, VTI, M3ID, TETHER
python -m tether.experiments.chair join --table test
python -m tether.eval.chair data/sets/chair_test500.jsonl outputs/runs/chair_test500/qwen

python -m tether.experiments.chair queue --table ablation         # -steering, -penalty, -belief (Qwen2.5, Gemma)
python -m tether.experiments.chair queue --table dev              # CHAIR on the 100 development images

python -m tether.experiments.throne queue --gpus 4                # THRONE: every row on all 5,000 images
python -m tether.experiments.throne join
python -m tether.experiments.throne judges --gpus 4               # Flan-T5 base, large, xl
python -m tether.eval.throne table --qwen outputs/runs/throne/qwen --gemma outputs/runs/throne/gemma
```

Every TETHER row, on CHAIR and THRONE, uses the setting of its model in `configs/tether.json`. VCD runs with
`python -m tether.baselines.contrastive --method vcd`.

## Steering sites

`tether.decode` applies the steering update `x_real += α·(1 − r(t))·(x_real − x_gray)` at the point given by
`--site` (default `resid`, the TETHER rows of the paper):

| site | point in decoder layer `--layer` |
|---|---|
| `resid` | layer output (TETHER) |
| `attn`, `mlp` | what attention and the MLP add to the residual stream (after the post-norms in Gemma) |
| `ple` | Gemma's per-layer input block |
| `heads` | input of `o_proj`, one unit per head |
| `q`, `k`, `v` | queries, keys, values (after `q_norm`, `k_norm`, `v_norm` where the model has them), one unit per head |
| `gate`, `up`, `act` | `g = W_gate x`, `u = W_up x` and `a = act(g) ⊙ u`, the input of `down_proj` |

`--units` limits the update to the given heads or neurons, and `--topk K` to the K units with the largest real–gray
difference at each step. `--layer` takes several layers. In Qwen2.5-VL and Qwen3-VL, `o_proj` and `down_proj` are
linear without bias, so `heads` and `act` over all units equal `attn` and `mlp`. Gemma 4 E4B has keys and values
only in layers 1–24; layers 25–42 reuse those of layers 23 and 24.

## Released code

| | |
|---|---|
| RUDDER | Akko000/RUDDER-Residual-Update-Directed-DEcoding-Regulation- `bd149d8` |
| GrAInS | duykhuongnguyen/GrAInS `8ad78db` |
| VTI | shengliu66/VTI `bb5dfee` |
| CHAIR | Maxlinn/CHAIR-metric-standalone `4087a26` |
| THRONE | amazon-science/THRONE `45badbb` |

Models: Qwen2.5-VL-7B-Instruct `cc59489`, Qwen3-VL-8B-Instruct `0c351dd`, Gemma 4 E4B-it `ee0ef60`.
