#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONPATH=$PWD/src${PYTHONPATH:+:$PYTHONPATH}
$PY -m tether.upstream chair throne
$PY -c "import nltk; [nltk.download(r, quiet=True) for r in ('wordnet', 'punkt', 'punkt_tab', 'averaged_perceptron_tagger_eng')]"
python3 -m venv .venv-throne
.venv-throne/bin/pip install -q "torch==2.8.0" "transformers==4.37.2" "tokenizers<0.19" "huggingface-hub<0.25" \
  "numpy==1.26.4" "pycocotools==2.0.8" "accelerate==0.21.0" "sentencepiece==0.2.0" "safetensors<0.5"
for spec in base:7bcac572ce56db69c1ea7c8af255c5d7c9672fc2 large:0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a xl:7d6315df2c2fb742f0f5b556879d730926ca9001; do
  .venv-throne/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('google/flan-t5-${spec%%:*}', revision='${spec#*:}', local_dir='outputs/judges/flan-t5-${spec%%:*}')"
done
echo "export THRONE_PYTHON=$PWD/.venv-throne/bin/python"
