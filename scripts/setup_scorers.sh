#!/bin/bash
set -euo pipefail
L=/home/project-b/LJW; B=$L/bench; V=$L/venvs; UV=$L/.local/bin/uv
export UV_CACHE_DIR=$L/.cache/uv PIP_CACHE_DIR=$L/.cache/pip HF_HOME=$L/.cache/huggingface UV_PYTHON_INSTALL_DIR=$L/.local/share/uv/python UV_HTTP_TIMEOUT=600
PYBIN=$(readlink -f /home/project-b/LJW/vlm-grounding-steering/.venv/bin/python)
mkdir -p $B/models $B/coco2017/annotations $B/pkgs $V
echo "$(date +%T) THRONE repo"
[ -d $B/THRONE/.git ] || git clone -q https://github.com/amazon-science/THRONE $B/THRONE
git -C $B/THRONE checkout -q 45badbb71c2b3e8d6619f6b45cccdac24dd20b46; git -C $B/THRONE log -1 --format='THRONE %H'
echo "$(date +%T) throne venv"
[ -x $V/throne/bin/python ] || $UV venv -q --python $PYBIN $V/throne
$UV pip install -q --python $V/throne/bin/python 'torch==2.8.0+cu128' --index-url https://download.pytorch.org/whl/cu128
$UV pip install -q --python $V/throne/bin/python "transformers==4.37.2" "tokenizers<0.19" "huggingface-hub<0.25" "numpy==1.26.4" \
  "pycocotools==2.0.8" "accelerate==0.21.0" "sentencepiece==0.2.0" "safetensors<0.5"
$V/throne/bin/python -c "import torch,transformers,pycocotools,accelerate;print('throne env ok',torch.__version__,transformers.__version__)"
echo "$(date +%T) coco2017 annotations"
J=$B/coco2017/annotations/instances_val2017.json
if [ ! -s $J ]; then Z=$B/coco2017/annotations_trainval2017.zip
  [ -s $Z ] || curl -sSL --retry 5 -o $Z http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  $V/throne/bin/python -c "import zipfile;z=zipfile.ZipFile('$Z');open('$J','wb').write(z.read('annotations/instances_val2017.json'))"; rm -f $Z; fi
echo "$(date +%T) judges"
for spec in base:7bcac572ce56db69c1ea7c8af255c5d7c9672fc2 large:0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a xl:7d6315df2c2fb742f0f5b556879d730926ca9001; do
  s=${spec%%:*}; rev=${spec#*:}
  $V/throne/bin/python -c "
from huggingface_hub import snapshot_download
p=snapshot_download('google/flan-t5-$s',revision='$rev',local_dir='$B/models/flan-t5-$s',local_dir_use_symlinks=False,
  allow_patterns=['*.safetensors','config.json','generation_config.json','spiece.model','tokenizer.json','tokenizer_config.json','special_tokens_map.json','model.safetensors.index.json'])
print('judge ok',p)"
done
du -sh $B/models/* $V/*
echo "$(date +%T) BENCH_SETUP_DONE"
