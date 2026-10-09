set -euo pipefail
export PATH=$HOME/.local/bin:$PATH UV_LINK_MODE=copy
ROOT=~/robonav; cd $ROOT/robo-nav
[ -d .venv-loc ] || uv venv .venv-loc --python 3.11
uv pip install --python .venv-loc/bin/python -r requirements.txt \
  mujoco scipy anthropic kornia einops psutil faiss-cpu fast_pytorch_kmeans \
  pytorch_lightning pytorch_metric_learning \
  "lightglue @ git+https://github.com/cvg/LightGlue.git" \
  -e "$ROOT/LightNav-0/mujoco_demo"
