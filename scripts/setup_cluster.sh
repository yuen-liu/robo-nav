#!/usr/bin/env bash
# One-time setup on a fresh Linux GPU box: robo-nav + LightNav-0 (server + patched MuJoCo sim).
#
#   ROOT=~/robonav bash setup_cluster.sh        # both repos end up under $ROOT
#
# Creates two environments:
#   $ROOT/LightNav-0/.venv     LightNav-0 model server (vLLM)
#   $ROOT/robo-nav/.venv-loc   everything else: sim, robo-nav experiments, localization methods
set -euo pipefail

ROOT=${ROOT:-$HOME/robonav}
BRANCH=${BRANCH:-lightnav-sim-nav}
LIGHTNAV_COMMIT=3015508
mkdir -p "$ROOT"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

[ -d robo-nav ] || git clone -b "$BRANCH" https://github.com/yuen-liu/robo-nav.git
[ -d LightNav-0 ] || git clone https://github.com/lightorigins/LightNav-0.git

# LightNav-0 at the commit the patch was made against, plus wall collisions + set_pose for the sim
cd "$ROOT/LightNav-0"
git checkout -q "$LIGHTNAV_COMMIT"
if git apply --check "$ROOT/robo-nav/patches/lightnav0_sim.patch" 2>/dev/null; then
  git apply "$ROOT/robo-nav/patches/lightnav0_sim.patch"
else
  echo "lightnav0_sim.patch already applied (or conflicts) -- check 'git -C $ROOT/LightNav-0 diff --stat'"
fi

echo "== LightNav-0 server env"
[ -d .venv ] || uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e ".[vllm,video]"
[ -d checkpoints/LightNav-0 ] || .venv/bin/hf download LightOriginsHQ/LightNav-0 --local-dir checkpoints/LightNav-0

echo "== robo-nav eval env"
cd "$ROOT/robo-nav"
[ -d .venv-loc ] || uv venv .venv-loc --python 3.11
uv pip install --python .venv-loc/bin/python -r requirements.txt \
  mujoco scipy anthropic kornia einops psutil faiss-cpu fast_pytorch_kmeans \
  pytorch_lightning pytorch_metric_learning \
  "lightglue @ git+https://github.com/cvg/LightGlue.git" \
  -e "$ROOT/LightNav-0/mujoco_demo"

echo "== smoke test"
MUJOCO_GL=egl .venv-loc/bin/python -c "
import torch, mujoco, vln_mujoco, robo_nav.sim.render, robo_nav.loc.benchmark
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| mujoco', mujoco.__version__)
from robo_nav.sim.render import OfflineCamera
OfflineCamera().render(6.5, 13.8, 0.0); print('headless sim render ok')
"
echo "Done. Next: scripts/start_stack.sh to launch the LightNav server + sim in tmux."
