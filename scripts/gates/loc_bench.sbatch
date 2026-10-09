#!/bin/bash
#SBATCH -J robonav-loc
#SBATCH -p gpu
#SBATCH --gres=gpu:a4500:1
#SBATCH -c 8
#SBATCH -o /mnt/beegfs/home/friesner/bgl2126/robonav/jobs/logs/loc-%j.out
# Usage: sbatch loc_bench.sbatch [methods...]   (default: the non-API methods)
# VLM methods read ANTHROPIC_API_KEY from ~/.secrets/anthropic.env.
set -euo pipefail
ROOT=$HOME/robonav; cd $ROOT/robo-nav
PY=.venv-loc/bin/python; DS=sim_data/loc/val2
SCENE=$ROOT/LightNav-0/mujoco_demo/vln_mujoco/assets/scenes/procthor-10k-val/val_2.json
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
echo "node $(hostname) $(date)"
[ -f $DS/queries.csv ] || $PY -m robo_nav.sim.render $DS --scene $SCENE
METHODS=${@:-prior dinov2 dinov2_full dinov2_look dinov2_full_look salad salad_look anyloc anyloc_look salad_lg salad_lg_look}
if [[ "$METHODS" == *vlm* ]]; then
  [ -f $HOME/.secrets/anthropic.env ] && source $HOME/.secrets/anthropic.env
  [ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "no ANTHROPIC_API_KEY (~/.secrets/anthropic.env)"; exit 1; }
fi
$PY -m robo_nav.loc.benchmark $DS --methods $METHODS
