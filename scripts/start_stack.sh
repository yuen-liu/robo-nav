#!/usr/bin/env bash
# Launch the LightNav-0 server and the headless MuJoCo sim in detached tmux sessions.
#
#   GPU=0 ROOT=~/robonav bash scripts/start_stack.sh
#
# tmux sessions: "lightnav" (model server, :8050) and "sim" (sim + web UI, :8088).
# Logs: $ROOT/LightNav-0/lightnav.log and $ROOT/LightNav-0/sim.log.
# To watch the sim from a laptop: ssh -N -L 8088:localhost:8088 <this host>, open http://127.0.0.1:8088
set -euo pipefail

ROOT=${ROOT:-$HOME/robonav}
GPU=${GPU:?set GPU to a free GPU index (see nvidia-smi)}

if ! tmux has-session -t lightnav 2>/dev/null; then
  tmux new-session -d -s lightnav \
    "cd $ROOT/LightNav-0 && PORT=8050 CUDA_VISIBLE_DEVICES=$GPU .venv/bin/lightnav-serve \
       --task vln --model_path checkpoints/LightNav-0 --backend vllm_local 2>&1 | tee lightnav.log"
  echo "started lightnav server on GPU $GPU"
fi

echo -n "waiting for the LightNav server on :8050 "
for _ in $(seq 1 180); do
  if (echo > /dev/tcp/127.0.0.1/8050) 2>/dev/null; then echo " up"; break; fi
  echo -n "."; sleep 5
done

if ! tmux has-session -t sim 2>/dev/null; then
  tmux new-session -d -s sim \
    "cd $ROOT/LightNav-0/mujoco_demo && MUJOCO_GL=egl $ROOT/robo-nav/.venv-loc/bin/vln-mujoco \
       --vln-server ws://127.0.0.1:8050 2>&1 | tee $ROOT/LightNav-0/sim.log"
  echo "started sim on :8088"
fi
tmux ls
