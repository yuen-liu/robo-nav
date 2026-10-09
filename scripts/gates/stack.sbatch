#!/bin/bash
#SBATCH -J robonav-stack
#SBATCH -p gpu
#SBATCH --gres=gpu:a4500:1
#SBATCH -c 8
#SBATCH -o /mnt/beegfs/home/friesner/bgl2126/robonav/jobs/logs/stack-%j.out
# LightNav-0 server (vLLM, inside the glibc-2.36 container) + headless MuJoCo sim with web UI on :8088.
# Both bind 127.0.0.1 on the compute node; reach the UI from a laptop via an SSH jump through gates.
set -uo pipefail
ROOT=$HOME/robonav
module load singularity/3.6.3
echo "node $(hostname)  gpus $CUDA_VISIBLE_DEVICES  $(date)"
echo "$(hostname)" > $ROOT/jobs/stack_node

cd $ROOT/LightNav-0
HOST=127.0.0.1 PORT=8050 singularity exec --nv $ROOT/containers/py311.sif \
  .venv/bin/lightnav-serve --task vln --model_path checkpoints/LightNav-0 --backend vllm_local \
  > lightnav.log 2>&1 &
SERVER=$!

echo -n "waiting for LightNav server on :8050 "
for _ in $(seq 1 240); do
  (echo > /dev/tcp/127.0.0.1/8050) 2>/dev/null && { echo " up"; break; }
  kill -0 $SERVER 2>/dev/null || { echo " server died, see lightnav.log"; exit 1; }
  sleep 5
done

cd $ROOT/LightNav-0/mujoco_demo
MUJOCO_GL=egl $ROOT/robo-nav/.venv-loc/bin/vln-mujoco --host 127.0.0.1 --port 8088 \
  --vln-server ws://127.0.0.1:8050 > $ROOT/LightNav-0/sim.log 2>&1 &
SIM=$!
echo "sim pid $SIM on :8088"
wait -n $SERVER $SIM
echo "a process exited at $(date); stopping"; kill $SERVER $SIM 2>/dev/null
