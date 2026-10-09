#!/bin/bash
#SBATCH -J robonav-batch
#SBATCH -p gpu
#SBATCH --gres=gpu:a4500:4
#SBATCH -c 32
#SBATCH -o /mnt/beegfs/home/friesner/bgl2126/robonav/jobs/logs/batch-%j.out
# N parallel stacks on one node: LightNav server k (GPU k, :8050+k) + headless sim k (:8088+k),
# then one landmark.navigate worker per stack (--shards N --shard k). No --mem: SLURM here reports 1 MB.
# Usage: sbatch [--gres=gpu:a4500:2] batch.sbatch <run_name> [N=4]
set -uo pipefail
RUN=${1:-batch1}; N=${2:-4}
ROOT=$HOME/robonav; LOGS=$ROOT/jobs/logs
module load singularity/3.6.3
IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
echo "node $(hostname)  gpus ${CUDA_VISIBLE_DEVICES}  run $RUN  shards $N  $(date)"
echo "$(hostname)" > $ROOT/jobs/batch_node

PIDS=()
for k in $(seq 0 $((N-1))); do
  ( cd $ROOT/LightNav-0 && CUDA_VISIBLE_DEVICES=${GPUS[$k]} HOST=127.0.0.1 PORT=$((8050+k)) \
      singularity exec --nv $ROOT/containers/py311.sif \
      .venv/bin/lightnav-serve --task vln --model_path checkpoints/LightNav-0 --backend vllm_local \
      > $LOGS/${RUN}_lightnav$k.log 2>&1 ) &
  PIDS+=($!)
done

for k in $(seq 0 $((N-1))); do
  echo -n "waiting for LightNav $k on :$((8050+k)) "
  for _ in $(seq 1 240); do
    (echo > /dev/tcp/127.0.0.1/$((8050+k))) 2>/dev/null && { echo "up $(date +%T)"; break; }
    kill -0 ${PIDS[$k]} 2>/dev/null || { echo "server $k died, see ${RUN}_lightnav$k.log"; exit 1; }
    sleep 5
  done
done

cd $ROOT/LightNav-0/mujoco_demo
for k in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$k]} MUJOCO_EGL_DEVICE_ID=$k MUJOCO_GL=egl \
    $ROOT/robo-nav/.venv-loc/bin/vln-mujoco --host 127.0.0.1 --port $((8088+k)) \
    --vln-server ws://127.0.0.1:$((8050+k)) > $LOGS/${RUN}_sim$k.log 2>&1 &
  PIDS+=($!)
done

for k in $(seq 0 $((N-1))); do
  echo -n "waiting for sim $k health "
  for _ in $(seq 1 120); do
    curl -s localhost:$((8088+k))/api/health | grep -q '"ok": *true' && { echo "ok $(date +%T)"; break; }
    sleep 5
  done
  curl -s localhost:$((8088+k))/api/health; echo
done

set -a; source $HOME/.secrets/anthropic.env; set +a
cd $ROOT/robo-nav
WORKERS=()
for k in $(seq 0 $((N-1))); do
  .venv-loc/bin/python -m robo_nav.landmark.navigate \
    --scene $ROOT/LightNav-0/mujoco_demo/vln_mujoco/assets/scenes/procthor-10k-val/val_2.json \
    --map sim_data/landmark/val2 --starts sim_data/loc/val2/queries.csv \
    --n_starts 6 --goals_per_start 2 \
    --modes zeroshot zeroshot_retry landmark landmark_nounstick \
    --shards $N --shard $k --sim_ws ws://127.0.0.1:$((8088+k))/ws \
    --out sim_data/landmark/runs/$RUN > $LOGS/${RUN}_shard$k.log 2>&1 &
  WORKERS+=($!)
done
echo "workers ${WORKERS[*]} started $(date +%T)"
wait "${WORKERS[@]}"
echo "all workers done $(date)"
kill "${PIDS[@]}" 2>/dev/null
