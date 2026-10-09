# Running on gates (Columbia SLURM cluster)

`scripts/setup_cluster.sh` and `start_stack.sh` assume a GPU box with glibc >= 2.31 and tmux. On
gates the login node has no GPU, nodes are RHEL 8 (glibc 2.28, driver 535 / CUDA 12.2), and
everything runs as SLURM jobs. Background and benchmark results: `docs/gates-cluster.md`.

Layout assumed by these scripts (paths are hard-coded to `~/robonav`):

    ~/robonav/robo-nav       this repo (.venv-loc: sim + localization + navigate, native)
    ~/robonav/LightNav-0     LightNav-0 checkout (.venv: vLLM server, built inside the container)
    ~/robonav/containers     py311.sif + build scripts
    ~/robonav/jobs           *.sbatch, logs/
    ~/.secrets/anthropic.env ANTHROPIC_API_KEY=... (workspace-scoped key; mode 600)

## One-time setup

vLLM 0.19.1 has no wheels for glibc 2.28, so the LightNav server env lives in a Debian container:

    module load singularity/3.6.3
    export SINGULARITY_CACHEDIR=~/robonav/containers/cache SINGULARITY_TMPDIR=~/robonav/containers/tmp
    cd ~/robonav/containers && singularity pull py311.sif docker://python:3.11-bookworm
    singularity exec py311.sif bash build_server_env.sh    # LightNav-0/.venv (vllm + torch cu128)
    bash build_loc_env.sh                                  # robo-nav/.venv-loc, native

`.venv-loc` must get a cu128 torch (`torch==2.11.0+cu128`, `torchvision==0.26.0+cu128` from
`https://download.pytorch.org/whl/cu128`, with `--reinstall-package torch` if a cu130 build is
already there); a bare `torch` silently runs on CPU with driver 535.

## Jobs

Copy the `.sbatch` files to `~/robonav/jobs/` (their `-o` log paths point there; edit the absolute
`/mnt/beegfs/home/...` prefix for another user). Never pass `--mem`: SLURM reports 1 MB per node and
rejects any memory request.

| file | what |
|---|---|
| `stack.sbatch` | one LightNav server (:8050) + sim (:8088) on 1 GPU, runs until cancelled |
| `batch.sbatch` | N stacks on N GPUs of one node (:8050+k / :8088+k) and N `landmark.navigate` workers (`--shards N --shard k`); `sbatch batch.sbatch <run_name> [N]`, with `--gres=gpu:a4500:N` for N != 4 |
| `loc_bench.sbatch` | localization benchmark; `sbatch loc_bench.sbatch [methods...]` |

Everything binds 127.0.0.1 on the compute node. To watch a sim from a laptop:

    ssh -N -J $USER@gates.shapirolab.zi.columbia.edu -L 8088:localhost:8088 $USER@<node>

The node of the last job is written to `~/robonav/jobs/stack_node` / `batch_node`.
