# robonav cluster status (gates) — 2026-10-08

Written for a second Claude Code session. Read this file instead of being pasted a summary.

## Cluster reality check

`gates.shapirolab.zi.columbia.edu` is a SLURM **login node with no GPU**. Verified: the only
display adapter is the ASPEED BMC chip, there is no `/dev/nvidia*` and no `nvidia` kernel module.
`/usr/bin/nvidia-smi` *does* exist (cluster-wide image), so running it here fails with
"couldn't communicate with the NVIDIA driver" — that is a missing card, not a broken driver.

GPUs live on the compute nodes, 128 total: `gpu001-gpu008` = `gpu:a4000:8`,
`gpu009-gpu016` = `gpu:a4500:8` (20 GB each). Partition `gpu`, timelimit infinite.
Compute nodes have outbound internet (HuggingFace, GitHub) and accept SSH from the login node.

## Four deviations from scripts/setup_cluster.sh

1. **`setup_cluster.sh` fails here.** vLLM 0.19.1 ships only manylinux_2_31+ wheels; this cluster
   is glibc 2.28 (RHEL 8). uv fell back to a source build that died on `CUDA_HOME is not set`.
   Fix: the LightNav server env runs inside Singularity — `~/robonav/containers/py311.sif`
   (Debian, glibc 2.36, gcc 12), launched with `singularity exec --nv`, `module load singularity/3.6.3`.
   Inside it: vllm 0.19.1 + torch 2.10.0+cu128. `.venv-loc` stays native.

2. **Unpinned `torch` silently disables the GPU.** `requirements.txt:1` is a bare `torch`, so uv
   installs a cu130 build; the driver here is 535.104.12 (CUDA 12.2). Result:
   `torch.cuda.is_available()` is False with no error and everything runs on CPU — the first
   benchmark run did exactly this. Fix applied to `.venv-loc`:
   `torch==2.11.0+cu128` / `torchvision==0.26.0+cu128` from `https://download.pytorch.org/whl/cu128`.
   uv treats an installed cu130 build as satisfying `torch`, so forcing it needs
   `--reinstall-package torch` plus the explicit `+cu128` version. **Not yet pinned in the repo** —
   the brief said not to change repo code, but this is the one change worth making.

3. **`start_stack.sh` / tmux does not apply.** The stack runs as a SLURM job:
   `~/robonav/jobs/stack.sbatch` (`-p gpu --gres=gpu:a4500:1 -c 8`). SLURM reports **1 MB** of node
   memory, so any `--mem` request fails with "Memory specification can not be satisfied" — omit it.
   Currently job 13000924 on **gpu009**: lightnav `:8050` + sim `:8088`, `/api/health` → `ok: true`,
   robot connected, camera 18.5 fps, zero errors in `lightnav.log`.
   Benchmark job template: `~/robonav/jobs/loc_bench.sbatch`. Logs: `~/robonav/jobs/logs/`.

4. **The Anthropic API key must be workspace-scoped.** An org/default-scoped key returns
   400 "This API key is not scoped to a workspace, so this request must include the
   anthropic-workspace-id header". No env var supplies that header for plain API-key auth
   (`ANTHROPIC_WORKSPACE_ID` is read only by the federation path in
   `anthropic/lib/credentials/_workload.py`); only an `ant auth login` profile or a code-level
   `default_headers` would. Resolved by swapping in a workspace-scoped key.
   Key lives in `~/.secrets/anthropic.env` (mode 600), sourced only by jobs that need it.

## Ports / viewing the sim

Both services bind `127.0.0.1` on the compute node. From a laptop:

    ssh -N -J bgl2126@gates.shapirolab.zi.columbia.edu -L 8088:localhost:8088 bgl2126@gpu009

then open http://127.0.0.1:8088. In `-L 8088:localhost:8088` the `localhost` resolves on the
*final* destination, so the destination must be the GPU node, not gates.

`navigate.py:306` defaults to `ws://127.0.0.1:8088/ws`, which cannot reach gpu009 from the login
node, so a `ssh -N -L 8088:localhost:8088 gpu009` forward runs on gates (tmux session `simtunnel`).

## Done: localization benchmark

All 11 methods, on GPU. `sim_data/loc/val2/results.json`, per-query `preds_<method>.csv`.
Dataset: 1487 map keyframes, 100 queries x 8 views, scene `val_2`.

| method | room | pos median | <1m | yaw median | secs |
|---|---|---|---|---|---|
| prior | 10% | - | - | - | 0 |
| dinov2 | 79% | 0.91 m | 55% | 11° | 47 |
| dinov2_full | 78% | 0.84 m | 56% | 15° | 19 |
| dinov2_look | 99% | 0.57 m | 76% | 8° | cached |
| dinov2_full_look | 96% | 0.50 m | 73% | 8° | cached |
| salad | 85% | 0.77 m | 58% | 8° | 52 |
| salad_look | 99% | 0.52 m | 77% | 4° | cached |
| anyloc | 91% | 0.71 m | 70% | 11° | 483 |
| anyloc_look | 99% | 0.52 m | 79% | 7° | cached |
| salad_lg | 87% | 0.74 m | 65% | 2° | 38 |
| salad_lg_look | 97% | 0.50 m | 80% | 1° | 188 |

Takeaways: look-around dominates descriptor choice (every `_look` >= 96%; the best single-view
method, anyloc, is 91%). LightGlue owns heading (1-2° vs 7-15°). AnyLoc costs 483 s vs SALAD's
52 s for +6 points and worse yaw. Confidence gating is well-calibrated — the confident half is
at or near 100% for every learned method. Caveat: one scene, 100 queries, so a point or two
between the `_look` variants is noise.

## Done: landmark map

`sim_data/landmark/val2/landmark_map.md` — 42 landmarks, 24 connections, from 149 keyframes
(`--every 10`, 6 chunks). Holds the no-rooms constraint throughout.

## Done: 12-episode pilot

`sim_data/landmark/runs/pilot`, 2 starts x 2 goals x 3 modes. Success = within 1.5 m of the goal
object and in the same room.

| mode | success | median final | median closest | median time |
|---|---|---|---|---|
| landmark | **2/4** | 6.41 m | 4.65 m | 300.1 s |
| zeroshot | 0/4 | 9.81 m | 7.09 m | 14.3 s |
| zeroshot_retry | 0/4 | 6.97 m | 5.81 m | 85.4 s |

| start | goal | zeroshot | zeroshot_retry | landmark |
|---|---|---|---|---|
| q23 livingroom_2 | fridge | 13.34 m | 6.90 m | **0.91 m OK** |
| q23 livingroom_2 | laundry hamper | 9.81 m | 6.97 m | 12.46 m (timeout) |
| q8 kitchen | dog bed | 5.82 m | 7.50 m | 6.41 m (timeout) |
| q8 kitchen | armchair | 4.28 m | 3.39 m | **0.73 m OK** |

Landmark mode is the only mode that ever arrives (2/4 vs 0/4 and 0/4), and when it does it is
decisive — 0.73-0.91 m, ending on Claude's own `arrived`. The medians understate this: they are
dominated by the two timeouts, so landmark's 6.41 m median looks worse than zeroshot_retry's
6.97 m despite being the only mode with successes. n=4 per mode — directional only.

Reading the traces (full `where_am_i`/`instruction` sequences in the episode JSONs):

- Both successes took 3 planner calls, no `look_around`, no backtracking.
- **Both failures are instruction execution, not planning or localization.** `where_am_i` stays
  accurate throughout and Claude diagnoses its own errors ("This is not on the route to the
  hamper") and re-plans correctly, but the robot keeps ending up back in the counter area. The
  hamper episode issues the same "through the plain brown-trimmed doorway" instruction four times
  (65 s, 158 s, 207 s, 249 s) and twice reports being physically stuck — "blocked by the door frame
  post", "wedged between the open white glass-panel door and the end of the black TV stand".
- `look_around` works as designed: used twice in the dog-bed run, both after genuinely losing
  track, and recovered a confident position each time.

### Two issues found, neither fixed (both would need repo changes)

1. **`--timeout 300.0` is the binding constraint for landmark mode.** Both failures hit it while
   still progressing sensibly — the dog-bed run's final instruction at 305 s was correctly aimed at
   the goal. Zero-shot finishes in 5-25 s, so the default was never tested against a mode that
   costs a planner call plus a drive segment per step. A larger timeout is an experiment-design
   decision.
2. **Video recording fails on every episode** (non-fatal). OpenCV picks the `h264_v4l2m2m` encoder,
   which has no device on these nodes: `Could not open codec h264_v4l2m2m` ->
   `Failed to initialize VideoWriter`. Episodes run and score normally; only the per-episode videos
   are lost. Fixing it means changing the encoder in `robo_nav/sim/video.py`.
