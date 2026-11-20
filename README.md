# robo-nav

`robo-nav` end goal: light weight engine for robots to perform visual navigation and relocalization using egocentric data

lingbot-based 3d spatial memory with graph where nodes are rooms/regions and edges are hallways/paths

uses dinov2 for coarse cos similarity to find the most similar frames as query images and then use local subsequence 6dof pose estimation to refine the relative poses (usingLingbot's relative pose estimator)

metric scale calibration: anchors spatial predictions to physical meters using external metric cues: first ~8 frames -> lingbot anchor reconstruction + ONE metric cue -> solve scale alpha -> metric anchor -> all subsequent lingbot poses -> metrical spatial map



---

## Key Features

1. **Self-Contained LingBot-Map Core Engine (`lingbot_map/`)**:
   - Streaming inference (`gct_stream.py`) with causal KV cache.
   - Overlapping windowed reconstruction (`gct_stream_window.py`) for long video sequences.
   - Standalone 3D visualization and postprocessing pipelines.

2. **Metric Scale Calibration Engine (`robo_nav/scale_calibration.py`)**:
   - Solves the global scale scalar $\alpha = \frac{d_{\text{metric}}}{d_{\text{predicted}}}$ using LingBot's initial ~8 anchor frames + 1 external ground-truth metric cue:
     - `DEPTH_POINT`: Known metric depth at a target frame & pixel.
     - `TRANSLATION_STEP`: Known physical distance between camera poses (e.g. wheel odometry step).
     - `STEREO_BASELINE`: Known baseline distance or target marker size.
   - Rescales all predicted point clouds, depth maps, and camera positions $\mathbf{t}_{\text{metric}} = \alpha \cdot \mathbf{t}_{\text{predicted}}$.

3. **Coarse-to-Fine Metric 6DoF Relocalization (`localize.py`)**:
   - Stage 1: Fast GPU batched DINOv2 visual similarity matching, with blur/quality gating
     (excludes low-sharpness frames from *retrieval eligibility* only — they remain part of the
     reconstructed map, so gating never leaves a gap in the geometry) and retrieval-confidence
     gating (flags ambiguous top1/top2 similarity margins).
   - Stage 2: Full map reconstruction integrated with metric scale calibration ($\alpha$).
   - Stage 3: k-independent-hypothesis local 6DoF pose estimation — one relative-pose estimate
     per top-k retrieved reference, fused via distance-based consensus (outlier references that
     disagree with the rest are discarded rather than trusted blindly). The result includes a
     `low_confidence` flag combining retrieval ambiguity and consensus disagreement.
   - See [Confidence Gating & Diagnostics](#confidence-gating--diagnostics) below for the exact
     scores, thresholds, and how they combine into `low_confidence`.

4. **Standalone Demos**:
   - `demo.py`: Run streaming or windowed 3D reconstruction on image folders or MP4 video (from the original lingbot repo, plus some extra flags to choose .png from).
   - `graph.py`: Topological graph search and room zone path planning.

---

## Installation & Setup

### 1. Environment & Setup
```bash
git clone https://github.com/your-org/robo-nav.git
cd robo-nav

conda create -n robo-nav python=3.10 -y
conda activate robo-nav
```

### 2. Dependencies
```bash
# Install PyTorch with CUDA support (adjust for your CUDA version if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install requirements & package in editable mode
pip install -r requirements.txt
pip install -e .
```

### 3. Model Checkpoints
```bash
mkdir -p checkpoints/
# Place LingBot-Map model checkpoint at checkpoints/lingbot-map-long.pt
```

---

## Quickstart

### 1. Streaming / Windowed 3D Reconstruction Demo (`demo.py`)
```bash
# Streaming 3D reconstruction from image folder
python demo.py --model_path checkpoints/lingbot-map-long.pt --image_folder /path/to/images/

# Windowed reconstruction for long video files
python demo.py --model_path checkpoints/lingbot-map-long.pt --video_path video.mp4 --mode windowed --window_size 64
```

### 2. Metric Scale Relocalization (`localize.py`)
```bash
# Relocalize query image into map with translation step metric calibration (e.g. 0.5 meter odometry step)
python localize.py /path/to/query.jpg /path/to/map_frames/ \
  --model_path checkpoints/lingbot-map-long.pt \
  --metric_cue_type translation_step \
  --metric_val 0.50 \
  --frame_idx_a 0 --frame_idx_b 1

# Tune retrieval/consensus/blur gating (defaults shown)
python localize.py /path/to/query.jpg /path/to/map_frames/ \
  --top_k 4 \
  --margin_threshold 0.05 \
  --blur_threshold 30.0 \
  --consensus_radius 0.3 \
  --window_radius 2
```
- `--margin_threshold`: top1/top2 DINOv2 similarity gap below which retrieval is flagged ambiguous.
- `--blur_threshold`: Laplacian-variance sharpness cutoff; frames below it are excluded from
  retrieval matching only — they're still used for map reconstruction, so the map stays gapless
  (`0` disables gating). The default is a mild heuristic — tune per camera/lighting setup.
- `--consensus_radius`: max pairwise disagreement (meters) between the k independent pose
  hypotheses for them to be treated as agreeing; outliers beyond this are discarded during fusion.
- `--window_radius`: number of map frames before each top-k candidate to include in its local
  Stage 3 pose-estimation window (window = `window_radius` map frames + the candidate + the
  query). Larger values give the local streaming pass more frames/parallax to estimate depth
  and pose from.

### Confidence Gating & Diagnostics

`localize.py` never returns a bare pose — every result comes with the scores that produced it,
so a caller can decide whether to trust it. Two independent gates feed the final flag:

**1. Retrieval-margin gating (Stage 1)** — computed by `compute_confidence_and_uncertainty()`:
| Score | Meaning |
|---|---|
| `top1_similarity` / `top2_similarity` | Best and runner-up DINOv2 cosine similarity among map candidates. |
| `margin` | `top1_similarity - top2_similarity`. |
| `margin_ratio` | `margin / top1_similarity`. |
| `is_ambiguous` | `True` if `margin < --margin_threshold` (default `0.05`) — multiple map locations look visually similar. |
| `softmax_entropy` | Shannon entropy of the temperature-scaled (`tau=0.1`) similarity distribution over *all* map candidates — high entropy means no single frame stands out. |
| `spatial_covariance` / `spatial_trace` / `spatial_std` | Covariance of the top-k hypotheses' 3D positions (computed after Stage 3) — spread-out hypotheses indicate the retrieved candidates don't agree on where the query actually is. |

**2. Consensus-fusion gating (Stage 3)** — computed by `fuse_pose_hypotheses()`:
- Each of the `top_k` retrieved candidates produces an independent 6DoF pose hypothesis.
- Hypotheses within `--consensus_radius` meters of the largest mutually-agreeing cluster are kept
  as inliers; the rest are discarded as outliers rather than averaged in blindly.
- Returns `num_inliers` / `num_hypotheses` and the discarded `outlier_map_indices`.

**Final flag**: `low_confidence = is_ambiguous OR (num_inliers <= max(1, num_hypotheses // 2))` —
i.e. it trips if retrieval itself was ambiguous, *or* if half or more of the k pose hypotheses
disagreed enough to be thrown out during consensus fusion. Either condition alone is enough to
flag the result — always check `result["low_confidence"]` before acting on `result["position"]`.

### Blur/Quality Gating

`compute_image_sharpness()` scores each map frame via OpenCV's Laplacian-variance heuristic
(`cv2.Laplacian(image, cv2.CV_64F).var()` — lower means blurrier). Frames scoring below
`--blur_threshold` are excluded from *retrieval eligibility* in Stage 1 only: they can never
become the top-k matched keyframe, but Stage 2/3 map reconstruction always uses the full,
ungapped frame set. (Earlier versions dropped blurry frames from reconstruction too, which left
gaps in the sequence and shifted the reconstructed geometry — this is why gating is retrieval-only
now.) Set `--blur_threshold 0` to disable gating entirely.

> **Test coverage note**: neither the blur-sharpness scoring nor the confidence-gating functions
> above have unit tests yet (`tests/` currently only covers `MetricScaleSolver`). They've been
> validated by inspecting real run output, not by an automated test suite.

### 3. Python API for Metric Scale Solver
```python
from robo_nav import MetricScaleSolver, MetricCue, MetricCueType

# Initialize solver
solver = MetricScaleSolver()

# Define 1 external metric cue (e.g. known 1.25m metric depth at frame 0, pixel (250, 300))
cue = MetricCue(
    cue_type=MetricCueType.DEPTH_POINT,
    metric_val=1.25,
    frame_idx_a=0,
    pixel_coords=(250, 300)
)

# Solve scale factor alpha from LingBot predictions
alpha = solver.solve_scale(predictions, cue)

# Rescale predictions to physical meters
metric_predictions = solver.apply_scale(predictions, alpha)
print(f"Metric scale alpha: {alpha:.6f}")
```

---

## LightNav-0 Sim Navigation

robo-nav does the global, room-level planning (`graph.py` over the house's room/door graph) and
hands [LightNav-0](https://github.com/lightorigins/LightNav-0) one room-to-room hop at a time
("Go through the doorway ahead on your right, about 5 meters away, into the bedroom."), in
LightNav-0's MuJoCo demo house. Localization is currently the sim's ground-truth pose.

```bash
# 1. LightNav-0 server on a GPU box (see their docs/DEPLOYMENT.md), tunneled to localhost:8050
ssh -N -L 8050:localhost:8050 <gpu-host>

# 2. The MuJoCo demo, with wall/furniture collisions (the stock TurtleBot drives through walls)
git clone https://github.com/lightorigins/LightNav-0.git && cd LightNav-0
git checkout 3015508 && git apply ../robo-nav/patches/lightnav0_sim.patch
cd mujoco_demo && ./run.sh --vln-server ws://127.0.0.1:8050      # UI: http://127.0.0.1:8088

# 3. Navigate (scene = mujoco_demo/vln_mujoco/assets/scenes/procthor-10k-val/val_2.json)
python -m robo_nav.sim.nav bedroom_1 --reset --scene $SCENE \
    --out sim_data/runs/bedroom_1.json --video sim_data/runs/bedroom_1.mp4
python -m robo_nav.sim.summary sim_data/runs                       # success-rate table
```

**Everything on one GPU box (no laptop needed):** `ROOT=~/robonav bash scripts/setup_cluster.sh`
(clones both repos, applies the patch, builds the LightNav server env and `.venv-loc`), then
`GPU=0 bash scripts/start_stack.sh` (LightNav server + headless sim in tmux sessions `lightnav` / `sim`).

Code layout:

| Path | What |
|---|---|
| `robo_nav/sim/` | MuJoCo-house tooling: `scene.py` (GT rooms/doors/camera), `nav.py` (planner + LightNav loop), `video.py`, `compare.py`, `summary.py`, `record.py` (record while driving), `render.py` (offline GT dataset) |
| `robo_nav/loc/` | kidnapped-robot localization benchmark: `benchmark.py` (DINOv2 / SALAD / AnyLoc / LightGlue), `vlm.py` (Claude) |
| `robo_nav/landmark/` | landmark-only navigation (no GT, no room names): LLM-annotated house video + LightNav |
| `robo_nav/habitat/` | Habitat evals (GOAT-Bench, RxR-CE): `goat_env.py` (habitat-sim server), `goat_nav.py` (controller), `goal_report.py`, `goat_difficulty.py`, `far_goals.py`, `floors.py` (see [Habitat Evals](#habitat-evals-goat-bench-and-rxr-ce)) |
| `evals/` | fixed eval sets + the landmark maps the Habitat runs used |
| `patches/lightnav0_sim.patch` | LightNav-0 sim changes: wall/furniture collisions, `set_pose` for evaluation starts |
| `scripts/` | cluster setup and launch; `scripts/habitat/` Habitat setup, map building, eval runs |

- `robo_nav/sim/nav.py`: plan → instruct → re-instruct when LightNav stops or a hop times out;
  replans after wrong turns. Episodes end as `success`, `gave_up`, `timeout`, or as
  `wall_crossing` / `left_house` (sanity checks for runs without the collision patch).
  `--hint none` sends plain "Go to the X." instead of doorway directions.
- `robo_nav/sim/video.py`: episode MP4s (robot cam + chase cam + floor plan with route/trajectory).
- `robo_nav/sim/record.py`: records keyframes + GT pose + room label while you drive (mapping data).
- `robo_nav/lightnav_client.py`: minimal LightNav-0 WebSocket client for open-loop runs on recorded video (`--show` for a live overlay).

## Habitat Evals: GOAT-Bench and RxR-CE

The same LightNav-0 + landmark-planner stack, scored on standard benchmarks in habitat-sim:

| Benchmark | Scenes | Split | Success | Notes |
|---|---|---|---|---|
| GOAT-Bench (`hm3d/v1`) | HM3D v0.2 val (36) | `val_unseen` (also `val_seen`, `val_seen_synonyms`) | 0.25 m to a goal view point | lifelong: each subtask starts where the last ended. **Image-goal subtasks are skipped**, so numbers are text-goal-only and not comparable to official GOAT-Bench results |
| RxR-CE (`RxR_VLNCE_v0`) | MP3D (11 val_unseen) | `val_unseen`, English guide instructions (3,669) | 3 m to the path end | one subtask per episode |
| R2R-CE (`R2R_VLNCE_v1-3_preprocessed`) | MP3D | `val_unseen` | 3 m | supported, not used in reported runs |

Two processes talk over ZeroMQ, in two environments (habitat-sim and vLLM don't share one):

- `robo_nav/habitat/goat_env.py serve`: the scene, in a micromamba env `habitat` (python 3.7, habitat-sim 0.2.3 headless). It owns all GT, which only shows up in scores, never in observations.
- `robo_nav.habitat.goat_nav`: the controller, in `LightNav-0/.venv`. It runs LightNav-0 in-process (vLLM) and/or calls the VLM planner (`ROBO_NAV_VLM=claude|gemini`).

Modes (`--modes`): `zeroshot` (LightNav-0 alone with the goal text), `landmark` (planner + the scene's landmark map
instructs LightNav-0 one hop at a time), `landmark_vlm` (as landmark, but the VLM drives with coarse moves),
`vlm` / `vlm_map` (VLM drives alone, without or with the map). `zeroshot` and `landmark` need a GPU for LightNav-0;
the other three only need one for habitat-sim.

### 1. Setup (once per machine)

```bash
ROOT=~/robonav bash scripts/setup_cluster.sh       # robo-nav + LightNav-0 + its checkpoint (see above)
cd ~/robonav/robo-nav

# habitat env + datasets into ~/robonav/data (override with DATA=...); each step skips what's already there
bash scripts/habitat/setup_habitat.sh env

# GOAT-Bench: HM3D needs a Matterport API token (request HM3D access first, see the script header)
MATTERPORT_TOKEN_ID=... MATTERPORT_TOKEN_SECRET=... bash scripts/habitat/setup_habitat.sh hm3d goat

# RxR-CE: MP3D needs Matterport's personal download_mp.py (sign the MP3D terms of use to get it)
MP_DOWNLOAD=/path/to/download_mp.py bash scripts/habitat/setup_habitat.sh mp3d rxr
#   (or MP3D_ZIP=/path/to/mp3d_habitat.zip if you already have it)

bash scripts/habitat/setup_habitat.sh check        # layout + one habitat-sim render
```

Resulting layout (`goat_env.py` reads `~/robonav/data`, or `$ROBO_NAV_DATA`):

```
~/robonav/data/
├── scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json, val/00800-TEEsavR23oF/...
├── scene_datasets/mp3d/<scene>/<scene>.glb ...                      (90 scenes)
├── datasets/goat_bench/hm3d/v1/{val_seen,val_seen_synonyms,val_unseen}/content/<scene>.json.gz
├── datasets/RxR_VLNCE_v0/val_unseen/val_unseen_guide.json.gz
└── datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz (optional)
```

API keys: export `GEMINI_API_KEY` and/or `ANTHROPIC_API_KEY`, or point `ROBO_NAV_ENV_FILE` at a `KEY=value` file
that the run scripts source. The reported runs used `ROBO_NAV_VLM=gemini ROBO_NAV_GEMINI_MODEL=gemini-4-argon`.

### 2. Eval sets and landmark maps (in `evals/`)

| Path | What |
|---|---|
| `evals/goat/far_goals_10m.json` | 120 GOAT `val_unseen` text subtasks that start >= 10 m (geodesic) from the goal, with fixed start poses so every mode runs the same goals (built with `robo_nav.habitat.far_goals` from a zeroshot reference run) |
| `evals/goat/far_goals_smoke.json` | one of those, for a smoke test |
| `evals/rxr/rxr_far_12m.json` | 120 English RxR-CE `val_unseen` episodes with a geodesic start-goal distance >= 12 m, 12 per scene over 10 scenes |
| `evals/goat/maps/<scene>/`, `evals/rxr/maps/<scene>/` | the landmark maps the runs used (`landmark_map.md` is what the planner reads; `floors.json` for `--floors`). Text only: keyframes and tour renders are Matterport-derived images, so they stay out of the repo |

To rebuild maps (new scenes or a new annotator model; costs API calls): `bash scripts/habitat/build_maps.sh goat`
(or `rxr`). This renders each scene's walkthrough tour in habitat, annotates it with the VLM, and adds floor labels, writing into `sim_data/<ds>/maps`.

### 3. Run

`scripts/habitat/run_eval.sh` starts the env server, runs K controller clients against it, then does one resume pass for episodes that hit API errors. Rerunning the same command resumes; finished results are skipped.

```bash
export ROBO_NAV_VLM=gemini ROBO_NAV_GEMINI_MODEL=gemini-4-argon GEMINI_API_KEY=...

# smoke test: one GOAT goal, VLM-only mode (no LightNav GPU needed)
bash scripts/habitat/run_eval.sh goat sim_data/goat/runs/smoke --goals evals/goat/far_goals_smoke.json -- --modes vlm

# GOAT far goals, VLM modes, 6 clients sharing one server, step budget 2x shortest-path time
bash scripts/habitat/run_eval.sh goat sim_data/goat/runs/far10_b2 --goals evals/goat/far_goals_10m.json --clients 6 \
    -- --modes vlm vlm_map landmark_vlm --budget_factor 2.0

# LightNav-0 modes: habitat on GPU 0, LightNav-0 on GPU 1 (one client per LightNav GPU)
bash scripts/habitat/run_eval.sh goat sim_data/goat/runs/far10_b2 --goals evals/goat/far_goals_10m.json \
    --env_gpu 0 --nav_gpu 1 -- --modes zeroshot landmark --budget_factor 2.0

# RxR-CE far episodes
bash scripts/habitat/run_eval.sh rxr sim_data/vlnce/runs/rxr12_b2 --episode_list evals/rxr/rxr_far_12m.json \
    --clients 6 -- --modes vlm vlm_map landmark_vlm --budget_factor 2.0

# full lifelong GOAT episodes instead of single goals
bash scripts/habitat/run_eval.sh goat sim_data/goat/runs/val_unseen --episodes "0 30 60" --env_gpu 0 --nav_gpu 1 \
    -- --modes zeroshot landmark
```

Useful `goat_nav` flags (after `--`): `--budget_factor F` (step budget = F x shortest-path time at 0.25 m/s),
`--subtask_steps N`, `--floors`, `--verify_arrival N`, `--final_approach N`, `--arrive_prompt old|new`,
`--actuation_noise LIN_SD ANG_SD_DEG`. Run `python -m robo_nav.habitat.goat_nav --help` for the full list.

### 4. Report

```bash
PY=~/robonav/LightNav-0/.venv/bin/python   # needs torch
$PY -m robo_nav.habitat.goal_report --runs sim_data/goat/runs/far10_b2 --factors 1.5 2.0

# success by difficulty (start distance, goal visible at start, doorways, floor change); needs the habitat env
micromamba run -n habitat python -I robo_nav/habitat/goat_difficulty.py --goals evals/goat/far_goals_10m.json \
    --out sim_data/goat/difficulty_far_10m.json
$PY -m robo_nav.habitat.goal_report --runs sim_data/goat/runs/far10_b2 --difficulty sim_data/goat/difficulty_far_10m.json
```

Each result is `<out>/{ep<idx>|g<idx>_<subtask>}_<mode>.json`, with the per-subtask success, SPL, the GT track, and the planner transcript.

### Troubleshooting

- **No EGL device / `unable to create context`** (habitat-sim or MuJoCo): the node's driver may be compute-only, with no `libEGL_nvidia`. Install the matching `libnvidia-gl`. Without root, extract the EGL libs from NVIDIA's `.run` installer for the same driver version, then point `LD_LIBRARY_PATH` and `__EGL_VENDOR_LIBRARY_FILENAMES` at them.
- **GPUs in Exclusive_Process mode** allow one process per GPU. Put the env server and LightNav-0 on different GPUs (`--env_gpu` / `--nav_gpu`). Run the VLM-only modes with `--nav_gpu` unset.
- **HM3D navmesh**: GOAT runs use `--navmesh_cell_height 0.05`, the run script sets it. The stock HM3D v0.2 navmeshes sit slightly off the floor, which shifts GT distances. See LightNav-0's `docs/EVAL_HABITAT.md`.
- **Proxies**: if `http_proxy` is set, keep `localhost` out of it (`no_proxy=localhost,127.0.0.1`), or the ZeroMQ and websocket traffic goes through the proxy. The scripts set this.

## Repository Layout

```
robo-nav/
├── lingbot_map/             # LingBot-Map vision transformer trunk & heads
├── robo_nav/                # Metric scale solver & navigation package
│   ├── __init__.py
│   └── scale_calibration.py # MetricScaleSolver & MetricCue classes
├── demo.py                  # Standalone streaming & windowed 3D reconstruction
├── graph.py                 # Topological graph search and path planning
├── localize.py              # 3-stage coarse-to-fine metric 6DoF relocalizer
├── requirements.txt         # Package dependencies
└── README.md
```
