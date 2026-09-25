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
