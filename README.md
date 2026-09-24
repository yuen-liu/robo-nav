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
     (drops low-sharpness map frames before they can corrupt retrieval) and retrieval-confidence
     gating (flags ambiguous top1/top2 similarity margins).
   - Stage 2: Full map reconstruction integrated with metric scale calibration ($\alpha$).
   - Stage 3: k-independent-hypothesis local 6DoF pose estimation — one relative-pose estimate
     per top-k retrieved reference, fused via distance-based consensus (outlier references that
     disagree with the rest are discarded rather than trusted blindly). The result includes a
     `low_confidence` flag combining retrieval ambiguity and consensus disagreement.

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
  --consensus_radius 0.3
```
- `--margin_threshold`: top1/top2 DINOv2 similarity gap below which retrieval is flagged ambiguous.
- `--blur_threshold`: Laplacian-variance sharpness cutoff; frames below it are excluded from
  retrieval matching only — they're still used for map reconstruction, so the map stays gapless
  (`0` disables gating). The default is a mild heuristic — tune per camera/lighting setup.
- `--consensus_radius`: max pairwise disagreement (meters) between the k independent pose
  hypotheses for them to be treated as agreeing; outliers beyond this are discarded during fusion.

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
