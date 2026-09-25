"""
MoGe-3 based automatic metric cue construction.

MoGe-3 (https://github.com/microsoft/MoGe) predicts metric-scale depth directly
from a single RGB image. This lets a DEPTH_POINT MetricCue be built automatically
from an anchor frame's MoGe-3 depth instead of requiring a manually-measured
ground-truth depth at a chosen pixel.

Requires MoGe-3 to be installed separately (not a robo-nav dependency):
    pip install git+https://github.com/microsoft/MoGe.git
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from robo_nav.scale_calibration import MetricCue, MetricCueType


def load_moge_model(checkpoint_path: str, device: Optional[torch.device] = None):
    """Load a MoGe-3 model from a checkpoint, for reuse across multiple cue calls."""
    from moge.model.v3 import MoGeModel

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return MoGeModel.from_pretrained(checkpoint_path).to(device).eval()


def moge_depth_at_pixel(
    image_path: str,
    pixel_coords: Tuple[int, int],
    model=None,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> float:
    """
    Run MoGe-3 on one image and return its metric depth at (y, x), median-filtered
    over a 3x3 patch for robustness (matches MetricScaleSolver's own patch handling).

    Args:
        image_path: Path to the RGB image.
        pixel_coords: (y, x) pixel position to sample depth at.
        model: A preloaded MoGe-3 model (from load_moge_model), for reuse across calls.
            If omitted, checkpoint_path is used to load one for this call only.
        checkpoint_path: Path to a MoGe-3 checkpoint, used only when model is None.
        device: Device to load a freshly-created model onto (ignored if model is given).
    """
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Provide either a preloaded `model` or a `checkpoint_path`.")
        model = load_moge_model(checkpoint_path, device)

    model_device = next(model.parameters()).device
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_tensor = torch.tensor(image / 255, dtype=torch.float32).permute(2, 0, 1).to(model_device)

    with torch.no_grad():
        output = model.infer(image_tensor)
    depth = output["depth"].detach().cpu().numpy()

    py, px = pixel_coords
    H, W = depth.shape[:2]
    y_min, y_max = max(0, py - 1), min(H, py + 2)
    x_min, x_max = max(0, px - 1), min(W, px + 2)
    patch = depth[y_min:y_max, x_min:x_max]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        raise ValueError(f"No valid MoGe-3 depth near pixel {pixel_coords} in {image_path}")
    return float(np.median(valid))


def moge_metric_cue(
    image_path: str,
    pixel_coords: Tuple[int, int],
    frame_idx_a: int = 0,
    model=None,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> MetricCue:
    """
    Build a DEPTH_POINT MetricCue automatically from MoGe-3's metric depth at
    (y, x) in `image_path`, instead of a manually-measured ground-truth depth.

    `frame_idx_a` should match the index of `image_path` within whatever frame
    sequence the resulting cue is later applied to (e.g. MetricScaleSolver.solve_scale
    looks up predictions["depth"][frame_idx_a] to compare against this cue's value).
    """
    depth_val = moge_depth_at_pixel(
        image_path, pixel_coords, model=model, checkpoint_path=checkpoint_path, device=device
    )
    return MetricCue(
        cue_type=MetricCueType.DEPTH_POINT,
        metric_val=depth_val,
        frame_idx_a=frame_idx_a,
        pixel_coords=pixel_coords,
    )
