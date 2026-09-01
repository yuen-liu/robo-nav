"""
Metric Scale Calibration Module for LingBot / robo-nav
------------------------------------------------------
Solves absolute real-world metric scale alpha using LingBot-Map's initial ~8 anchor
reconstruction frames + ONE external ground-truth metric cue.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch


class MetricCueType(Enum):
    DEPTH_POINT = "depth_point"            # Ground truth metric depth at a specific frame & pixel
    TRANSLATION_STEP = "translation_step"  # Ground truth metric distance between 2 frame poses (e.g. wheel odometry)
    STEREO_BASELINE = "stereo_baseline"    # Ground truth baseline distance or known target metric size


@dataclass
class MetricCue:
    cue_type: MetricCueType
    metric_val: float                      # Physical metric value in meters
    frame_idx_a: int = 0                   # Reference frame index (default: first frame)
    frame_idx_b: Optional[int] = None      # Target frame index (for TRANSLATION_STEP / STEREO_BASELINE)
    pixel_coords: Optional[Tuple[int, int]] = None  # (y, x) pixel position for DEPTH_POINT


class MetricScaleSolver:
    """
    Solves the global scale scalar alpha = d_metric / d_predicted
    from anchor predictions and applies it to 3D point clouds, depth maps, and camera poses.
    """

    def __init__(self, default_num_anchor_frames: int = 8):
        self.default_num_anchor_frames = default_num_anchor_frames

    def solve_scale(self, predictions: Dict[str, Any], cue: MetricCue) -> float:
        """
        Compute metric scale factor alpha from predictions and 1 metric cue.

        Returns:
            scale_factor (float): Multiplicative scalar alpha where X_metric = alpha * X_predicted.
        """
        if cue.metric_val <= 0:
            raise ValueError(f"Metric cue value must be > 0 meters, got {cue.metric_val}")

        if cue.cue_type == MetricCueType.DEPTH_POINT:
            if "depth" not in predictions:
                raise KeyError("Predictions dict missing 'depth' map required for DEPTH_POINT cue")
            depths = predictions["depth"]
            if isinstance(depths, torch.Tensor):
                depths = depths.detach().cpu().numpy()
            
            while isinstance(depths, np.ndarray) and depths.ndim > 3 and depths.shape[0] == 1:
                depths = depths[0]
            
            frame_depth = depths[cue.frame_idx_a]
            if isinstance(frame_depth, np.ndarray) and frame_depth.ndim == 3 and frame_depth.shape[-1] == 1:
                frame_depth = frame_depth.squeeze(-1)

            if cue.pixel_coords is not None:
                py, px = cue.pixel_coords
                H, W = frame_depth.shape[:2]
                y_min, y_max = max(0, py - 1), min(H, py + 2)
                x_min, x_max = max(0, px - 1), min(W, px + 2)
                patch = frame_depth[y_min:y_max, x_min:x_max]
                pred_val = float(np.median(patch[patch > 0])) if np.any(patch > 0) else float(frame_depth[py, px])
            else:
                valid_depths = frame_depth[frame_depth > 0]
                pred_val = float(np.median(valid_depths))

            if pred_val <= 1e-6:
                raise ValueError(f"Predicted depth at anchor is non-positive or near zero: {pred_val}")
            
            alpha = float(cue.metric_val / pred_val)

        elif cue.cue_type == MetricCueType.TRANSLATION_STEP:
            poses = predictions.get("extrinsic")
            if poses is None:
                raise KeyError("Predictions dict missing 'extrinsic' poses required for TRANSLATION_STEP cue")
            if isinstance(poses, torch.Tensor):
                poses = poses.detach().cpu().numpy()
            while isinstance(poses, np.ndarray) and poses.ndim > 3 and poses.shape[0] == 1:
                poses = poses[0]

            idx_a = cue.frame_idx_a
            idx_b = cue.frame_idx_b if cue.frame_idx_b is not None else idx_a + 1
            if idx_b >= len(poses):
                raise IndexError(f"frame_idx_b {idx_b} out of bounds for {len(poses)} poses")

            pos_a = poses[idx_a, :3, 3] if poses.shape[-2:] in [(4, 4), (3, 4)] else poses[idx_a, :3]
            pos_b = poses[idx_b, :3, 3] if poses.shape[-2:] in [(4, 4), (3, 4)] else poses[idx_b, :3]

            pred_dist = float(np.linalg.norm(pos_b - pos_a))
            if pred_dist <= 1e-6:
                raise ValueError(f"Predicted translation distance between frames {idx_a} and {idx_b} is near zero: {pred_dist}")

            alpha = float(cue.metric_val / pred_dist)

        elif cue.cue_type == MetricCueType.STEREO_BASELINE:
            poses = predictions.get("extrinsic")
            if poses is None:
                raise KeyError("Predictions dict missing 'extrinsic' poses required for STEREO_BASELINE cue")
            if isinstance(poses, torch.Tensor):
                poses = poses.detach().cpu().numpy()
            while isinstance(poses, np.ndarray) and poses.ndim > 3 and poses.shape[0] == 1:
                poses = poses[0]

            idx_a = cue.frame_idx_a
            idx_b = cue.frame_idx_b if cue.frame_idx_b is not None else 1

            pos_a = poses[idx_a, :3, 3] if poses.shape[-2:] in [(4, 4), (3, 4)] else poses[idx_a, :3]
            pos_b = poses[idx_b, :3, 3] if poses.shape[-2:] in [(4, 4), (3, 4)] else poses[idx_b, :3]

            pred_baseline = float(np.linalg.norm(pos_b - pos_a))
            if pred_baseline <= 1e-6:
                raise ValueError(f"Predicted stereo baseline is near zero: {pred_baseline}")

            alpha = float(cue.metric_val / pred_baseline)

        else:
            raise NotImplementedError(f"Unsupported MetricCueType: {cue.cue_type}")

        return alpha

    def apply_scale(self, predictions: Dict[str, Any], scale_factor: float) -> Dict[str, Any]:
        """
        Rescale predictions dict in-place or return scaled dictionary.
        Scales extrinsics translation, depth maps, and world points.
        """
        scaled = {}
        for k, v in predictions.items():
            if k == "extrinsic":
                if isinstance(v, torch.Tensor):
                    v_scaled = v.clone()
                    v_scaled[..., :3, 3] = v_scaled[..., :3, 3] * scale_factor
                elif isinstance(v, np.ndarray):
                    v_scaled = v.copy()
                    v_scaled[..., :3, 3] = v_scaled[..., :3, 3] * scale_factor
                else:
                    v_scaled = v
                scaled[k] = v_scaled
            elif k in ["depth", "world_points"]:
                if isinstance(v, (torch.Tensor, np.ndarray)):
                    scaled[k] = v * scale_factor
                else:
                    scaled[k] = v
            else:
                scaled[k] = v

        scaled["metric_scale_alpha"] = scale_factor
        return scaled
