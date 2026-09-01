"""
Coarse-to-Fine 6DoF Visual Relocalization & Metric-Scaled 3D Mapping
----------------------------------------------------------------------
Integrates LingBot-Map's streaming & windowed 3D geometry engine with
robo-nav's MetricScaleSolver (solving scale alpha via 1 metric cue).

1. Stage 1 (Fast GPU Batched DINOv2 Coarse Search): Finds top-k matching keyframes.
2. Stage 2 (Local Subsequence 6DoF Relative Transform): Computes local delta T.
3. Stage 3 (Full Windowed Reconstruction + Metric Scale Calibration):
   Reconstructs full map, solves metric scale alpha if metric cue is supplied,
   and places query image at its true localized 6DoF pose in metric space.
"""

import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import glob
import time
import argparse
import shutil
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo import (
    load_model as demo_load_model,
    postprocess as demo_postprocess,
    prepare_for_visualization as demo_prepare_vis,
)
from graph import TopologicalGraph

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general

from robo_nav.scale_calibration import MetricScaleSolver, MetricCue, MetricCueType


def compute_confidence_and_uncertainty(
    sim_scores: np.ndarray,
    top_k_candidates: list,
    margin_threshold: float = 0.05,
    top_k_positions: np.ndarray = None,
) -> dict:
    top1_sim = top_k_candidates[0][0]
    top2_sim = top_k_candidates[1][0] if len(top_k_candidates) > 1 else 0.0

    margin = float(top1_sim - top2_sim)
    margin_ratio = float(margin / (top1_sim + 1e-8))
    is_ambiguous = margin < margin_threshold

    tau = 0.1
    logits = sim_scores / tau
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    softmax_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    cov_matrix = None
    cov_trace = None
    spatial_std = None

    if top_k_positions is not None and len(top_k_positions) >= 2:
        cov_matrix = np.cov(top_k_positions, rowvar=False)
        if cov_matrix.ndim == 2:
            cov_trace = float(np.trace(cov_matrix))
            spatial_std = np.sqrt(np.maximum(0.0, np.diag(cov_matrix)))
        elif cov_matrix.ndim == 0:
            cov_trace = float(cov_matrix)
    return {
        "top1_similarity": float(top1_sim),
        "top2_similarity": float(top2_sim),
        "margin": margin,
        "margin_ratio": margin_ratio,
        "margin_threshold": margin_threshold,
        "is_ambiguous": is_ambiguous,
        "softmax_entropy": softmax_entropy,
        "spatial_covariance": cov_matrix,
        "spatial_trace": cov_trace,
        "spatial_std": spatial_std,
    }


class DemoArgs:
    """Args container matching demo.py configuration."""
    def __init__(self, model_path, mode="windowed", use_sdpa=None):
        self.mode = mode
        self.model_path = model_path
        self.image_size = 518
        self.patch_size = 14
        self.enable_3d_rope = True
        self.max_frame_num = 1024
        self.kv_cache_sliding_window = 64
        self.num_scale_frames = 8
        self.use_sdpa = use_sdpa if use_sdpa is not None else (shutil.which("ninja") is None)
        self.camera_num_iterations = 4
        self.window_size = 64
        self.overlap_size = 16
        self.overlap_keyframes = None
        self.keyframe_interval = 1
        self.compile = False
        self.offload_to_cpu = False


def run_gct_stream_inference(gct_model, images_tensor, device):
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32
    if dtype != torch.float32 and hasattr(gct_model, "aggregator"):
        gct_model.aggregator = gct_model.aggregator.to(dtype=dtype)

    output_device = torch.device("cpu")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        return gct_model.inference_streaming(
            images_tensor,
            num_scale_frames=min(8, images_tensor.shape[0]),
            keyframe_interval=1,
            output_device=output_device,
        )


def run_demo_window_inference(gct_window_model, images_tensor, demo_args):
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    if dtype != torch.float32 and hasattr(gct_window_model, "aggregator"):
        gct_window_model.aggregator = gct_window_model.aggregator.to(dtype=dtype)

    output_device = torch.device("cpu")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        return gct_window_model.inference_windowed(
            images_tensor,
            window_size=demo_args.window_size,
            overlap_size=demo_args.overlap_size,
            overlap_keyframes=demo_args.overlap_keyframes,
            num_scale_frames=demo_args.num_scale_frames,
            keyframe_interval=demo_args.keyframe_interval,
            output_device=output_device,
        )


def localize_query(
    query_path: str,
    map_folder: str,
    model_path: str = "checkpoints/lingbot-map-long.pt",
    top_k: int = 4,
    max_map_frames: int = None,
    margin_threshold: float = 0.05,
    metric_cue: Optional[MetricCue] = None,
    use_sdpa: bool = None,
    visualize: bool = True,
    port: int = 8080,
):
    if use_sdpa is None:
        use_sdpa = (shutil.which("ninja") is None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" 🚀 robo-nav: Metric-Scaled 6DoF Relocalization & 3D Render")
    print(f" Query Image: {os.path.basename(query_path)}")
    print(f" Map Folder:  {map_folder}")
    if metric_cue is not None:
        print(f" Metric Cue:  Type={metric_cue.cue_type.value}, MetricVal={metric_cue.metric_val}m")
    print(f"=======================================================\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── STAGE 1: DINOv2 GPU Coarse Search ────────────────────────────────────
    print(" [Stage 1] Extracting DINOv2 embeddings in GPU batch pass...")
    t0 = time.time()
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_exts = (".png", ".jpg", ".jpeg")
    all_candidates = []
    for root, _, files in os.walk(map_folder):
        for f in files:
            if f.lower().endswith(valid_exts):
                all_candidates.append(os.path.join(root, f))
    all_candidates = sorted(list(set(all_candidates)))
    map_candidate_paths = all_candidates[:max_map_frames] if max_map_frames else all_candidates
    all_paths = [query_path] + map_candidate_paths

    batch_tensors = torch.stack([transform(Image.open(p).convert('RGB')) for p in all_paths]).to(device)
    with torch.no_grad():
        all_feats = F.normalize(dinov2(batch_tensors), dim=-1)

    query_feat = all_feats[0:1]
    map_feats = all_feats[1:]
    sim_scores = (query_feat @ map_feats.T)[0].cpu().numpy()

    del dinov2, batch_tensors, all_feats, query_feat, map_feats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sims = [(float(sim_scores[idx]), idx, p) for idx, p in enumerate(map_candidate_paths)]
    sims.sort(key=lambda x: x[0], reverse=True)
    top_k_candidates = sims[:top_k]
    best_sim, best_map_idx, best_map_path = top_k_candidates[0]
    print(f"  Stage 1 GPU coarse match completed in {time.time() - t0:.2f}s!")

    # ── STAGE 2: Local Subsequence Inference ──────────────────────────────────
    window_radius = 2
    start_idx = max(0, best_map_idx - window_radius)
    end_idx = min(len(map_candidate_paths), best_map_idx + 1)
    contiguous_window_paths = map_candidate_paths[start_idx:end_idx]
    subsequence_paths = contiguous_window_paths + [query_path]
    anchor_sub_idx = len(contiguous_window_paths) - 1

    sub_images = load_and_preprocess_images(subsequence_paths, mode="crop", image_size=518, patch_size=14).to(device)
    stream_args = DemoArgs(model_path=model_path, mode="streaming", use_sdpa=use_sdpa)
    gct_stream_model = demo_load_model(stream_args, device)
    sub_preds = run_gct_stream_inference(gct_stream_model, sub_images, device)

    sub_ext, sub_int = pose_encoding_to_extri_intri(sub_preds["pose_enc"], sub_images.shape[-2:])
    sub_ext4x4 = torch.zeros((*sub_ext.shape[:-2], 4, 4), device=sub_ext.device, dtype=sub_ext.dtype)
    sub_ext4x4[..., :3, :4] = sub_ext
    sub_ext4x4[..., 3, 3] = 1.0
    sub_c2w = closed_form_inverse_se3_general(sub_ext4x4)[0].cpu().numpy()

    delta_T = np.linalg.inv(sub_c2w[anchor_sub_idx]) @ sub_c2w[-1]
    del gct_stream_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── STAGE 3: Full Map Reconstruction + Metric Calibration ────────────────
    print(f"\n [Stage 3] Building Full Map Reconstruction...")
    map_images = load_and_preprocess_images(map_candidate_paths, mode="crop", image_size=518, patch_size=14).to(device)
    window_args = DemoArgs(model_path=model_path, mode="windowed", use_sdpa=use_sdpa)
    gct_window_model = demo_load_model(window_args, device)
    map_preds = run_demo_window_inference(gct_window_model, map_images, window_args)
    map_preds_post, map_images_cpu = demo_postprocess(map_preds, map_images)

    # Solve Metric Scale Alpha if metric cue provided
    metric_alpha = 1.0
    if metric_cue is not None:
        solver = MetricScaleSolver()
        metric_alpha = solver.solve_scale(map_preds_post, metric_cue)
        map_preds_post = solver.apply_scale(map_preds_post, metric_alpha)
        print(f"  📐 Solved Metric Scale Alpha: {metric_alpha:.6f} (map scaled to physical meters)")

    map_c2w_3x4 = map_preds_post["extrinsic"]
    if isinstance(map_c2w_3x4, torch.Tensor):
        map_c2w_3x4 = map_c2w_3x4.detach().cpu().numpy()
    map_c2w = np.zeros((map_c2w_3x4.shape[0], 4, 4), dtype=np.float32)
    map_c2w[:, :3, :4] = map_c2w_3x4
    map_c2w[:, 3, 3] = 1.0

    def prepare_array(arr):
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        while isinstance(arr, np.ndarray) and arr.ndim > 0 and arr.shape[0] == 1 and arr.ndim > 3:
            arr = arr[0]
        return arr

    map_depths_all = prepare_array(map_preds_post["depth"])
    sub_depths_all = prepare_array(sub_preds["depth"])

    # Relative depth scale factor between Stage 3 (map) and Stage 2 (subsequence)
    anchor_map_depth = map_depths_all[best_map_idx]
    anchor_sub_depth = sub_depths_all[anchor_sub_idx]
    rel_scale_factor = float(np.median(anchor_map_depth) / np.median(anchor_sub_depth))

    delta_T_scaled = delta_T.copy()
    delta_T_scaled[:3, 3] = delta_T[:3, 3] * rel_scale_factor

    T_map_anchor = map_c2w[best_map_idx]
    T_map_query = T_map_anchor @ delta_T_scaled
    query_pos = T_map_query[:3, 3]
    query_rot = T_map_query[:3, :3]

    print("\n=======================================================")
    print(" 🎯 FINAL PREDICTED 6DOF CAMERA POSE IN METRIC MAP")
    print("=======================================================")
    print(f" Query Image:            {os.path.basename(query_path)}")
    print(f" Matched Keyframe:       {os.path.basename(best_map_path)} (Index #{best_map_idx})")
    print(f" Metric Scale Alpha:     {metric_alpha:.6f}")
    print(f" Query 3D Position (m):  [{query_pos[0]:.4f}, {query_pos[1]:.4f}, {query_pos[2]:.4f}]")
    print(f" Query 3D Rotation:\n{query_rot}")
    print("=======================================================")

    return {
        "query_path": query_path,
        "anchor_keyframe": best_map_path,
        "position": query_pos,
        "rotation": query_rot,
        "c2w_matrix": T_map_query,
        "metric_scale_alpha": metric_alpha,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="robo-nav Metric-Scaled 6DoF Relocalization")
    parser.add_argument("query_path", type=str)
    parser.add_argument("map_folder", type=str)
    parser.add_argument("--model_path", type=str, default="checkpoints/lingbot-map-long.pt")
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--metric_cue_type", type=str, default=None, choices=["depth_point", "translation_step", "stereo_baseline"])
    parser.add_argument("--metric_val", type=float, default=None, help="Physical metric cue value in meters")
    parser.add_argument("--frame_idx_a", type=int, default=0)
    parser.add_argument("--frame_idx_b", type=int, default=1)
    args = parser.parse_args()

    metric_cue = None
    if args.metric_cue_type and args.metric_val:
        metric_cue = MetricCue(
            cue_type=MetricCueType(args.metric_cue_type),
            metric_val=args.metric_val,
            frame_idx_a=args.frame_idx_a,
            frame_idx_b=args.frame_idx_b,
        )

    localize_query(
        args.query_path,
        args.map_folder,
        model_path=args.model_path,
        top_k=args.top_k,
        metric_cue=metric_cue,
        visualize=False,
    )
