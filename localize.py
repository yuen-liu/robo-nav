"""
Coarse-to-Fine 6DoF Visual Relocalization & Metric-Scaled 3D Mapping
----------------------------------------------------------------------
Integrates LingBot-Map's streaming & windowed 3D geometry engine with
robo-nav's MetricScaleSolver (solving metric scale alpha via 1 metric cue).

Pipeline Stages:
1. Stage 1 (GPU Batched DINOv2 Coarse Search): Rapid visual similarity keyframe
   retrieval, with blur/quality gating to drop low-sharpness map frames and
   retrieval-confidence gating (margin/entropy) to flag ambiguous matches.
2. Stage 2 (Full Map Reconstruction + Metric Scale Calibration): Reconstructs
   the full map and solves scale scalar alpha (converting predictions to meters).
3. Stage 3 (k-Hypothesis Local Pose Estimation + Consensus Fusion): Computes an
   independent 6DoF global pose hypothesis from each of the top-k retrieved
   references, then fuses them via distance-based consensus rather than
   trusting a single reference blindly.
"""

import os
import sys

# Enable PyTorch CUDA memory segment expansion to mitigate fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import glob
import time
import argparse
import shutil
from typing import Optional, Tuple, Dict, Any, List

import cv2
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
    top_k_positions: Optional[np.ndarray] = None,
) -> dict:
    """
    Computes visual similarity confidence metrics, softmax entropy, and spatial uncertainty.

    Args:
        sim_scores: Array of cosine similarity scores across all map candidates.
        top_k_candidates: List of tuples (sim_score, index, filepath) sorted by score.
        margin_threshold: Score gap threshold below which localizations are deemed ambiguous.
        top_k_positions: Optional (k, 3) spatial coordinates of top-k candidates for covariance calculation.

    Returns:
        Dict containing top1/top2 similarity, margin, ambiguity flag, entropy, and spatial variance.
    """
    top1_sim = top_k_candidates[0][0]
    top2_sim = top_k_candidates[1][0] if len(top_k_candidates) > 1 else 0.0

    # Similarity score gap between #1 match and runner-up match
    margin = float(top1_sim - top2_sim)
    margin_ratio = float(margin / (top1_sim + 1e-8))
    is_ambiguous = margin < margin_threshold

    # Softmax temperature-scaled distribution & Shannon entropy across map candidates
    tau = 0.1
    logits = sim_scores / tau
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    softmax_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    # Spatial dispersion (covariance matrix & std) across top-k candidate positions
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


def compute_image_sharpness(image_path: str) -> float:
    """Laplacian-variance sharpness score for a single image; lower means blurrier."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def filter_blurry_frames(
    paths: List[str],
    blur_threshold: float,
) -> Tuple[List[str], List[Tuple[str, float]]]:
    """
    Blur/quality gating at ingestion: drops map frames whose Laplacian-variance
    sharpness falls below blur_threshold, before they can silently corrupt
    downstream retrieval or pose estimation.

    Returns:
        (kept_paths, dropped) where dropped is a list of (path, sharpness_score).
    """
    kept, dropped = [], []
    for p in paths:
        score = compute_image_sharpness(p)
        if score < blur_threshold:
            dropped.append((p, score))
        else:
            kept.append(p)
    return kept, dropped


def average_rotations(rotations: List[np.ndarray]) -> np.ndarray:
    """Chordal L2 mean of a set of rotation matrices, projected back onto SO(3) via SVD."""
    if len(rotations) == 1:
        return rotations[0]
    R_sum = np.sum(rotations, axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg


def fuse_pose_hypotheses(
    hypotheses: List[Dict[str, Any]],
    consensus_radius: float = 0.3,
) -> Dict[str, Any]:
    """
    Fuses k independently-computed global pose hypotheses (one per top-k retrieved
    reference) via distance-based consensus: finds the largest mutually-agreeing
    cluster (within consensus_radius meters of each other) and averages over it,
    discarding any reference whose hypothesis disagrees with the consensus rather
    than trusting a single reference blindly.

    Args:
        hypotheses: List of dicts each containing "position" (3,), "rotation" (3,3),
            and "map_idx".
        consensus_radius: Max pairwise translation distance (meters) for two
            hypotheses to be considered in agreement.

    Returns:
        Dict with fused position/rotation plus consensus diagnostics.
    """
    positions = np.stack([h["position"] for h in hypotheses])
    n = len(hypotheses)

    if n == 1:
        inlier_idxs = [0]
    else:
        dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        support = (dists < consensus_radius).sum(axis=1)
        seed_idx = int(np.argmax(support))
        inlier_idxs = [i for i in range(n) if dists[seed_idx, i] < consensus_radius]

    outlier_idxs = [i for i in range(n) if i not in inlier_idxs]

    inlier_positions = positions[inlier_idxs]
    inlier_rotations = [hypotheses[i]["rotation"] for i in inlier_idxs]

    fused_position = inlier_positions.mean(axis=0)
    fused_rotation = average_rotations(inlier_rotations)
    spatial_std = inlier_positions.std(axis=0) if len(inlier_idxs) > 1 else np.zeros(3)

    return {
        "position": fused_position,
        "rotation": fused_rotation,
        "num_hypotheses": n,
        "num_inliers": len(inlier_idxs),
        "inlier_map_indices": [hypotheses[i]["map_idx"] for i in inlier_idxs],
        "outlier_map_indices": [hypotheses[i]["map_idx"] for i in outlier_idxs],
        "consensus_spatial_std": spatial_std,
    }


class DemoArgs:
    """Configuration container for LingBot-Map model initialization and inference."""
    def __init__(self, model_path: str, mode: str = "windowed", use_sdpa: Optional[bool] = None):
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


def run_gct_stream_inference(gct_model, images_tensor: torch.Tensor, device: torch.device):
    """Executes causal streaming inference with KV caching for fast local window alignment."""
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


def run_demo_window_inference(gct_window_model, images_tensor: torch.Tensor, demo_args: DemoArgs):
    """Executes overlapping windowed inference for large-scale full map sequence reconstruction."""
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


def prepare_array(arr):
    """Standardize tensor/ndarray shapes for consistent depth slicing."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    while isinstance(arr, np.ndarray) and arr.ndim > 0 and arr.shape[0] == 1 and arr.ndim > 3:
        arr = arr[0]
    return arr


def localize_query(
    query_path: str,
    map_folder: str,
    model_path: str = "checkpoints/lingbot-map-long.pt",
    top_k: int = 4,
    max_map_frames: Optional[int] = None,
    margin_threshold: float = 0.05,
    blur_threshold: float = 30.0,
    consensus_radius: float = 0.3,
    metric_cue: Optional[MetricCue] = None,
    use_sdpa: Optional[bool] = None,
    visualize: bool = True,
    port: int = 8080,
) -> dict:
    """
    Localizes a query image against a spatial map folder in 3 coarse-to-fine stages.

    Returns:
        Dict containing query filepath, top retrieval match, consensus-fused 3D metric
        position, 3x3 rotation matrix, 4x4 SE(3) c2w matrix, solved metric scale alpha,
        retrieval/consensus confidence diagnostics, and a low_confidence flag.
    """
    if use_sdpa is None:
        use_sdpa = (shutil.which("ninja") is None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" robo-nav: Metric-Scaled 6DoF Relocalization & 3D Render")
    print(f" Query Image: {os.path.basename(query_path)}")
    print(f" Map Folder:  {map_folder}")
    if metric_cue is not None:
        print(f" Metric Cue:  Type={metric_cue.cue_type.value}, MetricVal={metric_cue.metric_val}m")
    print(f"=======================================================\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── STAGE 1: Fast GPU Batched DINOv2 Visual Similarity Search ──────────────
    print(" [Stage 1] Extracting DINOv2 embeddings in GPU batch pass...")
    t0 = time.time()

    # Discover map image candidates
    valid_exts = (".png", ".jpg", ".jpeg")
    all_candidates = []
    for root, _, files in os.walk(map_folder):
        for f in files:
            if f.lower().endswith(valid_exts):
                all_candidates.append(os.path.join(root, f))
    all_candidates = sorted(list(set(all_candidates)))
    map_candidate_paths = all_candidates[:max_map_frames] if max_map_frames else all_candidates

    # Blur/quality gating at ingestion: drop low-sharpness map frames before they can
    # silently corrupt retrieval or downstream pose estimation.
    if blur_threshold > 0:
        map_candidate_paths, dropped = filter_blurry_frames(map_candidate_paths, blur_threshold)
        if dropped:
            dropped_names = [os.path.basename(p) for p, _ in dropped[:5]]
            print(f"  Blur gating: dropped {len(dropped)}/{len(dropped) + len(map_candidate_paths)} "
                  f"low-sharpness map frames (threshold={blur_threshold}): {dropped_names}"
                  f"{' ...' if len(dropped) > 5 else ''}")
    if not map_candidate_paths:
        raise ValueError(
            "No map frames remain after blur gating; lower --blur_threshold or check map_folder."
        )

    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()

    # Preprocessing pipeline for DINOv2 vision transformer
    transform = transforms.Compose([
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_paths = [query_path] + map_candidate_paths

    # Batch GPU embedding extraction & L2 normalization
    batch_tensors = torch.stack([transform(Image.open(p).convert('RGB')) for p in all_paths]).to(device)
    with torch.no_grad():
        all_feats = F.normalize(dinov2(batch_tensors), dim=-1)

    # Compute dot-product cosine similarity matrix between query and map candidates
    query_feat = all_feats[0:1]
    map_feats = all_feats[1:]
    sim_scores = (query_feat @ map_feats.T)[0].cpu().numpy()

    # Free DINOv2 model and temporary feature tensors from GPU memory
    del dinov2, batch_tensors, all_feats, query_feat, map_feats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Rank candidate keyframes by visual similarity
    sims = [(float(sim_scores[idx]), idx, p) for idx, p in enumerate(map_candidate_paths)]
    sims.sort(key=lambda x: x[0], reverse=True)
    top_k_candidates = sims[:top_k]
    best_sim, best_map_idx, best_map_path = top_k_candidates[0]
    print(f"  Stage 1 GPU coarse match completed in {time.time() - t0:.2f}s!")

    # Retrieval-quality gating: flag ambiguous top-1/top-2 similarity margins before
    # committing to a match — never trust coarse retrieval as a final answer alone.
    retrieval_confidence = compute_confidence_and_uncertainty(
        sim_scores=sim_scores,
        top_k_candidates=top_k_candidates,
        margin_threshold=margin_threshold,
    )
    if retrieval_confidence["is_ambiguous"]:
        print(f"  ⚠️  Retrieval ambiguity: top1/top2 similarity margin "
              f"({retrieval_confidence['margin']:.4f}) is below threshold ({margin_threshold}) — "
              f"multiple map locations look visually similar.")

    # ── STAGE 2: Full Map Reconstruction + Metric Scale Calibration ───────────
    print(f"\n [Stage 2] Building Full Map Reconstruction...")
    map_images = load_and_preprocess_images(map_candidate_paths, mode="crop", image_size=518, patch_size=14).to(device)
    window_args = DemoArgs(model_path=model_path, mode="windowed", use_sdpa=use_sdpa)
    gct_window_model = demo_load_model(window_args, device)
    map_preds = run_demo_window_inference(gct_window_model, map_images, window_args)
    map_preds_post, map_images_cpu = demo_postprocess(map_preds, map_images)

    # Solve metric scale factor alpha (X_metric = alpha * X_unit) if a metric cue is provided
    metric_alpha = 1.0
    if metric_cue is not None:
        solver = MetricScaleSolver()
        metric_alpha = solver.solve_scale(map_preds_post, metric_cue)
        map_preds_post = solver.apply_scale(map_preds_post, metric_alpha)
        print(f"  Solved Metric Scale Alpha: {metric_alpha:.6f} (map scaled to physical meters)")

    del gct_window_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Extract global camera-to-world (c2w) matrices for all map keyframes
    map_c2w_3x4 = map_preds_post["extrinsic"]
    if isinstance(map_c2w_3x4, torch.Tensor):
        map_c2w_3x4 = map_c2w_3x4.detach().cpu().numpy()
    map_c2w = np.zeros((map_c2w_3x4.shape[0], 4, 4), dtype=np.float32)
    map_c2w[:, :3, :4] = map_c2w_3x4
    map_c2w[:, 3, 3] = 1.0

    map_depths_all = prepare_array(map_preds_post["depth"])

    # ── STAGE 3: k-Independent-Hypothesis Local Pose Estimation + Consensus ───
    print(f"\n [Stage 3] Computing {len(top_k_candidates)} independent pose hypotheses "
          f"for consensus fusion...")
    window_radius = 2
    stream_args = DemoArgs(model_path=model_path, mode="streaming", use_sdpa=use_sdpa)
    gct_stream_model = demo_load_model(stream_args, device)

    hypotheses = []
    for sim, cand_map_idx, cand_map_path in top_k_candidates:
        # Construct a contiguous keyframe window around this candidate reference + query image
        start_idx = max(0, cand_map_idx - window_radius)
        end_idx = min(len(map_candidate_paths), cand_map_idx + 1)
        contiguous_window_paths = map_candidate_paths[start_idx:end_idx]
        subsequence_paths = contiguous_window_paths + [query_path]
        anchor_sub_idx = len(contiguous_window_paths) - 1

        sub_images = load_and_preprocess_images(
            subsequence_paths, mode="crop", image_size=518, patch_size=14
        ).to(device)
        sub_preds = run_gct_stream_inference(gct_stream_model, sub_images, device)

        sub_ext, sub_int = pose_encoding_to_extri_intri(sub_preds["pose_enc"], sub_images.shape[-2:])
        sub_ext4x4 = torch.zeros((*sub_ext.shape[:-2], 4, 4), device=sub_ext.device, dtype=sub_ext.dtype)
        sub_ext4x4[..., :3, :4] = sub_ext
        sub_ext4x4[..., 3, 3] = 1.0
        sub_c2w = closed_form_inverse_se3_general(sub_ext4x4)[0].cpu().numpy()

        # Compute relative pose delta: delta_T = (T_anchor)^(-1) @ T_query
        delta_T = np.linalg.inv(sub_c2w[anchor_sub_idx]) @ sub_c2w[-1]

        # Compute depth scale ratio between this candidate's map depth and its local subsequence depth
        sub_depths_all = prepare_array(sub_preds["depth"])
        anchor_map_depth = map_depths_all[cand_map_idx]
        anchor_sub_depth = sub_depths_all[anchor_sub_idx]
        rel_scale_factor = float(np.median(anchor_map_depth) / np.median(anchor_sub_depth))

        delta_T_scaled = delta_T.copy()
        delta_T_scaled[:3, 3] = delta_T[:3, 3] * rel_scale_factor

        # Compose this hypothesis's 6DoF pose in global map coordinate space
        T_map_anchor = map_c2w[cand_map_idx]
        T_map_query = T_map_anchor @ delta_T_scaled

        hypotheses.append({
            "map_idx": cand_map_idx,
            "map_path": cand_map_path,
            "similarity": sim,
            "position": T_map_query[:3, 3],
            "rotation": T_map_query[:3, :3],
            "c2w": T_map_query,
        })

    del gct_stream_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Consensus fusion across k independent hypotheses — discard outlier references
    # rather than trusting the single top-1 retrieval match blindly.
    fused = fuse_pose_hypotheses(hypotheses, consensus_radius=consensus_radius)
    query_pos = fused["position"]
    query_rot = fused["rotation"]
    T_map_query = np.eye(4, dtype=np.float64)
    T_map_query[:3, :3] = query_rot
    T_map_query[:3, 3] = query_pos

    if fused["num_inliers"] < fused["num_hypotheses"]:
        print(f"  ⚠️  Consensus fusion discarded {fused['num_hypotheses'] - fused['num_inliers']} "
              f"outlier hypothesis(es) (map indices {fused['outlier_map_indices']}) that disagreed by "
              f">{consensus_radius}m — kept map indices {fused['inlier_map_indices']}.")

    # Final confidence: retrieval margin/entropy plus spatial dispersion across the k
    # independently-computed pose hypotheses (not just candidate image similarity).
    hypothesis_positions = np.stack([h["position"] for h in hypotheses])
    final_confidence = compute_confidence_and_uncertainty(
        sim_scores=sim_scores,
        top_k_candidates=top_k_candidates,
        margin_threshold=margin_threshold,
        top_k_positions=hypothesis_positions,
    )
    low_confidence = bool(retrieval_confidence["is_ambiguous"]) or fused["num_inliers"] <= max(1, fused["num_hypotheses"] // 2)

    print("\n=======================================================")
    print(" FINAL PREDICTED 6DOF CAMERA POSE IN METRIC MAP")
    print("=======================================================")
    print(f" Query Image:            {os.path.basename(query_path)}")
    print(f" Top Retrieval Match:    {os.path.basename(best_map_path)} (Index #{best_map_idx})")
    print(f" Consensus Inliers:      {fused['num_inliers']}/{fused['num_hypotheses']}")
    print(f" Metric Scale Alpha:     {metric_alpha:.6f}")
    print(f" Query 3D Position (m):  [{query_pos[0]:.4f}, {query_pos[1]:.4f}, {query_pos[2]:.4f}]")
    print(f" Query 3D Rotation:\n{query_rot}")
    if low_confidence:
        print(f" ⚠️  LOW CONFIDENCE localization — verify before acting on this pose.")
    print("=======================================================")

    # ── Interactive 3D Viewer ───────────────────────────────────────────────
    if visualize:
        try:
            from lingbot_map.vis import PointCloudViewer
            import viser.transforms as tf

            print(f"\n Launching 3D viewer on port {port}...")
            vis_pred_dict = demo_prepare_vis(map_preds_post, map_images_cpu)
            viewer = PointCloudViewer(
                pred_dict=vis_pred_dict,
                port=port,
                vis_threshold=1.5,
                image_folder=map_folder,
            )

            # Highlight the consensus-fused query pose as a distinct frustum,
            # separate from the map's own per-frame cameras.
            intrinsic_np = map_preds_post["intrinsic"]
            if isinstance(intrinsic_np, torch.Tensor):
                intrinsic_np = intrinsic_np.detach().cpu().numpy()
            focal = float(intrinsic_np[best_map_idx, 0, 0])
            pp = (float(intrinsic_np[best_map_idx, 0, 2]), float(intrinsic_np[best_map_idx, 1, 2]))
            fov = 2 * np.arctan(pp[0] / focal)
            aspect = pp[0] / pp[1]
            query_wxyz = tf.SO3.from_matrix(query_rot).wxyz

            viewer.server.scene.add_camera_frustum(
                name="/query/camera",
                fov=fov,
                aspect=aspect,
                wxyz=query_wxyz,
                position=query_pos,
                scale=0.08,
                color=(255, 0, 0) if not low_confidence else (255, 165, 0),
            )
            viewer.server.scene.add_label(
                "/query/label",
                text=f"QUERY: {os.path.basename(query_path)}"
                     f"{' (LOW CONFIDENCE)' if low_confidence else ''}",
                position=query_pos,
            )
            print(f" Viewer ready — open http://localhost:{port} "
                  f"(tunnel with: ssh -L {port}:localhost:{port} <cluster>)")
            viewer.run()
        except ImportError:
            print("viser not installed. Install with: pip install viser")

    return {
        "query_path": query_path,
        "anchor_keyframe": best_map_path,
        "position": query_pos,
        "rotation": query_rot,
        "c2w_matrix": T_map_query,
        "metric_scale_alpha": metric_alpha,
        "confidence": final_confidence,
        "consensus": fused,
        "hypotheses": hypotheses,
        "low_confidence": low_confidence,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="robo-nav Metric-Scaled 6DoF Relocalization")
    parser.add_argument("query_path", type=str, help="Path to query image file")
    parser.add_argument("map_folder", type=str, help="Directory containing map keyframe images")
    parser.add_argument("--model_path", type=str, default="checkpoints/lingbot-map-long.pt", help="Path to model checkpoint")
    parser.add_argument("--top_k", type=int, default=4, help="Number of top visual candidates to retrieve and fuse via consensus")
    parser.add_argument("--margin_threshold", type=float, default=0.05, help="Top1/top2 similarity margin below which retrieval is flagged ambiguous")
    parser.add_argument("--blur_threshold", type=float, default=30.0, help="Laplacian-variance sharpness threshold; map frames below this are dropped at ingestion (0 disables)")
    parser.add_argument("--consensus_radius", type=float, default=0.3, help="Max pairwise disagreement (meters) between pose hypotheses to be treated as consensus inliers")
    parser.add_argument("--metric_cue_type", type=str, default=None, choices=["depth_point", "translation_step", "stereo_baseline"], help="Metric cue type for scale calibration")
    parser.add_argument("--metric_val", type=float, default=None, help="Physical metric cue value in meters")
    parser.add_argument("--frame_idx_a", type=int, default=0, help="First frame index for metric cue")
    parser.add_argument("--frame_idx_b", type=int, default=1, help="Second frame index for metric cue")
    parser.add_argument("--port", type=int, default=8080, help="Viser port for the interactive 3D viewer")
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=True,
                        help="Launch the interactive 3D viewer after localizing (on by default; use --no-visualize to skip)")
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
        margin_threshold=args.margin_threshold,
        blur_threshold=args.blur_threshold,
        consensus_radius=args.consensus_radius,
        metric_cue=metric_cue,
        visualize=args.visualize,
        port=args.port,
    )
