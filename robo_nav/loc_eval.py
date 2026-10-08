"""
Kidnapped-robot localization benchmark on a `sim_render` dataset: for every query start, each
method predicts the room (and, where it can, the position/heading); scored against GT.

Methods (pick with --methods):
  prior            always the room with the most map frames (chance-level floor)
  dinov2           localize.py's Stage 1: DINOv2 ViT-S/14 CLS, Resize(518)+CenterCrop(518),
                   room = similarity-weighted vote of the top-k map frames, pose = top-1 frame
  dinov2_full      same, but the whole 16:9 frame (no center crop)
  dinov2_look      dinov2 on all look-around views; votes pooled across views
  dinov2_full_look dinov2_full on all look-around views

  python -m robo_nav.loc_eval sim_data/loc/val2 --methods prior dinov2 dinov2_full dinov2_look
"""

import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from glob import glob
from typing import Dict, List

import numpy as np


def load_dataset(root: str) -> dict:
    def rows(path):
        with open(path) as f:
            return list(csv.DictReader(f))
    m = rows(os.path.join(root, "map", "poses.csv"))
    q = rows(os.path.join(root, "queries.csv"))
    views = len(glob(os.path.join(root, "queries", "q000_v*.jpg")))
    return {
        "root": root,
        "map_paths": [os.path.join(root, "map", "frames", f"{int(r['idx']):06d}.jpg") for r in m],
        "map_xy": np.array([[float(r["x"]), float(r["y"])] for r in m]),
        "map_yaw": np.array([float(r["yaw"]) for r in m]),
        "map_room": [r["room_name"] for r in m],
        "q_xy": np.array([[float(r["x"]), float(r["y"])] for r in q]),
        "q_yaw": np.array([float(r["yaw"]) for r in q]),
        "q_room": [r["room_name"] for r in q],
        "q_paths": [[os.path.join(root, "queries", f"q{int(r['idx']):03d}_v{v}.jpg") for v in range(views)] for r in q],
        "views": views,
    }


def angle_err_deg(a, b):
    return abs(math.degrees(math.atan2(math.sin(a - b), math.cos(a - b))))


# --------------------------------------------------------------------------- DINOv2 retrieval

class DinoEmbedder:
    def __init__(self, crop: bool, device: str, cache_dir: str):
        import torch
        from torchvision import transforms
        self.torch, self.device = torch, device
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if crop:  # exactly localize.py
            self.tf = transforms.Compose([transforms.Resize(518), transforms.CenterCrop(518), transforms.ToTensor(), norm])
        else:     # keep the full 16:9 frame; both sides multiples of the 14 px patch
            self.tf = transforms.Compose([transforms.Resize((294, 518)), transforms.ToTensor(), norm])
        self.tag = "dinov2_crop" if crop else "dinov2_full"
        self.cache_dir = cache_dir

    def embed(self, paths: List[str], cache_name: str = None, batch: int = 32) -> np.ndarray:
        from PIL import Image
        if cache_name:
            cache = os.path.join(self.cache_dir, f"{self.tag}_{cache_name}.npy")
            if os.path.exists(cache):
                feats = np.load(cache)
                if len(feats) == len(paths):
                    return feats
        out = []
        with self.torch.no_grad():
            for i in range(0, len(paths), batch):
                x = self.torch.stack([self.tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch]]).to(self.device)
                f = self.torch.nn.functional.normalize(self.model(x), dim=-1)
                out.append(f.float().cpu().numpy())
        feats = np.concatenate(out)
        if cache_name:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(cache, feats)
        return feats


def retrieval_predict(ds: dict, map_feats: np.ndarray, q_feats: np.ndarray, top_k: int, temp: float = 0.02) -> dict:
    """q_feats: (V, D) for one query start (V=1 for single view). Pools a softmax-weighted
    room vote over every view's top-k; position = weighted mean of the strongest room's
    top matches; heading = view-0 top-1 yaw (single view) or the yaw implied by the best view."""
    sims = q_feats @ map_feats.T                                  # (V, M)
    votes: Dict[str, float] = defaultdict(float)
    cands = []
    for v, s in enumerate(sims):
        top = np.argsort(-s)[:top_k]
        w = np.exp((s[top] - s[top[0]]) / temp)
        for j, wj in zip(top, w):
            votes[ds["map_room"][j]] += wj
            cands.append((s[j], wj, j, v))
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    room = ranked[0][0]
    total = sum(votes.values())
    in_room = [(s, w, j, v) for s, w, j, v in cands if ds["map_room"][j] == room]
    w = np.array([c[1] for c in in_room])
    xy = (ds["map_xy"][[c[2] for c in in_room]] * w[:, None]).sum(0) / w.sum()
    best = max(cands, key=lambda c: c[0])
    yaw = ds["map_yaw"][best[2]] - 2 * math.pi * best[3] / max(1, ds["views"])  # undo that view's offset
    top2 = np.sort(sims.max(0))[-2:]
    return {"room": room, "xy": xy, "yaw": yaw, "conf": ranked[0][1] / total,
            "margin": float(top2[1] - top2[0])}


# --------------------------------------------------------------------------- evaluation

def evaluate(ds: dict, method: str, args) -> List[dict]:
    n = len(ds["q_room"])
    if method == "prior":
        room = Counter(ds["map_room"]).most_common(1)[0][0]
        return [{"room": room, "xy": None, "yaw": None, "conf": 0.0} for _ in range(n)]

    if method.startswith("dinov2"):
        crop = not method.startswith("dinov2_full")
        look = method.endswith("_look")
        key = "crop" if crop else "full"
        if key not in args.embedders:
            args.embedders[key] = DinoEmbedder(crop, args.device, os.path.join(ds["root"], "cache"))
        emb = args.embedders[key]
        map_feats = emb.embed(ds["map_paths"], "map")
        flat = [p for views in ds["q_paths"] for p in views]
        q_all = emb.embed(flat, "queries").reshape(n, ds["views"], -1)
        return [retrieval_predict(ds, map_feats, q_all[i] if look else q_all[i, :1], args.top_k) for i in range(n)]

    raise SystemExit(f"unknown method {method}")


def score(ds: dict, preds: List[dict]) -> dict:
    correct = [p["room"] == r for p, r in zip(preds, ds["q_room"])]
    out = {"room_acc": float(np.mean(correct)), "n": len(preds)}
    if preds[0]["xy"] is not None:
        pos = np.array([np.linalg.norm(p["xy"] - xy) for p, xy in zip(preds, ds["q_xy"])])
        yaw = np.array([angle_err_deg(p["yaw"], y) for p, y in zip(preds, ds["q_yaw"])])
        out.update(pos_median_m=float(np.median(pos)), pos_within_1m=float(np.mean(pos < 1.0)),
                   yaw_median_deg=float(np.median(yaw)), yaw_within_30=float(np.mean(yaw < 30)))
        conf = np.array([p["conf"] for p in preds])
        hi = conf >= np.median(conf)
        out.update(acc_confident_half=float(np.mean(np.array(correct)[hi])),
                   acc_unsure_half=float(np.mean(np.array(correct)[~hi])))
    # same-type confusions (e.g. bedroom_1 vs bedroom_3) vs wrong room type entirely
    same_type = sum(1 for p, r in zip(preds, ds["q_room"]) if p["room"] != r and p["room"].split("_")[0] == r.split("_")[0])
    out["wrong_same_type"] = same_type
    out["wrong_other_type"] = int(len(preds) - sum(correct) - same_type)
    return out


def main():
    parser = argparse.ArgumentParser(description="Kidnapped-robot localization benchmark")
    parser.add_argument("dataset", type=str)
    parser.add_argument("--methods", nargs="+", default=["prior", "dinov2", "dinov2_full", "dinov2_look", "dinov2_full_look"])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default=None, help="Results JSON (default <dataset>/results.json)")
    args = parser.parse_args()
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    args.embedders = {}

    ds = load_dataset(args.dataset)
    print(f"{len(ds['map_paths'])} map frames, {len(ds['q_room'])} queries x {ds['views']} views, device={args.device}")
    out_path = args.out or os.path.join(args.dataset, "results.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for m in args.methods:
        t = time.perf_counter()
        preds = evaluate(ds, m, args)
        res = score(ds, preds)
        res["seconds"] = round(time.perf_counter() - t, 1)
        results[m] = res
        with open(os.path.join(args.dataset, f"preds_{m}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "gt_room", "pred_room", "conf", "pred_x", "pred_y", "pred_yaw"])
            for i, p in enumerate(preds):
                xy = p["xy"] if p["xy"] is not None else (None, None)
                w.writerow([i, ds["q_room"][i], p["room"], f"{p['conf']:.3f}", xy[0], xy[1], p["yaw"]])
        print(f"{m:<18} room {res['room_acc']:.0%}" + (
            f" | pos median {res['pos_median_m']:.2f} m, <1m {res['pos_within_1m']:.0%}"
            f" | yaw median {res['yaw_median_deg']:.0f}°" if "pos_median_m" in res else "") +
            f" | wrong: {res['wrong_same_type']} same-type, {res['wrong_other_type']} other")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
