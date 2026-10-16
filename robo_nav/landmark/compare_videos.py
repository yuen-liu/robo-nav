"""
Side-by-side episode videos for two landmark-nav runs (e.g. Claude vs Gemini planners) on the same
start/goal pairs, matched by (start pose, goal, mode) from each episode's JSON.

Both sides play in robot time by default: a stretch where the robot camera doesn't change for longer
than --idle_s (the robot waiting on the planner API) is cut to one second tagged "idle N s cut", so a
slow API doesn't make one side look slower than it drove. --keep_idle plays wall time.

  python -m robo_nav.landmark.compare_videos sim_data/landmark/runs/batch1 sim_data/landmark/runs/gemini_batch1 \\
      --labels "Claude Opus 5.5" "Gemini 3.8 Flash" --modes landmark --out sim_data/landmark/compare/claude_vs_gemini

Writes <out>/<episode>_<mode>.mp4 per matched pair and <out>/summary.md.
"""

import argparse
import json
import os
from glob import glob
from typing import Dict, Iterator, Optional, Tuple

import cv2
import numpy as np

from robo_nav.sim.video import CAM_H, CAM_W, FONT, open_video_writer, text

HEADER_H = 70
GREEN, RED, GREY = (90, 210, 90), (80, 80, 240), (200, 200, 200)


def load_run(run_dir: str) -> Dict[Tuple, Tuple[dict, str]]:
    """(start query, goal, mode) -> (episode json, mp4 path)."""
    out = {}
    for p in sorted(glob(os.path.join(run_dir, "e*.json"))):
        mp4 = p[:-5] + ".mp4"
        if not os.path.exists(mp4):
            continue
        e = json.load(open(p))
        out[(e["start"]["query_idx"], e["goal"], e["mode"])] = (e, mp4)
    return out


def robot_time(path: str, idle_s: float, keep_idle: bool, diff_thresh: float = 0.8) -> Iterator[Tuple[np.ndarray, str]]:
    """Frames of `path`, with long static stretches (robot camera unchanged) cut to one second."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    prev, still, still_n = None, None, 0

    def flush():
        if still_n == 0:
            return
        if keep_idle or still_n <= idle_s * fps:
            for _ in range(still_n):
                yield still, ""
        else:
            for _ in range(int(fps)):
                yield still, f"idle {still_n / fps:.0f} s cut"

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cam = cv2.cvtColor(frame[30:CAM_H, :CAM_W], cv2.COLOR_BGR2GRAY).astype(np.int16)
        if prev is not None and np.abs(cam - prev).mean() < diff_thresh:
            still, still_n = frame, still_n + 1
        else:
            yield from flush()
            still, still_n = None, 0
            yield frame, ""
        prev = cam
    yield from flush()
    cap.release()


def header(width: int, label: str, e: dict) -> np.ndarray:
    img = np.full((HEADER_H, width, 3), (32, 28, 26), np.uint8)
    ended = e["ended"].replace("claude_", "planner ").replace("_", " ")
    outcome = f"SUCCESS {e['final_dist_m']:.2f} m" if e["success"] else f"miss {e['final_dist_m']:.2f} m ({ended})"
    text(img, label, (10, 26), 0.7, (255, 255, 255), 2)
    (w, _), _ = cv2.getTextSize(label, FONT, 0.7, 2)
    text(img, outcome, (24 + w, 26), 0.65, GREEN if e["success"] else RED, 2)
    calls = sum(1 for x in e["log"] if x["event"] == "plan")
    if "wall_s" in e:
        timing = f"robot {e['time_s']:.0f} s   planner wait {e['planner_s']:.0f} s   {calls} planner calls"
    else:
        timing = f"{e['time_s']:.0f} s wall clock (older run, includes planner waits)   {calls} planner calls"
    cost = (e.get("usage") or {}).get("cost_usd")
    if cost is not None:
        timing += f"   ${cost:.2f}"
    text(img, timing, (10, 56), 0.55, GREY)
    return img


def compare(a: Tuple[dict, str], b: Tuple[dict, str], labels, out_path: str, args):
    sides = []
    for (e, mp4), label in zip((a, b), labels):
        frames = robot_time(mp4, args.idle_s, args.keep_idle)
        first, _ = next(frames)
        sides.append({"frames": frames, "last": first, "tag": "", "head": header(first.shape[1], label, e),
                      "done": False, "pending": first})
    h, w = sides[0]["last"].shape[:2]
    writer = open_video_writer(out_path, 10.0, (2 * w + 8, h + HEADER_H))
    step = max(1, int(args.speed))
    n = 0
    while not all(s["done"] for s in sides):
        panels = []
        for s in sides:
            if s["pending"] is not None:
                s["last"], s["pending"] = s["pending"], None
            elif not s["done"]:
                for _ in range(step):
                    try:
                        s["last"], s["tag"] = next(s["frames"])
                    except StopIteration:
                        s["done"], s["tag"] = True, ""
                        break
            panel = s["last"].copy()
            if s["tag"]:
                text(panel, s["tag"], (CAM_W - 200, 22), 0.6, (80, 220, 255), 2)
            panels.append(np.vstack([s["head"], panel]))
        gap = np.full((h + HEADER_H, 8, 3), (90, 90, 90), np.uint8)
        writer.write(np.hstack([panels[0], gap, panels[1]]))
        n += 1
    for _ in range(20):  # hold the last frame
        writer.write(np.hstack([panels[0], gap, panels[1]]))
    writer.release()
    return n


def main():
    parser = argparse.ArgumentParser(description="Side-by-side videos of two landmark-nav runs")
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--labels", nargs=2, help="Default: the run folder names")
    parser.add_argument("--modes", nargs="+", default=["landmark"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--speed", type=float, default=2.0, help="Playback speed-up (whole number >= 1)")
    parser.add_argument("--idle_s", type=float, default=3.0, help="Cut static stretches longer than this")
    parser.add_argument("--keep_idle", action="store_true", help="Play wall time (no idle cuts)")
    args = parser.parse_args()
    labels = args.labels or [os.path.basename(os.path.normpath(r)) for r in (args.run_a, args.run_b)]
    A, B = load_run(args.run_a), load_run(args.run_b)
    keys = sorted((k for k in A.keys() & B.keys() if k[2] in args.modes),
                  key=lambda k: (os.path.basename(A[k][1])[:3], args.modes.index(k[2])))
    if not keys:
        raise SystemExit("no episodes match on (start, goal, mode)")
    os.makedirs(args.out, exist_ok=True)
    rows = [f"| episode | goal | mode | {labels[0]} | {labels[1]} |", "|---|---|---|---|---|"]
    for k in keys:
        (ea, pa), (eb, pb) = A[k], B[k]
        name = os.path.basename(pa)[:3] + f"_{k[1].replace(' ', '_')}_{k[2]}"
        out_path = os.path.join(args.out, name + ".mp4")
        n = compare(A[k], B[k], labels, out_path, args)
        cell = lambda e: ("**success**" if e["success"] else "miss") + f" {e['final_dist_m']:.2f} m"
        rows.append(f"| {name[:3]} | {k[1]} | {k[2]} | {cell(ea)} | {cell(eb)} |")
        print(f"{out_path}: {n / 10:.0f} s", flush=True)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
