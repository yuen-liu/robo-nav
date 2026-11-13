"""
Pick long-horizon GOAT subtasks for single-subtask runs (goat_nav --goals): every text subtask that
started >= --min_m (geodesic) from the robot in a reference run, with the pose it started from there --
the previous scored subtask's end pose, or the episode start (position None) for the first one. Every
mode then starts each goal from the same pose, so results pair goal for goal.

  python -m robo_nav.habitat.far_goals --run sim_data/goat/runs/val_unseen --mode zeroshot \\
      --min_m 10 --n 120 --per_scene 6 --out sim_data/goat/far_goals_10m.json
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict
from glob import glob


def candidates(run_dir: str, mode: str, min_m: float):
    out = []
    for p in sorted(glob(os.path.join(run_dir, f"ep*_{mode}.json"))):
        ep = json.load(open(p))
        if ep["mode"] != mode or ep["ended"] == "planner_error":
            continue
        start = {"position": None, "heading": None}
        for s in ep["subtasks"]:
            if s["ended"] == "skipped":
                continue  # the robot stays put
            geo = s["geodesic_m"]
            if math.isfinite(geo) and geo >= min_m:
                out.append({"episode": ep["episode"], "subtask": s["subtask"], "scene": ep["scene"],
                            "type": s["type"], "text": s["text"], "geodesic_m": round(geo, 3), **start})
            start = {"position": s["pose"]["position"], "heading": s["pose"]["heading"]}
    return out


def pick(cands, n: int, per_scene: int, seed: int = 0):
    rng = random.Random(seed)
    by_scene = defaultdict(list)
    for c in cands:
        by_scene[c["scene"]].append(c)
    pool = []
    for scene in sorted(by_scene):
        g = by_scene[scene][:]
        rng.shuffle(g)
        pool += g[:per_scene]
    rng.shuffle(pool)
    return sorted(pool[:n], key=lambda c: (c["episode"], c["subtask"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", default="zeroshot")
    ap.add_argument("--min_m", type=float, default=10.0)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--per_scene", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = candidates(a.run, a.mode, a.min_m)
    goals = pick(c, a.n, a.per_scene, a.seed)
    with open(a.out, "w") as f:
        json.dump(goals, f, indent=1)
    scenes = {g["scene"] for g in goals}
    print(f"{len(c)} candidates >= {a.min_m} m; picked {len(goals)} from {len(scenes)} scenes -> {a.out}")


if __name__ == "__main__":
    main()
