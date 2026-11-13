"""
GOAT / VLN-CE run report (no sim needed):

  episode   episode-level success for lifelong GOAT runs: all of an episode's scored subtasks succeed,
            plus the mean per-episode share of subtasks done
  budget    success / SPL rescored at step budgets of F x shortest-path time (geodesic / 0.25 m/s), from
            the env's per-step track (runs made with --budget_factor >= F)
  slices    success / SPL by difficulty (goat_difficulty.py rows): start distance, goal visible at start,
            doorways on the shortest path, floor change

  python -m robo_nav.habitat.goal_report --runs sim_data/goat/runs/val_unseen --difficulty sim_data/goat/difficulty_all.json
  python -m robo_nav.habitat.goal_report --runs sim_data/goat/runs/far10_b2 --factors 1.5 2.0
"""

import argparse
import json
import math
import os
from collections import defaultdict
from glob import glob

MAX_LIN = 0.25  # m/s


def load(run_dir):
    eps = [json.load(open(p)) for p in sorted(glob(os.path.join(run_dir, "ep*.json")) + glob(os.path.join(run_dir, "g*.json")))]
    return [e for e in eps if e["ended"] != "planner_error"]


def scored(e):
    return [s for s in e["subtasks"] if s["ended"] != "skipped" and "success" in s]


def at_budget(s, factor):
    """(success, spl) of subtask s had its budget been ceil(factor x geodesic / MAX_LIN) steps."""
    geo = s["geodesic_m"]
    b = max(1, math.ceil(factor * geo / MAX_LIN))
    track = s.get("track")
    if track is None:
        return None
    if s.get("budget_steps") is not None and b > s["budget_steps"] and s["steps"] >= s["budget_steps"]:
        return None  # the run was cut before this budget: unknown
    if s["steps"] <= b:
        dtg, path = s["distance_to_goal"], s["path_m"]
    else:
        dtg, path = track[b - 1]
    ok = dtg < s.get("success_distance", 0.25)
    spl = float(ok) * geo / max(path, geo) if geo > 0 else float(ok)
    return ok, spl


def pct(xs):
    return f"{100 * sum(xs) / len(xs):5.1f}%" if xs else "   -  "


def mean(xs):
    return f"{sum(xs) / len(xs):.3f}" if xs else "  -  "


def bin_of(key, v):
    if v is None:
        return None
    if key == "geodesic_m":
        for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 1e9)]:
            if lo <= v < hi:
                return f"{lo}-{hi if hi < 1e9 else ''}m"
    if key == "doorways":
        return str(min(v, 3)) + ("+" if v >= 3 else "")
    return str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--difficulty")
    ap.add_argument("--factors", type=float, nargs="*", default=[])
    a = ap.parse_args()
    diff = {}
    if a.difficulty:
        for r in json.load(open(a.difficulty)):
            diff[(r["run"], r["mode"], r["episode"], r["subtask"])] = r
            if r["mode"] == "fixed":
                diff[("fixed", r["episode"], r["subtask"])] = r
    for d in a.runs:
        run = os.path.basename(os.path.normpath(d))
        eps = load(d)
        modes = sorted({e["mode"] for e in eps})
        print(f"\n===== {run} ({len(eps)} result files)")
        print(f"{'mode':<14}{'subtasks':>9}{'success':>9}{'SPL':>7}   {'episodes':>8}{'ep. success':>12}{'mean share':>11}")
        for m in modes:
            E = [e for e in eps if e["mode"] == m]
            S = [s for e in E for s in scored(e)]
            multi = [e for e in E if len(scored(e)) > 1]
            ep_ok = [all(s["success"] for s in scored(e)) for e in multi]
            share = [sum(s["success"] for s in scored(e)) / len(scored(e)) for e in multi]
            print(f"{m:<14}{len(S):>9}{pct([s['success'] for s in S]):>9}{mean([s['spl'] for s in S]):>7}   "
                  f"{len(multi):>8}{pct(ep_ok):>12}{mean(share):>11}")
        for f in a.factors:
            print(f"-- budget {f:g}x shortest-path time")
            for m in modes:
                R = [at_budget(s, f) for e in eps if e["mode"] == m for s in scored(e)]
                R = [r for r in R if r is not None]
                print(f"   {m:<14}{len(R):>6}  success {pct([r[0] for r in R])}  SPL {mean([r[1] for r in R])}")
        if diff:
            for key in ["geodesic_m", "visible_start", "doorways", "floor_change"]:
                print(f"-- by {key}")
                rows = defaultdict(lambda: defaultdict(list))
                for e in eps:
                    for s in scored(e):
                        r = diff.get((run, e["mode"], e["episode"], s["subtask"])) or diff.get(("fixed", e["episode"], s["subtask"]))
                        b = bin_of(key, r[key]) if r else None
                        if b is not None:
                            rows[b][e["mode"]].append(s)
                for b in sorted(rows):
                    cells = "  ".join(f"{m}: {pct([s['success'] for s in rows[b][m]])} SPL {mean([s['spl'] for s in rows[b][m]])} (n={len(rows[b][m])})"
                                      for m in modes)
                    print(f"   {b:<8} {cells}")


if __name__ == "__main__":
    main()
