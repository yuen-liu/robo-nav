"""
Floor labels for a scene's landmark map, from the tour the map was annotated from.

The tour (goat_env.render_tour) walks the navmesh; its poses.json gives the agent base height of every
frame. Floors are the heights the walk dwells on (navmesh floors are flat, stairs are not); each
landmark gets the floor(s) of the keyframes it was seen at (keyframe k = frame k * every).

  python -m robo_nav.habitat.floors --maps sim_data/goat/maps --tours ~/robonav/data/goat_tours
writes <maps>/<scene>/floors.json: {"levels": [y, ...] lowest first, "landmarks": {name: [floor, ...]}},
floors numbered from 1 (lowest).
"""

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Optional

LEVEL_MERGE_M = 1.0   # dwell heights closer than this are one floor
ON_LEVEL_M = 0.5      # a height within this of a level is on that floor (else: stairs)
MIN_SHARE = 0.05      # a floor needs at least this share of the tour's frames
MIN_ON_LEVEL = 0.6    # if fewer of the tour's frames are on a floor (sloped site, terraces), report no floors


def levels_from_heights(ys: List[float]) -> List[float]:
    """Floor heights, lowest first: 0.1 m height bins holding >= MIN_SHARE of the frames, merged within LEVEL_MERGE_M."""
    bins = Counter(round(y, 1) for y in ys)
    dwell = sorted(b for b, n in bins.items() if n >= MIN_SHARE * len(ys))
    levels: List[List[float]] = []
    for b in dwell:
        if levels and b - levels[-1][-1] < LEVEL_MERGE_M:
            levels[-1].append(b)
        else:
            levels.append([b])
    out = []
    for group in levels:  # frame-weighted mean of the group's bins
        n = sum(bins[b] for b in group)
        out.append(round(sum(b * bins[b] for b in group) / n, 2))
    on = sum(1 for y in ys if out and min(abs(lv - y) for lv in out) <= ON_LEVEL_M)
    return out if ys and on >= MIN_ON_LEVEL * len(ys) else []


def floor_of(y: float, levels: List[float]) -> Optional[int]:
    """1-based floor whose level is within ON_LEVEL_M of y, else None (stairs / between floors)."""
    if not levels:
        return None
    i = min(range(len(levels)), key=lambda j: abs(levels[j] - y))
    return i + 1 if abs(levels[i] - y) <= ON_LEVEL_M else None


def describe_floor(y: float, levels: List[float]) -> str:
    """What the planner is told about the robot's height (an altimeter reading, not a position)."""
    f = floor_of(y, levels)
    if f is not None:
        return f"The robot is on floor {f} of {len(levels)} (numbered from the lowest)."
    below = [i + 1 for i, lv in enumerate(levels) if lv < y]
    lo = below[-1] if below else 1
    return f"The robot is between floor {lo} and floor {min(lo + 1, len(levels))} (on stairs or a ramp)."


def landmark_floors(m: dict, poses: List[List[float]], every: int) -> Dict[str, List[int]]:
    ys = [p[1] for p in poses]
    levels = levels_from_heights(ys)
    out = {}
    for lm in m["landmarks"]:
        fs = Counter(f for k in lm.get("seen_at", []) if k * every < len(ys)
                     for f in [floor_of(ys[k * every], levels)] if f is not None)
        # floors it was seen from at least twice (all of them if none was), lowest first
        keep = [f for f, n in fs.items() if n >= 2] or list(fs)
        out[lm["name"]] = sorted(keep)
    return out


def annotate_map_md(map_md: str, floors: dict) -> str:
    """Add '(floor k)' to each landmark line of landmark_map.md and a note on the numbering."""
    if len(floors["levels"]) < 2:
        return map_md  # single-floor house: nothing to add
    lines = []
    for line in map_md.splitlines():
        if line.startswith("- **") and "**:" in line:
            name = line[4:line.index("**:")]
            fs = floors["landmarks"].get(name)
            if fs:
                line = line.replace("**:", f"** (floor {' and '.join(map(str, fs))}):", 1)
        lines.append(line)
    note = (f"This building has {len(floors['levels'])} floors, numbered from the lowest (1). Each landmark "
            "is tagged with the floor(s) it was seen on during the walk.\n\n")
    return note + "\n".join(lines) + ("\n" if map_md.endswith("\n") else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--maps", required=True)
    parser.add_argument("--tours", required=True, help="<tours>/<scene>/poses.json")
    parser.add_argument("--every", type=int, default=10, help="annotate --every used for the maps")
    args = parser.parse_args()
    for scene in sorted(os.listdir(args.maps)):
        mp, pp = os.path.join(args.maps, scene, "landmark_map.json"), os.path.join(args.tours, scene, "poses.json")
        if not (os.path.exists(mp) and os.path.exists(pp)):
            print(f"{scene}: missing map or poses, skipped")
            continue
        m, poses = json.load(open(mp)), json.load(open(pp))
        levels = levels_from_heights([p[1] for p in poses])
        lf = landmark_floors(m, poses, args.every)
        with open(os.path.join(args.maps, scene, "floors.json"), "w") as f:
            json.dump({"levels": levels, "landmarks": lf}, f, indent=1)
        print(f"{scene}: {len(levels)} floor(s) {levels}, {sum(1 for v in lf.values() if not v)} of {len(lf)} landmarks untagged")


if __name__ == "__main__":
    main()
