"""
Record a mapping run from the LightNav-0 MuJoCo demo (`mujoco_demo/run.sh`) for robo-nav.

Drive the robot around in the web UI (Take control + WASD) while this polls the sim's HTTP API
and saves a keyframe whenever the robot has moved/turned enough. Because the sim knows the
ground truth, every frame is saved with its exact pose and the ProcTHOR room it was taken in.

Output layout (<out>/):
  frames/000000.jpg ...  sequential first-person RGB (480x270), feed to demo.py / localize.py
  poses.csv              idx, x, y, yaw (sim world, meters/rad), room_id, room_name
  intrinsics.json        pinhole K for the robot camera + camera mount offset
  rooms.json             GT room polygons (sim x/y) and door adjacency -> room graph

Usage (sim running on :8088):
  python -m robo_nav.sim_record sim_data/val2_map \
      --scene ~/Desktop/projects/LightNav-0/mujoco_demo/vln_mujoco/assets/scenes/procthor-10k-val/val_2.json
Ctrl+C to stop.
"""

import argparse
import csv
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import requests

# TurtleBot camera in LightNav-0's mujoco_demo (robots/turtlebot.py, model.py)
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOVY_DEG = 480, 270, 79.865
CAMERA_OFFSET_XYZ = (0.090, 0.0, 0.165 + 0.033)  # forward, left, up from the ground under base_link


def camera_intrinsics() -> dict:
    fy = (CAMERA_HEIGHT / 2) / math.tan(math.radians(CAMERA_FOVY_DEG) / 2)
    return {"width": CAMERA_WIDTH, "height": CAMERA_HEIGHT, "fx": fy, "fy": fy,
            "cx": CAMERA_WIDTH / 2, "cy": CAMERA_HEIGHT / 2,
            "camera_offset_xyz": CAMERA_OFFSET_XYZ}


def room_label(room: dict, index: Dict[str, int]) -> str:
    """'Kitchen' + room|2 -> 'kitchen'; repeated types get a suffix: 'livingroom_2'."""
    base = room["roomType"].lower()
    return base if index[base] == 1 else f"{base}_{room['_n']}"


def door_center(door: dict, walls: Dict[str, dict]) -> Tuple[float, float]:
    """Doorway midpoint in sim (x, y). The hole offset runs from the first vertex of wall0's
    *polygon*; the endpoint order in the wall id is sometimes the reverse."""
    hole = door["holePolygon"]
    offset = (hole[0]["x"] + hole[1]["x"]) / 2
    (x0, z0), (x1, z1) = [(p["x"], p["z"]) for p in walls[door["wall0"]]["polygon"][:2]]
    length = math.hypot(x1 - x0, z1 - z0)
    return x0 + offset / length * (x1 - x0), z0 + offset / length * (z1 - z0)


def load_rooms(scene_json: str) -> Tuple[List[dict], List[dict]]:
    """GT rooms + interior doors from a ProcTHOR house JSON.
    THOR floor (x, z) maps to MuJoCo world (x, y)."""
    with open(scene_json) as f:
        house = json.load(f)
    counts: Dict[str, int] = {}
    for r in house["rooms"]:
        t = r["roomType"].lower()
        counts[t] = counts.get(t, 0) + 1
        r["_n"] = counts[t]
    rooms = [{"id": r["id"], "name": room_label(r, counts),
              "polygon": [(p["x"], p["z"]) for p in r["floorPolygon"]]} for r in house["rooms"]]
    walls = {w["id"]: w for w in house["walls"]}
    doors = [{"id": d["id"], "rooms": (d["room0"], d["room1"]), "center": door_center(d, walls),
              "width": abs(d["holePolygon"][1]["x"] - d["holePolygon"][0]["x"])}
             for d in house.get("doors", []) if d["room0"] != d["room1"]]
    return rooms, doors


def point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    inside = False
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def locate(x: float, y: float, rooms: List[dict]) -> Optional[dict]:
    return next((r for r in rooms if point_in_polygon(x, y, r["polygon"])), None)


def angle_diff(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def main():
    parser = argparse.ArgumentParser(description="Record keyframes + GT poses from the LightNav MuJoCo sim")
    parser.add_argument("out", type=str, help="Output directory")
    parser.add_argument("--scene", type=str, required=True, help="ProcTHOR house JSON (val_2.json)")
    parser.add_argument("--sim", type=str, default="http://127.0.0.1:8088")
    parser.add_argument("--min_move", type=float, default=0.15, help="Save after moving this many meters")
    parser.add_argument("--min_turn_deg", type=float, default=10.0, help="...or turning this many degrees")
    parser.add_argument("--hz", type=float, default=10.0, help="Polling rate")
    args = parser.parse_args()

    rooms, doors = load_rooms(args.scene)
    frame_dir = os.path.join(args.out, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    with open(os.path.join(args.out, "intrinsics.json"), "w") as f:
        json.dump(camera_intrinsics(), f, indent=1)
    with open(os.path.join(args.out, "rooms.json"), "w") as f:
        by_id = {r["id"]: r["name"] for r in rooms}
        json.dump({"rooms": rooms, "doors": [{"id": d["id"], "rooms": [by_id[r] for r in d["rooms"]],
                                              "center": d["center"]} for d in doors]}, f, indent=1)

    session = requests.Session()
    pose_path = os.path.join(args.out, "poses.csv")
    resume = os.path.exists(pose_path)
    idx = sum(1 for _ in open(pose_path)) - 1 if resume else 0
    last = None
    with open(pose_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not resume:
            writer.writerow(["idx", "x", "y", "yaw", "room_id", "room_name"])
        print(f"Recording to {args.out} (Ctrl+C to stop). Drive with WASD in the sim page.")
        try:
            while True:
                time.sleep(1.0 / args.hz)
                p0 = session.get(f"{args.sim}/api/health", timeout=2).json()["simulation"]["pose"]
                if last is not None and math.hypot(p0["x"] - last[0], p0["y"] - last[1]) < args.min_move \
                        and angle_diff(p0["yaw"], last[2]) < math.radians(args.min_turn_deg):
                    continue
                jpeg = session.get(f"{args.sim}/api/camera.jpg", timeout=2).content
                p1 = session.get(f"{args.sim}/api/health", timeout=2).json()["simulation"]["pose"]
                # Frame was rendered between the two pose reads; skip if the robot moved a lot in between
                if math.hypot(p1["x"] - p0["x"], p1["y"] - p0["y"]) > 0.05 or angle_diff(p1["yaw"], p0["yaw"]) > 0.05:
                    continue
                x, y = (p0["x"] + p1["x"]) / 2, (p0["y"] + p1["y"]) / 2
                yaw = math.atan2(math.sin(p0["yaw"]) + math.sin(p1["yaw"]), math.cos(p0["yaw"]) + math.cos(p1["yaw"]))
                room = locate(x, y, rooms)
                with open(os.path.join(frame_dir, f"{idx:06d}.jpg"), "wb") as img:
                    img.write(jpeg)
                writer.writerow([idx, f"{x:.4f}", f"{y:.4f}", f"{yaw:.5f}",
                                 room["id"] if room else "", room["name"] if room else ""])
                f.flush()
                print(f"[{idx:05d}] ({x:6.2f}, {y:6.2f}) yaw={math.degrees(yaw):6.1f}  {room['name'] if room else '?'}")
                last = (x, y, yaw)
                idx += 1
        except KeyboardInterrupt:
            pass
    print(f"Saved {idx} frames to {frame_dir}")


if __name__ == "__main__":
    main()
