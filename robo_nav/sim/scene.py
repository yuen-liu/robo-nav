"""
Ground truth for the LightNav-0 MuJoCo house (ProcTHOR `val_2`): room polygons, interior doors,
and the robot camera's intrinsics. THOR floor coordinates (x, z) are MuJoCo world (x, y).
"""

import json
import math
from typing import Dict, List, Optional, Tuple

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


def load_objects(scene_json: str) -> List[dict]:
    """Every placed object (children included) with its type and sim (x, y) position."""
    with open(scene_json) as f:
        house = json.load(f)

    def walk(objs):
        for o in objs:
            yield o
            yield from walk(o.get("children", []))

    return [{"id": o["id"], "type": o["id"].split("|")[0], "x": o["position"]["x"], "y": o["position"]["z"]}
            for o in walk(house["objects"])]
