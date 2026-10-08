"""
Build a localization dataset from the LightNav-0 MuJoCo house by rendering the robot camera
offline (no live sim, no real-time driving), with exact ground-truth poses.

  map/      a continuous mapping tour through every room: transit along collision-free paths
            plus a small driven circle at a few spots per room so every heading is seen with
            parallax (LingBot-Map needs a video-like sequence, retrieval needs view coverage)
  queries/  "kidnapped robot" starts: random collision-free poses per room, each rendered as an
            8-view look-around (view 0 = the single-image query, views k = yaw + k*45 deg)

Output (<out>/):
  map/frames/000000.jpg ..., map/poses.csv          idx, x, y, yaw, room_id, room_name
  queries/q000_v0.jpg ... q000_v7.jpg, queries.csv  idx, x, y, yaw, room_id, room_name
  intrinsics.json, rooms.json, overview.png         (overview: floor plan + tour + query starts)

Run with an env that has both robo-nav and LightNav-0's `vln_mujoco` (patched TurtleBot):
  python -m robo_nav.sim_render sim_data/loc/val2 --scene $SCENE
On headless Linux prefix with MUJOCO_GL=egl.
"""

import argparse
import csv
import heapq
import json
import math
import os
from typing import Dict, List, Tuple

import cv2
import mujoco
import numpy as np
from PIL import Image
from scipy import ndimage

from robo_nav.sim_record import camera_intrinsics, load_rooms, locate, point_in_polygon
from robo_nav.sim_video import FloorPlan

GRID_RES = 0.1  # m per occupancy cell


class OfflineCamera:
    """The demo TurtleBot's first-person camera, posed directly instead of driven."""

    def __init__(self):
        from vln_mujoco.model import CAMERA_HEIGHT, CAMERA_WIDTH
        from vln_mujoco.robots.turtlebot import TurtleBotBackend
        self.robot = TurtleBotBackend()
        self.model, self.data = self.robot.model, self.robot.data
        self.qpos = self.model.jnt_qposadr[self.model.joint("base_joint").id]
        self.renderer = mujoco.Renderer(self.model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH)

    def set_pose(self, x: float, y: float, yaw: float = 0.0) -> None:
        q = self.qpos
        self.data.qpos[q:q + 2] = (x, y)
        self.data.qpos[q + 3:q + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        mujoco.mj_forward(self.model, self.data)

    def penetration(self, x: float, y: float) -> float:
        self.set_pose(x, y)
        return self.robot._penetration()

    def render(self, x: float, y: float, yaw: float) -> np.ndarray:
        self.set_pose(x, y, yaw)
        self.renderer.update_scene(self.data, camera=self.robot.camera_name)
        return self.renderer.render().copy()


class Grid:
    """Free-space grid: inside some room polygon and the chassis touches nothing."""

    def __init__(self, cam: OfflineCamera, rooms: List[dict]):
        pts = np.array([p for r in rooms for p in r["polygon"]], dtype=float)
        self.lo = pts.min(0)
        self.shape = tuple(np.ceil((pts.max(0) - self.lo) / GRID_RES).astype(int) + 1)
        self.room = np.full(self.shape, -1, int)
        free = np.zeros(self.shape, bool)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                x, y = self.world(i, j)
                for k, r in enumerate(rooms):
                    if point_in_polygon(x, y, r["polygon"]):
                        self.room[i, j] = k
                        free[i, j] = cam.penetration(x, y) < 2e-3
                        break
        # Doorway cells sit exactly on polygon edges; let them through if the chassis fits
        for i, j in zip(*np.nonzero(self.room < 0)):
            x, y = self.world(i, j)
            if any(self.room[max(0, i - 1):i + 2, max(0, j - 1):j + 2].ravel() >= 0):
                free[i, j] = cam.penetration(x, y) < 2e-3
        self.free = free
        self.clearance = ndimage.distance_transform_edt(free) * GRID_RES  # m to nearest blocked cell

    def world(self, i: int, j: int) -> Tuple[float, float]:
        return self.lo[0] + i * GRID_RES, self.lo[1] + j * GRID_RES

    def cell(self, x: float, y: float) -> Tuple[int, int]:
        return int(round((x - self.lo[0]) / GRID_RES)), int(round((y - self.lo[1]) / GRID_RES))

    def nearest_free(self, x: float, y: float, min_clear: float = 0.0) -> Tuple[int, int]:
        ok = self.free & (self.clearance >= min_clear)
        ii, jj = np.nonzero(ok)
        ci, cj = self.cell(x, y)
        k = np.argmin((ii - ci) ** 2 + (jj - cj) ** 2)
        return int(ii[k]), int(jj[k])

    def astar(self, a: Tuple[int, int], b: Tuple[int, int], keep_away: float = 0.5) -> List[Tuple[int, int]]:
        """8-connected A*; cost grows near obstacles so paths keep to the middle of rooms/doors."""
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        penalty = 1.0 + keep_away / np.maximum(self.clearance, GRID_RES)
        dist = {a: 0.0}
        parent = {a: None}
        heap = [(0.0, a)]
        while heap:
            _, cur = heapq.heappop(heap)
            if cur == b:
                break
            for di, dj in moves:
                nb = (cur[0] + di, cur[1] + dj)
                if not (0 <= nb[0] < self.shape[0] and 0 <= nb[1] < self.shape[1]) or not self.free[nb]:
                    continue
                nd = dist[cur] + math.hypot(di, dj) * penalty[nb]
                if nd < dist.get(nb, math.inf):
                    dist[nb], parent[nb] = nd, cur
                    heapq.heappush(heap, (nd + math.hypot(nb[0] - b[0], nb[1] - b[1]), nb))
        if b not in parent:
            raise RuntimeError(f"no free path {a} -> {b}")
        path, cur = [], b
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        return path[::-1]


def smooth(points: np.ndarray, window: int = 7) -> np.ndarray:
    if len(points) < window:
        return points
    kernel = np.ones(window) / window
    padded = np.pad(points, ((window // 2, window // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(2)], 1)


def room_spots(grid: Grid, room_index: int, room_area: float, min_clear: float = 0.55) -> List[Tuple[float, float]]:
    """1-3 well-separated, open spots in a room (farthest-point sampling over roomy cells)."""
    ii, jj = np.nonzero((grid.room == room_index) & grid.free & (grid.clearance >= min_clear))
    if len(ii) == 0:
        ii, jj = np.nonzero((grid.room == room_index) & grid.free)
        order = np.argsort(-grid.clearance[ii, jj])[:1]
        return [grid.world(ii[order[0]], jj[order[0]])]
    pts = np.stack([ii, jj], 1)
    n = 1 if room_area < 8 else (2 if room_area < 20 else 3)
    chosen = [pts[np.argmax(grid.clearance[ii, jj])]]
    while len(chosen) < n:
        d = np.min([np.hypot(*(pts - c).T) for c in chosen], axis=0)
        chosen.append(pts[np.argmax(d)])
    return [grid.world(*c) for c in chosen]


def polygon_area(poly) -> float:
    x, y = np.array(poly).T
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def mapping_tour(grid: Grid, rooms: List[dict], doors: List[dict], start_room: str,
                 circle_r: float = 0.35) -> np.ndarray:
    """DFS over the door graph; at each room's open spots drive one circle. Returns (N, 2) xy."""
    idx = {r["id"]: k for k, r in enumerate(rooms)}
    adj: Dict[str, List[dict]] = {r["id"]: [] for r in rooms}
    for d in doors:
        adj[d["rooms"][0]].append(d)
        adj[d["rooms"][1]].append(d)

    waypoints: List[Tuple[str, Tuple[float, float]]] = []  # ("go", xy) or ("circle", xy)
    seen = set()

    def visit(rid: str):
        seen.add(rid)
        for spot in room_spots(grid, idx[rid], polygon_area(rooms[idx[rid]]["polygon"])):
            waypoints.append(("circle", spot))
        for d in adj[rid]:
            other = d["rooms"][1] if d["rooms"][0] == rid else d["rooms"][0]
            if other not in seen:
                waypoints.append(("go", tuple(d["center"])))
                visit(other)
                waypoints.append(("go", tuple(d["center"])))

    visit(next(r["id"] for r in rooms if r["name"] == start_room))
    while waypoints and waypoints[-1][0] == "go":  # no need to walk back out at the end
        waypoints.pop()

    xy: List[Tuple[float, float]] = []
    cur = grid.nearest_free(*waypoints[0][1])
    for kind, target in waypoints:
        goal = grid.nearest_free(*target, min_clear=0.15 if kind == "go" else circle_r + 0.15)
        seg = [grid.world(*c) for c in grid.astar(cur, goal)]
        xy.extend(seg)
        cur = goal
        if kind == "circle":
            cx, cy = grid.world(*goal)
            # drive once around (CCW) a circle centered on the spot; its clearance >= circle_r + 0.15
            for t in np.linspace(0, 2 * math.pi, 60):
                xy.append((cx + circle_r * math.cos(t), cy + circle_r * math.sin(t)))
            xy.append((cx, cy))
    return smooth(np.array(xy))


def keyframes(xy: np.ndarray, min_move: float, min_turn_deg: float) -> List[Tuple[float, float, float]]:
    """Resample a path into (x, y, yaw) keyframes, yaw = direction of travel; turns in place
    (e.g. leaving a dead-end room) are emitted as rotation-only frames every `min_turn_deg`."""
    dense = [xy[0]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / 0.02)))
        dense.extend(a + (b - a) * k / n for k in range(1, n + 1))
    dense = np.array(dense)
    heading = np.unwrap(np.arctan2(*np.gradient(smooth(dense, 15), axis=0).T[::-1]))
    frames = [(dense[0][0], dense[0][1], heading[0])]
    for p, h in zip(dense[1:], heading[1:]):
        lx, ly, lh = frames[-1]
        turn = h - lh
        if abs(turn) > math.radians(45):  # sharp reversal: rotate in place first
            steps = int(abs(turn) // math.radians(min_turn_deg))
            frames.extend((lx, ly, lh + turn * k / (steps + 1)) for k in range(1, steps + 1))
            lh = frames[-1][2]
        if math.hypot(p[0] - lx, p[1] - ly) >= min_move or abs(h - lh) >= math.radians(min_turn_deg):
            frames.append((p[0], p[1], h))
    return [(x, y, math.atan2(math.sin(h), math.cos(h))) for x, y, h in frames]


def write_poses(path: str, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "x", "y", "yaw", "room_id", "room_name"])
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Render a GT localization dataset from the MuJoCo house")
    parser.add_argument("out", type=str)
    parser.add_argument("--scene", type=str, required=True, help="ProcTHOR house JSON (val_2.json)")
    parser.add_argument("--start_room", type=str, default="livingroom_2")
    parser.add_argument("--min_move", type=float, default=0.2, help="Map keyframe spacing (m)")
    parser.add_argument("--min_turn_deg", type=float, default=15.0, help="...or heading change (deg)")
    parser.add_argument("--queries_per_room", type=int, default=10)
    parser.add_argument("--query_clearance", type=float, default=0.25, help="Min free space around query starts (m)")
    parser.add_argument("--views", type=int, default=8, help="Look-around views per query")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rooms, doors = load_rooms(args.scene)
    cam = OfflineCamera()
    print("Building free-space grid ...")
    grid = Grid(cam, rooms)
    print(f"  {grid.free.sum()} free cells of {grid.free.size}")

    # --- map tour ---
    tour = mapping_tour(grid, rooms, doors, args.start_room)
    frames = keyframes(tour, args.min_move, args.min_turn_deg)
    map_dir = os.path.join(args.out, "map", "frames")
    os.makedirs(map_dir, exist_ok=True)
    rows = []
    for k, (x, y, yaw) in enumerate(frames):
        Image.fromarray(cam.render(x, y, yaw)).save(os.path.join(map_dir, f"{k:06d}.jpg"), quality=92)
        r = locate(x, y, rooms)
        rows.append([k, f"{x:.4f}", f"{y:.4f}", f"{yaw:.5f}", r["id"] if r else "", r["name"] if r else ""])
    write_poses(os.path.join(args.out, "map", "poses.csv"), rows)
    print(f"Map: {len(frames)} keyframes, {np.sum(np.hypot(*np.diff(tour, axis=0).T)):.0f} m tour")

    # --- kidnapped-robot query starts ---
    rng = np.random.default_rng(args.seed)
    q_dir = os.path.join(args.out, "queries")
    os.makedirs(q_dir, exist_ok=True)
    q_rows = []
    for k, room in enumerate(rooms):
        ii, jj = np.nonzero((grid.room == k) & grid.free & (grid.clearance >= args.query_clearance))
        pick = rng.choice(len(ii), size=min(args.queries_per_room, len(ii)), replace=False)
        for c in pick:
            x, y = grid.world(ii[c], jj[c])
            x, y = x + rng.uniform(-0.04, 0.04), y + rng.uniform(-0.04, 0.04)
            yaw = rng.uniform(-math.pi, math.pi)
            q = len(q_rows)
            for v in range(args.views):
                img = cam.render(x, y, yaw + 2 * math.pi * v / args.views)
                Image.fromarray(img).save(os.path.join(q_dir, f"q{q:03d}_v{v}.jpg"), quality=92)
            q_rows.append([q, f"{x:.4f}", f"{y:.4f}", f"{yaw:.5f}", room["id"], room["name"]])
    write_poses(os.path.join(args.out, "queries.csv"), q_rows)
    print(f"Queries: {len(q_rows)} starts x {args.views} views")

    with open(os.path.join(args.out, "intrinsics.json"), "w") as f:
        json.dump(camera_intrinsics(), f, indent=1)
    with open(os.path.join(args.out, "rooms.json"), "w") as f:
        by_id = {r["id"]: r["name"] for r in rooms}
        json.dump({"rooms": rooms, "doors": [{**d, "rooms": [by_id[r] for r in d["rooms"]]} for d in doors]}, f, indent=1)

    plan = FloorPlan(rooms, doors, size=900)
    img = plan.draw(None, [], [tuple(p) for p in tour], None)
    for _, x, y, *_ in q_rows:
        cv2.drawMarker(img, plan.px(float(x), float(y)), (80, 80, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
    cv2.imwrite(os.path.join(args.out, "overview.png"), img)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
