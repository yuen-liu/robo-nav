"""
Room-to-room navigation in the LightNav-0 MuJoCo demo: robo-nav plans, LightNav-0 drives.

  localization : sim ground-truth pose -> ProcTHOR room polygon   (oracle; swap later)
  global plan  : graph.py TopologicalGraph BFS over the GT door graph
  local policy : LightNav-0, given one hop at a time ("go through the doorway ... into the X")

A hop is re-issued (with a fresh direction hint) if LightNav says stop, or the hop times out,
before the robot has crossed into the next room. Entering an off-plan room just triggers a
replan from wherever the robot is.

The demo's TurtleBot is kinematic and passes through walls, so every room change is checked
against the door graph: crossing anywhere but a connecting doorway ends the episode as
"wall_crossing" (or is only logged with --allow_wall_crossing), and driving out of every room
polygon ends it as "left_house".

Usage (sim on :8088 configured with the LightNav server, tunnel up):
  python -m robo_nav.sim_nav bathroom_1 \
      --scene ~/Desktop/projects/LightNav-0/mujoco_demo/vln_mujoco/assets/scenes/procthor-10k-val/val_2.json
Release manual control on the sim page first (the sim lets only one client drive).
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

import websockets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph import TopologicalGraph  # noqa: E402
from robo_nav.sim_record import load_rooms, locate  # noqa: E402
from robo_nav.sim_video import EpisodeRecorder, FloorPlan  # noqa: E402


def build_graph(rooms: List[dict], doors: List[dict]) -> TopologicalGraph:
    graph = TopologicalGraph()
    graph.nodes, graph.hallways, graph.adj = set(), set(), {}  # drop the office demo defaults
    names = {r["id"]: r["name"] for r in rooms}
    for r in rooms:
        graph.add_node(r["name"])
    for d in doors:
        graph.add_edge(names[d["rooms"][0]], names[d["rooms"][1]], d["id"])
    return graph


def crossing_door(doors: List[dict], room_a: str, room_b: str, a_xy: tuple, b_xy: tuple,
                  margin: float = 0.35) -> Optional[dict]:
    """The doorway between room ids a/b that the step a_xy -> b_xy passed through, if any."""
    mid = ((a_xy[0] + b_xy[0]) / 2, (a_xy[1] + b_xy[1]) / 2)
    for d in doors:
        if set(d["rooms"]) == {room_a, room_b} and \
                math.hypot(mid[0] - d["center"][0], mid[1] - d["center"][1]) <= d["width"] / 2 + margin:
            return d
    return None


def spoken(room_name: str) -> str:
    """'livingroom_2' -> 'living room'. The direction hint is what disambiguates duplicates."""
    base = room_name.split("_")[0]
    return {"livingroom": "living room"}.get(base, base)


def door_bearing(pose: dict, target: tuple) -> tuple:
    """(bearing_deg, distance_m) of a world point in the robot frame; + bearing is left (CCW)."""
    dx, dy = target[0] - pose["x"], target[1] - pose["y"]
    bearing = math.degrees(math.atan2(dy, dx) - pose["yaw"])
    return (bearing + 180) % 360 - 180, math.hypot(dx, dy)


def hop_instruction(pose: dict, door: dict, next_room: str, hint: str) -> str:
    room = spoken(next_room)
    if hint == "none":
        return f"Go to the {room}."
    bearing, dist = door_bearing(pose, door["center"])
    side = "left" if bearing > 0 else "right"
    # LightNav ignores "behind you" and drives forward, so make the turn an explicit first step
    if abs(bearing) >= 120:
        return f"Turn around, then go through the doorway into the {room}."
    where = {abs(bearing) < 25: "straight ahead",
             25 <= abs(bearing) < 70: f"ahead on your {side}",
             70 <= abs(bearing) < 120: f"on your {side}"}[True]
    far = f", about {dist:.0f} meters away," if dist >= 1.5 else ""
    return f"Go through the doorway {where}{far} into the {room}."


class SimSession:
    """Thin client for the mujoco_demo page protocol (/ws): keeps the latest snapshot."""

    def __init__(self, url: str):
        self.url = url
        self.snapshot: Optional[dict] = None
        self.last_result: Optional[dict] = None

    async def __aenter__(self):
        self.ws = await websockets.connect(self.url, max_size=16 * 1024 * 1024)
        self._reader = asyncio.create_task(self._read())
        while self.snapshot is None:
            await asyncio.sleep(0.05)
        return self

    async def __aexit__(self, *exc):
        self._reader.cancel()
        await self.ws.close()

    async def _read(self):
        async for raw in self.ws:
            msg = json.loads(raw)
            if msg.get("type") in ("snapshot", "runtime"):
                self.snapshot = msg["data"]
            elif msg.get("type") == "command_result":
                self.last_result = msg

    async def command(self, payload: dict) -> dict:
        self.last_result = None
        await self.ws.send(json.dumps(payload))
        for _ in range(100):
            if self.last_result is not None:
                return self.last_result
            await asyncio.sleep(0.02)
        raise TimeoutError(f"no reply to {payload['type']}")

    @property
    def pose(self) -> dict:
        return self.snapshot["simulation"]["pose"]

    @property
    def vln(self) -> dict:
        return self.snapshot["vln"]


async def navigate(args) -> dict:
    rooms, doors = load_rooms(args.scene)
    graph = build_graph(rooms, doors)
    by_name: Dict[str, dict] = {r["name"]: r for r in rooms}
    if args.goal not in by_name:
        raise SystemExit(f"Unknown goal {args.goal!r}; rooms: {sorted(by_name)}")
    door_by_id = {d["id"]: d for d in doors}

    log = []
    t0 = time.monotonic()
    status = {"goal": args.goal}  # read by the video recorder

    def route_points(pose: dict, path) -> list:
        centroid = tuple(sum(c) / len(c) for c in zip(*by_name[args.goal]["polygon"]))
        return [(pose["x"], pose["y"])] + [door_by_id[d]["center"] for _, d in path if d] + [centroid]

    def event(kind: str, **kw):
        entry = {"t": round(time.monotonic() - t0, 1), "event": kind, **kw}
        log.append(entry)
        print(f"[{entry['t']:6.1f}s] {kind}: " + ", ".join(f"{k}={v}" for k, v in kw.items()))

    async with SimSession(args.sim_ws) as sim:
        if args.reset:
            await sim.command({"type": "reset"})
            await asyncio.sleep(0.5)
        if args.vln_server:
            await sim.command({"type": "set_server_url", "server_url": args.vln_server})

        room = locate(sim.pose["x"], sim.pose["y"], rooms)
        current = room["name"] if room else None
        current_id = room["id"] if room else None
        last_in_room_xy = (sim.pose["x"], sim.pose["y"])
        if current is None:
            raise SystemExit(f"Robot is not inside any room polygon: {sim.pose}")
        plan = graph.plan_path(current, args.goal)
        event("start", room=current, goal=args.goal, plan=" -> ".join(n for n, _ in plan))
        status.update(room=current, to_go=len(plan) - 1, route=route_points(sim.pose, plan))

        recorder = None
        if args.video:
            os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
            sim_http = args.sim_ws.replace("ws://", "http://").rsplit("/ws", 1)[0]
            recorder = EpisodeRecorder(args.video, sim_http, FloorPlan(rooms, doors), status, lambda: sim.pose)
            recorder.start()

        last_xy = (sim.pose["x"], sim.pose["y"])
        distance = 0.0
        hop_target = None
        hop_started = 0.0
        attempts = 0
        issue = True
        outcome = "timeout"
        deadline = time.monotonic() + args.timeout

        while time.monotonic() < deadline:
            await asyncio.sleep(0.1)
            pose = sim.pose
            distance += math.hypot(pose["x"] - last_xy[0], pose["y"] - last_xy[1])
            last_xy = (pose["x"], pose["y"])

            room = locate(pose["x"], pose["y"], rooms)
            if room is not None and room["name"] != current:  # doorway cells match no polygon: keep last room
                xy = (pose["x"], pose["y"])
                if crossing_door(doors, current_id, room["id"], last_in_room_xy, xy) is None:
                    event("wall_crossing", frm=current, to=room["name"],
                          at=f"({(xy[0] + last_in_room_xy[0]) / 2:.2f}, {(xy[1] + last_in_room_xy[1]) / 2:.2f})")
                    if not args.allow_wall_crossing:
                        outcome = "wall_crossing"
                        break
                current, current_id = room["name"], room["id"]
                status["room"] = current
                if current == args.goal:
                    event("reached", room=current)
                    status["to_go"] = 0
                    outcome = "success"
                    break
                on_plan = current == hop_target
                event("entered", room=current, on_plan=on_plan)
                attempts, issue = 0, True

            if room is not None:
                last_in_room_xy = (pose["x"], pose["y"])
            elif math.hypot(pose["x"] - last_in_room_xy[0], pose["y"] - last_in_room_xy[1]) > args.outside_limit:
                # Doorway gaps between polygons are a few cm; this far out means it went through an exterior wall
                event("left_house", frm=current, at=f"({pose['x']:.2f}, {pose['y']:.2f})")
                outcome = "left_house"
                break

            if not issue and time.monotonic() - hop_started > 1.0:  # let the post-command snapshot arrive
                vln = sim.vln
                if vln.get("error"):
                    event("vln_error", error=vln["error"])
                    issue = True
                elif sim.snapshot["control"]["source"] != "vln":
                    event("lightnav_stopped", room=current, target=hop_target)
                    issue = True
                elif time.monotonic() - hop_started > args.hop_timeout:
                    event("hop_timeout", room=current, target=hop_target)
                    issue = True

            if issue:
                if attempts >= args.max_attempts:
                    outcome = "gave_up"
                    event("gave_up", room=current, target=hop_target)
                    break
                path = graph.plan_path(current, args.goal)
                if path is None or len(path) < 2:
                    outcome = "no_path"
                    break
                (_, door_id), (hop_target, _) = path[0], path[1]
                instruction = hop_instruction(pose, door_by_id[door_id], hop_target, args.hint)
                result = await sim.command({"type": "set_vln", "enabled": True, "instruction": instruction})
                if not result.get("ok"):
                    raise SystemExit(f"Sim refused VLN start: {result.get('message')} "
                                     "(release manual control on the sim page)")
                event("instruct", target=hop_target, attempt=attempts + 1, instruction=repr(instruction))
                status.update(instruction=instruction, route=route_points(pose, path), to_go=len(path) - 1)
                hop_started = time.monotonic()
                attempts += 1
                issue = False

        await sim.command({"type": "stop"})
        if recorder is not None:
            status["outcome"] = outcome
            await recorder.close()
        summary = {"goal": args.goal, "outcome": outcome, "hint": args.hint,
                   "time_s": round(time.monotonic() - t0, 1), "distance_m": round(distance, 2),
                   "instructions": sum(e["event"] == "instruct" for e in log), "log": log}
        print(f"\n{outcome.upper()}: {summary['time_s']}s, {summary['distance_m']} m driven, "
              f"{summary['instructions']} instructions")
        return summary


def main():
    parser = argparse.ArgumentParser(description="robo-nav room planner driving LightNav-0 in the MuJoCo sim")
    parser.add_argument("goal", type=str, help="Goal room name, e.g. kitchen, bedroom_2 (see rooms.json)")
    parser.add_argument("--scene", type=str, required=True, help="ProcTHOR house JSON (val_2.json)")
    parser.add_argument("--sim_ws", type=str, default="ws://127.0.0.1:8088/ws")
    parser.add_argument("--vln_server", type=str, default=None, help="Override the sim's LightNav server URL")
    parser.add_argument("--hint", choices=["direction", "none"], default="direction",
                        help="'direction': name where the doorway is; 'none': just name the next room")
    parser.add_argument("--hop_timeout", type=float, default=60.0, help="Re-instruct after this many seconds")
    parser.add_argument("--max_attempts", type=int, default=4, help="Instructions per hop before giving up")
    parser.add_argument("--timeout", type=float, default=600.0, help="Whole-episode limit (s)")
    parser.add_argument("--allow_wall_crossing", action="store_true",
                        help="Log wall crossings instead of ending the episode")
    parser.add_argument("--outside_limit", type=float, default=0.5,
                        help="End as left_house once this far (m) outside every room polygon")
    parser.add_argument("--reset", action="store_true", help="Reset the robot to the spawn pose first")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON path for the episode log")
    parser.add_argument("--video", type=str, default=None, help="Optional MP4 path for an episode video")
    args = parser.parse_args()

    summary = asyncio.run(navigate(args))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
