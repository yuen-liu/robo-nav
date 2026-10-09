"""
Landmark-only navigation in the LightNav-0 MuJoCo house: no ground-truth pose, no localization,
no room names. Claude holds a landmark map it annotated itself from a walkthrough video
(`landmark.annotate`), looks at the robot's camera, and hands LightNav-0 one short landmark
instruction at a time ("go forward past the grey couch toward the wooden doors").

Modes (all start from the same random poses, set with the sim's set_pose):
  zeroshot        LightNav-0 alone: "Go to the <goal>." once; ends when it stops
  zeroshot_retry  same instruction, re-sent whenever LightNav stops (no map, no planner)
  landmark        Claude + landmark map -> step-by-step landmark instructions -> LightNav-0

Ground truth is used only to score (final distance to the goal object, same room) and to draw
the floor-plan panel in the episode videos -- never in what the robot or Claude sees.

  python -m robo_nav.landmark.navigate --scene $SCENE --map sim_data/landmark/val2 \
      --starts sim_data/loc/val2/queries.csv --n_starts 10 --modes zeroshot zeroshot_retry landmark \
      --out sim_data/landmark/runs/batch1
Needs the sim (scripts/start_stack.sh) and Anthropic credentials (landmark mode).
"""

import argparse
import asyncio
import csv
import json
import math
import os
import random
import time
from typing import Dict, List, Optional

import requests

from robo_nav.claude import ask_json, client as make_client, image_block, obj_schema
from robo_nav.sim.nav import SimSession
from robo_nav.sim.scene import load_objects, load_rooms, locate
from robo_nav.sim.video import EpisodeRecorder, FloorPlan

GOALS = {  # what the robot is asked for -> sim object type (each appears once in val_2)
    "fridge": "Fridge",
    "armchair": "ArmChair",
    "dog bed": "DogBed",
    "laundry hamper": "LaundryHamper",
    "side table": "SideTable",
}
SUCCESS_RADIUS = 1.5  # m from the goal object's center, and in the same room

PLAN_SCHEMA = obj_schema({
    "observation": {"type": "string", "description": "what the robot camera shows right now"},
    "where_am_i": {"type": "string", "description": "position relative to landmarks in the map, or 'unsure'"},
    "action": {"type": "string", "enum": ["instruct", "look_around", "arrived", "give_up"]},
    "instruction": {"type": "string", "description": "the single next instruction for the driving policy ('' unless action is instruct)"},
    "plan": {"type": "array", "items": {"type": "string"}, "description": "remaining landmark steps after this one"},
})


def planner_system(map_md: str) -> List[dict]:
    text = (
        "You guide a small wheeled robot through a building using only landmarks. Its camera is about "
        "20 cm above the floor and faces forward. You never know its coordinates; work out where it is "
        "from what the camera shows and the landmark map below, which you wrote earlier from a video of a "
        "walk through this building.\n\n"
        "A local driving policy executes your instructions. Give it one short instruction at a time that "
        "covers one landmark or a few meters, e.g. 'Go forward past the grey couch toward the wooden "
        "doors.', 'Turn left and go through the open doorway.', 'Go to the white fridge and stop.' It "
        "cannot see behind itself: to go somewhere behind, start the instruction with 'Turn around'. It "
        "stops when it thinks the instruction is done, then you see a new camera view.\n\n"
        "Rules: describe only physical things -- furniture, appliances, doors, wall corners, colors. Never "
        "name rooms or room types. Use action 'look_around' to get eight views (45 degrees apart) when you "
        "cannot tell where the robot is, 'arrived' only when the goal is right in front of the robot "
        "(within about a meter), and 'give_up' only if the goal cannot be found.\n\n# Landmark map\n\n" + map_md)
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


class Episode:
    def __init__(self, sim: SimSession, sim_http: str, args, goal: str, goal_obj: dict, rooms: List[dict]):
        self.sim, self.sim_http, self.args = sim, sim_http, args
        self.goal, self.goal_obj, self.rooms = goal, goal_obj, rooms
        self.log: List[dict] = []
        self.t0 = time.monotonic()
        self.status: Dict = {"goal_label": goal, "goal_xy": (goal_obj["x"], goal_obj["y"]), "instruction": ""}
        self.min_dist = math.inf
        self.distance = 0.0
        self._last_xy = None

    # -- bookkeeping (GT only for scoring/video) ---------------------------------------------
    def event(self, kind: str, **kw):
        entry = {"t": round(time.monotonic() - self.t0, 1), "event": kind, **kw}
        self.log.append(entry)
        print(f"    [{entry['t']:6.1f}s] {kind}: " + ", ".join(f"{k}={str(v)[:110]}" for k, v in kw.items()), flush=True)

    def track(self):
        p = self.sim.pose
        xy = (p["x"], p["y"])
        if self._last_xy is not None:
            self.distance += math.hypot(xy[0] - self._last_xy[0], xy[1] - self._last_xy[1])
        self._last_xy = xy
        self.min_dist = min(self.min_dist, self.goal_dist())
        r = locate(p["x"], p["y"], self.rooms)
        if r is not None:
            self.status["room"] = r["name"]

    def goal_dist(self) -> float:
        p = self.sim.pose
        return math.hypot(p["x"] - self.goal_obj["x"], p["y"] - self.goal_obj["y"])

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    # -- robot I/O (what a real robot would have) ----------------------------------------------
    def camera(self) -> bytes:
        return requests.get(f"{self.sim_http}/api/camera.jpg", timeout=2).content

    async def look_around(self) -> List[bytes]:
        """Spin in place 8 x 45 deg (counter-clockwise) with velocity commands, grabbing a view each stop."""
        views = []
        await self.sim.command({"type": "acquire_control"})
        for _ in range(8):
            await asyncio.sleep(0.4)
            views.append(await asyncio.to_thread(self.camera))
            t_end = time.monotonic() + (math.pi / 4) / 1.0
            while time.monotonic() < t_end:
                await self.sim.ws.send(json.dumps({"type": "twist", "linear": 0.0, "angular": 1.0}))
                await asyncio.sleep(0.05)
            await self.sim.ws.send(json.dumps({"type": "twist", "linear": 0.0, "angular": 0.0}))
        await self.sim.command({"type": "release_control"})
        return views

    async def drive(self, instruction: str) -> str:
        """Run one instruction on LightNav until it stops itself, stalls, or times out."""
        self.status["instruction"] = instruction
        res = await self.sim.command({"type": "set_vln", "enabled": True, "instruction": instruction})
        if not res.get("ok"):
            raise RuntimeError(f"sim refused VLN start: {res.get('message')}")
        started = time.monotonic()
        moved_at, anchor = started, (self.sim.pose["x"], self.sim.pose["y"])
        while True:
            await asyncio.sleep(0.1)
            self.track()
            now = time.monotonic()
            p = self.sim.pose
            if math.hypot(p["x"] - anchor[0], p["y"] - anchor[1]) > 0.1 or abs(self.sim.snapshot["simulation"]["velocity"]["angular"]) > 0.1:
                moved_at, anchor = now, (p["x"], p["y"])
            if now - started > 1.0 and self.sim.snapshot["control"]["source"] != "vln":
                return "stopped"
            if now - moved_at > self.args.stall_s:
                await self.sim.command({"type": "stop"})
                return "stalled"
            if now - started > self.args.step_timeout or self.elapsed() > self.args.timeout:
                await self.sim.command({"type": "stop"})
                return "timeout"

    # -- modes ----------------------------------------------------------------------------------
    async def run_zeroshot(self, retry: bool) -> str:
        instruction = f"Go to the {self.goal}."
        for attempt in range(self.args.max_instructions if retry else 1):
            self.event("instruct", attempt=attempt + 1, instruction=instruction)
            ended = await self.drive(instruction)
            self.event("step_end", how=ended)
            if self.elapsed() > self.args.timeout:
                return "timeout"
        return "done"

    async def run_landmark(self, cl, system: List[dict]) -> str:
        history: List[str] = []
        views = await self.look_around()
        calls = 0
        while calls < self.args.max_planner_calls and self.elapsed() < self.args.timeout:
            content = [{"type": "text", "text": f"Goal: get the robot to the {self.goal}."}]
            if history:
                content.append({"type": "text", "text": "Steps so far:\n" + "\n".join(history)})
            if len(views) > 1:
                content.append({"type": "text", "text": "Look-around: view 0 is the current heading; each next view is 45 degrees further left (counter-clockwise). After the look-around the robot faces view 0 again."})
                for v, img in enumerate(views):
                    content += [{"type": "text", "text": f"view {v}"}, image_block(img)]
            else:
                content += [{"type": "text", "text": "Current camera view:"}, image_block(views[0])]
            content.append({"type": "text", "text": "Decide the next action."})
            plan = await asyncio.to_thread(ask_json, cl, content, PLAN_SCHEMA, system, 16000, self.args.effort)
            calls += 1
            self.event("plan", where_am_i=plan["where_am_i"], action=plan["action"], instruction=plan["instruction"],
                       plan=plan["plan"])
            if plan["action"] == "arrived":
                return "claude_arrived"
            if plan["action"] == "give_up":
                return "claude_gave_up"
            if plan["action"] == "look_around":
                history.append(f"{len(history) + 1}. (looked around)")
                views = await self.look_around()
                continue
            ended = await self.drive(plan["instruction"])
            self.event("step_end", how=ended)
            history.append(f"{len(history) + 1}. \"{plan['instruction']}\" -> driver {ended}; "
                           f"you had seen: {plan['observation']}")
            views = [await asyncio.to_thread(self.camera)]
        return "planner_limit" if calls >= self.args.max_planner_calls else "timeout"

    def score(self, ended: str) -> dict:
        p = self.sim.pose
        here = locate(p["x"], p["y"], self.rooms)
        there = locate(self.goal_obj["x"], self.goal_obj["y"], self.rooms)
        final = self.goal_dist()
        return {"ended": ended, "success": bool(final <= SUCCESS_RADIUS and here is not None and there is not None
                                                and here["id"] == there["id"]),
                "final_dist_m": round(final, 2), "min_dist_m": round(self.min_dist, 2),
                "time_s": round(self.elapsed(), 1), "distance_m": round(self.distance, 2)}


def pick_episodes(args, objects, rooms) -> List[dict]:
    with open(args.starts) as f:
        starts = list(csv.DictReader(f))
    rng = random.Random(args.seed)
    rng.shuffle(starts)
    goal_objs = {}
    for g, t in GOALS.items():
        matches = [o for o in objects if o["type"] == t]
        if len(matches) != 1:
            raise SystemExit(f"goal {g!r} ({t}) is not unique in this scene")
        goal_objs[g] = matches[0]
    episodes = []
    for s in starts[:args.n_starts]:
        start_room = s["room_name"]
        options = [g for g, o in goal_objs.items()
                   if (locate(o["x"], o["y"], rooms) or {}).get("name") != start_room]  # never start next to the goal
        for g in rng.sample(options, min(args.goals_per_start, len(options))):
            episodes.append({"start": {"x": float(s["x"]), "y": float(s["y"]), "yaw": float(s["yaw"]),
                                       "room": start_room, "query_idx": int(s["idx"])},
                             "goal": g, "goal_obj": goal_objs[g]})
    return episodes


async def run(args):
    rooms, doors = load_rooms(args.scene)
    objects = load_objects(args.scene)
    episodes = pick_episodes(args, objects, rooms)
    os.makedirs(args.out, exist_ok=True)
    sim_http = args.sim_ws.replace("ws://", "http://").rsplit("/ws", 1)[0]
    cl, system = None, None
    if "landmark" in args.modes:
        with open(os.path.join(args.map, "landmark_map.md")) as f:
            system = planner_system(f.read())
        cl = make_client()
    plan_view = FloorPlan(rooms, doors)
    print(f"{len(episodes)} start/goal pairs x {len(args.modes)} modes")

    async with SimSession(args.sim_ws) as sim:
        for i, ep in enumerate(episodes):
            for mode in args.modes:
                name = f"e{i:02d}_{ep['goal'].replace(' ', '_')}_{mode}"
                path = os.path.join(args.out, name + ".json")
                if os.path.exists(path):
                    continue
                print(f"== {name}: start q{ep['start']['query_idx']} in {ep['start']['room']} -> {ep['goal']}", flush=True)
                res = await sim.command({"type": "set_pose", **{k: ep["start"][k] for k in ("x", "y", "yaw")}})
                if not res.get("ok"):
                    raise SystemExit(f"set_pose failed: {res.get('message')} (sim needs patches/lightnav0_sim.patch)")
                await asyncio.sleep(0.5)
                e = Episode(sim, sim_http, args, ep["goal"], ep["goal_obj"], rooms)
                goal_room = locate(ep["goal_obj"]["x"], ep["goal_obj"]["y"], rooms)
                e.status["goal"] = goal_room["name"] if goal_room else None  # floor-plan shading only
                e.track()
                rec = EpisodeRecorder(os.path.join(args.out, name + ".mp4"), sim_http, plan_view, e.status, lambda: sim.pose)
                rec.start()
                try:
                    if mode == "landmark":
                        ended = await e.run_landmark(cl, system)
                    else:
                        ended = await e.run_zeroshot(retry=mode == "zeroshot_retry")
                except Exception as exc:  # keep the batch going; the episode is recorded as an error
                    e.event("error", error=repr(exc))
                    ended = "error"
                await sim.command({"type": "stop"})
                score = e.score(ended)
                e.status["outcome"] = "success" if score["success"] else f"miss {score['final_dist_m']} m"
                await rec.close()
                out = {"mode": mode, "goal": ep["goal"], "start": ep["start"], **score, "log": e.log}
                with open(path, "w") as f:
                    json.dump(out, f, indent=1)
                print(f"   -> {'SUCCESS' if score['success'] else 'miss'}: final {score['final_dist_m']} m "
                      f"(closest {score['min_dist_m']} m), {score['time_s']} s, ended={ended}", flush=True)
    summarize(args.out)


def summarize(run_dir: str):
    from glob import glob
    eps = [json.load(open(p)) for p in sorted(glob(os.path.join(run_dir, "e*.json")))]
    if not eps:
        return
    print("\nmode             success   median final dist   median closest   median time")
    for mode in sorted({e["mode"] for e in eps}):
        g = [e for e in eps if e["mode"] == mode]
        med = lambda k: sorted(e[k] for e in g)[len(g) // 2]
        print(f"{mode:<16} {sum(e['success'] for e in g):>3}/{len(g):<4}  {med('final_dist_m'):>12.2f} m  "
              f"{med('min_dist_m'):>12.2f} m  {med('time_s'):>9.1f} s")


def main():
    parser = argparse.ArgumentParser(description="Landmark-only navigation (no GT, no room names) vs zero-shot LightNav-0")
    parser.add_argument("--scene", type=str, required=True, help="ProcTHOR house JSON (val_2.json)")
    parser.add_argument("--map", type=str, help="landmark.annotate output dir (landmark mode)")
    parser.add_argument("--starts", type=str, required=True, help="CSV of start poses (e.g. sim_data/loc/val2/queries.csv)")
    parser.add_argument("--n_starts", type=int, default=10)
    parser.add_argument("--goals_per_start", type=int, default=2)
    parser.add_argument("--modes", nargs="+", default=["zeroshot", "zeroshot_retry", "landmark"],
                        choices=["zeroshot", "zeroshot_retry", "landmark"])
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--sim_ws", type=str, default="ws://127.0.0.1:8088/ws")
    parser.add_argument("--timeout", type=float, default=300.0, help="Episode limit (s)")
    parser.add_argument("--step_timeout", type=float, default=45.0, help="Limit per LightNav instruction (s)")
    parser.add_argument("--stall_s", type=float, default=12.0, help="End a step after this long without moving")
    parser.add_argument("--max_instructions", type=int, default=6, help="zeroshot_retry re-sends")
    parser.add_argument("--max_planner_calls", type=int, default=20)
    parser.add_argument("--effort", type=str, default="medium")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if "landmark" in args.modes and not args.map:
        parser.error("--map is required for landmark mode")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
