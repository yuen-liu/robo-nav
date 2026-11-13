"""
GOAT-Bench with LightNav-0 driving: zero-shot LightNav vs the landmark planner (Claude or Gemini,
ROBO_NAV_VLM) handing it one landmark instruction at a time, plus VLM-only baselines that drive the
robot themselves with coarse moves.

Runs in LightNav-0's venv: the policy is built in-process exactly like `lightnav-eval-habitat`
(vLLM engine, TrajVocabVLNCEPolicy, velocity_control at dt = 1 s); the scene runs in
`robo_nav/habitat/goat_env.py serve` (habitat env) on --env.

GOAT episodes are lifelong: subtasks run back to back from wherever the last one ended. A subtask
ends when the robot stops for good (zeroshot: LightNav's own STOP; landmark: the planner says
'arrived' or 'give_up'; vlm/vlm_map: the VLM says 'stop' or 'give_up') or runs out of --subtask_steps policy steps, and is scored by the env at
that pose (GT, 0.25 m to a goal view point). Image-goal subtasks are skipped for now (counted as
skipped, the robot stays put).

Modes:
  zeroshot   LightNav-0 alone with the subtask text ("Find the bed." / the language description)
  landmark   planner + the scene's landmark map (annotated from its tour) -> LightNav-0
  landmark_vlm  as landmark, but the VLM drives each instruction with coarse moves instead of LightNav-0
  vlm        the VLM alone: camera views in, 'forward x m' / 'turn y deg' out, no map, no LightNav
  vlm_map    as vlm, plus the scene's landmark map

Step budget is policy steps (sim time), so planner API latency never costs the robot anything.

  python -m robo_nav.habitat.goat_nav --model_path checkpoints/LightNav-0 --env tcp://127.0.0.1:5560 \\
      --maps sim_data/goat/maps --episodes 0 1 2 --modes zeroshot landmark --out sim_data/goat/runs/pilot
"""

import argparse
import json
import math
import os
import pickle
import time
from typing import Dict, List, Optional

import numpy as np

from robo_nav.claude import ask_json, client as make_client, image_block, obj_schema, print_cost, usage_summary
from robo_nav.habitat.floors import annotate_map_md, describe_floor
from robo_nav.landmark.planner import ARRIVE, PLAN_SCHEMA, planner_system

CAMERA = "about 0.9 m above the floor"

DIRECT_SCHEMA = obj_schema({
    "observation": {"type": "string", "description": "what the robot camera shows right now"},
    "where_am_i": {"type": "string", "description": "position relative to landmarks seen so far (or in the map), or 'unsure'"},
    "action": {"type": "string", "enum": ["forward", "turn", "look_around", "stop", "give_up"]},
    "amount": {"type": "number", "description": "meters for forward (0.25 to 3), degrees for turn (+ = left), 0 otherwise"},
})


def direct_system(map_md: Optional[str], arrive: str = "old") -> List[dict]:
    """System prompt for the VLM-only modes; arrival wording matches planner_system for a fair comparison."""
    text = (
        f"You drive a wheeled robot through a building to find a goal. Its camera is {CAMERA}, faces "
        "forward and sees about 120 degrees. You never know its coordinates; work out where it is from what "
        "the camera shows" + (" and the landmark map below, which you wrote earlier from a video of a walk "
                              "through this building." if map_md else ".") + "\n\n"
        "Each turn, pick one move. 'forward' drives straight ahead `amount` meters (0.25 to 3); it stops "
        "early if something blocks it. 'turn' turns in place `amount` degrees, positive = left "
        "(counter-clockwise), negative = right, up to 180. Then you see a new camera view.\n\n"
        "Use 'look_around' to get eight views (45 degrees apart) when you cannot tell where the robot is, "
        f"'stop' only when {ARRIVE[arrive]}, and 'give_up' only if the goal cannot be found.")
    if map_md:
        text += "\n\n# Landmark map\n\n" + map_md
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


DRIVER_SCHEMA = obj_schema({
    "observation": {"type": "string", "description": "what the robot camera shows right now"},
    "action": {"type": "string", "enum": ["forward", "turn", "done"]},
    "amount": {"type": "number", "description": "meters for forward (0.25 to 3), degrees for turn (+ = left), 0 for done"},
})


def driver_system() -> List[dict]:
    """System prompt for the VLM standing in for LightNav-0 in landmark_vlm: it only sees the planner's
    instruction and the camera, like LightNav, no goal and no map."""
    text = (
        f"You are the local driving policy of a wheeled robot. Its camera is {CAMERA}, faces forward and "
        "sees about 120 degrees. You get one short instruction at a time, e.g. 'Go forward past the grey "
        "couch toward the wooden doors.' or 'Turn left and go through the open doorway.', and carry it out.\n\n"
        "Each turn, pick one move. 'forward' drives straight ahead `amount` meters (0.25 to 3); it stops "
        "early if something blocks it. 'turn' turns in place `amount` degrees, positive = left "
        "(counter-clockwise), negative = right, up to 180. Then you see a new camera view. Say 'done' as soon "
        "as the instruction has been carried out (or cannot be carried out any further).")
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


VERIFY_SCHEMA = obj_schema({
    "observation": {"type": "string", "description": "what the views show near the robot"},
    "candidate": {"type": "string", "description": "the object in the views that best matches the goal, and which view it is in, or 'none'"},
    "cues": {"type": "string", "description": "each detail the goal gives (color, material, what it is next to, where it is) and whether the candidate fits it"},
    "matches": {"type": "boolean", "description": "the candidate is the goal: the right kind of object and, for a description, every detail fits"},
    "close_enough": {"type": "boolean", "description": "the robot is right next to it: about an arm's length away"},
    "reason": {"type": "string", "description": "one sentence for the planner if this is not the goal or not close enough"},
})


def verify_system() -> List[dict]:
    text = (
        f"You check a robot's claim that it has reached its goal. Its camera is {CAMERA} and sees about 120 "
        "degrees; you get eight views 45 degrees apart from where it stands. Be strict: a house often has "
        "several objects of the same kind (mirrors, pictures, dressers, rugs), and a description names one of "
        "them. Check every detail the goal gives -- color, material, what it is next to or above, where it is -- "
        "against what the views show. If the views do not show the details, it does not match.")
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


class PlannerError(RuntimeError):
    """The planner API failed after the client's retries: an outage, not a navigation miss."""


class EnvClient:
    def __init__(self, address: str, timeout_ms: int = 600000):
        import zmq
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(address)

    def call(self, command: str, data=None) -> dict:
        self.sock.send(pickle.dumps({"command": command, "data": data}, protocol=4))
        reply = pickle.loads(self.sock.recv())
        if reply.get("status") != "success":
            raise RuntimeError(f"env {command} failed: {reply.get('message')}")
        return reply

    def close(self):
        try:
            self.call("close")
        finally:
            self.sock.close()
            self.ctx.term()


def build_policy(args, info: dict):
    from lightnav.habitat.runner import HabitatEvalConfig, _default_engine_factory, build_velocity_policy
    cfg = HabitatEvalConfig(model_path=args.model_path, backend=args.backend,
                            gpu_memory_utilization=args.gpu_memory_utilization)
    engine, bundle = _default_engine_factory(cfg)
    return build_velocity_policy(cfg, engine, bundle, info)


def jpeg(rgb: np.ndarray) -> bytes:
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class Run:
    """One GOAT episode in one mode."""

    def __init__(self, env: EnvClient, policy, args, obs: dict, info: dict):
        self.env, self.policy, self.args = env, policy, args
        self.obs, self.info = obs, info
        self.log: List[dict] = []
        self.planner_s = 0.0
        self.usage: List[dict] = []  # planner token usage, one entry per call
        self.steps = 0  # policy/env steps this subtask
        self.last_latency = 0.0  # s, the latest planner call
        self.floor_levels: Optional[List[float]] = None  # --floors: tell the planner which floor it is on

    def event(self, kind: str, **kw):
        self.log.append({"event": kind, **kw})
        print(f"    {kind}: " + ", ".join(f"{k}={str(v)[:110]}" for k, v in kw.items()), flush=True)

    def step(self, action: dict) -> dict:
        reply = self.env.call("step", action)
        self.obs, self.steps = reply["obs"], self.steps + 1
        return reply["info"]

    def turn(self, deg: float):
        """Turn in place (+ = left/ccw) in <= 30 deg velocity steps."""
        lo, hi = self.info["ang_vel_range"]
        dt = self.info["habitat_time_step"]
        left = deg
        while abs(left) > 0.5:
            d = max(min(left, hi * dt), lo * dt)
            raw = 2 * (d / dt - lo) / (hi - lo) - 1
            lin_raw = -1.0  # = lin_vel_range[0] = 0 m/s
            self.step({"action": "velocity_control", "action_args": {"linear_velocity": lin_raw, "angular_velocity": raw}})
            left -= d

    def look_around(self) -> List[bytes]:
        views = []
        for _ in range(8):
            views.append(jpeg(self.obs["rgb"]))
            self.turn(45)
        return views

    def drive(self, instruction: str, budget: int, stall: bool = True) -> str:
        """LightNav-0 on one instruction until it stops, stalls (if `stall`), or the budget runs out."""
        self.policy.reset({"instruction": {"text": instruction}})
        stuck = 0
        for _ in range(budget):
            info = self.step(self.policy.act(self.obs, None))
            if info["stopped"]:
                return "stopped"
            stuck = stuck + 1 if info["collided"] else 0
            if stall and stuck >= self.args.stall_steps:
                return "stalled"
        return "step_limit"

    def zeroshot(self, text: str) -> str:
        return self.drive(text, self.args.subtask_steps, stall=False)  # as lightnav-eval-habitat: no stall rule

    def forward(self, meters: float, until: Optional[int] = None) -> str:
        """Drive straight ahead at full speed; 'blocked' after 2 collided steps in a row."""
        lo, hi = self.info["lin_vel_range"]
        alo, ahi = self.info["ang_vel_range"]
        dt = self.info["habitat_time_step"]
        ang_raw = 2 * (0.0 - alo) / (ahi - alo) - 1  # 0 deg/s
        left, stuck = meters, 0
        until = self.args.subtask_steps if until is None else min(until, self.args.subtask_steps)
        while left >= 0.05 and self.steps < until:  # slower than ~0.025 m/s would read as STOP
            v = min(left / dt, hi)
            info = self.step({"action": "velocity_control",
                              "action_args": {"linear_velocity": 2 * (v - lo) / (hi - lo) - 1, "angular_velocity": ang_raw}})
            left -= v * dt
            stuck = stuck + 1 if info["collided"] else 0
            if stuck >= 2:
                return "blocked"
        return "done"

    def ask(self, cl, system, schema: dict, text: str, history: List[str], views: List[bytes]) -> dict:
        """One planner call: goal, step history and the current view(s) -> JSON matching `schema`."""
        content = [{"type": "text", "text": f"Goal: {text}"}]
        if self.floor_levels:
            y = self.env.call("pose")["pose"]["position"][1]
            content.append({"type": "text", "text": describe_floor(y, self.floor_levels)})
        if history:
            content.append({"type": "text", "text": "Steps so far:\n" + "\n".join(history)})
        if len(views) > 1:
            content.append({"type": "text", "text": "Look-around: view 0 is the current heading; each next view is 45 degrees further left (counter-clockwise). After the look-around the robot faces view 0 again."})
            for v, img in enumerate(views):
                content += [{"type": "text", "text": f"view {v}"}, image_block(img)]
        else:
            content += [{"type": "text", "text": "Current camera view:"}, image_block(views[0])]
        content.append({"type": "text", "text": "Decide the next action."})
        asked = time.monotonic()
        try:
            return ask_json(cl, content, schema, system, 16000, self.args.effort, usage=self.usage)
        except Exception as exc:
            raise PlannerError(repr(exc)) from exc
        finally:
            self.planner_s += time.monotonic() - asked
            self.last_latency = time.monotonic() - asked

    def drive_vlm(self, cl, instruction: str, budget: int) -> str:
        """The VLM in LightNav-0's place on one instruction: coarse moves until it says 'done', its call cap,
        or the step budget (same per-instruction budget as LightNav)."""
        end = self.steps + budget
        moves: List[str] = []
        for _ in range(self.args.max_driver_calls):
            if self.steps >= end:
                return "step_limit"
            content = [{"type": "text", "text": f"Instruction: {instruction}"}]
            if moves:
                content.append({"type": "text", "text": "Moves so far:\n" + "\n".join(moves)})
            content += [{"type": "text", "text": "Current camera view:"}, image_block(jpeg(self.obs["rgb"])),
                        {"type": "text", "text": "Decide the next move."}]
            asked = time.monotonic()
            try:
                move = ask_json(cl, content, DRIVER_SCHEMA, self._driver_system, 16000, self.args.effort, usage=self.usage)
            except Exception as exc:
                raise PlannerError(repr(exc)) from exc
            finally:
                self.planner_s += time.monotonic() - asked
            self.usage[-1]["role"] = "driver"
            self.event("drive", action=move["action"], amount=move["amount"])
            if move["action"] == "done":
                return "stopped"
            if move["action"] == "forward":
                m = min(max(float(move["amount"]), 0.25), 3.0)
                how = self.forward(m, until=end)
                moves.append(f"{len(moves) + 1}. forward {m:.2f} m -> {how}")
            else:
                deg = min(max(float(move["amount"]), -180.0), 180.0)
                self.turn(deg)
                moves.append(f"{len(moves) + 1}. turn {deg:+.0f} deg")
        return "driver_limit"

    _driver_system = driver_system()

    def verify(self, cl, text: str, views: List[bytes]) -> dict:
        """--verify_arrival: a separate VLM call checks the planner's 'arrived' against a fresh look-around."""
        content = [{"type": "text", "text": f"Goal: {text}"},
                   {"type": "text", "text": "Look-around from where the robot stands: view 0 is its heading; each next view is 45 degrees further left."}]
        for v, img in enumerate(views):
            content += [{"type": "text", "text": f"view {v}"}, image_block(img)]
        content.append({"type": "text", "text": "Is the robot at the goal?"})
        asked = time.monotonic()
        try:
            out = ask_json(cl, content, VERIFY_SCHEMA, self._verify_system, 16000, self.args.effort, usage=self.usage)
        except Exception as exc:
            raise PlannerError(repr(exc)) from exc
        finally:
            self.planner_s += time.monotonic() - asked
        self.usage[-1]["role"] = "verify"
        return out

    _verify_system = verify_system()

    def approach(self, cl, text: str) -> str:
        """--final_approach: after an accepted 'arrived', the VLM makes small moves to get right next to the goal."""
        goal = (f"{text} (The robot should already be near it. Make small moves to get right next to it, "
                "about an arm's length away and facing it, then stop.)")
        ended = self.direct(cl, self._approach_system, goal, max_calls=self.args.final_approach, look=False)
        return "planner_arrived+approach_" + ended.split("_", 1)[1]

    _approach_system = direct_system(None, "new")

    def landmark(self, cl, system, text: str, vlm_driver: bool = False) -> str:
        history: List[str] = []
        views = self.look_around()
        checks = 0
        for _ in range(self.args.max_planner_calls):
            if self.steps >= self.args.subtask_steps:
                return "step_limit"
            plan = self.ask(cl, system, PLAN_SCHEMA, text, history, views)
            self.event("plan", latency_s=round(self.last_latency, 1), out_tokens=self.usage[-1]["output"], where_am_i=plan["where_am_i"],
                       action=plan["action"], instruction=plan["instruction"])
            if plan["action"] == "arrived" and checks < self.args.verify_arrival:
                checks += 1
                views = self.look_around()
                v = self.verify(cl, text, views)
                ok = v["matches"] and (v["close_enough"] or self.args.final_approach > 0)
                self.event("verify", ok=ok, matches=v["matches"], close_enough=v["close_enough"],
                           candidate=v["candidate"], reason=v["reason"])
                if not ok:
                    history.append(f"{len(history) + 1}. You said 'arrived', but a check of a fresh look-around "
                                   f"rejected it: {v['reason']} (you had seen: {plan['observation']})")
                    continue
            if plan["action"] == "arrived" and self.args.final_approach > 0:
                return self.approach(cl, text)
            if plan["action"] in ("arrived", "give_up"):
                return f"planner_{plan['action']}"
            if plan["action"] == "look_around":
                history.append(f"{len(history) + 1}. (looked around)")
                views = self.look_around()
                continue
            budget = min(self.args.instruction_steps, self.args.subtask_steps - self.steps)
            ended = (self.drive_vlm(cl, plan["instruction"], budget) if vlm_driver
                     else self.drive(plan["instruction"], budget))
            self.event("step_end", how=ended, steps=self.steps)
            history.append(f"{len(history) + 1}. \"{plan['instruction']}\" -> driver {ended}; "
                           f"you had seen: {plan['observation']}")
            views = [jpeg(self.obs["rgb"])]
        return "planner_limit"

    def direct(self, cl, system, text: str, max_calls: Optional[int] = None, look: bool = True) -> str:
        """VLM-only baseline: the VLM picks every move itself (no LightNav). Also the --final_approach driver."""
        history: List[str] = []
        views = self.look_around() if look else [jpeg(self.obs["rgb"])]
        for _ in range(max_calls or self.args.max_direct_calls):
            if self.steps >= self.args.subtask_steps:
                return "step_limit"
            plan = self.ask(cl, system, DIRECT_SCHEMA, text, history, views)
            self.event("plan", latency_s=round(self.last_latency, 1), out_tokens=self.usage[-1]["output"], where_am_i=plan["where_am_i"],
                       action=plan["action"], amount=plan["amount"])
            if plan["action"] in ("stop", "give_up"):
                return f"vlm_{plan['action']}"
            if plan["action"] == "look_around":
                history.append(f"{len(history) + 1}. (looked around)")
                views = self.look_around()
                continue
            if plan["action"] == "forward":
                m = min(max(float(plan["amount"]), 0.25), 3.0)
                how = self.forward(m)
                history.append(f"{len(history) + 1}. forward {m:.2f} m -> {how}; you had seen: {plan['observation']}")
            else:
                deg = min(max(float(plan["amount"]), -180.0), 180.0)
                self.turn(deg)
                history.append(f"{len(history) + 1}. turn {deg:+.0f} deg; you had seen: {plan['observation']}")
            views = [jpeg(self.obs["rgb"])]
        return "vlm_limit"


def main():
    parser = argparse.ArgumentParser(description="GOAT-Bench: zero-shot LightNav-0 vs landmark planner + LightNav-0 vs VLM-only")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--backend", default="vllm_local")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.65)
    parser.add_argument("--env", default="tcp://127.0.0.1:5560")
    parser.add_argument("--env_timeout_s", type=int, default=600, help="raise for a shared env server (resets wait for other clients)")
    parser.add_argument("--shard", type=int, nargs=2, metavar=("I", "N"), help="--goals: run only goals i, i+N, i+2N, ...")
    parser.add_argument("--maps", help="Dir of landmark.annotate outputs, one per scene: <maps>/<scene>/landmark_map.md")
    parser.add_argument("--episodes", type=int, nargs="+", help="Episode indices in the env's split")
    parser.add_argument("--goals", help="JSON list of {episode, subtask, position, heading}: run only that subtask, "
                                        "starting from that pose (robo_nav.habitat.far_goals); instead of --episodes")
    parser.add_argument("--modes", nargs="+", default=["zeroshot", "landmark"], choices=["zeroshot", "landmark", "landmark_vlm", "vlm", "vlm_map"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--subtask_steps", type=int, default=500, help="Policy steps per subtask (dt = 1 s)")
    parser.add_argument("--budget_factor", type=float, default=None,
                        help="Per-subtask step budget = ceil(F x shortest-path time at full speed) = ceil(F x geodesic / 0.25 m/s), "
                             "instead of --subtask_steps; the env's per-step track rescores any smaller F (goal_report)")
    parser.add_argument("--instruction_steps", type=int, default=60, help="Policy steps per planner instruction")
    parser.add_argument("--stall_steps", type=int, default=8, help="End an instruction after this many blocked steps")
    parser.add_argument("--max_planner_calls", type=int, default=20, help="Per subtask")
    parser.add_argument("--max_driver_calls", type=int, default=10, help="VLM driver calls per planner instruction in landmark_vlm")
    parser.add_argument("--max_direct_calls", type=int, default=60, help="VLM calls per subtask in vlm/vlm_map")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--actuation_noise", type=float, nargs=2, metavar=("LIN_SD", "ANG_SD_DEG"),
                        help="Gaussian actuation noise in the env: speed x (1 + N(0, LIN_SD)), turn rate + N(0, ANG_SD_DEG) deg/s; off by default")
    parser.add_argument("--noise_seed", type=int, default=0)
    parser.add_argument("--verify_arrival", type=int, default=0,
                        help="landmark modes: check up to N 'arrived' calls per subtask against a fresh look-around (0 = off)")
    parser.add_argument("--final_approach", type=int, default=0,
                        help="landmark modes: after 'arrived', up to N VLM calls of small moves to get right next to the goal (0 = off)")
    parser.add_argument("--floors", action="store_true",
                        help="tag map landmarks with floors (<maps>/<scene>/floors.json, robo_nav.habitat.floors) and tell the planner the robot's floor")
    parser.add_argument("--arrive_prompt", default="old", choices=sorted(ARRIVE),
                        help="When the planner/VLM may stop (landmark.planner.ARRIVE); 'old' = the first val_unseen run")
    args = parser.parse_args()
    if bool(args.episodes) == bool(args.goals):
        parser.error("give exactly one of --episodes / --goals")
    os.makedirs(args.out, exist_ok=True)
    if args.goals:
        with open(args.goals) as f:
            jobs = [(g["episode"], g["subtask"], g) for g in json.load(f)]
        if args.shard:
            jobs = jobs[args.shard[0]::args.shard[1]]
    else:
        jobs = [(idx, None, None) for idx in args.episodes]

    env = EnvClient(args.env, timeout_ms=args.env_timeout_s * 1000)
    policy, cl = None, None
    try:
        for idx, only, start in jobs:
            for mode in args.modes:
                name = f"ep{idx:04d}" if only is None else f"g{idx:04d}_{only:02d}"
                path = os.path.join(args.out, f"{name}_{mode}.json")
                if os.path.exists(path) and json.load(open(path))["ended"] != "planner_error":
                    continue
                noise = ({"lin": args.actuation_noise[0], "ang_deg": args.actuation_noise[1], "seed": args.noise_seed}
                         if args.actuation_noise else None)
                reply = env.call("reset", {"episode": idx, **({"noise": noise} if noise else {})})
                obs, info = reply["obs"], reply["info"]
                if start is not None and start.get("position") is not None:
                    obs = env.call("teleport", {"position": start["position"], "heading": start["heading"]})["obs"]
                scene = os.path.basename(os.path.normpath(info["scene_id"])).split(".")[0]
                print(f"== ep {idx} ({scene}) {mode}: {len(info['subtasks'])} subtasks", flush=True)
                if policy is None and mode in ("zeroshot", "landmark"):
                    policy = build_policy(args, info)
                map_md, system = None, None
                floors = None
                if mode in ("landmark", "landmark_vlm", "vlm_map"):
                    with open(os.path.join(args.maps, scene, "landmark_map.md")) as f:
                        map_md = f.read()
                    if args.floors:
                        with open(os.path.join(args.maps, scene, "floors.json")) as f:
                            floors = json.load(f)
                        map_md = annotate_map_md(map_md, floors)
                if mode in ("landmark", "landmark_vlm"):
                    system = planner_system(map_md, CAMERA, args.arrive_prompt)
                elif mode in ("vlm", "vlm_map"):
                    system = direct_system(map_md, args.arrive_prompt)
                if mode != "zeroshot":
                    cl = cl or make_client()
                run = Run(env, policy, args, obs, info)
                if floors and len(floors["levels"]) > 1:
                    run.floor_levels = floors["levels"]
                results, ended_ep = [], "done"
                for k, sub in enumerate(info["subtasks"]):
                    if only is not None and k != only:
                        continue
                    if sub["text"] is None:
                        results.append({"subtask": k, **sub, "ended": "skipped"})
                        continue
                    run.steps = 0
                    geo = env.call("begin", {"subtask": k})["geodesic_m"]
                    if args.budget_factor:
                        args.subtask_steps = max(1, math.ceil(args.budget_factor * geo / info["lin_vel_range"][1]))
                    run.event("subtask", k=k, type=sub["type"], text=sub["text"], geodesic_m=round(geo, 2))
                    try:
                        if mode == "zeroshot":
                            ended = run.zeroshot(sub["text"])
                        elif mode in ("landmark", "landmark_vlm"):
                            ended = run.landmark(cl, system, sub["text"], vlm_driver=mode == "landmark_vlm")
                        else:
                            ended = run.direct(cl, system, sub["text"])
                    except PlannerError as exc:
                        run.event("planner_error", error=str(exc))
                        ended_ep = "planner_error"
                        break
                    score = env.call("score", {"subtask": k})
                    score.pop("status", None)
                    results.append({"subtask": k, **sub, "ended": ended, "steps": run.steps,
                                    "budget_steps": args.subtask_steps, **score})
                    print(f"   -> subtask {k} {sub['type']}: {'SUCCESS' if score['success'] else 'miss'} "
                          f"{score['distance_to_goal']:.2f} m, SPL {score['spl']:.2f}, {run.steps} steps, {ended}", flush=True)
                out = {"episode": idx, "only_subtask": only, "start": start, "scene": scene, "mode": mode, "arrive_prompt": args.arrive_prompt, "noise": noise,
                       "budget_factor": args.budget_factor, "verify_arrival": args.verify_arrival, "final_approach": args.final_approach, "floors": args.floors, "ended": ended_ep,
                       "planner_s": round(run.planner_s, 1), "usage": usage_summary(run.usage), "subtasks": results, "log": run.log}
                with open(path, "w") as f:
                    json.dump(out, f, indent=1, default=float)
    finally:
        env.close()
    summarize(args.out)


def summarize(run_dir: str):
    from glob import glob
    eps = [json.load(open(p)) for p in sorted(glob(os.path.join(run_dir, "ep*.json")) + glob(os.path.join(run_dir, "g*.json")))]
    for mode in sorted({e["mode"] for e in eps}):
        if any(e.get("usage", {}).get("calls") for e in eps if e["mode"] == mode):
            print(f"\n[{mode}]", end="")
            print_cost([e for e in eps if e["mode"] == mode])
    outages = [e for e in eps if e["ended"] == "planner_error"]
    if outages:
        print(f"\n{len(outages)} episode(s) cut short by a planner API failure, left out below")
    eps = [e for e in eps if e["ended"] != "planner_error"]
    print("\nmode       type          subtasks  success   SPL")
    for mode in sorted({e["mode"] for e in eps}):
        subs = [s for e in eps if e["mode"] == mode for s in e["subtasks"] if s["ended"] != "skipped"]
        for kind in ["all"] + sorted({s["type"] for s in subs}):
            g = [s for s in subs if kind == "all" or s["type"] == kind]
            if g:
                print(f"{mode:<10} {kind:<12} {len(g):>8}  {np.mean([s['success'] for s in g]):>7.1%}  "
                      f"{np.mean([s['spl'] for s in g]):>5.3f}")


if __name__ == "__main__":
    main()
