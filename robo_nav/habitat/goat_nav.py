"""
GOAT-Bench with LightNav-0 driving: zero-shot LightNav vs the landmark planner (Claude or Gemini,
ROBO_NAV_VLM) handing it one landmark instruction at a time.

Runs in LightNav-0's venv: the policy is built in-process exactly like `lightnav-eval-habitat`
(vLLM engine, TrajVocabVLNCEPolicy, velocity_control at dt = 1 s); the scene runs in
`robo_nav/habitat/goat_env.py serve` (habitat env) on --env.

GOAT episodes are lifelong: subtasks run back to back from wherever the last one ended. A subtask
ends when the robot stops for good (zeroshot: LightNav's own STOP; landmark: the planner says
'arrived' or 'give_up') or runs out of --subtask_steps policy steps, and is scored by the env at
that pose (GT, 0.25 m to a goal view point). Image-goal subtasks are skipped for now (counted as
skipped, the robot stays put).

Modes:
  zeroshot   LightNav-0 alone with the subtask text ("Find the bed." / the language description)
  landmark   planner + the scene's landmark map (annotated from its tour) -> LightNav-0

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

from robo_nav.claude import ask_json, client as make_client, image_block
from robo_nav.landmark.planner import PLAN_SCHEMA, planner_system

CAMERA = "about 0.9 m above the floor"


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
        self.steps = 0  # policy/env steps this subtask

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

    def landmark(self, cl, system, text: str) -> str:
        history: List[str] = []
        views = self.look_around()
        for _ in range(self.args.max_planner_calls):
            if self.steps >= self.args.subtask_steps:
                return "step_limit"
            content = [{"type": "text", "text": f"Goal: {text}"}]
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
                plan = ask_json(cl, content, PLAN_SCHEMA, system, 16000, self.args.effort)
            except Exception as exc:
                raise PlannerError(repr(exc)) from exc
            finally:
                self.planner_s += time.monotonic() - asked
            self.event("plan", latency_s=round(time.monotonic() - asked, 1), where_am_i=plan["where_am_i"],
                       action=plan["action"], instruction=plan["instruction"])
            if plan["action"] in ("arrived", "give_up"):
                return f"planner_{plan['action']}"
            if plan["action"] == "look_around":
                history.append(f"{len(history) + 1}. (looked around)")
                views = self.look_around()
                continue
            ended = self.drive(plan["instruction"], min(self.args.instruction_steps,
                                                        self.args.subtask_steps - self.steps))
            self.event("step_end", how=ended, steps=self.steps)
            history.append(f"{len(history) + 1}. \"{plan['instruction']}\" -> driver {ended}; "
                           f"you had seen: {plan['observation']}")
            views = [jpeg(self.obs["rgb"])]
        return "planner_limit"


def main():
    parser = argparse.ArgumentParser(description="GOAT-Bench: zero-shot LightNav-0 vs landmark planner + LightNav-0")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--backend", default="vllm_local")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.65)
    parser.add_argument("--env", default="tcp://127.0.0.1:5560")
    parser.add_argument("--maps", help="Dir of landmark.annotate outputs, one per scene: <maps>/<scene>/landmark_map.md")
    parser.add_argument("--episodes", type=int, nargs="+", required=True, help="Episode indices in the env's split")
    parser.add_argument("--modes", nargs="+", default=["zeroshot", "landmark"], choices=["zeroshot", "landmark"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--subtask_steps", type=int, default=500, help="Policy steps per subtask (dt = 1 s)")
    parser.add_argument("--instruction_steps", type=int, default=60, help="Policy steps per planner instruction")
    parser.add_argument("--stall_steps", type=int, default=8, help="End an instruction after this many blocked steps")
    parser.add_argument("--max_planner_calls", type=int, default=20, help="Per subtask")
    parser.add_argument("--effort", default="medium")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = EnvClient(args.env)
    policy, cl = None, None
    try:
        for idx in args.episodes:
            for mode in args.modes:
                path = os.path.join(args.out, f"ep{idx:04d}_{mode}.json")
                if os.path.exists(path) and json.load(open(path))["ended"] != "planner_error":
                    continue
                reply = env.call("reset", {"episode": idx})
                obs, info = reply["obs"], reply["info"]
                scene = os.path.basename(os.path.normpath(info["scene_id"])).split(".")[0]
                print(f"== ep {idx} ({scene}) {mode}: {len(info['subtasks'])} subtasks", flush=True)
                if policy is None:
                    policy = build_policy(args, info)
                system = None
                if mode == "landmark":
                    with open(os.path.join(args.maps, scene, "landmark_map.md")) as f:
                        system = planner_system(f.read(), CAMERA)
                    cl = cl or make_client()
                run = Run(env, policy, args, obs, info)
                results, ended_ep = [], "done"
                for k, sub in enumerate(info["subtasks"]):
                    if sub["text"] is None:
                        results.append({"subtask": k, **sub, "ended": "skipped"})
                        continue
                    run.steps = 0
                    geo = env.call("begin", {"subtask": k})["geodesic_m"]
                    run.event("subtask", k=k, type=sub["type"], text=sub["text"], geodesic_m=round(geo, 2))
                    try:
                        ended = run.zeroshot(sub["text"]) if mode == "zeroshot" else run.landmark(cl, system, sub["text"])
                    except PlannerError as exc:
                        run.event("planner_error", error=str(exc))
                        ended_ep = "planner_error"
                        break
                    score = env.call("score", {"subtask": k})
                    score.pop("status", None)
                    results.append({"subtask": k, **sub, "ended": ended, "steps": run.steps, **score})
                    print(f"   -> subtask {k} {sub['type']}: {'SUCCESS' if score['success'] else 'miss'} "
                          f"{score['distance_to_goal']:.2f} m, SPL {score['spl']:.2f}, {run.steps} steps, {ended}", flush=True)
                out = {"episode": idx, "scene": scene, "mode": mode, "ended": ended_ep,
                       "planner_s": round(run.planner_s, 1), "subtasks": results, "log": run.log}
                with open(path, "w") as f:
                    json.dump(out, f, indent=1, default=float)
    finally:
        env.close()
    summarize(args.out)


def summarize(run_dir: str):
    from glob import glob
    eps = [json.load(open(p)) for p in sorted(glob(os.path.join(run_dir, "ep*.json")))]
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
