"""
GOAT-Bench (HM3D v0.2 val) environment for LightNav-0, on raw habitat-sim, served over ZeroMQ.

Runs in the habitat env (python 3.9, habitat-sim; see LightNav-0/habitat_server/environment.yml).
The controller (`robo_nav.habitat.goat_nav`) runs in LightNav-0's venv and talks to it with the
same pickle REQ/REP wire format as LightNav-0's habitat_server, plus a few GOAT commands:

  reset        {"episode": i}       load the scene if needed, put the agent at the episode start
  step         velocity_control     one LightNav-0 action (dt = 1 s)
  begin        {"subtask": k}       start scoring subtask k from the current pose (GOAT is lifelong:
                                    each subtask starts where the previous one ended)
  score        {"subtask": k}       GT result for subtask k at the current pose (success, SPL)
  close

Agent, camera and velocity control match LightNav-0's HM3D ObjectNav/OVON configs: 0.88 m camera,
480x270, 120 deg hfov, radius 0.18 m, 0.25 m/s and 30 deg/s max at dt = 1 s, sliding on the navmesh.
GT (pose, goal distances) only goes into `score`/logs, never into observations.

Also renders each scene's walkthrough tour -- the "house video" `robo_nav.landmark.annotate` turns
into a landmark map. The tour is planned on the navmesh (like the MuJoCo mapping tour); the map
built from it is all the planner ever gets.

Standalone on purpose (importing the robo_nav package pulls torch/cv2, which the habitat env lacks):
  python robo_nav/habitat/goat_env.py serve --split val_seen --port 5560
  python robo_nav/habitat/goat_env.py tour --split val_seen --out ~/robonav/data/goat_tours
"""

import argparse
import gzip
import json
import logging
import math
import os
import pickle
from glob import glob
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DATA = os.path.expanduser("~/robonav/data")
SCENE_CONFIG = "scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
EPISODES = "datasets/goat_bench/hm3d/v1/{split}/content"

CAMERA_HEIGHT = 0.88
IMAGE_HW = (270, 480)
HFOV = 120.0
AGENT_RADIUS, AGENT_HEIGHT = 0.18, 0.88
DT = 1.0
LIN_VEL_RANGE = (0.0, 0.25)  # m/s
ANG_VEL_RANGE = (-30.0, 30.0)  # deg/s
MIN_ABS_LIN, MIN_ABS_ANG = 0.025, 1.0  # below both = the policy's STOP
SUCCESS_DISTANCE = 0.25  # m to the nearest goal view point (GOAT-Bench / OVON standard)


def load_episodes(split: str, data: str = DATA) -> List[dict]:
    """All episodes of a split, scene-sorted; each carries its scene's `goals` dict."""
    episodes = []
    for path in sorted(glob(os.path.join(data, EPISODES.format(split=split), "*.json.gz"))):
        with gzip.open(path, "rt") as f:
            content = json.load(f)
        for ep in content["episodes"]:
            ep["_goals"] = content["goals"]
            episodes.append(ep)
    if not episodes:
        raise SystemExit(f"no episodes under {os.path.join(data, EPISODES.format(split=split))}")
    return episodes


def subtask_goals(ep: dict, task: list) -> List[dict]:
    """Goal objects a subtask accepts: every instance of the category for an object goal,
    the referenced instance for a description / image goal."""
    category, kind, ref = task[0], task[1], task[2]
    scene_file = os.path.basename(os.path.normpath(ep["scene_id"]))
    goals = ep["_goals"].get(f"{scene_file}_{category}", [])
    if kind != "object":
        goals = [g for g in goals if g.get("object_id") == ref]
    if not goals:
        raise KeyError(f"no goal for subtask {task} in {scene_file}")
    return goals


def subtask_text(ep: dict, task: list) -> Optional[str]:
    """What the robot is asked: the category for an object goal, the description for a language
    goal; None for image goals (not supported yet)."""
    category, kind = task[0], task[1]
    if kind == "object":
        return f"Find the {category.replace('_', ' ')}."
    if kind == "description":
        goal = subtask_goals(ep, task)[0]
        text = goal.get("lang_desc") or goal.get("description")
        if not text:
            raise KeyError(f"description goal {task[2]} has no lang_desc; goal keys: {sorted(goal)}")
        return text
    return None


def scene_name(ep: dict) -> str:
    """'4ok3usBNeis' from 'hm3d/val//00877-4ok3usBNeis/4ok3usBNeis.basis.glb'."""
    return os.path.basename(os.path.normpath(ep["scene_id"])).split(".")[0]


def view_points(goals: List[dict]) -> List[List[float]]:
    pts = []
    for g in goals:
        for vp in g.get("view_points", []):
            pts.append((vp.get("agent_state") or vp)["position"])
    return pts


class GoatEnv:
    def __init__(self, split: str, gpu_id: int = 0, data: str = DATA,
                 navmesh_cell_height: Optional[float] = None):
        self.episodes = load_episodes(split, data)
        self.split, self.gpu_id, self.data = split, gpu_id, data
        self.navmesh_cell_height = navmesh_cell_height
        self.sim = None
        self.scene = None
        self.ep: Optional[dict] = None
        self.steps = 0
        self.sub: Dict = {}

    # -- simulator ------------------------------------------------------------------------------
    def _config(self, scene: str):
        import habitat_sim
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_dataset_config_file = os.path.join(self.data, SCENE_CONFIG)
        sim_cfg.scene_id = scene
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = self.gpu_id
        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid, rgb.sensor_type = "rgb", habitat_sim.SensorType.COLOR
        rgb.resolution = list(IMAGE_HW)
        rgb.position = [0.0, CAMERA_HEIGHT, 0.0]
        rgb.hfov = HFOV
        agent = habitat_sim.agent.AgentConfiguration(height=AGENT_HEIGHT, radius=AGENT_RADIUS,
                                                     sensor_specifications=[rgb])
        return habitat_sim.Configuration(sim_cfg, [agent])

    def _load_scene(self, scene: str):
        import habitat_sim
        if scene == self.scene:
            return
        cfg = self._config(scene)
        if self.sim is None:
            self.sim = habitat_sim.Simulator(cfg)
        else:
            self.sim.reconfigure(cfg)
        if self.navmesh_cell_height is not None:  # see LightNav-0 objectnav.py: v2-style episodes
            s = habitat_sim.nav.NavMeshSettings()
            s.set_defaults()
            s.agent_radius, s.agent_height, s.cell_height = AGENT_RADIUS, AGENT_HEIGHT, self.navmesh_cell_height
            ok = self.sim.recompute_navmesh(self.sim.pathfinder, s)
            logger.info("rebaked navmesh for %s at cell_height=%.2f: ok=%s", scene, self.navmesh_cell_height, ok)
        self.scene = scene
        self.vel = habitat_sim.physics.VelocityControl()
        self.vel.controlling_lin_vel = self.vel.lin_vel_is_local = True
        self.vel.controlling_ang_vel = self.vel.ang_vel_is_local = True

    def _obs(self) -> dict:
        return {"rgb": self.sim.get_sensor_observations()["rgb"][..., :3]}

    def pose(self) -> dict:
        st = self.sim.get_agent(0).get_state()
        q = st.rotation
        heading = math.atan2(2 * (q.w * q.y + q.x * q.z), 1 - 2 * (q.y * q.y + q.z * q.z))
        return {"position": [float(v) for v in st.position], "heading": heading}

    def _geodesic(self, src, targets) -> float:
        import habitat_sim
        path = habitat_sim.MultiGoalShortestPath()
        path.requested_start = np.asarray(src, dtype=np.float32)
        path.requested_ends = [np.asarray(t, dtype=np.float32) for t in targets]
        return float(path.geodesic_distance) if self.sim.pathfinder.find_path(path) else math.inf

    # -- commands -------------------------------------------------------------------------------
    def reset(self, episode: int) -> dict:
        from habitat_sim.utils.common import quat_from_coeffs
        import habitat_sim
        self.ep = self.episodes[episode]
        self._load_scene(os.path.normpath(os.path.join(self.data, "scene_datasets", self.ep["scene_id"])))
        state = habitat_sim.AgentState()
        state.position = np.asarray(self.ep["start_position"], dtype=np.float32)
        state.rotation = quat_from_coeffs(self.ep["start_rotation"])
        self.sim.get_agent(0).set_state(state)
        self.steps, self.sub = 0, {}
        subtasks = [{"category": t[0], "type": t[1], "text": subtask_text(self.ep, t)} for t in self.ep["tasks"]]
        info = {"episode_index": episode, "episode_id": self.ep["episode_id"], "scene_id": self.ep["scene_id"],
                "subtasks": subtasks, "habitat_time_step": DT, "lin_vel_range": LIN_VEL_RANGE,
                "ang_vel_range": ANG_VEL_RANGE, "pose": self.pose()}
        return {"obs": self._obs(), "info": info}

    def step(self, action: dict) -> dict:
        from habitat_sim.utils.common import quat_from_magnum, quat_to_magnum
        import habitat_sim
        args = action["action_args"]
        lin = LIN_VEL_RANGE[0] + (float(args["linear_velocity"]) + 1) / 2 * (LIN_VEL_RANGE[1] - LIN_VEL_RANGE[0])
        ang = ANG_VEL_RANGE[0] + (float(args["angular_velocity"]) + 1) / 2 * (ANG_VEL_RANGE[1] - ANG_VEL_RANGE[0])
        self.steps += 1
        if abs(lin) < MIN_ABS_LIN and abs(ang) < MIN_ABS_ANG:
            return {"obs": self._obs(), "info": {"stopped": True, "collided": False, "steps": self.steps}}
        # Same integration as habitat-lab's VelocityAction (no physics, sliding on the navmesh).
        agent = self.sim.get_agent(0)
        st = agent.get_state()
        self.vel.linear_velocity = np.array([0.0, 0.0, -lin])
        self.vel.angular_velocity = np.array([0.0, math.radians(ang), 0.0])
        current = habitat_sim.RigidState(quat_to_magnum(st.rotation.normalized()), st.position)
        goal = self.vel.integrate_transform(DT, current)
        final = np.array(self.sim.pathfinder.try_step(st.position, goal.translation), dtype=np.float32)
        moved = np.linalg.norm(final - st.position)
        wanted = np.linalg.norm(np.array(goal.translation, dtype=np.float32) - st.position)
        new = habitat_sim.AgentState()
        new.position = final
        new.rotation = quat_from_magnum(goal.rotation)
        agent.set_state(new, reset_sensors=False)
        if self.sub:
            self.sub["path"] += float(np.linalg.norm(final - np.asarray(self.sub["last"])))
            self.sub["last"] = [float(v) for v in final]
        return {"obs": self._obs(), "info": {"stopped": False, "collided": bool(moved < wanted - 1e-3),
                                             "steps": self.steps}}

    def begin(self, subtask: int) -> dict:
        pts = view_points(subtask_goals(self.ep, self.ep["tasks"][subtask]))
        start = self.pose()["position"]
        self.sub = {"k": subtask, "points": pts, "start": start, "last": start, "path": 0.0,
                    "geo": self._geodesic(start, pts)}
        return {"geodesic_m": self.sub["geo"]}

    def score(self, subtask: int) -> dict:
        assert self.sub.get("k") == subtask, f"score({subtask}) without begin({subtask})"
        dtg = self._geodesic(self.pose()["position"], self.sub["points"])
        success = dtg <= SUCCESS_DISTANCE
        geo, path = self.sub["geo"], self.sub["path"]
        spl = float(success) * geo / max(path, geo) if math.isfinite(geo) and geo > 0 else float(success)
        return {"success": bool(success), "distance_to_goal": dtg, "spl": spl, "path_m": path,
                "geodesic_m": geo, "pose": self.pose()}

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None


def serve(env: GoatEnv, address: str):
    import zmq
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(address)
    logger.info("GOAT env on %s (%d episodes, split %s)", address, len(env.episodes), env.split)
    handlers = {"reset": lambda d: env.reset(**d), "step": env.step, "begin": lambda d: env.begin(**d),
                "score": lambda d: env.score(**d)}
    try:
        while True:
            msg = pickle.loads(sock.recv())
            cmd = msg.get("command")
            if cmd == "close":
                sock.send(pickle.dumps({"status": "success"}, protocol=4))
                break
            try:
                if cmd not in handlers:
                    raise ValueError(f"unknown command {cmd!r}")
                reply = {"status": "success", **handlers[cmd](msg.get("data"))}
            except Exception as exc:  # report to the client; keep serving
                logger.exception("%s failed", cmd)
                reply = {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
            sock.send(pickle.dumps(reply, protocol=4))
    finally:
        env.close()
        sock.close()
        ctx.term()


def render_tour(env: GoatEnv, episode: int, out_dir: str, n_points: int = 40, min_sep: float = 2.0,
                step_m: float = 0.25, seed: int = 0) -> int:
    """Walk the navmesh island of the episode's start: visit spread-out navigable points in
    nearest-neighbour order along shortest paths, facing the direction of travel, with a slow
    360 at each point. Saves f00000.jpg ... (camera frames only) and returns the frame count."""
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis
    from PIL import Image
    env.reset(episode)
    pf = env.sim.pathfinder
    pf.seed(seed)
    start = np.asarray(env.ep["start_position"], dtype=np.float32)
    island = pf.get_island(start)
    pts = [pf.snap_point(start)]
    for _ in range(n_points * 50):
        if len(pts) > n_points:
            break
        p = pf.get_random_navigable_point(island_index=island)
        if np.all(np.linalg.norm(np.asarray(pts) - p, axis=1) >= min_sep):
            pts.append(p)
    order, left = [0], list(range(1, len(pts)))
    while left:
        last = pts[order[-1]]
        nxt = min(left, key=lambda i: np.linalg.norm(pts[i] - last))
        order.append(nxt)
        left.remove(nxt)

    os.makedirs(out_dir, exist_ok=True)
    agent = env.sim.get_agent(0)
    n = [0]

    def shoot(pos, yaw):
        st = habitat_sim.AgentState()
        st.position = np.asarray(pos, dtype=np.float32)
        st.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
        agent.set_state(st)
        Image.fromarray(env._obs()["rgb"]).save(os.path.join(out_dir, f"f{n[0]:05d}.jpg"), quality=90)
        n[0] += 1

    yaw = 0.0
    for a, b in zip(order, order[1:]):
        path = habitat_sim.ShortestPath()
        path.requested_start, path.requested_end = pts[a], pts[b]
        if not pf.find_path(path):
            continue
        for p0, p1 in zip(path.points, path.points[1:]):
            p0, p1 = np.asarray(p0), np.asarray(p1)
            seg = np.linalg.norm(p1 - p0)
            if seg < 1e-3:
                continue
            yaw = math.atan2(-(p1[0] - p0[0]), -(p1[2] - p0[2]))  # habitat: forward is -z
            for t in np.arange(0.0, seg, step_m):
                shoot(p0 + (p1 - p0) * t / seg, yaw)
        for k in range(1, 13):  # look around at the point
            shoot(pts[b], yaw + k * math.pi / 6)
    return n[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("serve", "tour"):
        p = sub.add_parser(name)
        p.add_argument("--split", default="val_seen", choices=["val_seen", "val_seen_synonyms", "val_unseen"])
        p.add_argument("--gpu", type=int, default=0)
        p.add_argument("--data", default=DATA)
        p.add_argument("--navmesh_cell_height", type=float, default=None,
                       help="Re-bake each scene's navmesh (0.05 if episodes sit below the shipped mesh)")
        if name == "serve":
            p.add_argument("--port", type=int, default=5560)
        else:
            p.add_argument("--out", required=True, help="Writes <out>/<scene>/frames/")
            p.add_argument("--scenes", nargs="*", help="Scene ids (default: all in the split)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    env = GoatEnv(args.split, args.gpu, args.data, args.navmesh_cell_height)
    if args.cmd == "serve":
        serve(env, f"tcp://127.0.0.1:{args.port}")
        return
    first = {}  # scene -> first episode index
    for i, ep in enumerate(env.episodes):
        first.setdefault(scene_name(ep), i)
    for scene, i in first.items():
        if args.scenes and scene not in args.scenes:
            continue
        out = os.path.join(args.out, scene, "frames")
        if os.path.isdir(out) and os.listdir(out):
            logger.info("%s: tour exists, skipping", scene)
            continue
        logger.info("%s: %d frames", scene, render_tour(env, i, out))
    env.close()


if __name__ == "__main__":
    main()
