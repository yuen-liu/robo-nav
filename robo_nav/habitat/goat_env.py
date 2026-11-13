"""
GOAT-Bench (HM3D v0.2 val) environment for LightNav-0, on raw habitat-sim, served over ZeroMQ.

Runs in the habitat env (python 3.9, habitat-sim; see LightNav-0/habitat_server/environment.yml).
The controller (`robo_nav.habitat.goat_nav`) runs in LightNav-0's venv and talks to it with the
same pickle REQ/REP wire format as LightNav-0's habitat_server, plus a few GOAT commands:

  reset        {"episode": i}       load the scene if needed, put the agent at the episode start
  step         velocity_control     one LightNav-0 action (dt = 1 s)
  begin        {"subtask": k}       start scoring subtask k from the current pose (GOAT is lifelong:
                                    each subtask starts where the previous one ended)
  score        {"subtask": k}       GT result for subtask k at the current pose (success, SPL, per-step track)
  teleport     {"position", "heading"}  move the robot (single-subtask runs from a fixed start)
  pose
  close

`serve --shared` serves several clients on one sim (one pose per client; scene switches when all wait).
`--dataset r2r|rxr` serves VLN-CE episodes on MP3D as one-subtask episodes (3 m success).

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
VLN_SUCCESS_DISTANCE = 3.0  # m to the path's end (VLN-CE standard)
# VLN-CE (MP3D) episodes as one-subtask GOAT episodes; --split picks the file. RxR: English guide only.
VLNCE = {"r2r": "datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz",
         "rxr": "datasets/RxR_VLNCE_v0/{split}/{split}_guide.json.gz"}
VLN_PROMPT = "Follow these directions and stop where they end: "


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


def load_vlnce(dataset: str, split: str, data: str = DATA) -> List[dict]:
    """VLN-CE episodes, scene-sorted, each as a GOAT-style episode with a single 'vln' subtask whose
    goal is the path's end point (scored at VLN_SUCCESS_DISTANCE, geodesic)."""
    with gzip.open(os.path.join(data, VLNCE[dataset].format(split=split)), "rt") as f:
        eps = json.load(f)["episodes"]
    if dataset == "rxr":
        eps = [e for e in eps if e["instruction"]["language"].startswith("en")]
    out = []
    for e in sorted(eps, key=lambda e: (e["scene_id"], e["episode_id"])):
        e = dict(e)
        e["tasks"] = [["path_end", "vln", None]]
        e["_goals"] = {}
        out.append(e)
    return out


def subtask_goals(ep: dict, task: list) -> List[dict]:
    """Goal objects a subtask accepts: every instance of the category for an object goal,
    the referenced instance for a description / image goal."""
    category, kind, ref = task[0], task[1], task[2]
    if kind == "vln":
        return [{"view_points": [{"position": ep["goals"][0]["position"]}]}]
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
    if kind == "vln":
        return VLN_PROMPT + " ".join(ep["instruction"]["instruction_text"].split())
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
                 navmesh_cell_height: Optional[float] = None, dataset: str = "goat"):
        self.dataset = dataset
        self.episodes = load_episodes(split, data) if dataset == "goat" else load_vlnce(dataset, split, data)
        self.success_distance = SUCCESS_DISTANCE if dataset == "goat" else VLN_SUCCESS_DISTANCE
        self.split, self.gpu_id, self.data = split, gpu_id, data
        self.navmesh_cell_height = navmesh_cell_height
        self.sim = None
        self.scene = None
        self.ep: Optional[dict] = None
        self.steps = 0
        self.sub: Dict = {}
        self.noise, self.rng = None, np.random.default_rng(0)  # set per episode by reset()

    # -- simulator ------------------------------------------------------------------------------
    def _config(self, scene: str):
        import habitat_sim
        sim_cfg = habitat_sim.SimulatorConfiguration()
        if self.dataset == "goat":  # MP3D (VLN-CE) loads its .glb + .navmesh with the default config
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
    def reset(self, episode: int, noise: Optional[dict] = None) -> dict:
        """`noise` (optional, off by default): {"lin": sd, "ang_deg": sd, "seed": int}, Gaussian actuation
        noise on every moving step -- linear speed x (1 + N(0, lin)), turn rate + N(0, ang_deg) deg/s --
        seeded per (seed, episode) so a noisy run is reproducible."""
        self.noise = dict(noise) if noise else None
        self.rng = np.random.default_rng([int((noise or {}).get("seed", 0)), episode])
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
            if self.sub:
                self._track()
            return {"obs": self._obs(), "info": {"stopped": True, "collided": False, "steps": self.steps}}
        if self.noise:
            lin = max(0.0, lin * (1 + self.rng.normal(0, self.noise.get("lin", 0.0))))
            ang += self.rng.normal(0, self.noise.get("ang_deg", 0.0))
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
            self._track()
        return {"obs": self._obs(), "info": {"stopped": False, "collided": bool(moved < wanted - 1e-3),
                                             "steps": self.steps}}

    def teleport(self, position: List[float], heading: float) -> dict:
        """Put the robot at `position` facing `heading` (rad about +y, as pose()) in the current episode's
        scene -- for single-subtask runs from a fixed start (goat_nav --goals)."""
        import habitat_sim
        from habitat_sim.utils.common import quat_from_angle_axis
        state = habitat_sim.AgentState()
        state.position = np.asarray(position, dtype=np.float32)
        state.rotation = quat_from_angle_axis(float(heading), np.array([0.0, 1.0, 0.0]))
        self.sim.get_agent(0).set_state(state)
        return {"obs": self._obs(), "pose": self.pose()}

    def _track(self):
        """Per-step [distance to goal, path so far] of the open subtask, so a run can be rescored at any
        smaller step budget (the agent never sees its budget, so its first B steps don't depend on it)."""
        self.sub["track"].append([round(self._geodesic(self.pose()["position"], self.sub["points"]), 3),
                                  round(self.sub["path"], 3)])

    def begin(self, subtask: int) -> dict:
        pts = view_points(subtask_goals(self.ep, self.ep["tasks"][subtask]))
        start = self.pose()["position"]
        self.sub = {"k": subtask, "points": pts, "start": start, "last": start, "path": 0.0,
                    "geo": self._geodesic(start, pts), "track": []}
        return {"geodesic_m": self.sub["geo"]}

    def score(self, subtask: int) -> dict:
        assert self.sub.get("k") == subtask, f"score({subtask}) without begin({subtask})"
        dtg = self._geodesic(self.pose()["position"], self.sub["points"])
        success = dtg < self.success_distance  # strict, as GoatSuccess
        geo, path = self.sub["geo"], self.sub["path"]
        spl = float(success) * geo / max(path, geo) if math.isfinite(geo) and geo > 0 else float(success)
        return {"success": bool(success), "distance_to_goal": dtg, "spl": spl, "path_m": path,
                "geodesic_m": geo, "pose": self.pose(), "success_distance": self.success_distance,
                "track": self.sub["track"]}

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
                "score": lambda d: env.score(**d), "pose": lambda d: {"pose": env.pose()},
                "teleport": lambda d: env.teleport(**d)}
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


SLOT_ATTRS = ("ep", "steps", "sub", "noise", "rng")


def serve_shared(env: GoatEnv, address: str):
    """Like serve(), but for several clients at once on one simulator (GPUs in Exclusive_Process mode fit one
    sim process each, and runs are bound by planner latency, not rendering). Each client (REQ socket) is a
    slot with its own episode state and robot pose, swapped into the sim around every command. All slots
    share the loaded scene: a reset that needs another scene is held until every live slot is waiting for a
    new scene too, then the scene most of them want loads -- so give clients goal lists in the same scene
    order. A client's 'close' ends its slot; the server exits when the last slot has closed."""
    import zmq
    import habitat_sim
    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(address)
    logger.info("GOAT env on %s (%d episodes, split %s, shared)", address, len(env.episodes), env.split)
    slots: Dict[bytes, dict] = {}  # client id -> saved state
    held: Dict[bytes, dict] = {}  # client id -> reset waiting for another scene
    seen_any = False

    def reply(cid, msg):
        sock.send_multipart([cid, b"", pickle.dumps(msg, protocol=4)])

    def scene_of(d):
        ep = env.episodes[d["episode"]]
        return os.path.normpath(os.path.join(env.data, "scene_datasets", ep["scene_id"]))

    def run(cid, cmd, data):
        st = slots.setdefault(cid, {})
        for a in SLOT_ATTRS:
            setattr(env, a, st.get(a, getattr(GoatEnv, a, None) if a != "rng" else np.random.default_rng(0)))
        if "agent" in st and cmd != "reset":
            env.sim.get_agent(0).set_state(st["agent"], reset_sensors=False)
        handlers = {"reset": lambda d: env.reset(**d), "step": env.step, "begin": lambda d: env.begin(**d),
                    "score": lambda d: env.score(**d), "pose": lambda d: {"pose": env.pose()},
                    "teleport": lambda d: env.teleport(**d)}
        try:
            if cmd not in handlers:
                raise ValueError(f"unknown command {cmd!r}")
            out = {"status": "success", **handlers[cmd](data)}
        except Exception as exc:
            logger.exception("%s failed", cmd)
            out = {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
        for a in SLOT_ATTRS:
            st[a] = getattr(env, a, None)
        if env.sim is not None:
            st["agent"] = env.sim.get_agent(0).get_state()
        reply(cid, out)

    def release():
        """Load the next scene once every live slot is held, and serve the resets that want it."""
        if not held or any(c not in held for c in slots):
            return
        want: Dict[str, int] = {}
        for d in held.values():
            want[scene_of(d)] = want.get(scene_of(d), 0) + 1
        scene = max(sorted(want), key=lambda k: want[k])
        logger.info("all %d slots waiting: loading %s for %d of them", len(slots), os.path.basename(scene), want[scene])
        for cid in [c for c, d in held.items() if scene_of(d) == scene]:
            run(cid, "reset", held.pop(cid))

    try:
        while True:
            cid, _, raw = sock.recv_multipart()
            msg = pickle.loads(raw)
            cmd, data = msg.get("command"), msg.get("data")
            seen_any = True
            if cmd == "close":
                slots.pop(cid, None)
                held.pop(cid, None)
                reply(cid, {"status": "success"})
                if not slots:
                    break
                release()
                continue
            slots.setdefault(cid, {})
            if cmd == "reset" and env.scene is not None and scene_of(data) != env.scene:
                held[cid] = data
                release()
                continue
            run(cid, cmd, data)
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
        p.add_argument("--dataset", default="goat", choices=["goat", *VLNCE],
                       help="goat (HM3D GOAT-Bench) or a VLN-CE set on MP3D, one 'vln' subtask per episode")
        p.add_argument("--gpu", type=int, default=0)
        p.add_argument("--data", default=DATA)
        p.add_argument("--navmesh_cell_height", type=float, default=None,
                       help="Re-bake each scene's navmesh (0.05 if episodes sit below the shipped mesh)")
        if name == "serve":
            p.add_argument("--port", type=int, default=5560)
            p.add_argument("--shared", action="store_true", help="serve several clients on one sim (serve_shared)")
        else:
            p.add_argument("--out", required=True, help="Writes <out>/<scene>/frames/")
            p.add_argument("--scenes", nargs="*", help="Scene ids (default: all in the split)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    env = GoatEnv(args.split, args.gpu, args.data, args.navmesh_cell_height, args.dataset)
    if args.cmd == "serve":
        (serve_shared if args.shared else serve)(env, f"tcp://127.0.0.1:{args.port}")
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
