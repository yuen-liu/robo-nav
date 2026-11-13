"""
Difficulty features for GOAT subtasks from where each one actually started (habitat env, standalone like
goat_env):

  geodesic_m      shortest-path distance from the start pose to the nearest goal view point
  visible_start   some goal instance's center is in line of sight from the start: a 360 deg depth sweep
                  (8 x 45 deg, the robot's camera) sees it unoccluded within VISIBLE_MAX_M
  doorways        narrow spots crossed by the shortest path: stretches <= DOOR_MAX_LEN_M long where the
                  navmesh is < DOOR_WIDTH_M wide across the path (HM3D has no room labels; doorways stand in)
  floor_change    |height of the path's end - start height| > FLOOR_DY_M (another floor)
  dy_m            that height difference

Start poses: a run's previous scored subtask end pose (episode start for the first), per mode -- or the
fixed starts of a --goals file.

  python robo_nav/habitat/goat_difficulty.py --runs sim_data/goat/runs/val_unseen --out sim_data/goat/difficulty_val_unseen.json
  python robo_nav/habitat/goat_difficulty.py --goals sim_data/goat/far_goals_10m.json --out sim_data/goat/difficulty_far_10m.json
"""

import argparse
import json
import math
import os
import sys
from glob import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import goat_env  # noqa: E402

VISIBLE_MAX_M = 10.0
VIS_TOL_M = 0.6      # depth may stop this much short of the center (the object's own surface)
DOOR_WIDTH_M = 0.42  # navmesh width (eroded by the 0.18 m agent radius): ~0.8 m wall to wall; doors ~0.3, halls ~0.5+
DOOR_MERGE_M = 1.0   # narrow spots closer than this are one doorway
DOOR_MAX_LEN_M = 1.2
SAMPLE_M = 0.1
FLOOR_DY_M = 1.0


def starts_from_runs(run_dirs):
    """(episode, subtask, position|None, heading|None, mode, run) for every scored subtask."""
    out = []
    for d in run_dirs:
        for p in sorted(glob(os.path.join(d, "ep*.json"))):
            ep = json.load(open(p))
            if ep["ended"] == "planner_error":
                continue
            pos, head = None, None
            for s in ep["subtasks"]:
                if s["ended"] == "skipped":
                    continue
                out.append((ep["episode"], s["subtask"], pos, head, ep["mode"], os.path.basename(os.path.normpath(d))))
                pos, head = s["pose"]["position"], s["pose"]["heading"]
    return out


class Probe:
    def __init__(self, gpu: int):
        self.env = goat_env.GoatEnv("val_unseen", gpu, navmesh_cell_height=0.05)
        self.env._config = self._config_with_depth(self.env._config)

    @staticmethod
    def _config_with_depth(base):
        def cfg(scene):
            import habitat_sim
            c = base(scene)
            d = habitat_sim.CameraSensorSpec()
            d.uuid, d.sensor_type = "depth", habitat_sim.SensorType.DEPTH
            d.resolution = list(goat_env.IMAGE_HW)
            d.position = [0.0, goat_env.CAMERA_HEIGHT, 0.0]
            d.hfov = goat_env.HFOV
            c.agents[0].sensor_specifications.append(d)
            return c
        return cfg

    def place(self, episode, pos, heading):
        self.env.reset(episode)
        if pos is not None:
            self.env.teleport(pos, heading)
        return self.env.pose()

    def visible(self, goals) -> bool:
        import habitat_sim
        from habitat_sim.utils.common import quat_from_angle_axis
        agent = self.env.sim.get_agent(0)
        st0 = agent.get_state()
        centers = [np.asarray(g["position"], dtype=np.float64) for g in goals if "position" in g]
        h, w = goat_env.IMAGE_HW
        fx = (w / 2) / math.tan(math.radians(goat_env.HFOV) / 2)
        cam = np.asarray(st0.position, dtype=np.float64) + [0, goat_env.CAMERA_HEIGHT, 0]
        base = self.env.pose()["heading"]
        seen = False
        for k in range(8):
            yaw = base + k * math.pi / 4
            st = habitat_sim.AgentState()
            st.position, st.rotation = st0.position, quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
            agent.set_state(st)
            depth = self.env.sim.get_sensor_observations()["depth"]
            for c in centers:
                v = c - cam  # world -> camera (habitat: camera looks down -z, y up)
                cy, sy = math.cos(-yaw), math.sin(-yaw)
                x, z = cy * v[0] + sy * v[2], -sy * v[0] + cy * v[2]
                y = v[1]
                if z >= -0.1:
                    continue  # behind
                zc = -z
                u, r = w / 2 + fx * x / zc, h / 2 - fx * y / zc
                if not (0 <= u < w and 0 <= r < h):
                    continue
                dist = float(np.linalg.norm(v))
                if zc > VISIBLE_MAX_M:
                    continue
                dz = float(depth[int(r), int(u)])  # z-depth along the optical axis
                if dz <= 0 or dz >= zc - VIS_TOL_M:
                    seen = True
            if seen:
                break
        agent.set_state(st0)
        return seen

    def path(self, start, targets):
        import habitat_sim
        p = habitat_sim.MultiGoalShortestPath()
        p.requested_start = np.asarray(start, dtype=np.float32)
        p.requested_ends = [np.asarray(t, dtype=np.float32) for t in targets]
        ok = self.env.sim.pathfinder.find_path(p)
        return (float(p.geodesic_distance), [np.asarray(q) for q in p.points]) if ok else (math.inf, [])

    def width(self, pt, d) -> float:
        """Navmesh width across horizontal direction d at pt (no sliding), capped at 2 x 1.5 m."""
        pf = self.env.sim.pathfinder
        n = np.array([-d[2], 0.0, d[0]])
        tot = 0.0
        for s in (1, -1):
            end = pf.try_step_no_sliding(pt, pt + s * 1.5 * n)
            tot += float(np.linalg.norm((np.asarray(end) - pt)[[0, 2]]))
        return tot

    def doorways(self, pts) -> int:
        samples = []  # (arc length, narrow?)
        acc = 0.0
        for a, b in zip(pts, pts[1:]):
            seg = b - a
            L = float(np.linalg.norm(seg[[0, 2]]))
            if L < 1e-3:
                continue
            d = seg / max(np.linalg.norm(seg), 1e-6)
            d = np.array([d[0], 0.0, d[2]]) / max(np.linalg.norm([d[0], d[2]]), 1e-6)
            for t in np.arange(0.0, L, SAMPLE_M):
                samples.append((acc + t, self.width(a + seg * (t / L), d) < DOOR_WIDTH_M))
            acc += L
        total = acc
        runs, run_start = [], None
        for s, narrow in samples + [(total, False)]:
            if narrow and run_start is None:
                run_start = s
            elif not narrow and run_start is not None:
                if runs and run_start - runs[-1][1] < DOOR_MERGE_M:
                    runs[-1][1] = s
                else:
                    runs.append([run_start, s])
                run_start = None
        return sum(1 for a, b in runs if b - a <= DOOR_MAX_LEN_M and a > 0.3 and b < total - 0.3)

    def features(self, episode, subtask, pos, heading) -> dict:
        start = self.place(episode, pos, heading)["position"]
        ep = self.env.ep
        task = ep["tasks"][subtask]
        goals = goat_env.subtask_goals(ep, task)
        geo, pts = self.path(start, goat_env.view_points(goals))
        if not pts:
            return {"geodesic_m": geo, "visible_start": None, "doorways": None, "floor_change": None, "dy_m": None}
        dy = float(pts[-1][1] - start[1])
        return {"geodesic_m": round(geo, 3), "visible_start": self.visible(goals), "doorways": self.doorways(pts),
                "floor_change": abs(dy) > FLOOR_DY_M, "dy_m": round(dy, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--goals")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = starts_from_runs(a.runs)
    if a.goals:
        items += [(g["episode"], g["subtask"], g["position"], g["heading"], "fixed", os.path.basename(a.goals))
                  for g in json.load(open(a.goals))]
    items.sort(key=lambda t: (t[0], t[1]))  # episodes are scene-sorted: one scene load each
    items = items[:a.limit] if a.limit else items
    probe, cache, rows = Probe(a.gpu), {}, []
    for i, (e, k, pos, head, mode, run) in enumerate(items):
        key = (e, k, None if pos is None else tuple(round(v, 3) for v in pos), None if head is None else round(head, 3))
        if key not in cache:
            cache[key] = probe.features(e, k, pos, head)
        rows.append({"run": run, "mode": mode, "episode": e, "subtask": k, **cache[key]})
        if i % 200 == 0:
            print(f"{i}/{len(items)}", flush=True)
    probe.env.close()
    json.dump(rows, open(a.out, "w"), indent=0)
    print(f"{len(rows)} rows ({len(cache)} distinct starts) -> {a.out}")


if __name__ == "__main__":
    main()
