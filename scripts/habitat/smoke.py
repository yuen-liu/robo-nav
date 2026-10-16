"""habitat-sim smoke test: load HM3D via the GOAT-Bench scene dataset config, place the agent at a
val_seen episode's start pose, save one RGB + one semantic frame, and report navmesh status.

usage: python scripts/habitat/smoke.py <episode .json.gz> <out_dir>
"""
import gzip
import json
import os
import sys

import numpy as np
from PIL import Image

import habitat_sim
from habitat_sim.utils.common import quat_from_coeffs

DATA = os.path.expanduser("~/robonav/data")
CONFIG = f"{DATA}/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
ep_file, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

ep = json.load(gzip.open(ep_file))["episodes"][0]
scene = os.path.normpath(f"{DATA}/scene_datasets/{ep['scene_id']}")
print("episode", ep["episode_id"], "scene", scene)
print("start", ep["start_position"], ep["start_rotation"])

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_dataset_config_file = CONFIG
sim_cfg.scene_id = scene
sim_cfg.enable_physics = False
sim_cfg.gpu_device_id = 0

sensors = []
for uuid, kind in (("rgb", habitat_sim.SensorType.COLOR), ("semantic", habitat_sim.SensorType.SEMANTIC)):
    s = habitat_sim.CameraSensorSpec()
    s.uuid, s.sensor_type = uuid, kind
    s.resolution = [480, 640]
    s.position = [0.0, 1.31, 0.0]
    s.hfov = 79
    sensors.append(s)
agent_cfg = habitat_sim.agent.AgentConfiguration(sensor_specifications=sensors)

sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
mm = sim.metadata_mediator
print("active dataset:", mm.active_dataset)
print("scene handles in dataset:", len(mm.get_scene_handles()))
print("current scene:", sim.curr_scene_name)

pf = sim.pathfinder
print(f"navmesh is_loaded={pf.is_loaded} navigable_area={pf.navigable_area:.2f} m^2")
start = np.array(ep["start_position"], dtype=np.float32)
print("start is_navigable:", pf.is_navigable(start))

state = habitat_sim.AgentState()
state.position = start
state.rotation = quat_from_coeffs(ep["start_rotation"])  # [x, y, z, w]
sim.get_agent(0).set_state(state)

obs = sim.get_sensor_observations()
rgb = obs["rgb"][..., :3]
sem = obs["semantic"].astype(np.int64)
Image.fromarray(rgb).save(f"{out_dir}/rgb.png")
ids = np.unique(sem)
palette = np.random.RandomState(0).randint(0, 255, size=(int(sem.max()) + 1, 3), dtype=np.uint8)
palette[0] = 0
Image.fromarray(palette[sem]).save(f"{out_dir}/semantic.png")
print(f"rgb mean={rgb.mean():.1f} std={rgb.std():.1f}")
print(f"semantic unique ids in frame={len(ids)} semantic_scene objects={len(sim.semantic_scene.objects)}")
names = {o.semantic_id: o.category.name() for o in sim.semantic_scene.objects if o is not None}
print("labels in frame:", sorted({names.get(int(i), "?") for i in ids if i != 0})[:20])
print("saved", f"{out_dir}/rgb.png", f"{out_dir}/semantic.png")
sim.close()
