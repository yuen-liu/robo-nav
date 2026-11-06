import math
import types
import unittest
from unittest import mock

import numpy as np

from robo_nav.habitat import goat_nav

LIN, ANG = (0.0, 0.25), (-30.0, 30.0)
INFO = {"lin_vel_range": LIN, "ang_vel_range": ANG, "habitat_time_step": 1.0}


class FakeEnv:
    """goat_env's velocity decoding on a plane, with a wall at x = 2 m."""

    def __init__(self):
        self.x, self.y, self.yaw, self.steps = 0.0, 0.0, 0.0, 0

    def call(self, command, data=None):
        assert command == "step"
        a = data["action_args"]
        lin = LIN[0] + (a["linear_velocity"] + 1) / 2 * (LIN[1] - LIN[0])
        ang = ANG[0] + (a["angular_velocity"] + 1) / 2 * (ANG[1] - ANG[0])
        self.steps += 1
        if abs(lin) < 0.025 and abs(ang) < 1.0:
            return {"obs": self.obs(), "info": {"stopped": True, "collided": False}}
        self.yaw += math.radians(ang)
        nx = self.x + lin * math.cos(self.yaw)
        collided = nx > 2.0
        self.x, self.y = min(nx, 2.0), self.y + lin * math.sin(self.yaw)
        return {"obs": self.obs(), "info": {"stopped": False, "collided": collided}}

    def obs(self):
        return {"rgb": np.zeros((4, 4, 3), np.uint8)}


def make_run(**args):
    env = FakeEnv()
    a = types.SimpleNamespace(**{"subtask_steps": 500, "max_direct_calls": 60, "effort": "medium",
                                   "max_planner_calls": 20, "max_driver_calls": 10, "instruction_steps": 60,
                                   "verify_arrival": 0, "final_approach": 0, **args})
    return goat_nav.Run(env, None, a, env.obs(), INFO), env


class TestDirect(unittest.TestCase):
    def test_forward_exact_distance(self):
        run, env = make_run()
        self.assertEqual(run.forward(1.1), "done")
        self.assertAlmostEqual(env.x, 1.1, places=6)
        self.assertEqual(run.steps, 5)  # 4 full-speed steps + 0.1 m

    def test_forward_blocked_by_wall(self):
        run, env = make_run()
        self.assertEqual(run.forward(3.0), "blocked")
        self.assertAlmostEqual(env.x, 2.0)

    def test_forward_respects_step_budget(self):
        run, env = make_run(subtask_steps=3)
        run.forward(3.0)
        self.assertEqual(run.steps, 3)

    def test_direct_loop(self):
        plans = iter([
            {"observation": "hall", "where_am_i": "unsure", "action": "forward", "amount": 9.0},
            {"observation": "wall", "where_am_i": "end", "action": "turn", "amount": 90},
            {"observation": "bed", "where_am_i": "by bed", "action": "stop", "amount": 0},
        ])
        seen = []

        def fake_ask(cl, content, schema, system, max_tokens, effort, usage):
            self.assertIs(schema, goat_nav.DIRECT_SCHEMA)
            seen.append(content)
            usage.append({"output": 1})
            return next(plans)

        run, env = make_run()
        with mock.patch.object(goat_nav, "ask_json", fake_ask), \
                mock.patch.object(goat_nav, "jpeg", lambda rgb: b""), \
                mock.patch.object(goat_nav, "image_block", lambda img: {"type": "text", "text": "img"}):
            ended = run.direct(None, goat_nav.direct_system(None), "Find the bed.")
        self.assertEqual(ended, "vlm_stop")
        self.assertAlmostEqual(env.x, 2.0)  # 9 m clamped to 3, then the wall
        self.assertAlmostEqual(math.degrees(env.yaw) % 360, 90.0)
        history = seen[2][1]["text"]
        self.assertIn("forward 3.00 m -> blocked", history)
        self.assertIn("turn +90 deg", history)

    def test_system_prompt_map(self):
        self.assertNotIn("Landmark map", goat_nav.direct_system(None)[0]["text"])
        self.assertIn("# Landmark map\n\nfridge", goat_nav.direct_system("fridge")[0]["text"])

    def test_arrive_prompt(self):
        from robo_nav.landmark.planner import planner_system
        old = planner_system("m", goat_nav.CAMERA)[0]["text"]
        self.assertIn("'arrived' only when the goal is right in front of the robot (within about a meter), and", old)
        self.assertEqual(old, planner_system("m", goat_nav.CAMERA, "old")[0]["text"])
        self.assertIn("arm's length", planner_system("m", goat_nav.CAMERA, "new")[0]["text"])
        self.assertIn("'stop' only when the robot is right next to the goal", goat_nav.direct_system(None, "new")[0]["text"])

    def test_landmark_vlm_driver(self):
        plans = iter([
            {"observation": "hall", "where_am_i": "unsure", "action": "instruct", "instruction": "Go to the wall.", "plan": []},
            {"observation": "wall", "where_am_i": "at wall", "action": "arrived", "instruction": "", "plan": []},
        ])
        moves = iter([
            {"observation": "hall", "action": "forward", "amount": 3.0},
            {"observation": "wall", "action": "done", "amount": 0},
        ])
        driver_contents = []

        def fake_ask(cl, content, schema, system, max_tokens, effort, usage):
            usage.append({"output": 1})
            if schema is goat_nav.DRIVER_SCHEMA:
                self.assertNotIn("Landmark map", system[0]["text"])
                driver_contents.append(content)
                return next(moves)
            return next(plans)

        run, env = make_run(max_planner_calls=20, max_driver_calls=10, instruction_steps=60)
        with mock.patch.object(goat_nav, "ask_json", fake_ask), \
                mock.patch.object(goat_nav, "jpeg", lambda rgb: b""), \
                mock.patch.object(goat_nav, "image_block", lambda img: {"type": "text", "text": "img"}):
            ended = run.landmark(None, [{"type": "text", "text": "planner"}], "Find the wall.", vlm_driver=True)
        self.assertEqual(ended, "planner_arrived")
        self.assertAlmostEqual(env.x, 2.0)
        self.assertEqual(driver_contents[0][0]["text"], "Instruction: Go to the wall.")
        self.assertIn("forward 3.00 m -> blocked", driver_contents[1][1]["text"])
        self.assertEqual([u.get("role") for u in run.usage], [None, "driver", "driver", None])
        self.assertEqual([e["how"] for e in run.log if e["event"] == "step_end"], ["stopped"])

    def test_driver_respects_instruction_budget(self):
        run, env = make_run(max_driver_calls=10)
        with mock.patch.object(goat_nav, "ask_json", lambda *a, usage, **k: usage.append({}) or
                               {"observation": "", "action": "forward", "amount": 3.0}), \
                mock.patch.object(goat_nav, "jpeg", lambda rgb: b""), \
                mock.patch.object(goat_nav, "image_block", lambda img: {}):
            env.x = -100.0  # no wall in reach
            self.assertEqual(run.drive_vlm(None, "Go.", budget=7), "step_limit")
        self.assertEqual(run.steps, 7)

    def _patched(self, fake_ask):
        return (mock.patch.object(goat_nav, "ask_json", fake_ask), mock.patch.object(goat_nav, "jpeg", lambda rgb: b""),
                mock.patch.object(goat_nav, "image_block", lambda img: {"type": "text", "text": "img"}))

    def test_verify_rejects_then_accepts(self):
        plans = iter([{"observation": "a mirror", "where_am_i": "x", "action": "arrived", "instruction": "", "plan": []},
                      {"observation": "another mirror", "where_am_i": "y", "action": "arrived", "instruction": "", "plan": []}])
        checks = iter([{"observation": "", "candidate": "mirror, view 0", "cues": "", "matches": False, "close_enough": True,
                        "reason": "this mirror is not above a sink"},
                       {"observation": "", "candidate": "mirror, view 2", "cues": "", "matches": True, "close_enough": True, "reason": ""}])
        seen = []

        def fake_ask(cl, content, schema, system, max_tokens, effort, usage):
            usage.append({"output": 1})
            seen.append(content)
            return next(checks) if schema is goat_nav.VERIFY_SCHEMA else next(plans)

        run, env = make_run(verify_arrival=2)
        p = self._patched(fake_ask)
        with p[0], p[1], p[2]:
            ended = run.landmark(None, [{"type": "text", "text": "planner"}], "mirror above the sink.")
        self.assertEqual(ended, "planner_arrived")
        self.assertEqual([e["ok"] for e in run.log if e["event"] == "verify"], [False, True])
        self.assertIn("rejected it: this mirror is not above a sink", seen[2][1]["text"])  # planner sees why
        self.assertEqual(run.steps, 3 * 16)  # 3 look-arounds (start + one per check), 8 x 45 deg at 30 deg/step = 16 steps

    def test_verify_off_by_default(self):
        def fake_ask(cl, content, schema, system, max_tokens, effort, usage):
            usage.append({"output": 1})
            self.assertIs(schema, goat_nav.PLAN_SCHEMA)
            return {"observation": "", "where_am_i": "", "action": "arrived", "instruction": "", "plan": []}

        run, env = make_run()
        p = self._patched(fake_ask)
        with p[0], p[1], p[2]:
            self.assertEqual(run.landmark(None, [], "Find the bed."), "planner_arrived")

    def test_final_approach(self):
        calls = iter([{"observation": "", "where_am_i": "", "action": "arrived", "instruction": "", "plan": []},
                      {"observation": "bed", "where_am_i": "", "action": "forward", "amount": 0.5},
                      {"observation": "bed", "where_am_i": "", "action": "stop", "amount": 0}])
        systems = []

        def fake_ask(cl, content, schema, system, max_tokens, effort, usage):
            usage.append({"output": 1})
            systems.append((schema, system, content))
            return next(calls)

        run, env = make_run(final_approach=4)
        p = self._patched(fake_ask)
        with p[0], p[1], p[2]:
            ended = run.landmark(None, [{"type": "text", "text": "planner"}], "Find the bed.")
        self.assertEqual(ended, "planner_arrived+approach_stop")
        self.assertAlmostEqual(env.x, 0.5)
        self.assertIs(systems[1][0], goat_nav.DIRECT_SCHEMA)
        self.assertIn("arm's length", systems[1][1][0]["text"])
        self.assertIn("already be near it", systems[1][2][0]["text"])
        self.assertEqual(sum(1 for c in systems[1][2] if c.get("text") == "img"), 1)  # no look-around


class TestFloors(unittest.TestCase):
    def test_levels_and_tags(self):
        from robo_nav.habitat import floors as F
        ys = [0.0] * 50 + [i * 0.3 for i in range(1, 10)] + [3.0] * 40
        self.assertEqual(F.levels_from_heights(ys), [0.0, 3.0])
        self.assertEqual(F.floor_of(0.2, [0.0, 3.0]), 1)
        self.assertIsNone(F.floor_of(1.5, [0.0, 3.0]))
        self.assertIn("floor 2 of 2", F.describe_floor(3.1, [0.0, 3.0]))
        self.assertIn("between floor 1 and floor 2", F.describe_floor(1.5, [0.0, 3.0]))
        poses = [[0, y, 0] for y in ys]
        m = {"landmarks": [{"name": "rug", "seen_at": [0, 1, 2]}, {"name": "bed", "seen_at": [8, 9]},
                           {"name": "stairs", "seen_at": [5, 9]}]}
        lf = F.landmark_floors(m, poses, every=10)
        self.assertEqual(lf, {"rug": [1], "bed": [2], "stairs": [1, 2]})
        md = "# Landmarks\n- **rug**: red.\n- **bed**: big.\n"
        out = F.annotate_map_md(md, {"levels": [0.0, 3.0], "landmarks": lf})
        self.assertIn("- **rug** (floor 1): red.", out)
        self.assertIn("2 floors", out)
        self.assertEqual(F.annotate_map_md(md, {"levels": [0.0], "landmarks": {}}), md)


if __name__ == "__main__":
    unittest.main()
