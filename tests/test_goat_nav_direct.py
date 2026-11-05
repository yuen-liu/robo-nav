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
    a = types.SimpleNamespace(**{"subtask_steps": 500, "max_direct_calls": 60, "effort": "medium", **args})
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


if __name__ == "__main__":
    unittest.main()
