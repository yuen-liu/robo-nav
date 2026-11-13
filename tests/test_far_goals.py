import json
import tempfile
import unittest

from robo_nav.habitat import far_goals


def sub(k, geo, ended="stopped", pos=None):
    s = {"subtask": k, "type": "object", "text": f"Find {k}.", "ended": ended}
    if ended != "skipped":
        s.update(geodesic_m=geo, pose={"position": pos or [k, 0, 0], "heading": 0.1 * k})
    return s


class FarGoalsTest(unittest.TestCase):
    def test_start_pose_is_previous_scored_end(self):
        with tempfile.TemporaryDirectory() as d:
            ep = {"episode": 3, "scene": "S", "mode": "zeroshot", "ended": "done",
                  "subtasks": [sub(0, 12.0), sub(1, 3.0), sub(2, None, "skipped"), sub(3, 15.0),
                               sub(4, float("inf"))]}
            json.dump(ep, open(f"{d}/ep0003_zeroshot.json", "w"))
            c = far_goals.candidates(d, "zeroshot", 10.0)
        self.assertEqual([x["subtask"] for x in c], [0, 3])
        self.assertIsNone(c[0]["position"])  # first subtask: episode start
        self.assertEqual(c[1]["position"], [1, 0, 0])  # subtask 2 skipped -> robot still at subtask 1's end
        self.assertAlmostEqual(c[1]["heading"], 0.1)

    def test_pick_caps_per_scene(self):
        c = [{"episode": i, "subtask": 0, "scene": "AB"[i % 2]} for i in range(20)]
        g = far_goals.pick(c, 100, 3)
        self.assertEqual(len(g), 6)
        self.assertEqual(sorted(g, key=lambda x: (x["episode"], x["subtask"])), g)


if __name__ == "__main__":
    unittest.main()
