import unittest
import numpy as np
import torch
from robo_nav.scale_calibration import MetricScaleSolver, MetricCue, MetricCueType


class TestMetricScaleSolver(unittest.TestCase):
    def setUp(self):
        self.solver = MetricScaleSolver()

    def test_solve_scale_depth_point(self):
        # Synthetic depth predictions (unscaled median depth = 2.0)
        unscaled_depth = np.full((1, 5, 100, 100, 1), 2.0, dtype=np.float32)
        preds = {"depth": unscaled_depth}

        # True metric depth is 5.0 meters
        cue = MetricCue(
            cue_type=MetricCueType.DEPTH_POINT,
            metric_val=5.0,
            frame_idx_a=0,
            pixel_coords=(50, 50),
        )

        alpha = self.solver.solve_scale(preds, cue)
        self.assertAlmostEqual(alpha, 2.5, places=5)

        scaled_preds = self.solver.apply_scale(preds, alpha)
        self.assertAlmostEqual(float(scaled_preds["depth"][0, 0, 50, 50, 0]), 5.0, places=5)

    def test_solve_scale_translation_step(self):
        # Synthetic pose translation step (predicted dist = 0.5)
        extrinsic = np.zeros((1, 5, 4, 4), dtype=np.float32)
        extrinsic[0, 0] = np.eye(4)
        extrinsic[0, 1] = np.eye(4)
        extrinsic[0, 1, 0, 3] = 0.5  # 0.5 units translation along X

        preds = {"extrinsic": extrinsic}

        # True metric translation step is 1.0 meters
        cue = MetricCue(
            cue_type=MetricCueType.TRANSLATION_STEP,
            metric_val=1.0,
            frame_idx_a=0,
            frame_idx_b=1,
        )

        alpha = self.solver.solve_scale(preds, cue)
        self.assertAlmostEqual(alpha, 2.0, places=5)

        scaled_preds = self.solver.apply_scale(preds, alpha)
        self.assertAlmostEqual(float(scaled_preds["extrinsic"][0, 1, 0, 3]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
