import unittest

import numpy as np

from pipeline.benchmark_metrics import (
    body_proportion_error,
    composite_fidelity_score,
    pose_angle_error_deg,
)


class BenchmarkMetricTests(unittest.TestCase):
    def test_pose_angle_error_detects_elbow_difference(self):
        target = np.zeros((18, 2), dtype=np.float32)
        generated = np.zeros((18, 2), dtype=np.float32)
        target_valid = np.zeros(18, dtype=bool)
        generated_valid = np.zeros(18, dtype=bool)

        # Right shoulder=2, elbow=3, wrist=4. Target elbow is 90 degrees,
        # generated elbow is straight (180 degrees).
        target[2] = [0, 0]
        target[3] = [10, 0]
        target[4] = [10, 10]
        generated[2] = [0, 0]
        generated[3] = [10, 0]
        generated[4] = [20, 0]
        target_valid[[2, 3, 4]] = True
        generated_valid[[2, 3, 4]] = True

        error = pose_angle_error_deg(generated, generated_valid, target, target_valid)
        self.assertIsNotNone(error)
        self.assertAlmostEqual(error, 90.0, places=4)

    def _skeleton(self):
        points = np.zeros((18, 2), dtype=np.float32)
        points[1] = [50, 20]   # neck
        points[2] = [65, 25]
        points[3] = [78, 38]
        points[4] = [88, 54]
        points[5] = [35, 25]
        points[6] = [22, 38]
        points[7] = [12, 54]
        points[8] = [60, 58]
        points[9] = [62, 82]
        points[10] = [64, 108]
        points[11] = [40, 58]
        points[12] = [38, 82]
        points[13] = [36, 108]
        valid = np.zeros(18, dtype=bool)
        valid[1:14] = True
        return points, valid

    def test_body_proportion_error_ignores_global_scale(self):
        base, valid = self._skeleton()
        generated = base * 1.8
        error = body_proportion_error(generated, valid, base, valid)
        self.assertIsNotNone(error)
        self.assertLess(error, 1e-5)

    def test_body_proportion_error_detects_distortion(self):
        base, valid = self._skeleton()
        generated = base.copy()
        generated[4] = generated[3] + (generated[4] - generated[3]) * 2.0
        generated[7] = generated[6] + (generated[7] - generated[6]) * 2.0
        error = body_proportion_error(generated, valid, base, valid)
        self.assertIsNotNone(error)
        self.assertGreater(error, 0.05)

    def test_composite_score_prefers_better_variant(self):
        good = composite_fidelity_score(0.75, 8.0, 0.08)
        weak = composite_fidelity_score(0.40, 28.0, 0.25)
        self.assertIsNotNone(good)
        self.assertIsNotNone(weak)
        self.assertGreater(good, weak)


if __name__ == "__main__":
    unittest.main()
