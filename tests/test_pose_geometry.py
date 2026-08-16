import unittest

import numpy as np
from PIL import Image

from pipeline.pose_geometry import (
    assess_pose_pair,
    openpose_points,
    piecewise_affine_warp,
    retarget_skeleton,
)


def _skeleton(scale_x=1.0, scale_y=1.0):
    pts = np.zeros((18, 2), dtype=np.float32)
    pts[0] = [100, 80]
    pts[1] = [100, 100]
    pts[2] = [80, 110]
    pts[3] = [60, 140]
    pts[4] = [45, 170]
    pts[5] = [120, 110]
    pts[6] = [140, 140]
    pts[7] = [155, 170]
    pts[8] = [90, 180]
    pts[9] = [90, 240]
    pts[10] = [90, 300]
    pts[11] = [110, 180]
    pts[12] = [110, 240]
    pts[13] = [110, 300]
    pts[14] = [94, 75]
    pts[15] = [106, 75]
    pts[16] = [88, 78]
    pts[17] = [112, 78]
    pts[:, 0] = 100 + (pts[:, 0] - 100) * scale_x
    pts[:, 1] = 100 + (pts[:, 1] - 100) * scale_y
    return pts


class PoseGeometryTests(unittest.TestCase):
    def test_openpose_points_scales_leffa_coordinates(self):
        raw = np.zeros((18, 2), dtype=np.float32)
        raw[1] = [100, 200]
        points, valid = openpose_points({"pose_keypoints_2d": raw.tolist()})
        self.assertTrue(valid[1])
        self.assertAlmostEqual(float(points[1, 0]), 200.0)
        self.assertAlmostEqual(float(points[1, 1]), 400.0)

    def test_pose_pair_is_safe_with_complete_moderate_proportions(self):
        donor = _skeleton()
        base = _skeleton(scale_x=1.15, scale_y=1.08)
        valid = np.ones(18, dtype=bool)
        diag = assess_pose_pair(base, valid, donor, valid)
        self.assertTrue(diag.safe_to_retarget)
        self.assertGreaterEqual(diag.common_body_joints, 14)

    def test_pose_pair_skips_when_core_joint_missing(self):
        donor = _skeleton()
        base = _skeleton()
        base_valid = np.ones(18, dtype=bool)
        donor_valid = np.ones(18, dtype=bool)
        donor_valid[2] = False
        diag = assess_pose_pair(base, base_valid, donor, donor_valid)
        self.assertFalse(diag.safe_to_retarget)
        self.assertIn("core", diag.reason)

    def test_retarget_preserves_bone_direction_while_changing_length(self):
        donor = _skeleton()
        base = _skeleton(scale_x=1.25, scale_y=1.10)
        valid = np.ones(18, dtype=bool)
        target = retarget_skeleton(base, valid, donor, valid, strength=1.0)

        donor_vec = donor[3] - donor[2]
        target_vec = target[3] - target[2]
        donor_dir = donor_vec / np.linalg.norm(donor_vec)
        target_dir = target_vec / np.linalg.norm(target_vec)
        self.assertGreater(float(np.dot(donor_dir, target_dir)), 0.999)
        self.assertGreater(float(np.linalg.norm(target_vec)), float(np.linalg.norm(donor_vec)))

    def test_piecewise_warp_identity_mapping_preserves_image(self):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[16:48, 16:48] = [10, 120, 240]
        points = np.asarray(
            [[16, 16], [48, 16], [48, 48], [16, 48], [32, 20], [32, 44]],
            dtype=np.float32,
        )
        out = piecewise_affine_warp(Image.fromarray(arr), points, points)
        self.assertLess(float(np.abs(np.asarray(out).astype(int) - arr.astype(int)).mean()), 0.1)


if __name__ == "__main__":
    unittest.main()
