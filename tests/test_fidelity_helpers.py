import unittest

import numpy as np
from PIL import Image

from pipeline.body_lock import match_result_body_to_base, soft_preserve_torso
from pipeline.face_lock import _adaptive_face_blend, _identity_adjusted_blend
from pipeline.leffa_sequential import _resolve_vton_model


class _Face:
    def __init__(self, pose):
        self.pose = np.asarray(pose, dtype=np.float32)


class FidelityPolicyTests(unittest.TestCase):
    def test_auto_vton_model_uses_viton_hd_for_upper(self):
        self.assertEqual(_resolve_vton_model("upper_body", "auto"), "viton_hd")

    def test_auto_vton_model_uses_dress_code_for_full_outfit(self):
        self.assertEqual(_resolve_vton_model("dresses", "auto"), "dress_code")
        self.assertEqual(_resolve_vton_model("lower_body", "auto"), "dress_code")

    def test_face_lock_skips_large_head_angle_mismatch(self):
        base = _Face([0, 0, 0])
        turned = _Face([0, 62, 0])
        self.assertEqual(_adaptive_face_blend(base, turned, 0.84), 0.0)

    def test_identity_pressure_cannot_override_pose_safety_reduction(self):
        self.assertEqual(_identity_adjusted_blend(0.32, 0.05), 0.32)
        self.assertEqual(_identity_adjusted_blend(0.52, 0.05), 0.52)

    def test_identity_pressure_can_strengthen_frontal_low_similarity(self):
        self.assertGreater(_identity_adjusted_blend(0.82, 0.05), 0.82)

    def test_legacy_post_body_warp_is_disabled(self):
        image = Image.new("RGB", (32, 32), "white")
        out = match_result_body_to_base(image, image)
        torso = soft_preserve_torso(image, image)
        self.assertEqual(np.asarray(out).tolist(), np.asarray(image).tolist())
        self.assertEqual(np.asarray(torso).tolist(), np.asarray(image).tolist())


if __name__ == "__main__":
    unittest.main()
