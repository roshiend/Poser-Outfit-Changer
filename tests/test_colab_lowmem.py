import os
import unittest

from colab_runner import configure_colab_environment
from pipeline import ColabAwareFidelityPipeline, PoseClothPipeline
from pipeline.colab_lowmem import choose_colab_policy


class ColabLowMemoryPolicyTests(unittest.TestCase):
    def test_public_pipeline_is_colab_aware_wrapper(self):
        self.assertIs(PoseClothPipeline, ColabAwareFidelityPipeline)

    def test_t4_sized_pose_uses_full_resolution_without_cfg_batch_doubling(self):
        policy = choose_colab_policy(15.0, "pose_transfer", requested_cfg=True)
        self.assertEqual((policy.transform_width, policy.transform_height), (768, 1024))
        self.assertFalse(policy.use_cfg)

    def test_small_gpu_automatically_reduces_resolution(self):
        policy = choose_colab_policy(10.0, "pose_transfer", requested_cfg=True)
        self.assertEqual((policy.transform_width, policy.transform_height), (576, 768))
        self.assertFalse(policy.use_cfg)

    def test_mid_size_gpu_uses_balanced_resolution(self):
        policy = choose_colab_policy(12.0, "pose_transfer", requested_cfg=True)
        self.assertEqual((policy.transform_width, policy.transform_height), (672, 896))

    def test_vton_keeps_cfg_on_normal_colab_vram(self):
        policy = choose_colab_policy(15.0, "virtual_tryon", requested_cfg=True)
        self.assertTrue(policy.use_cfg)
        self.assertEqual((policy.transform_width, policy.transform_height), (768, 1024))

    def test_explicit_safe_resolution_overrides_vram(self):
        policy = choose_colab_policy(
            24.0,
            "pose_transfer",
            requested_cfg=True,
            resolution_mode="safe",
        )
        self.assertEqual((policy.transform_width, policy.transform_height), (576, 768))

    def test_colab_runner_sets_memory_flags(self):
        old = {name: os.environ.get(name) for name in (
            "LEFFA_COLAB_LOW_MEMORY",
            "LEFFA_COLAB_RESOLUTION",
            "LEFFA_ALLOW_POSE",
            "MAX_JOBS",
        )}
        try:
            configure_colab_environment("balanced")
            self.assertEqual(os.environ["LEFFA_COLAB_LOW_MEMORY"], "1")
            self.assertEqual(os.environ["LEFFA_COLAB_RESOLUTION"], "balanced")
            self.assertEqual(os.environ["LEFFA_ALLOW_POSE"], "1")
        finally:
            for name, value in old.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
