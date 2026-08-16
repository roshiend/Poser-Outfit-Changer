import gc
import os
import unittest
import weakref

from colab_runner import (
    _detectron2_extension_setup_text,
    configure_colab_environment,
)
from pipeline import ColabAwareFidelityPipeline, PoseClothPipeline
from pipeline.colab_lowmem import choose_colab_policy
from pipeline.colab_pipeline import _drop_failed_load_objects


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

    def test_failed_meta_load_objects_become_collectible_before_fallback(self):
        class HeavyObject:
            pass

        model = HeavyObject()
        state = HeavyObject()
        model_ref = weakref.ref(model)
        state_ref = weakref.ref(state)

        model, state = _drop_failed_load_objects(model, state)
        gc.collect()

        self.assertIsNone(model)
        self.assertIsNone(state)
        self.assertIsNone(model_ref())
        self.assertIsNone(state_ref())

    def test_detectron2_builder_targets_leffa_06_package_in_place(self):
        setup_text = _detectron2_extension_setup_text()
        self.assertIn('package = root / "detectron2"', setup_text)
        self.assertIn('"detectron2._C"', setup_text)
        self.assertIn("CUDAExtension", setup_text)
        self.assertIn('version="0.6.0"', setup_text)
        self.assertNotIn("pip install", setup_text)
        self.assertNotIn("mhp_extension", setup_text)

    def test_colab_runner_sets_memory_flags(self):
        names = (
            "LEFFA_COLAB_LOW_MEMORY",
            "LEFFA_COLAB_RESOLUTION",
            "LEFFA_ALLOW_POSE",
            "MAX_JOBS",
        )
        old = {name: os.environ.get(name) for name in names}
        try:
            os.environ.pop("MAX_JOBS", None)
            configure_colab_environment("balanced")
            self.assertEqual(os.environ["LEFFA_COLAB_LOW_MEMORY"], "1")
            self.assertEqual(os.environ["LEFFA_COLAB_RESOLUTION"], "balanced")
            self.assertEqual(os.environ["LEFFA_ALLOW_POSE"], "1")
            self.assertEqual(os.environ["MAX_JOBS"], "1")
        finally:
            for name, value in old.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
