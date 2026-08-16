import tempfile
import unittest
from pathlib import Path

from pipeline.preflight import checkpoint_requirements, run_preflight


class PreflightTests(unittest.TestCase):
    def test_pose_checkpoint_requirements_extend_vton_requirements(self):
        vton = set(checkpoint_requirements(require_pose=False))
        pose = set(checkpoint_requirements(require_pose=True))
        self.assertTrue(vton < pose)
        self.assertIn("pose_transfer.pth", pose)
        self.assertIn("densepose/model_final_162be9.pkl", pose)
        self.assertNotIn("pose_transfer.pth", vton)

    def test_missing_leffa_source_is_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_preflight(leffa_root=Path(tmp) / "missing")
        self.assertFalse(report["ready"])
        self.assertIn("leffa_source", report["blocking"])

    def test_missing_pose_densepose_is_blocking_when_pose_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Leffa"
            (root / "leffa").mkdir(parents=True)
            (root / "preprocess").mkdir(parents=True)
            report = run_preflight(
                leffa_root=root,
                require_pose=True,
                require_checkpoints=True,
            )
        self.assertFalse(report["ready"])
        self.assertIn("checkpoints", report["blocking"])
        self.assertIn("detectron2_densepose", report["blocking"])


if __name__ == "__main__":
    unittest.main()
