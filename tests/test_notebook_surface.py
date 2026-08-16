import json
import unittest
from pathlib import Path


class NotebookSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(__file__).resolve().parents[1] / "Pose_Cloth_Changer.ipynb"
        cls.notebook = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.text = json.dumps(cls.notebook)

    def test_notebook_is_valid_v4_json(self):
        self.assertEqual(self.notebook.get("nbformat"), 4)
        self.assertIsInstance(self.notebook.get("cells"), list)

    def test_notebook_imports_canonical_pipeline(self):
        self.assertIn("from pipeline import PoseClothPipeline", self.text)
        self.assertNotIn("class FidelityPoseClothPipeline", self.text)
        self.assertNotIn("def retarget_densepose_control", self.text)

    def test_notebook_exposes_final_fidelity_controls(self):
        self.assertIn("Auto detect (recommended)", self.text)
        self.assertIn("pose_retarget_strength", self.text)
        self.assertIn("run_preflight", self.text)


if __name__ == "__main__":
    unittest.main()
