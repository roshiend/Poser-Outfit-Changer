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

    def test_notebook_uses_canonical_colab_runner(self):
        self.assertIn("from colab_runner import prepare_colab", self.text)
        self.assertIn("from colab_runner import run_colab_generation", self.text)
        self.assertNotIn("class FidelityPoseClothPipeline", self.text)
        self.assertNotIn("def retarget_densepose_control", self.text)

    def test_notebook_enables_low_memory_policy_before_torch(self):
        self.assertIn("LEFFA_COLAB_LOW_MEMORY", self.text)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF", self.text)
        self.assertIn("LEFFA_COLAB_RESOLUTION", self.text)
        self.assertIn("MAX_JOBS", self.text)

    def test_notebook_has_no_gradio_web_server(self):
        self.assertNotIn("gradio", self.text.lower())
        self.assertNotIn("demo.launch", self.text)
        self.assertNotIn("share=True", self.text)

    def test_notebook_exposes_final_fidelity_controls(self):
        self.assertIn("garment_type='auto'", self.text)
        self.assertIn("pose_retarget_strength=0.65", self.text)
        self.assertIn("mode='both'", self.text)
        self.assertIn("prepare_colab(resolution='safe'", self.text)


if __name__ == "__main__":
    unittest.main()
