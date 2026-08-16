import json
import unittest
from pathlib import Path


class NotebookSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(__file__).resolve().parents[1] / "Pose_Cloth_Changer.ipynb"
        cls.notebook = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.text = json.dumps(cls.notebook)
        cls.code_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )

    def test_notebook_is_valid_v4_json(self):
        self.assertEqual(self.notebook.get("nbformat"), 4)
        self.assertIsInstance(self.notebook.get("cells"), list)

    def test_notebook_uses_canonical_colab_runner(self):
        self.assertIn("colab_runner.prepare_colab", self.code_text)
        self.assertIn("from colab_runner import run_colab_generation", self.code_text)
        self.assertNotIn("class FidelityPoseClothPipeline", self.code_text)
        self.assertNotIn("def retarget_densepose_control", self.code_text)

    def test_notebook_enables_low_memory_policy_before_torch(self):
        self.assertIn("LEFFA_COLAB_LOW_MEMORY", self.code_text)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF", self.code_text)
        self.assertIn("LEFFA_COLAB_RESOLUTION", self.code_text)
        self.assertIn("os.environ['MAX_JOBS'] = '1'", self.code_text)

    def test_bootstrap_forces_fresh_project_modules(self):
        self.assertIn("sys.modules.pop(name, None)", self.code_text)
        self.assertIn("name == 'colab_runner'", self.code_text)
        self.assertIn("name.startswith('pipeline.')", self.code_text)
        self.assertIn("importlib.invalidate_caches()", self.code_text)
        self.assertIn("importlib.reload(colab_runner)", self.code_text)
        self.assertIn("VERSION", self.code_text)
        self.assertIn("rev-parse", self.code_text)

    def test_bootstrap_guards_numpy_binary_abi(self):
        self.assertIn("numpy==1.26.4", self.code_text)
        self.assertIn(".poser_outfit_colab_abi", self.code_text)
        self.assertIn("import numpy.random", self.code_text)
        self.assertIn("import torchvision.transforms", self.code_text)
        self.assertIn("Runtime -> Restart session", self.code_text)
        self.assertIn("numpy.dtype size changed", self.code_text)
        self.assertIn("clean subprocess ABI", self.code_text)

    def test_notebook_has_secure_hugging_face_token_step(self):
        self.assertIn("from getpass import getpass", self.code_text)
        self.assertIn("userdata.get('HF_TOKEN')", self.code_text)
        self.assertIn("os.environ['HF_TOKEN'] = HF_TOKEN", self.code_text)
        self.assertIn("os.environ['HUGGING_FACE_HUB_TOKEN'] = HF_TOKEN", self.code_text)
        self.assertIn("login(token=HF_TOKEN, add_to_git_credential=False)", self.code_text)
        self.assertIn("HfApi().whoami(token=HF_TOKEN)", self.code_text)
        self.assertNotIn("print(HF_TOKEN", self.code_text)
        token_position = self.code_text.index("HfApi().whoami(token=HF_TOKEN)")
        prepare_position = self.code_text.index("colab_runner.prepare_colab")
        self.assertLess(token_position, prepare_position)

    def test_notebook_has_no_gradio_web_server_code(self):
        code = self.code_text.lower()
        self.assertNotIn("import gradio", code)
        self.assertNotIn("gr.blocks", code)
        self.assertNotIn("demo.launch", code)
        self.assertNotIn("share=true", code.replace(" ", ""))

    def test_notebook_exposes_final_fidelity_controls(self):
        self.assertIn("garment_type='auto'", self.code_text)
        self.assertIn("pose_retarget_strength=0.65", self.code_text)
        self.assertIn("mode='both'", self.code_text)
        self.assertIn("prepare_colab(resolution='safe'", self.text)


if __name__ == "__main__":
    unittest.main()
