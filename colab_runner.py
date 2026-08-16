"""Notebook-native Google Colab runner for the Fidelity v3 pipeline.

This module deliberately does not launch Gradio. It keeps Colab interaction in
notebook cells and activates the staged low-memory Leffa path.
"""

from __future__ import annotations

import gc
import importlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any


def configure_colab_environment(resolution: str = "auto") -> None:
    if resolution not in {"auto", "full", "balanced", "safe"}:
        raise ValueError("resolution must be auto, full, balanced, or safe")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["LEFFA_COLAB_LOW_MEMORY"] = "1"
    os.environ["LEFFA_COLAB_RESOLUTION"] = resolution
    os.environ["LEFFA_ALLOW_POSE"] = "1"
    # Detectron2 C++/CUDA compilation is the largest setup-time RAM spike on
    # free Colab. One compile job is slower but far safer on ~13 GB runtimes.
    os.environ.setdefault("MAX_JOBS", "1")


def _run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, env=env, cwd=None if cwd is None else str(cwd))


def _ensure_vendor_links(leffa_root: Path) -> None:
    for name in ("SCHP", "densepose", "detectron2"):
        target = leffa_root / "3rdparty" / name
        link = leffa_root / name
        if target.exists() and not link.exists():
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError:
                shutil.copytree(target, link)


def _clear_detectron2_modules() -> None:
    for name in list(sys.modules):
        if name == "detectron2" or name.startswith("detectron2."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _detectron2_ready() -> bool:
    try:
        _clear_detectron2_modules()
        import detectron2  # noqa: F401
        from detectron2 import _C  # noqa: F401

        return True
    except Exception:
        return False


def _detectron2_extension_setup_text() -> str:
    """Return a minimal setup.py that builds Leffa's exact Detectron2 0.6 _C.

    Leffa's ``3rdparty/detectron2`` directory is only the Python package tree,
    not an installable project. Building in-place from that exact tree keeps
    the compiled extension aligned with the 0.6 Python code DensePose imports.
    """
    return textwrap.dedent(
        r"""
        import os
        from pathlib import Path

        import torch
        from setuptools import setup
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

        root = Path(__file__).resolve().parent
        package = root / "detectron2"
        csrc = package / "layers" / "csrc"
        main_source = csrc / "vision.cpp"

        if not main_source.exists():
            raise RuntimeError(f"Detectron2 C++ source missing: {main_source}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required to build Leffa DensePose support.")
        if CUDA_HOME is None:
            raise RuntimeError(
                "CUDA toolkit/nvcc was not found. Colab must provide a CUDA development runtime."
            )

        cpp_sources = [
            str(path)
            for path in csrc.rglob("*.cpp")
            if path.resolve() != main_source.resolve()
        ]
        cuda_sources = [str(path) for path in csrc.rglob("*.cu")]
        sources = [str(main_source)] + cpp_sources + cuda_sources

        extension = CUDAExtension(
            "detectron2._C",
            sources,
            include_dirs=[str(csrc)],
            define_macros=[("WITH_CUDA", None)],
            extra_compile_args={
                "cxx": ["-O1"],
                "nvcc": [
                    "-O1",
                    "-DCUDA_HAS_FP16=1",
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        )

        setup(
            name="leffa-detectron2-colab-extension",
            version="0.6.0",
            ext_modules=[extension],
            cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
        )
        """
    ).lstrip()


def _prepare_detectron2_build_env() -> dict[str, str]:
    import torch

    env = dict(os.environ)
    env.setdefault("MAX_JOBS", "1")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        # Build only for the GPU Colab actually assigned (e.g. T4 = 7.5).
        env.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    return env


def ensure_real_densepose(leffa_root: Path) -> None:
    """Build Leffa's exact Detectron2 0.6 extension with low-RAM settings."""
    if _detectron2_ready():
        print("Real Detectron2 DensePose runtime already available.")
        return

    source = leffa_root / "3rdparty" / "detectron2"
    csrc = source / "layers" / "csrc"
    if not source.exists() or not (csrc / "vision.cpp").exists():
        raise RuntimeError(f"Leffa Detectron2 0.6 source is incomplete: {source}")

    # The copied 0.6 package intentionally has no setup.py. Do not `pip -e`
    # this directory: compile detectron2._C directly into the package instead.
    parent = source.parent
    setup_path = parent / "_colab_detectron2_setup.py"
    build_dir = parent / "build"

    # Remove an incompatible/stale binary from a previous runtime attempt.
    for stale in source.glob("_C*.so"):
        try:
            stale.unlink()
        except OSError:
            pass
    shutil.rmtree(build_dir, ignore_errors=True)

    setup_path.write_text(_detectron2_extension_setup_text(), encoding="utf-8")
    env = _prepare_detectron2_build_env()
    print(
        "Building Leffa Detectron2 0.6 in place "
        f"(MAX_JOBS={env.get('MAX_JOBS')}, "
        f"TORCH_CUDA_ARCH_LIST={env.get('TORCH_CUDA_ARCH_LIST', 'auto')})"
    )
    try:
        _run(
            [
                sys.executable,
                setup_path.name,
                "build_ext",
                "--inplace",
                "--force",
            ],
            env=env,
            cwd=parent,
        )
    finally:
        try:
            setup_path.unlink()
        except OSError:
            pass
        shutil.rmtree(build_dir, ignore_errors=True)

    _clear_detectron2_modules()
    if not _detectron2_ready():
        raise RuntimeError(
            "Detectron2 _C finished building but is not importable. "
            "Restart the Colab runtime and rerun the setup cells."
        )
    print("Real Detectron2 DensePose: READY")


def runtime_summary() -> dict[str, Any]:
    import psutil
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. Choose Runtime → Change runtime type → GPU.")
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": float(torch.cuda.get_device_properties(0).total_memory) / 1024**3,
        "system_ram_gb": float(psutil.virtual_memory().total) / 1024**3,
        "resolution_policy": os.environ.get("LEFFA_COLAB_RESOLUTION", "auto"),
        "detectron2_build_jobs": os.environ.get("MAX_JOBS", "1"),
    }
    print(json.dumps(info, indent=2))
    return info


def prepare_colab(
    project_root: str | Path = "/content/Poser-Outfit-Changer",
    leffa_root: str | Path = "/content/Leffa",
    resolution: str = "auto",
    download_checkpoints: bool = True,
):
    """Prepare Leffa/DensePose and return a ready low-memory pipeline."""
    configure_colab_environment(resolution)
    project_root = Path(project_root)
    leffa_root = Path(leffa_root)

    if not project_root.exists():
        raise RuntimeError(
            f"Project clone not found at {project_root}. Run the notebook bootstrap cell first."
        )
    if not leffa_root.exists():
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/franciszzj/Leffa.git",
                str(leffa_root),
            ]
        )

    _ensure_vendor_links(leffa_root)
    for path in (str(project_root), str(leffa_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. Choose Runtime → Change runtime type → GPU.")
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    runtime_summary()
    ensure_real_densepose(leffa_root)
    gc.collect()
    torch.cuda.empty_cache()

    from pipeline import PoseClothPipeline
    from pipeline.preflight import run_preflight

    ckpt_dir = leffa_root / "ckpts"
    pipe = PoseClothPipeline(
        leffa_root=str(leffa_root),
        ckpt_dir=str(ckpt_dir),
        dtype="float16",
        enable_face_lock=True,
        default_ref_kind="clothed_person",
        preserve_body=True,
    )
    if download_checkpoints:
        pipe.download_checkpoints()

    report = run_preflight(
        leffa_root=leffa_root,
        ckpt_dir=ckpt_dir,
        require_pose=True,
        require_gpu=True,
        require_checkpoints=True,
    )
    for item in report["checks"]:
        mark = "PASS" if item["ok"] else ("FAIL" if item["required"] else "WARN")
        print(f"[{mark}] {item['name']}: {item['detail']}")
    if not report["ready"]:
        raise RuntimeError("Preflight failed: " + ", ".join(report["blocking"]))

    print("Pipeline ready. Staged Colab mode:", getattr(pipe, "colab_low_memory", False))
    return pipe


def upload_base_reference():
    """Use the native Colab upload dialog twice and return local file paths."""
    from google.colab import files

    print("Upload BASE image — identity/body to keep")
    base_upload = files.upload()
    if not base_upload:
        raise RuntimeError("No base image uploaded.")
    base_path = next(iter(base_upload))

    print("Upload REFERENCE image — pose/outfit to copy")
    ref_upload = files.upload()
    if not ref_upload:
        raise RuntimeError("No reference image uploaded.")
    ref_path = next(iter(ref_upload))
    return base_path, ref_path


def run_colab_generation(
    pipe,
    base_image,
    ref_image,
    *,
    mode: str = "both",
    garment_type: str = "auto",
    steps: int = 25,
    guidance_scale: float = 2.5,
    seed: int = 42,
    pose_retarget_strength: float = 0.65,
    output_path: str | Path = "/content/poser_outfit_result.png",
):
    """Run one generation and print peak-memory/debug diagnostics."""
    import torch
    from IPython.display import display

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    result, debug = pipe.generate(
        base_image=base_image,
        ref_image=ref_image,
        mode=mode,
        garment_type=garment_type,
        vt_model_type="auto",
        steps=int(steps),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
        ref_acceleration=True,
        face_lock=True,
        ref_kind="clothed_person",
        preserve_body=True,
        pose_retarget_strength=float(pose_retarget_strength),
        return_debug=True,
    )

    output_path = Path(output_path)
    result.save(output_path)
    elapsed = time.time() - started
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Finished in {elapsed / 60:.1f} minutes")
    print(f"Peak allocated CUDA memory: {peak_gb:.2f} GB")

    for key in ("virtual_tryon_memory", "pose_transfer_memory", "colab_memory"):
        if key in debug:
            print(f"{key}: {json.dumps(debug[key], indent=2, default=str)}")

    quality = debug.get("outfit_quality") or {}
    for warning in quality.get("warnings", []):
        print("OUTFIT WARNING:", warning)
    identity = debug.get("identity_info") or {}
    if identity:
        print("Identity diagnostics:", json.dumps(identity, indent=2, default=str))

    display(result)
    print("Saved:", output_path)
    return result, debug


def download_result(path: str | Path = "/content/poser_outfit_result.png") -> None:
    from google.colab import files

    files.download(str(path))
