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
    os.environ.setdefault("MAX_JOBS", "2")


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, env=env)


def _ensure_vendor_links(leffa_root: Path) -> None:
    for name in ("SCHP", "densepose", "detectron2"):
        target = leffa_root / "3rdparty" / name
        link = leffa_root / name
        if target.exists() and not link.exists():
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError:
                shutil.copytree(target, link)


def _detectron2_ready() -> bool:
    try:
        import detectron2  # noqa: F401
        from detectron2 import _C  # noqa: F401

        return True
    except Exception:
        return False


def ensure_real_densepose(leffa_root: Path) -> None:
    """Build Leffa's vendored Detectron2 with a low-RAM compile job count."""
    if _detectron2_ready():
        print("Real Detectron2 DensePose runtime already available.")
        return

    source = leffa_root / "3rdparty" / "detectron2"
    if not source.exists():
        raise RuntimeError(f"Detectron2 source missing: {source}")

    env = dict(os.environ)
    env.setdefault("MAX_JOBS", "2")
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-build-isolation",
            "-e",
            str(source),
        ],
        env=env,
    )

    for name in list(sys.modules):
        if name == "detectron2" or name.startswith("detectron2."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()
    if not _detectron2_ready():
        raise RuntimeError(
            "Detectron2 finished installing but detectron2._C is not importable. "
            "Restart the Colab runtime and rerun the setup cells."
        )
    print("Real Detectron2 DensePose: READY")


def runtime_summary() -> dict[str, Any]:
    import psutil
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. Choose Runtime → Change runtime type → GPU.")
    info = {
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": float(torch.cuda.get_device_properties(0).total_memory) / 1024**3,
        "system_ram_gb": float(psutil.virtual_memory().total) / 1024**3,
        "resolution_policy": os.environ.get("LEFFA_COLAB_RESOLUTION", "auto"),
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
