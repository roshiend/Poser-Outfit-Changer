"""Colab-aware wrapper around the validated Fidelity v3 pipeline."""

from __future__ import annotations

import gc
import os
from typing import Literal

from .colab_lowmem import StagedLeffaInference, _release_cuda
from .fidelity_v2 import FidelityPoseClothPipeline
from .leffa_sequential import _is_colab
from .memory import cuda_mem_report, free_vram


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _resolution_mode() -> str:
    mode = os.environ.get("LEFFA_COLAB_RESOLUTION", "auto").strip().lower()
    if mode not in {"auto", "full", "balanced", "safe"}:
        raise ValueError(
            "LEFFA_COLAB_RESOLUTION must be one of: auto, full, balanced, safe"
        )
    return mode


class _FP16StagedLeffaInference(StagedLeffaInference):
    """Cast mmap-backed modules only when they enter CUDA, not in system RAM."""

    @staticmethod
    def _to_cuda(module) -> None:
        import torch

        module.to("cuda", dtype=torch.float16)
        _release_cuda()


class ColabAwareFidelityPipeline(FidelityPoseClothPipeline):
    """Use staged component offloading automatically on managed Colab runtimes."""

    def __init__(self, *args, **kwargs) -> None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        super().__init__(*args, **kwargs)
        self.colab_low_memory = bool(_is_colab() and _env_bool("LEFFA_COLAB_LOW_MEMORY", True))
        self.colab_resolution_mode = _resolution_mode()
        self._last_inference_info: dict = {}

        explicit_pose = os.environ.get("LEFFA_ALLOW_POSE", "").strip().lower()
        if self.colab_low_memory and explicit_pose not in {"0", "false", "no", "off"}:
            self.pose_enabled = True

        if self.colab_low_memory:
            print(
                "[colab-lowmem] enabled; "
                f"resolution policy={self.colab_resolution_mode}; pose_enabled={self.pose_enabled}"
            )

    def _build_leffa_model(self, pretrained_dir: str, weight_path: str):
        if not self.colab_low_memory:
            return super()._build_leffa_model(pretrained_dir, weight_path)

        import torch
        from leffa.model import LeffaModel

        print(f"[colab-lowmem] Memory-mapped model load: {weight_path}")

        try:
            from accelerate import init_empty_weights

            with init_empty_weights(include_buffers=False):
                model = LeffaModel(
                    pretrained_model_name_or_path=pretrained_dir,
                    pretrained_model="",
                    dtype=self.dtype,
                )
            state = torch.load(
                weight_path,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
            model.load_state_dict(state, strict=True, assign=True)
            del state
            model.eval()
            gc.collect()
            free_vram()
            print("[colab-lowmem] Meta + mmap checkpoint load succeeded")
            return model
        except Exception as exc:
            print(f"[colab-lowmem] Meta + mmap load unavailable; using half-init fallback: {exc!r}")
            gc.collect()

        original_load = torch.load
        original_dtype = torch.get_default_dtype()

        def _safe_load(*args, **kwargs):
            base_kwargs = dict(kwargs)
            base_kwargs.setdefault("map_location", "cpu")
            optimized = dict(base_kwargs)
            optimized.setdefault("mmap", True)
            optimized.setdefault("weights_only", True)
            try:
                obj = original_load(*args, **optimized)
            except Exception as load_exc:
                print(f"[colab-lowmem] mmap/weights-only fallback: {load_exc!r}")
                obj = original_load(*args, **base_kwargs)
            gc.collect()
            return obj

        torch.load = _safe_load  # type: ignore[assignment]
        half_init = self.dtype == "float16" and _env_bool("LEFFA_COLAB_HALF_INIT", True)
        if half_init:
            torch.set_default_dtype(torch.float16)
        try:
            model = LeffaModel(
                pretrained_model_name_or_path=pretrained_dir,
                pretrained_model=weight_path,
                dtype=self.dtype,
            )
        finally:
            torch.load = original_load  # type: ignore[assignment]
            torch.set_default_dtype(original_dtype)
        gc.collect()
        free_vram()
        return model

    def _load_vton(
        self,
        model_type: Literal["viton_hd", "dress_code"] = "viton_hd",
    ) -> None:
        if not self.colab_low_memory:
            return super()._load_vton(model_type)
        if self._active_model == f"vton_{model_type}" and self._vt_inference is not None:
            return

        self._unload_diffusion()
        print(f"[colab-lowmem] Loading staged VTON ({model_type})...")
        cuda_mem_report("before staged VTON load")
        weight = (
            self.ckpt_dir / "virtual_tryon.pth"
            if model_type == "viton_hd"
            else self.ckpt_dir / "virtual_tryon_dc.pth"
        )
        model = self._build_leffa_model(
            str(self.ckpt_dir / "stable-diffusion-inpainting"),
            str(weight),
        )
        self._vt_inference = _FP16StagedLeffaInference(
            model=model,
            control_type="virtual_tryon",
            resolution_mode=self.colab_resolution_mode,
        )
        self._active_model = f"vton_{model_type}"
        free_vram()
        cuda_mem_report("after staged VTON load")

    def _load_pose(self) -> None:
        if not self.colab_low_memory:
            return super()._load_pose()
        if self._active_model == "pose" and self._pt_inference is not None:
            return

        self._unload_diffusion()
        print("[colab-lowmem] Loading staged SDXL pose model...")
        cuda_mem_report("before staged pose load")
        model = self._build_leffa_model(
            str(self.ckpt_dir / "stable-diffusion-xl-1.0-inpainting-0.1"),
            str(self.ckpt_dir / "pose_transfer.pth"),
        )
        self._pt_inference = _FP16StagedLeffaInference(
            model=model,
            control_type="pose_transfer",
            resolution_mode=self.colab_resolution_mode,
        )
        self._active_model = "pose"
        free_vram()
        cuda_mem_report("after staged pose load")

    def _run_control_v2(self, *args, **kwargs):
        control_type = kwargs.get("control_type")
        if control_type is None and len(args) >= 3:
            control_type = args[2]
        result = super()._run_control_v2(*args, **kwargs)
        if not self.colab_low_memory:
            return result

        generated, mask, densepose, debug = result
        inference = self._pt_inference if control_type == "pose_transfer" else self._vt_inference
        info = dict(getattr(inference, "last_run_info", {}) or {})
        self._last_inference_info = info
        if info:
            debug["colab_memory"] = info
        return generated, mask, densepose, debug
