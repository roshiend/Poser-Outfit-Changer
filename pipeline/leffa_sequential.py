"""
Sequential Leffa runner designed for identity-safe outfit + pose transfer.

Only one diffusion checkpoint is kept in VRAM at a time:
  VTON -> unload -> Pose transfer -> unload -> adaptive face lock.

Pose transfer intentionally requires real Detectron2 DensePose. A parsing-based
fallback is still allowed for virtual try-on on constrained hardware, but it is
not geometrically accurate enough to drive pose transfer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .body_lock import align_pose_donor_to_base_body, person_bbox_from_parse
from .face_lock import lock_face_identity
from .garment_extract import extract_garment_from_person
from .memory import cuda_mem_report, free_vram

PathLike = Union[str, Path, Image.Image]
Mode = Literal["both", "outfit_only", "pose_only"]
RefKind = Literal["clothed_person", "flat_garment"]
VtonModel = Literal["auto", "viton_hd", "dress_code"]


class DensePoseUnavailableError(RuntimeError):
    """Raised when a pose operation is requested without real DensePose."""


def _is_hf_space() -> bool:
    return bool(os.environ.get("SPACE_ID") or os.environ.get("SPACE_HW"))


def _is_colab() -> bool:
    return Path("/content").exists() and not _is_hf_space()


def _pose_allowed() -> bool:
    override = os.environ.get("LEFFA_ALLOW_POSE", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    return not _is_colab()


def _resolve_vton_model(
    garment_type: str,
    requested: VtonModel,
) -> Literal["viton_hd", "dress_code"]:
    if requested not in {"auto", "viton_hd", "dress_code"}:
        raise ValueError(f"Unknown VTON model: {requested}")
    if requested != "auto":
        return requested
    return "viton_hd" if garment_type == "upper_body" else "dress_code"


def _ensure_leffa_on_path(leffa_root: Path) -> None:
    root = str(leffa_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _as_pil(image: PathLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


class PoseClothPipeline:
    """Transfer outfit and/or pose while keeping the base person's appearance."""

    def __init__(
        self,
        leffa_root: str | Path = "./Leffa",
        ckpt_dir: str | Path | None = None,
        dtype: str = "float16",
        enable_face_lock: bool = True,
        default_ref_kind: RefKind = "clothed_person",
        preserve_body: bool = True,
    ) -> None:
        self.leffa_root = Path(leffa_root)
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else self.leffa_root / "ckpts"
        self.dtype = dtype
        self.enable_face_lock = enable_face_lock
        self.default_ref_kind = default_ref_kind
        self.preserve_body = preserve_body
        self.pose_enabled = _pose_allowed()

        _ensure_leffa_on_path(self.leffa_root)

        self._parsing = None
        self._openpose = None
        self._densepose = None
        self._densepose_is_real = False
        self._vt_inference = None
        self._pt_inference = None
        self._face_app = None
        self._active_model: Optional[str] = None

    def download_checkpoints(self) -> None:
        from huggingface_hub import snapshot_download

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        print("Downloading Leffa checkpoints (first run can take a while)...")
        snapshot_download(repo_id="franciszzj/Leffa", local_dir=str(self.ckpt_dir))
        print("Checkpoints ready at", self.ckpt_dir)

    def _load_preprocessors(self, require_real_densepose: bool = False) -> None:
        if self._parsing is None or self._openpose is None:
            from preprocess.humanparsing.run_parsing import Parsing
            from preprocess.openpose.run_openpose import OpenPose

            ckpt = self.ckpt_dir
            self._parsing = Parsing(
                atr_path=str(ckpt / "humanparsing" / "parsing_atr.onnx"),
                lip_path=str(ckpt / "humanparsing" / "parsing_lip.onnx"),
            )
            self._openpose = OpenPose(
                body_model_path=str(ckpt / "openpose" / "body_pose_model.pth"),
            )

        if self._densepose is None or (require_real_densepose and not self._densepose_is_real):
            self._densepose = self._make_densepose(require_real=require_real_densepose)

    def _make_densepose(self, require_real: bool = False):
        ckpt = self.ckpt_dir
        try:
            import av  # noqa: F401
            import detectron2  # noqa: F401
            from detectron2 import _C  # noqa: F401
            from leffa_utils.densepose_predictor import DensePosePredictor

            pred = DensePosePredictor(
                config_path=str(ckpt / "densepose" / "densepose_rcnn_R_50_FPN_s1x.yaml"),
                weights_path=str(ckpt / "densepose" / "model_final_162be9.pkl"),
            )
            self._densepose_is_real = True
            print("[densepose] Using Detectron2 DensePose")
            return pred
        except Exception as exc:
            self._densepose_is_real = False
            if require_real:
                raise DensePoseUnavailableError(
                    "Pose transfer requires real Detectron2 DensePose, but it could not be loaded. "
                    "Install/build Detectron2 for the current PyTorch environment, then retry. "
                    f"Original error: {exc!r}"
                ) from exc

            print(f"[densepose] Real DensePose unavailable ({exc!r}); using VTON-only fallback")
            from .densepose_fallback import FallbackDensePosePredictor

            return FallbackDensePosePredictor(parsing_fn=self._parsing)

    def _unload_preprocessors(self) -> None:
        self._densepose = None
        self._densepose_is_real = False
        self._parsing = None
        self._openpose = None
        free_vram()
        print("[mem] Unloaded preprocessors (DensePose/parsing/openpose)")
        cuda_mem_report("after preprocess unload")

    def _unload_diffusion(self) -> None:
        self._vt_inference = None
        self._pt_inference = None
        self._active_model = None
        free_vram()
        cuda_mem_report("after diffusion unload")

    def _build_leffa_model(self, pretrained_dir: str, weight_path: str):
        import gc
        import torch
        from leffa.model import LeffaModel

        print(f"[load] Reading checkpoint: {weight_path}")
        original_load = torch.load

        def _safe_load(*args, **kwargs):
            kwargs.setdefault("map_location", "cpu")
            obj = original_load(*args, **kwargs)
            gc.collect()
            return obj

        torch.load = _safe_load  # type: ignore[assignment]
        try:
            model = LeffaModel(
                pretrained_model_name_or_path=pretrained_dir,
                pretrained_model=weight_path,
                dtype=self.dtype,
            )
        finally:
            torch.load = original_load  # type: ignore[assignment]
        gc.collect()
        free_vram()
        return model

    def _load_vton(
        self,
        model_type: Literal["viton_hd", "dress_code"] = "viton_hd",
    ) -> None:
        if self._active_model == f"vton_{model_type}" and self._vt_inference is not None:
            return
        self._unload_diffusion()

        print(f"[load] Loading VTON ({model_type})...")
        cuda_mem_report("before VTON load")
        from leffa.inference import LeffaInference

        weight = (
            self.ckpt_dir / "virtual_tryon.pth"
            if model_type == "viton_hd"
            else self.ckpt_dir / "virtual_tryon_dc.pth"
        )
        model = self._build_leffa_model(
            str(self.ckpt_dir / "stable-diffusion-inpainting"),
            str(weight),
        )
        self._vt_inference = LeffaInference(model=model)
        self._active_model = f"vton_{model_type}"
        free_vram()
        cuda_mem_report("after VTON load")
        print(f"[load] VTON ready: {model_type}")

    def _load_pose(self) -> None:
        if self._active_model == "pose" and self._pt_inference is not None:
            return
        self._unload_diffusion()

        print("[load] Loading pose-transfer (SDXL)...")
        cuda_mem_report("before pose load")
        from leffa.inference import LeffaInference

        model = self._build_leffa_model(
            str(self.ckpt_dir / "stable-diffusion-xl-1.0-inpainting-0.1"),
            str(self.ckpt_dir / "pose_transfer.pth"),
        )
        self._pt_inference = LeffaInference(model=model)
        self._active_model = "pose"
        free_vram()
        cuda_mem_report("after pose load")
        print("[load] Pose-transfer model ready")

    def preload_colab(self, include_pose: bool = False) -> None:
        print("[preload] Warming VTON...")
        self._load_vton("viton_hd")
        if include_pose:
            if not self.pose_enabled:
                raise RuntimeError(
                    "Pose is disabled for this Colab runtime by default. Set LEFFA_ALLOW_POSE=1 "
                    "before constructing PoseClothPipeline to force it."
                )
            self._load_pose()
        print("[preload] Done.")

    def _get_face_app(self):
        if self._face_app is not None:
            return self._face_app
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            self._face_app = app
        except Exception as exc:
            print(f"[face_lock] Could not init InsightFace: {exc}")
            self._face_app = False
        return self._face_app if self._face_app is not False else None

    def _garment_ref_from_clothed_person(
        self,
        person: Image.Image,
        garment_type: str,
    ) -> Image.Image:
        from leffa_utils.utils import resize_and_center

        self._load_preprocessors(require_real_densepose=False)
        person = resize_and_center(person.convert("RGB"), 768, 1024)
        parse_map, _ = self._parsing(person.resize((384, 512)))
        garment = extract_garment_from_person(
            person_rgb=person,
            parse_map=parse_map,
            garment_type=garment_type,
            out_size=(768, 1024),
        )
        print(f"[garment] Extracted {garment_type} clothing from clothed-person ref")
        return garment

    def _run_control(
        self,
        src_image: Image.Image,
        ref_image: Image.Image,
        control_type: Literal["virtual_tryon", "pose_transfer"],
        step: int = 30,
        scale: float = 2.5,
        seed: int = 42,
        ref_acceleration: bool = True,
        vt_model_type: Literal["viton_hd", "dress_code"] = "viton_hd",
        vt_garment_type: str = "upper_body",
        vt_repaint: bool = False,
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
        from leffa.transform import LeffaTransform
        from leffa_utils.utils import (
            get_agnostic_mask_dc,
            get_agnostic_mask_hd,
            resize_and_center,
        )

        require_real = control_type == "pose_transfer"
        self._load_preprocessors(require_real_densepose=require_real)

        src_image = resize_and_center(src_image.convert("RGB"), 768, 1024)
        ref_image = resize_and_center(ref_image.convert("RGB"), 768, 1024)
        src_array = np.array(src_image)

        if control_type == "virtual_tryon":
            model_parse, _ = self._parsing(src_image.resize((384, 512)))
            keypoints = self._openpose(src_image.resize((384, 512)))
            if vt_model_type == "viton_hd":
                mask = get_agnostic_mask_hd(model_parse, keypoints, vt_garment_type)
            else:
                mask = get_agnostic_mask_dc(model_parse, keypoints, vt_garment_type)
            mask = mask.resize((768, 1024))

            if vt_model_type == "viton_hd":
                seg = self._densepose.predict_seg(src_array)[:, :, ::-1]
            else:
                iuv = self._densepose.predict_iuv(src_array)
                seg = np.concatenate([iuv[:, :, 0:1]] * 3, axis=-1)
            densepose = Image.fromarray(seg)
            self._unload_preprocessors()
            free_vram()
            self._load_vton(vt_model_type)
            inference = self._vt_inference
        else:
            if not self._densepose_is_real:
                raise DensePoseUnavailableError(
                    "Pose transfer cannot run with fallback DensePose."
                )
            mask = Image.fromarray(np.ones_like(src_array) * 255)
            iuv = self._densepose.predict_iuv(src_array)[:, :, ::-1]
            densepose = Image.fromarray(iuv)
            self._unload_preprocessors()
            free_vram()
            self._load_pose()
            inference = self._pt_inference

        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = LeffaTransform()(data)
        output = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=int(step),
            guidance_scale=float(scale),
            seed=int(seed),
            repaint=vt_repaint if control_type == "virtual_tryon" else False,
        )
        generated = output["generated_image"][0].convert("RGB")
        return generated, mask.convert("RGB"), densepose.convert("RGB")

    def generate(
        self,
        base_image: PathLike,
        ref_image: PathLike,
        mode: Mode = "both",
        garment_type: str = "upper_body",
        vt_model_type: VtonModel = "auto",
        steps: int = 30,
        guidance_scale: float = 2.5,
        seed: int = 42,
        ref_acceleration: bool = True,
        face_lock: Optional[bool] = None,
        ref_kind: Optional[RefKind] = None,
        preserve_body: Optional[bool] = None,
        return_debug: bool = False,
    ) -> Image.Image | Tuple[Image.Image, dict]:
        from leffa_utils.utils import resize_and_center

        if mode not in {"both", "outfit_only", "pose_only"}:
            raise ValueError(f"Unknown mode: {mode}")
        if garment_type not in {"upper_body", "lower_body", "dresses"}:
            raise ValueError(f"Unknown garment type: {garment_type}")
        if mode in {"both", "pose_only"} and not self.pose_enabled:
            raise RuntimeError(
                "Pose transfer is disabled on this Colab runtime because the SDXL pose model often "
                "exceeds free-tier memory. Nothing was silently downgraded. Use Outfit only, move "
                "to a larger GPU, or set LEFFA_ALLOW_POSE=1 before creating the pipeline."
            )

        base = _as_pil(base_image)
        ref = _as_pil(ref_image)
        kind = ref_kind or self.default_ref_kind
        do_face = self.enable_face_lock if face_lock is None else face_lock
        do_body = self.preserve_body if preserve_body is None else preserve_body
        selected_vton = _resolve_vton_model(garment_type, vt_model_type)
        debug: dict = {"selected_vton_model": selected_vton}

        base = resize_and_center(base, 768, 1024)
        ref = resize_and_center(ref, 768, 1024)

        current = base
        base_bbox = None
        if do_body and mode in {"both", "pose_only"}:
            self._load_preprocessors(require_real_densepose=False)
            base_parse, _ = self._parsing(base.resize((384, 512)))
            base_bbox = person_bbox_from_parse(
                base_parse.resize((768, 1024), Image.NEAREST)
            )

        if mode in {"both", "outfit_only"}:
            print(f"Step 1/2: Outfit transfer using {selected_vton}...")
            if kind == "clothed_person":
                vton_ref = self._garment_ref_from_clothed_person(ref, garment_type)
                debug["garment_ref"] = vton_ref.copy()
            else:
                vton_ref = ref

            current, vt_mask, vt_densepose = self._run_control(
                src_image=current,
                ref_image=vton_ref,
                control_type="virtual_tryon",
                step=steps,
                scale=guidance_scale,
                seed=seed,
                ref_acceleration=ref_acceleration,
                vt_model_type=selected_vton,
                vt_garment_type=garment_type,
            )
            debug["outfit_mask"] = vt_mask
            debug["outfit_densepose"] = vt_densepose
            debug["after_vton"] = current.copy()
            self._unload_diffusion()

        if mode in {"both", "pose_only"}:
            print("Step 2/2: Pose transfer with real DensePose...")
            appearance = current if mode == "both" else base
            pose_src = ref

            if do_body:
                self._load_preprocessors(require_real_densepose=False)
                donor_parse, _ = self._parsing(ref.resize((384, 512)))
                donor_bbox = person_bbox_from_parse(
                    donor_parse.resize((768, 1024), Image.NEAREST)
                )
                pose_src = align_pose_donor_to_base_body(
                    pose_donor=ref,
                    base_person=appearance,
                    donor_bbox=donor_bbox,
                    base_bbox=base_bbox,
                )
                debug["aligned_pose_donor"] = pose_src.copy()

            current, pose_mask, pose_densepose = self._run_control(
                src_image=pose_src,
                ref_image=appearance,
                control_type="pose_transfer",
                step=steps,
                scale=guidance_scale,
                seed=seed,
                ref_acceleration=ref_acceleration,
            )
            debug["pose_mask"] = pose_mask
            debug["pose_densepose"] = pose_densepose
            debug["after_pose"] = current.copy()
            self._unload_diffusion()

        if do_face:
            print("Adaptive face identity lock...")
            requested_blend = 0.72 if mode == "outfit_only" else 0.84
            current = lock_face_identity(
                base_image=base,
                generated_image=current,
                face_app=self._get_face_app(),
                blend=requested_blend,
                adaptive_pose=True,
            )
            debug["after_face_lock"] = current.copy()

        free_vram()
        if return_debug:
            return current, debug
        return current

    def generate_from_paths(
        self,
        base_path: str,
        ref_path: str,
        **kwargs,
    ) -> Image.Image:
        return self.generate(base_path, ref_path, **kwargs)  # type: ignore[return-value]
