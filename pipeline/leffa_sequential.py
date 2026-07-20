"""
Sequential Leffa runner for Colab T4 (~15GB).

Loads at most one diffusion checkpoint at a time:
  VTON -> unload -> Pose transfer -> unload -> face lock.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .face_lock import lock_face_identity
from .garment_extract import extract_garment_from_person
from .memory import free_vram

PathLike = Union[str, Path, Image.Image]
Mode = Literal["both", "outfit_only", "pose_only"]
RefKind = Literal["clothed_person", "flat_garment"]


def _ensure_leffa_on_path(leffa_root: Path) -> None:
    root = str(leffa_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _as_pil(image: PathLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


class PoseClothPipeline:
    """
    Identity-safe pose + outfit transfer.

    Inputs
    ------
    base_image : person whose face / body proportions must stay consistent
    ref_image  : usually a **person wearing clothes** (pose + outfit source).
                 Flat product garment photos also work if ref_kind='flat_garment'.
    """

    def __init__(
        self,
        leffa_root: str | Path = "./Leffa",
        ckpt_dir: str | Path | None = None,
        dtype: str = "float16",
        enable_face_lock: bool = True,
        default_ref_kind: RefKind = "clothed_person",
    ) -> None:
        self.leffa_root = Path(leffa_root)
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else self.leffa_root / "ckpts"
        self.dtype = dtype
        self.enable_face_lock = enable_face_lock
        self.default_ref_kind = default_ref_kind

        _ensure_leffa_on_path(self.leffa_root)

        # Lazy-loaded preprocess / model handles
        self._parsing = None
        self._openpose = None
        self._densepose = None
        self._vt_inference = None
        self._pt_inference = None
        self._face_app = None
        self._active_model: Optional[str] = None

    # ------------------------------------------------------------------ setup
    def download_checkpoints(self) -> None:
        from huggingface_hub import snapshot_download

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        print("Downloading Leffa checkpoints (first run can take a while)...")
        snapshot_download(repo_id="franciszzj/Leffa", local_dir=str(self.ckpt_dir))
        print("Checkpoints ready at", self.ckpt_dir)

    def _load_preprocessors(self) -> None:
        if self._parsing is not None:
            return

        from leffa_utils.densepose_predictor import DensePosePredictor
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
        self._densepose = DensePosePredictor(
            config_path=str(ckpt / "densepose" / "densepose_rcnn_R_50_FPN_s1x.yaml"),
            weights_path=str(ckpt / "densepose" / "model_final_162be9.pkl"),
        )

    def _unload_diffusion(self) -> None:
        self._vt_inference = None
        self._pt_inference = None
        self._active_model = None
        free_vram()

    def _load_vton(self, model_type: str = "viton_hd") -> None:
        if self._active_model == f"vton_{model_type}" and self._vt_inference is not None:
            return
        self._unload_diffusion()

        from leffa.inference import LeffaInference
        from leffa.model import LeffaModel

        weight = (
            self.ckpt_dir / "virtual_tryon.pth"
            if model_type == "viton_hd"
            else self.ckpt_dir / "virtual_tryon_dc.pth"
        )
        model = LeffaModel(
            pretrained_model_name_or_path=str(
                self.ckpt_dir / "stable-diffusion-inpainting"
            ),
            pretrained_model=str(weight),
            dtype=self.dtype,
        )
        self._vt_inference = LeffaInference(model=model)
        self._active_model = f"vton_{model_type}"
        print(f"Loaded VTON model: {model_type}")

    def _load_pose(self) -> None:
        if self._active_model == "pose" and self._pt_inference is not None:
            return
        self._unload_diffusion()

        from leffa.inference import LeffaInference
        from leffa.model import LeffaModel

        model = LeffaModel(
            pretrained_model_name_or_path=str(
                self.ckpt_dir / "stable-diffusion-xl-1.0-inpainting-0.1"
            ),
            pretrained_model=str(self.ckpt_dir / "pose_transfer.pth"),
            dtype=self.dtype,
        )
        self._pt_inference = LeffaInference(model=model)
        self._active_model = "pose"
        print("Loaded pose-transfer model")

    def _get_face_app(self):
        if self._face_app is not None:
            return self._face_app
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
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
        """Parse a clothed person and isolate their outfit for VTON."""
        from leffa_utils.utils import resize_and_center

        self._load_preprocessors()
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

    # --------------------------------------------------------------- inference
    def _run_control(
        self,
        src_image: Image.Image,
        ref_image: Image.Image,
        control_type: Literal["virtual_tryon", "pose_transfer"],
        step: int = 30,
        scale: float = 2.5,
        seed: int = 42,
        ref_acceleration: bool = True,
        vt_model_type: str = "viton_hd",
        vt_garment_type: str = "upper_body",
        vt_repaint: bool = False,
    ) -> Image.Image:
        from leffa.transform import LeffaTransform
        from leffa_utils.utils import (
            get_agnostic_mask_dc,
            get_agnostic_mask_hd,
            resize_and_center,
        )

        self._load_preprocessors()

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
            self._load_vton(vt_model_type)
            inference = self._vt_inference
        else:
            mask = Image.fromarray(np.ones_like(src_array) * 255)
            iuv = self._densepose.predict_iuv(src_array)[:, :, ::-1]
            densepose = Image.fromarray(iuv)
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
        return output["generated_image"][0].convert("RGB")

    def generate(
        self,
        base_image: PathLike,
        ref_image: PathLike,
        mode: Mode = "both",
        garment_type: str = "upper_body",
        vt_model_type: str = "viton_hd",
        steps: int = 30,
        guidance_scale: float = 2.5,
        seed: int = 42,
        ref_acceleration: bool = True,
        face_lock: Optional[bool] = None,
        ref_kind: Optional[RefKind] = None,
        return_debug: bool = False,
    ) -> Image.Image | Tuple[Image.Image, dict]:
        """
        Run outfit and/or pose transfer.

        Default ref_kind='clothed_person': the pose/outfit image is a person
        wearing clothes. Clothing is parsed out for VTON; the full person
        image is still used for pose transfer.

        Leffa pose-transfer convention
        ------------------------------
        src = target pose image
        ref = appearance (identity / clothes) image
        """
        base = _as_pil(base_image)
        ref = _as_pil(ref_image)
        kind = ref_kind or self.default_ref_kind
        do_face = self.enable_face_lock if face_lock is None else face_lock
        debug: dict = {}

        current = base

        if mode in ("both", "outfit_only"):
            print("Step 1/2: Outfit transfer (VTON) — clothes from ref onto base...")
            if kind == "clothed_person":
                vton_ref = self._garment_ref_from_clothed_person(ref, garment_type)
                debug["garment_ref"] = vton_ref.copy()
            else:
                vton_ref = ref
            # Person = base; garment ref = extracted clothes (or flat garment)
            current = self._run_control(
                src_image=current,
                ref_image=vton_ref,
                control_type="virtual_tryon",
                step=steps,
                scale=guidance_scale,
                seed=seed,
                ref_acceleration=ref_acceleration,
                vt_model_type=vt_model_type,
                vt_garment_type=garment_type,
            )
            debug["after_vton"] = current.copy()
            self._unload_diffusion()
        if mode in ("both", "pose_only"):
            print("Step 2/2: Pose transfer — appearance from dressed base, pose from ref...")
            # Appearance = current (base or dressed), pose = ref
            appearance = current if mode == "both" else base
            current = self._run_control(
                src_image=ref,
                ref_image=appearance,
                control_type="pose_transfer",
                step=steps,
                scale=guidance_scale,
                seed=seed,
                ref_acceleration=ref_acceleration,
            )
            debug["after_pose"] = current.copy()
            self._unload_diffusion()

        if do_face and mode != "outfit_only":
            # After pose warp, re-apply base face. For outfit-only, Leffa usually
            # already keeps the face; still allow explicit lock if requested.
            print("Face identity lock...")
            current = lock_face_identity(
                base_image=base,
                generated_image=current,
                face_app=self._get_face_app(),
            )
            debug["after_face_lock"] = current.copy()
        elif do_face and mode == "outfit_only":
            print("Face identity lock (outfit mode)...")
            current = lock_face_identity(
                base_image=base,
                generated_image=current,
                face_app=self._get_face_app(),
                blend=0.75,
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
