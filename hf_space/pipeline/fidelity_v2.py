"""Fidelity pipeline: garment intelligence, anatomy-aware pose and identity diagnostics."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from PIL import Image

from .body_lock import align_pose_donor_to_base_body, person_bbox_from_parse
from .face_lock import lock_face_identity
from .garment_extract import GarmentAnalysis, extract_garment_from_person
from .leffa_sequential import DensePoseUnavailableError, Mode, PathLike, PoseClothPipeline as BasePoseClothPipeline, RefKind, VtonModel, _as_pil, _resolve_vton_model
from .memory import free_vram
from .outfit_quality import assess_outfit_preservation
from .pose_geometry import draw_skeleton, openpose_points, retarget_densepose_control


class FidelityPoseClothPipeline(BasePoseClothPipeline):
    """Leffa pipeline with guarded garment extraction and pose-control retargeting."""

    def _garment_ref_from_clothed_person(self, person: Image.Image, garment_type: str) -> tuple[Image.Image, GarmentAnalysis]:
        from leffa_utils.utils import resize_and_center
        self._load_preprocessors(require_real_densepose=False)
        person = resize_and_center(person.convert("RGB"), 768, 1024)
        parse_map, _ = self._parsing(person.resize((384, 512)))
        garment, analysis = extract_garment_from_person(person, parse_map, garment_type, (768, 1024), return_analysis=True)
        print(f"[garment] Extracted {analysis.resolved_type} clothing (requested={garment_type}, confidence={analysis.confidence:.2f})")
        return garment, analysis

    def _run_control_v2(self, src_image: Image.Image, ref_image: Image.Image, control_type: Literal["virtual_tryon", "pose_transfer"], step: int = 30, scale: float = 2.5, seed: int = 42, ref_acceleration: bool = True, vt_model_type: Literal["viton_hd", "dress_code"] = "viton_hd", vt_garment_type: str = "upper_body", vt_repaint: bool = False, pose_base_keypoints: dict | None = None, pose_retarget_strength: float = 0.0) -> tuple[Image.Image, Image.Image, Image.Image, dict]:
        if control_type == "virtual_tryon":
            generated, mask, densepose = super()._run_control(src_image, ref_image, control_type, step, scale, seed, ref_acceleration, vt_model_type, vt_garment_type, vt_repaint)
            return generated, mask, densepose, {}

        from leffa.transform import LeffaTransform
        from leffa_utils.utils import resize_and_center
        self._load_preprocessors(require_real_densepose=True)
        if not self._densepose_is_real:
            raise DensePoseUnavailableError("Pose transfer cannot run with fallback DensePose.")
        src_image = resize_and_center(src_image.convert("RGB"), 768, 1024)
        ref_image = resize_and_center(ref_image.convert("RGB"), 768, 1024)
        src_array = np.asarray(src_image)
        mask = Image.fromarray(np.ones_like(src_array) * 255)
        iuv = self._densepose.predict_iuv(src_array)[:, :, ::-1]
        densepose_original = Image.fromarray(iuv).convert("RGB")
        densepose = densepose_original
        control_debug: dict = {"pose_densepose_original": densepose_original.copy()}
        if pose_base_keypoints is not None and float(pose_retarget_strength) > 0.0:
            donor_keypoints = self._openpose(src_image.resize((384, 512)))
            base_points, base_valid = openpose_points(pose_base_keypoints)
            donor_points, donor_valid = openpose_points(donor_keypoints)
            densepose, retargeted_points, diagnostics = retarget_densepose_control(densepose_original, base_points, base_valid, donor_points, donor_valid, float(pose_retarget_strength))
            control_debug["pose_retarget_diagnostics"] = diagnostics.to_dict()
            control_debug["pose_skeleton_before"] = draw_skeleton(src_image, donor_points, donor_valid)
            control_debug["pose_skeleton_retargeted"] = draw_skeleton(src_image, retargeted_points, donor_valid)
            if diagnostics.safe_to_retarget:
                print(f"[pose] Anatomy retarget applied: strength={float(pose_retarget_strength):.2f}, common_joints={diagnostics.common_body_joints}, proportion_gap={diagnostics.proportion_gap:.2f}")
            else:
                print(f"[pose] Anatomy retarget skipped safely: {diagnostics.reason}")
        else:
            control_debug["pose_retarget_diagnostics"] = {"safe_to_retarget": False, "reason": "disabled"}
        control_debug["pose_densepose_retargeted"] = densepose.copy()
        self._unload_preprocessors()
        free_vram()
        self._load_pose()
        data = LeffaTransform()({"src_image": [src_image], "ref_image": [ref_image], "mask": [mask], "densepose": [densepose]})
        output = self._pt_inference(data, ref_acceleration=ref_acceleration, num_inference_steps=int(step), guidance_scale=float(scale), seed=int(seed), repaint=False)
        return output["generated_image"][0].convert("RGB"), mask.convert("RGB"), densepose.convert("RGB"), control_debug

    def generate(self, base_image: PathLike, ref_image: PathLike, mode: Mode = "both", garment_type: str = "auto", vt_model_type: VtonModel = "auto", steps: int = 30, guidance_scale: float = 2.5, seed: int = 42, ref_acceleration: bool = True, face_lock: Optional[bool] = None, ref_kind: Optional[RefKind] = None, preserve_body: Optional[bool] = None, pose_retarget_strength: float = 0.65, return_debug: bool = False) -> Image.Image | Tuple[Image.Image, dict]:
        from leffa_utils.utils import resize_and_center
        if mode not in {"both", "outfit_only", "pose_only"}:
            raise ValueError(f"Unknown mode: {mode}")
        if garment_type not in {"auto", "upper_body", "lower_body", "dresses"}:
            raise ValueError(f"Unknown garment type: {garment_type}")
        kind = ref_kind or self.default_ref_kind
        if mode in {"both", "outfit_only"} and garment_type == "auto" and kind == "flat_garment":
            raise ValueError("Auto garment detection requires a clothed-person reference; choose upper/lower/full for flat garment photos.")
        if mode in {"both", "pose_only"} and not self.pose_enabled:
            raise RuntimeError("Pose transfer is disabled on this Colab runtime because the SDXL pose model often exceeds free-tier memory. Nothing was silently downgraded. Use Outfit only, move to a larger GPU, or set LEFFA_ALLOW_POSE=1 before creating the pipeline.")

        base, ref = _as_pil(base_image), _as_pil(ref_image)
        do_face = self.enable_face_lock if face_lock is None else face_lock
        do_body = self.preserve_body if preserve_body is None else preserve_body
        resolved_garment_type = garment_type
        debug: dict = {"requested_garment_type": garment_type, "pose_retarget_strength": float(np.clip(pose_retarget_strength, 0.0, 1.0))}
        base, ref = resize_and_center(base, 768, 1024), resize_and_center(ref, 768, 1024)
        current, base_bbox, base_keypoints, base_parse_for_quality = base, None, None, None
        if do_body and mode in {"both", "pose_only"}:
            self._load_preprocessors(require_real_densepose=False)
            base_parse, _ = self._parsing(base.resize((384, 512)))
            base_parse_for_quality = base_parse
            base_bbox = person_bbox_from_parse(base_parse.resize((768, 1024), Image.NEAREST))
            base_keypoints = self._openpose(base.resize((384, 512)))

        if mode in {"both", "outfit_only"}:
            if kind == "clothed_person":
                vton_ref, garment_analysis = self._garment_ref_from_clothed_person(ref, garment_type)
                resolved_garment_type = garment_analysis.resolved_type
                debug["garment_analysis"] = garment_analysis.to_dict()
                debug["garment_ref"] = vton_ref.copy()
            else:
                vton_ref = ref
                debug["garment_analysis"] = {"requested_type": garment_type, "resolved_type": garment_type, "confidence": None, "reason": "flat garment photo; explicit selection"}
            selected_vton = _resolve_vton_model(resolved_garment_type, vt_model_type)
            debug["resolved_garment_type"] = resolved_garment_type
            debug["selected_vton_model"] = selected_vton
            print(f"Step 1/2: Outfit transfer using {selected_vton} ({resolved_garment_type})...")
            if base_parse_for_quality is None:
                self._load_preprocessors(require_real_densepose=False)
                base_parse_for_quality, _ = self._parsing(base.resize((384, 512)))
            current, vt_mask, vt_densepose, vt_debug = self._run_control_v2(current, vton_ref, "virtual_tryon", steps, guidance_scale, seed, ref_acceleration, selected_vton, resolved_garment_type)
            debug.update(vt_debug)
            debug["outfit_mask"], debug["outfit_densepose"], debug["after_vton"] = vt_mask, vt_densepose, current.copy()
            outfit_quality = assess_outfit_preservation(base, current, base_parse_for_quality)
            debug["outfit_quality"] = outfit_quality.to_dict()
            for warning in outfit_quality.warnings:
                print(f"[outfit-quality] WARNING: {warning}")
            self._unload_diffusion()
        else:
            debug["selected_vton_model"], debug["resolved_garment_type"] = "not used", "not used"

        if mode in {"both", "pose_only"}:
            print("Step 2/2: Pose transfer with real DensePose...")
            appearance, pose_src = (current if mode == "both" else base), ref
            if do_body:
                self._load_preprocessors(require_real_densepose=False)
                donor_parse, _ = self._parsing(ref.resize((384, 512)))
                donor_bbox = person_bbox_from_parse(donor_parse.resize((768, 1024), Image.NEAREST))
                pose_src = align_pose_donor_to_base_body(ref, appearance, donor_bbox, base_bbox)
                debug["aligned_pose_donor"] = pose_src.copy()
            effective_strength = float(np.clip(pose_retarget_strength, 0.0, 1.0)) if do_body else 0.0
            current, pose_mask, pose_densepose, pose_debug = self._run_control_v2(pose_src, appearance, "pose_transfer", steps, guidance_scale, seed, ref_acceleration, pose_base_keypoints=base_keypoints, pose_retarget_strength=effective_strength)
            debug.update(pose_debug)
            debug["pose_mask"], debug["pose_densepose"], debug["after_pose"] = pose_mask, pose_densepose, current.copy()
            self._unload_diffusion()

        if do_face:
            current, identity_info = lock_face_identity(base, current, face_app=self._get_face_app(), blend=0.72 if mode == "outfit_only" else 0.84, adaptive_pose=True, adaptive_identity=True, return_info=True)
            debug["identity_info"], debug["after_face_lock"] = identity_info, current.copy()
        free_vram()
        return (current, debug) if return_debug else current
