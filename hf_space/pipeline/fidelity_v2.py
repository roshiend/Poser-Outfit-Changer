"""Second-generation fidelity pipeline: anatomy-aware pose control + identity diagnostics."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from PIL import Image

from .body_lock import align_pose_donor_to_base_body, person_bbox_from_parse
from .face_lock import lock_face_identity
from .garment_extract import extract_garment_from_person
from .leffa_sequential import (
    DensePoseUnavailableError,
    Mode,
    PathLike,
    PoseClothPipeline as BasePoseClothPipeline,
    RefKind,
    VtonModel,
    _as_pil,
    _resolve_vton_model,
)
from .memory import free_vram
from .pose_geometry import draw_skeleton, openpose_points, retarget_densepose_control


class FidelityPoseClothPipeline(BasePoseClothPipeline):
    """Leffa pipeline with conservative skeleton retargeting before pose diffusion."""

    def _garment_ref_from_clothed_person(self, person: Image.Image, garment_type: str) -> Image.Image:
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

    def _run_control_v2(
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
        pose_base_keypoints: dict | None = None,
        pose_retarget_strength: float = 0.0,
    ) -> tuple[Image.Image, Image.Image, Image.Image, dict]:
        if control_type == "virtual_tryon":
            generated, mask, densepose = super()._run_control(
                src_image=src_image,
                ref_image=ref_image,
                control_type=control_type,
                step=step,
                scale=scale,
                seed=seed,
                ref_acceleration=ref_acceleration,
                vt_model_type=vt_model_type,
                vt_garment_type=vt_garment_type,
                vt_repaint=vt_repaint,
            )
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
        control_debug = {"pose_densepose_original": densepose_original.copy()}

        if pose_base_keypoints is not None and float(pose_retarget_strength) > 0.0:
            donor_keypoints = self._openpose(src_image.resize((384, 512)))
            base_points, base_valid = openpose_points(pose_base_keypoints)
            donor_points, donor_valid = openpose_points(donor_keypoints)
            densepose, retargeted_points, diagnostics = retarget_densepose_control(
                densepose=densepose_original,
                base_points=base_points,
                base_valid=base_valid,
                donor_points=donor_points,
                donor_valid=donor_valid,
                strength=float(pose_retarget_strength),
            )
            control_debug["pose_retarget_diagnostics"] = diagnostics.to_dict()
            control_debug["pose_skeleton_before"] = draw_skeleton(src_image, donor_points, donor_valid)
            control_debug["pose_skeleton_retargeted"] = draw_skeleton(src_image, retargeted_points, donor_valid)
            if diagnostics.safe_to_retarget:
                print(
                    "[pose] Anatomy retarget applied: "
                    f"strength={float(pose_retarget_strength):.2f}, "
                    f"common_joints={diagnostics.common_body_joints}, "
                    f"proportion_gap={diagnostics.proportion_gap:.2f}"
                )
            else:
                print(f"[pose] Anatomy retarget skipped safely: {diagnostics.reason}")
        else:
            control_debug["pose_retarget_diagnostics"] = {"safe_to_retarget": False, "reason": "disabled"}

        control_debug["pose_densepose_retargeted"] = densepose.copy()
        self._unload_preprocessors()
        free_vram()
        self._load_pose()
        inference = self._pt_inference

        data = {"src_image": [src_image], "ref_image": [ref_image], "mask": [mask], "densepose": [densepose]}
        data = LeffaTransform()(data)
        output = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=int(step),
            guidance_scale=float(scale),
            seed=int(seed),
            repaint=False,
        )
        generated = output["generated_image"][0].convert("RGB")
        return generated, mask.convert("RGB"), densepose.convert("RGB"), control_debug

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
        pose_retarget_strength: float = 0.65,
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
        debug = {
            "selected_vton_model": selected_vton,
            "pose_retarget_strength": float(np.clip(pose_retarget_strength, 0.0, 1.0)),
        }

        base = resize_and_center(base, 768, 1024)
        ref = resize_and_center(ref, 768, 1024)
        current = base
        base_bbox = None
        base_keypoints = None

        if do_body and mode in {"both", "pose_only"}:
            self._load_preprocessors(require_real_densepose=False)
            base_parse, _ = self._parsing(base.resize((384, 512)))
            base_bbox = person_bbox_from_parse(base_parse.resize((768, 1024), Image.NEAREST))
            base_keypoints = self._openpose(base.resize((384, 512)))

        if mode in {"both", "outfit_only"}:
            print(f"Step 1/2: Outfit transfer using {selected_vton}...")
            if kind == "clothed_person":
                vton_ref = self._garment_ref_from_clothed_person(ref, garment_type)
                debug["garment_ref"] = vton_ref.copy()
            else:
                vton_ref = ref
            current, vt_mask, vt_densepose, vt_debug = self._run_control_v2(
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
            debug.update(vt_debug)
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
                donor_bbox = person_bbox_from_parse(donor_parse.resize((768, 1024), Image.NEAREST))
                pose_src = align_pose_donor_to_base_body(
                    pose_donor=ref,
                    base_person=appearance,
                    donor_bbox=donor_bbox,
                    base_bbox=base_bbox,
                )
                debug["aligned_pose_donor"] = pose_src.copy()

            effective_retarget_strength = float(np.clip(pose_retarget_strength, 0.0, 1.0)) if do_body else 0.0
            current, pose_mask, pose_densepose, pose_debug = self._run_control_v2(
                src_image=pose_src,
                ref_image=appearance,
                control_type="pose_transfer",
                step=steps,
                scale=guidance_scale,
                seed=seed,
                ref_acceleration=ref_acceleration,
                pose_base_keypoints=base_keypoints,
                pose_retarget_strength=effective_retarget_strength,
            )
            debug.update(pose_debug)
            debug["pose_mask"] = pose_mask
            debug["pose_densepose"] = pose_densepose
            debug["after_pose"] = current.copy()
            self._unload_diffusion()

        if do_face:
            print("Adaptive face identity lock with embedding diagnostics...")
            requested_blend = 0.72 if mode == "outfit_only" else 0.84
            current, identity_info = lock_face_identity(
                base_image=base,
                generated_image=current,
                face_app=self._get_face_app(),
                blend=requested_blend,
                adaptive_pose=True,
                adaptive_identity=True,
                return_info=True,
            )
            debug["identity_info"] = identity_info
            debug["after_face_lock"] = current.copy()

        free_vram()
        if return_debug:
            return current, debug
        return current
