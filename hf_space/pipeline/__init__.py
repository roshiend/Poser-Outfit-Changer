"""Pose & outfit transfer helpers for Hugging Face Spaces."""

from .body_lock import (
    align_pose_donor_to_base_body,
    match_result_body_to_base,
    soft_preserve_torso,
)
from .face_lock import face_identity_similarity, lock_face_identity
from .fidelity_v2 import FidelityPoseClothPipeline
from .garment_extract import (
    GarmentAnalysis,
    GarmentExtractionError,
    analyse_garment,
    extract_garment_from_person,
)
from .memory import free_vram
from .outfit_quality import OutfitQualityDiagnostics, assess_outfit_preservation
from .pose_geometry import assess_pose_pair, retarget_densepose_control, retarget_skeleton

PoseClothPipeline = FidelityPoseClothPipeline

__all__ = [
    "PoseClothPipeline",
    "FidelityPoseClothPipeline",
    "lock_face_identity",
    "face_identity_similarity",
    "GarmentAnalysis",
    "GarmentExtractionError",
    "analyse_garment",
    "extract_garment_from_person",
    "OutfitQualityDiagnostics",
    "assess_outfit_preservation",
    "align_pose_donor_to_base_body",
    "match_result_body_to_base",
    "soft_preserve_torso",
    "assess_pose_pair",
    "retarget_skeleton",
    "retarget_densepose_control",
    "free_vram",
]
