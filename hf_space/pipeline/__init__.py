"""Pose & outfit transfer helpers for Colab, local and Hugging Face use."""

from .body_lock import (
    align_pose_donor_to_base_body,
    match_result_body_to_base,
    soft_preserve_torso,
)
from .colab_pipeline import ColabAwareFidelityPipeline
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

# Public default. Outside Colab this behaves like FidelityPoseClothPipeline; on
# managed Colab it activates staged low-memory inference automatically.
PoseClothPipeline = ColabAwareFidelityPipeline

__all__ = [
    "PoseClothPipeline",
    "ColabAwareFidelityPipeline",
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
