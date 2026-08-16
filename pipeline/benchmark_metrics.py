"""Objective metrics for comparing pose-retarget strengths.

The metrics intentionally avoid raw pixel error because benchmark outputs are
supposed to change pose and clothing. Instead we compare pose directions via
joint angles, body proportions via normalized segment lengths, and identity via
InsightFace separately.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .pose_geometry import (
    BONES,
    L_HIP,
    L_SHOULDER,
    NECK,
    R_HIP,
    R_SHOULDER,
)

# Triplets are (point A, vertex B, point C). They measure the major limb and
# torso articulations while remaining invariant to global translation/scale.
ANGLE_TRIPLETS = (
    (NECK, R_SHOULDER, 3),
    (R_SHOULDER, 3, 4),
    (NECK, L_SHOULDER, 6),
    (L_SHOULDER, 6, 7),
    (NECK, R_HIP, 9),
    (R_HIP, 9, 10),
    (NECK, L_HIP, 12),
    (L_HIP, 12, 13),
)


def _valid_segment(points: np.ndarray, valid: np.ndarray, a: int, b: int) -> float | None:
    if not (bool(valid[a]) and bool(valid[b])):
        return None
    length = float(np.linalg.norm(points[b] - points[a]))
    return length if length > 2.0 else None


def _angle_deg(points: np.ndarray, valid: np.ndarray, a: int, b: int, c: int) -> float | None:
    if not (bool(valid[a]) and bool(valid[b]) and bool(valid[c])):
        return None
    u = points[a] - points[b]
    v = points[c] - points[b]
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu <= 2.0 or nv <= 2.0:
        return None
    cosine = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def pose_angle_error_deg(
    generated_points: np.ndarray,
    generated_valid: np.ndarray,
    target_points: np.ndarray,
    target_valid: np.ndarray,
    triplets: Iterable[tuple[int, int, int]] = ANGLE_TRIPLETS,
) -> float | None:
    """Median absolute articulation-angle error against the target pose."""
    errors: list[float] = []
    for a, b, c in triplets:
        got = _angle_deg(generated_points, generated_valid, a, b, c)
        wanted = _angle_deg(target_points, target_valid, a, b, c)
        if got is not None and wanted is not None:
            errors.append(abs(got - wanted))
    return float(np.median(errors)) if errors else None


def _body_scale(points: np.ndarray, valid: np.ndarray) -> float | None:
    """Stable per-person scale used only to normalize lengths before comparison."""
    lengths: list[float] = []
    shoulder = _valid_segment(points, valid, R_SHOULDER, L_SHOULDER)
    hip = _valid_segment(points, valid, R_HIP, L_HIP)
    if shoulder is not None:
        lengths.append(shoulder)
    if hip is not None:
        lengths.append(hip)
    if bool(valid[NECK]) and bool(valid[R_HIP]) and bool(valid[L_HIP]):
        hip_mid = (points[R_HIP] + points[L_HIP]) * 0.5
        torso = float(np.linalg.norm(hip_mid - points[NECK]))
        if torso > 2.0:
            lengths.append(torso)
    if not lengths:
        for a, b in BONES:
            value = _valid_segment(points, valid, a, b)
            if value is not None:
                lengths.append(value)
    return float(np.median(lengths)) if lengths else None


def normalized_bone_profile(points: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    """Return visible major segment lengths normalized by overall body scale."""
    scale = _body_scale(points, valid)
    if scale is None or scale <= 2.0:
        return {}
    profile: dict[str, float] = {}
    for a, b in BONES:
        value = _valid_segment(points, valid, a, b)
        if value is not None:
            profile[f"{a}-{b}"] = value / scale
    for name, a, b in (
        ("shoulder_width", R_SHOULDER, L_SHOULDER),
        ("hip_width", R_HIP, L_HIP),
    ):
        value = _valid_segment(points, valid, a, b)
        if value is not None:
            profile[name] = value / scale
    return profile


def body_proportion_error(
    generated_points: np.ndarray,
    generated_valid: np.ndarray,
    base_points: np.ndarray,
    base_valid: np.ndarray,
) -> float | None:
    """RMS multiplicative proportion error; 0.10 is roughly a 10% aggregate mismatch."""
    got = normalized_bone_profile(generated_points, generated_valid)
    wanted = normalized_bone_profile(base_points, base_valid)
    common = sorted(set(got) & set(wanted))
    if len(common) < 4:
        return None
    log_errors = np.asarray(
        [math.log(max(got[k], 1e-6) / max(wanted[k], 1e-6)) for k in common],
        dtype=np.float32,
    )
    rms_log_error = float(np.sqrt(np.mean(np.square(log_errors))))
    return float(math.expm1(rms_log_error))


def composite_fidelity_score(
    identity_similarity: float | None,
    pose_error_deg: float | None,
    proportion_error: float | None,
    identity_weight: float = 0.45,
    pose_weight: float = 0.35,
    body_weight: float = 0.20,
) -> float | None:
    """Convenience 0..100 score for ranking benchmark variants, not a scientific metric."""
    components: list[tuple[float, float]] = []
    if identity_similarity is not None:
        # InsightFace cosine values for same identity are normally positive. Keep
        # the score bounded instead of interpreting the cosine as a probability.
        components.append((float(np.clip(identity_similarity, 0.0, 1.0)), identity_weight))
    if pose_error_deg is not None:
        components.append((1.0 - float(np.clip(pose_error_deg / 60.0, 0.0, 1.0)), pose_weight))
    if proportion_error is not None:
        components.append((1.0 - float(np.clip(proportion_error / 0.50, 0.0, 1.0)), body_weight))
    if not components:
        return None
    weight_sum = sum(weight for _, weight in components)
    return 100.0 * sum(value * weight for value, weight in components) / weight_sum
