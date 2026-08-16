"""Conservative anatomy-aware retargeting for Leffa DensePose controls.

Leffa already ships an OpenPose wrapper that returns 18 COCO-style body points.
This module uses those landmarks to measure whether a base/reference pair is
suitable for anatomy retargeting, reconstructs the reference skeleton with
base-like bone lengths while preserving reference joint directions, and
piecewise-warps the DensePose IUV control rather than stretching the final RGB
result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

NOSE = 0
NECK = 1
R_SHOULDER, R_ELBOW, R_WRIST = 2, 3, 4
L_SHOULDER, L_ELBOW, L_WRIST = 5, 6, 7
R_HIP, R_KNEE, R_ANKLE = 8, 9, 10
L_HIP, L_KNEE, L_ANKLE = 11, 12, 13
R_EYE, L_EYE, R_EAR, L_EAR = 14, 15, 16, 17

BODY_JOINTS = tuple(range(14))
CORE_JOINTS = (NECK, R_SHOULDER, L_SHOULDER, R_HIP, L_HIP)

BONES = (
    (NECK, R_SHOULDER), (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (NECK, L_SHOULDER), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
)

SKELETON_EDGES = (
    (NOSE, NECK), (NECK, R_SHOULDER), (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (NECK, L_SHOULDER), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (NECK, R_HIP), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
    (NECK, L_HIP), (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
)


@dataclass(frozen=True)
class PoseDiagnostics:
    common_body_joints: int
    completeness: float
    proportion_gap: float
    safe_to_retarget: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def openpose_points(keypoints: dict, out_size=(768, 1024), source_size=(384, 512)):
    raw = np.asarray(keypoints.get("pose_keypoints_2d", []), dtype=np.float32)
    if raw.shape != (18, 2):
        padded = np.zeros((18, 2), dtype=np.float32)
        n = min(len(raw), 18) if raw.ndim == 2 else 0
        if n:
            padded[:n] = raw[:n, :2]
        raw = padded
    src_w, src_h = source_size
    out_w, out_h = out_size
    points = raw.copy()
    points[:, 0] *= float(out_w) / float(src_w)
    points[:, 1] *= float(out_h) / float(src_h)
    valid = (raw[:, 0] > 1.0) & (raw[:, 1] > 1.0)
    return points, valid


def _length(points, a, b):
    return float(np.linalg.norm(points[b] - points[a]))


def _safe_length(points, valid, a, b):
    if not (valid[a] and valid[b]):
        return None
    value = _length(points, a, b)
    return value if value > 2.0 else None


def assess_pose_pair(base_points, base_valid, donor_points, donor_valid):
    common = base_valid & donor_valid
    common_body = int(common[list(BODY_JOINTS)].sum())
    completeness = common_body / float(len(BODY_JOINTS))
    ratios = []
    for a, b in BONES:
        bl = _safe_length(base_points, base_valid, a, b)
        dl = _safe_length(donor_points, donor_valid, a, b)
        if bl is not None and dl is not None:
            ratios.append(bl / dl)
    if ratios:
        logs = np.abs(np.log(np.clip(np.asarray(ratios, dtype=np.float32), 1e-4, 1e4)))
        proportion_gap = float(np.expm1(np.median(logs)))
    else:
        proportion_gap = 1.0
    if not all(bool(common[i]) for i in CORE_JOINTS):
        return PoseDiagnostics(common_body, completeness, proportion_gap, False, "missing core shoulders/hips")
    if common_body < 10:
        return PoseDiagnostics(common_body, completeness, proportion_gap, False, "too few common body joints")
    if len(ratios) < 6:
        return PoseDiagnostics(common_body, completeness, proportion_gap, False, "too few measurable limb segments")
    if proportion_gap > 0.55:
        return PoseDiagnostics(common_body, completeness, proportion_gap, False, "body proportions differ too much for safe warp")
    return PoseDiagnostics(common_body, completeness, proportion_gap, True, "ok")


def _desired_length(base_points, base_valid, donor_points, donor_valid, a, b, strength, ratio_limits=(0.72, 1.38)):
    donor_len = _safe_length(donor_points, donor_valid, a, b)
    if donor_len is None:
        return None
    base_len = _safe_length(base_points, base_valid, a, b)
    if base_len is None:
        return donor_len
    ratio = float(np.clip(base_len / donor_len, ratio_limits[0], ratio_limits[1]))
    return donor_len * (1.0 + float(np.clip(strength, 0.0, 1.0)) * (ratio - 1.0))


def _project_child(out, donor, base, donor_valid, base_valid, parent, child, strength):
    if not (donor_valid[parent] and donor_valid[child]):
        return
    vec = donor[child] - donor[parent]
    norm = float(np.linalg.norm(vec))
    if norm <= 2.0:
        return
    target_len = _desired_length(base, base_valid, donor, donor_valid, parent, child, strength)
    if target_len is None:
        return
    out[child] = out[parent] + vec / norm * target_len


def retarget_skeleton(base_points, base_valid, donor_points, donor_valid, strength=0.65):
    strength = float(np.clip(strength, 0.0, 1.0))
    out = np.asarray(donor_points, dtype=np.float32).copy()
    if strength <= 0.0:
        return out
    for shoulder in (R_SHOULDER, L_SHOULDER):
        _project_child(out, donor_points, base_points, donor_valid, base_valid, NECK, shoulder, strength)
    for parent, child in ((R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST)):
        _project_child(out, donor_points, base_points, donor_valid, base_valid, parent, child, strength)
    if donor_valid[R_HIP] and donor_valid[L_HIP] and donor_valid[NECK]:
        donor_hip_mid = (donor_points[R_HIP] + donor_points[L_HIP]) * 0.5
        donor_torso = donor_hip_mid - donor_points[NECK]
        donor_torso_len = float(np.linalg.norm(donor_torso))
        if donor_torso_len > 2.0:
            target_torso_len = donor_torso_len
            if base_valid[R_HIP] and base_valid[L_HIP] and base_valid[NECK]:
                base_hip_mid = (base_points[R_HIP] + base_points[L_HIP]) * 0.5
                base_torso_len = float(np.linalg.norm(base_hip_mid - base_points[NECK]))
                ratio = float(np.clip(base_torso_len / donor_torso_len, 0.78, 1.28))
                target_torso_len *= 1.0 + strength * (ratio - 1.0)
            new_mid = out[NECK] + donor_torso / donor_torso_len * target_torso_len
            donor_half = (donor_points[R_HIP] - donor_points[L_HIP]) * 0.5
            half_len = float(np.linalg.norm(donor_half))
            if half_len > 2.0:
                target_half_len = half_len
                if base_valid[R_HIP] and base_valid[L_HIP]:
                    base_half_len = _length(base_points, L_HIP, R_HIP) * 0.5
                    ratio = float(np.clip(base_half_len / half_len, 0.75, 1.35))
                    target_half_len *= 1.0 + strength * (ratio - 1.0)
                direction = donor_half / half_len
                out[R_HIP] = new_mid + direction * target_half_len
                out[L_HIP] = new_mid - direction * target_half_len
    for parent, child in ((R_HIP, R_KNEE), (R_KNEE, R_ANKLE), (L_HIP, L_KNEE), (L_KNEE, L_ANKLE)):
        _project_child(out, donor_points, base_points, donor_valid, base_valid, parent, child, strength)
    return out


def _dedupe_control_points(src: Iterable[np.ndarray], dst: Iterable[np.ndarray], min_distance=1.5):
    src_out, dst_out = [], []
    for s, d in zip(src, dst):
        s = np.asarray(s, dtype=np.float32)
        d = np.asarray(d, dtype=np.float32)
        if any(float(np.linalg.norm(s - prev)) < min_distance for prev in src_out):
            continue
        src_out.append(s)
        dst_out.append(d)
    return np.asarray(src_out, dtype=np.float32), np.asarray(dst_out, dtype=np.float32)


def _triangle_indices(tri, points, tolerance=2.5):
    ids = []
    for vertex in tri.reshape(3, 2):
        distances = np.linalg.norm(points - vertex[None, :], axis=1)
        idx = int(np.argmin(distances))
        if float(distances[idx]) > tolerance:
            return None
        ids.append(idx)
    if len(set(ids)) != 3:
        return None
    return tuple(ids)


def piecewise_affine_warp(image, src_points, dst_points, interpolation=cv2.INTER_NEAREST):
    arr = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    h, w = arr.shape[:2]
    border = np.asarray([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1], [w * 0.5, 0], [w - 1, h * 0.5], [w * 0.5, h - 1], [0, h * 0.5]], dtype=np.float32)
    src_all = np.vstack([src_points, border])
    dst_all = np.vstack([dst_points, border])
    src_all[:, 0] = np.clip(src_all[:, 0], 0, w - 1)
    src_all[:, 1] = np.clip(src_all[:, 1], 0, h - 1)
    dst_all[:, 0] = np.clip(dst_all[:, 0], 0, w - 1)
    dst_all[:, 1] = np.clip(dst_all[:, 1], 0, h - 1)
    src_all, dst_all = _dedupe_control_points(src_all, dst_all)
    if len(src_all) < 6:
        return Image.fromarray(arr.astype(np.uint8))
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for x, y in src_all:
        subdiv.insert((float(np.clip(x, 0, w - 1.001)), float(np.clip(y, 0, h - 1.001))))
    output = np.zeros_like(arr)
    coverage = np.zeros((h, w), dtype=np.uint8)
    for tri in subdiv.getTriangleList():
        ids = _triangle_indices(np.asarray(tri, dtype=np.float32), src_all)
        if ids is None:
            continue
        src_tri = src_all[list(ids)]
        dst_tri = dst_all[list(ids)]
        x1, y1, w1, h1 = cv2.boundingRect(src_tri.astype(np.float32))
        x2, y2, w2, h2 = cv2.boundingRect(dst_tri.astype(np.float32))
        if min(w1, h1, w2, h2) <= 0:
            continue
        if x1 < 0 or y1 < 0 or x1 + w1 > w or y1 + h1 > h or x2 < 0 or y2 < 0 or x2 + w2 > w or y2 + h2 > h:
            continue
        src_rel = src_tri - np.array([x1, y1], dtype=np.float32)
        dst_rel = dst_tri - np.array([x2, y2], dtype=np.float32)
        matrix = cv2.getAffineTransform(src_rel.astype(np.float32), dst_rel.astype(np.float32))
        patch = arr[y1:y1 + h1, x1:x1 + w1]
        warped = cv2.warpAffine(patch, matrix, (w2, h2), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = np.zeros((h2, w2), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(dst_rel).astype(np.int32), 255)
        roi = output[y2:y2 + h2, x2:x2 + w2]
        roi_cov = coverage[y2:y2 + h2, x2:x2 + w2]
        use = mask > 0
        roi[use] = warped[use]
        roi_cov[use] = 255
    output[coverage == 0] = arr[coverage == 0]
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))


def retarget_densepose_control(densepose, base_points, base_valid, donor_points, donor_valid, strength=0.65):
    diagnostics = assess_pose_pair(base_points, base_valid, donor_points, donor_valid)
    target = retarget_skeleton(base_points, base_valid, donor_points, donor_valid, strength=strength)
    if not diagnostics.safe_to_retarget or strength <= 0.0:
        return densepose.convert("RGB"), target, diagnostics
    common = base_valid & donor_valid
    src = donor_points[common]
    dst = target[common]
    displacement = np.linalg.norm(dst - src, axis=1)
    if len(displacement) and float(np.percentile(displacement, 90)) > 140.0:
        diagnostics = PoseDiagnostics(diagnostics.common_body_joints, diagnostics.completeness, diagnostics.proportion_gap, False, "retarget displacement is too large")
        return densepose.convert("RGB"), target, diagnostics
    warped = piecewise_affine_warp(densepose, src, dst, interpolation=cv2.INTER_NEAREST)
    return warped, target, diagnostics


def draw_skeleton(image, points, valid, line_thickness=4):
    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    for a, b in SKELETON_EDGES:
        if valid[a] and valid[b]:
            pa = tuple(np.round(points[a]).astype(int))
            pb = tuple(np.round(points[b]).astype(int))
            cv2.line(bgr, pa, pb, (255, 255, 255), line_thickness, cv2.LINE_AA)
    for idx in BODY_JOINTS:
        if valid[idx]:
            p = tuple(np.round(points[idx]).astype(int))
            cv2.circle(bgr, p, max(3, line_thickness + 1), (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(bgr, p, max(2, line_thickness - 1), (255, 255, 255), -1, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
