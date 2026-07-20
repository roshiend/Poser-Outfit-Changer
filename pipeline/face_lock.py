"""Paste the base person's face onto a generated image to preserve identity."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def _to_bgr(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
    else:
        rgb = image
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        elif rgb.shape[2] == 4:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _largest_face(faces):
    if not faces:
        return None
    return max(
        faces,
        key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
    )


def _face_ellipse_mask(
    shape: Tuple[int, int],
    bbox: np.ndarray,
    feather: float = 0.35,
) -> np.ndarray:
    h, w = shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    # Slightly enlarge vertically to cover forehead / chin.
    ax = max(1.0, (x2 - x1) * 0.52)
    ay = max(1.0, (y2 - y1) * 0.62)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(
        mask,
        (int(cx), int(cy)),
        (int(ax), int(ay)),
        0,
        0,
        360,
        1.0,
        -1,
    )
    k = max(3, int(min(ax, ay) * feather) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def lock_face_identity(
    base_image: Image.Image | np.ndarray,
    generated_image: Image.Image | np.ndarray,
    face_app=None,
    blend: float = 0.92,
) -> Image.Image:
    """
    Overlay the base face onto the generated image with a feathered blend.

    Falls back to returning the generated image unchanged if faces cannot be
    detected or InsightFace is unavailable.
    """
    base_bgr = _to_bgr(base_image)
    gen_bgr = _to_bgr(generated_image)

    if face_app is None:
        try:
            from insightface.app import FaceAnalysis

            face_app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            face_app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as exc:
            print(f"[face_lock] InsightFace unavailable ({exc}); skipping face lock.")
            return (
                generated_image
                if isinstance(generated_image, Image.Image)
                else Image.fromarray(
                    cv2.cvtColor(_to_bgr(generated_image), cv2.COLOR_BGR2RGB)
                )
            )

    base_faces = face_app.get(base_bgr)
    gen_faces = face_app.get(gen_bgr)
    src_face = _largest_face(base_faces)
    dst_face = _largest_face(gen_faces)

    if src_face is None or dst_face is None:
        print("[face_lock] Face not found on base or result; skipping face lock.")
        return _to_pil(gen_bgr)

    src_pts = np.float32(src_face.kps)
    dst_pts = np.float32(dst_face.kps)
    matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if matrix is None:
        print("[face_lock] Could not estimate face warp; skipping.")
        return _to_pil(gen_bgr)

    warped = cv2.warpAffine(
        base_bgr,
        matrix,
        (gen_bgr.shape[1], gen_bgr.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Warp base bbox into generated space for the blend mask.
    x1, y1, x2, y2 = src_face.bbox
    corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
    warped_corners = cv2.transform(corners, matrix).reshape(-1, 2)
    wx1, wy1 = warped_corners.min(axis=0)
    wx2, wy2 = warped_corners.max(axis=0)
    mask = _face_ellipse_mask(gen_bgr.shape, np.array([wx1, wy1, wx2, wy2]))
    mask = np.clip(mask * float(blend), 0.0, 1.0)[..., None]

    # Match color loosely inside the face region.
    m = mask[..., 0] > 0.05
    if np.any(m):
        for c in range(3):
            src_mean = float(warped[..., c][m].mean())
            dst_mean = float(gen_bgr[..., c][m].mean())
            if src_mean > 1e-3:
                warped[..., c] = np.clip(
                    warped[..., c].astype(np.float32) * (dst_mean / src_mean),
                    0,
                    255,
                ).astype(np.uint8)

    out = (
        warped.astype(np.float32) * mask + gen_bgr.astype(np.float32) * (1.0 - mask)
    ).astype(np.uint8)
    return _to_pil(out)
