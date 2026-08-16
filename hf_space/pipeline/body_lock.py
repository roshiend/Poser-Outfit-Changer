"""Helpers for preserving base-person scale without warping the final image."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _to_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr


def person_bbox_from_parse(
    parse_map: Image.Image | np.ndarray,
    min_area: int = 500,
) -> tuple[int, int, int, int] | None:
    labels = np.asarray(parse_map)
    if labels.ndim == 3:
        labels = labels[..., 0]
    mask = labels > 0
    if int(mask.sum()) < min_area:
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def person_bbox_from_rgb(image: Image.Image | np.ndarray) -> tuple[int, int, int, int] | None:
    """Fallback bbox estimator used only when human parsing is unavailable."""
    rgb = _to_rgb(image)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    if bw * bh < 0.02 * h * w:
        return None
    return x, y, x + bw, y + bh


def align_pose_donor_to_base_body(
    pose_donor: Image.Image,
    base_person: Image.Image,
    donor_bbox: tuple[int, int, int, int] | None = None,
    base_bbox: tuple[int, int, int, int] | None = None,
    canvas_size: tuple[int, int] = (768, 1024),
) -> Image.Image:
    """
    Uniformly scale/recenter the target-pose person before DensePose extraction.

    The old implementation independently adjusted width after height scaling.
    That can change limb angles and torso geometry before DensePose is computed.
    This version uses one uniform scale, preserving the reference pose geometry,
    while matching the base person's overall on-canvas height and center.
    """
    w, h = canvas_size
    donor = pose_donor.convert("RGB").resize((w, h), Image.BICUBIC)
    base = base_person.convert("RGB").resize((w, h), Image.BICUBIC)

    db = donor_bbox or person_bbox_from_rgb(donor)
    bb = base_bbox or person_bbox_from_rgb(base)
    if db is None or bb is None:
        print("[body] Could not measure bodies; skipping donor alignment")
        return donor

    dx0, dy0, dx1, dy1 = db
    bx0, by0, bx1, by1 = bb
    donor_h = max(1, dy1 - dy0)
    base_h = max(1, by1 - by0)

    scale = float(np.clip(base_h / donor_h, 0.72, 1.40))
    crop = np.array(donor)[dy0:dy1, dx0:dx1]
    new_h = max(1, int(round(crop.shape[0] * scale)))
    new_w = max(1, int(round(crop.shape[1] * scale)))

    fit = min((h * 0.98) / new_h, (w * 0.98) / new_w, 1.0)
    new_h = max(1, int(round(new_h * fit)))
    new_w = max(1, int(round(new_w * fit)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    base_cx = (bx0 + bx1) / 2.0
    base_cy = (by0 + by1) / 2.0
    paste_x = int(np.clip(base_cx - new_w / 2.0, 0, max(0, w - new_w)))
    paste_y = int(np.clip(base_cy - new_h / 2.0, 0, max(0, h - new_h)))
    canvas[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = resized

    print(f"[body] Aligned pose donor with uniform scale={scale * fit:.2f}")
    return Image.fromarray(canvas)


def match_result_body_to_base(
    result: Image.Image,
    base_person: Image.Image,
    result_bbox: tuple[int, int, int, int] | None = None,
    base_bbox: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """Legacy compatibility shim. Final-image body warping is intentionally disabled."""
    print("[body] Final-image body resize is disabled; returning generated result unchanged")
    return result.convert("RGB")


def soft_preserve_torso(
    posed: Image.Image,
    dressed_base: Image.Image,
    strength: float = 0.22,
) -> Image.Image:
    """Legacy compatibility shim. Pixel-blending the old torso can undo a new pose."""
    print("[body] Torso pixel blending is disabled; returning posed result unchanged")
    return posed.convert("RGB")
