"""Keep base-person body proportions when transferring pose from another photo."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _to_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    arr = image
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
    # Non-background person pixels (SCHP/ATR: 0 = background)
    mask = labels > 0
    if int(mask.sum()) < min_area:
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def person_bbox_from_rgb(image: Image.Image | np.ndarray) -> tuple[int, int, int, int] | None:
    """Fallback bbox via simple foreground estimate when parsing is unavailable."""
    rgb = _to_rgb(image)
    h, w = rgb.shape[:2]
    # Rough center-weighted non-white / non-uniform region
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
    Rescale/recenter the pose donor so their body height matches the base person.

    Pose transfer uses DensePose from the donor image; matching body scale first
    keeps proportions closer to the base subject.
    """
    w, h = canvas_size
    donor = pose_donor.convert("RGB").resize((w, h), Image.BICUBIC)
    base = base_person.convert("RGB").resize((w, h), Image.BICUBIC)

    db = donor_bbox or person_bbox_from_rgb(donor)
    bb = base_bbox or person_bbox_from_rgb(base)
    if db is None or bb is None:
        print("[body] Could not measure bodies; skipping donor align")
        return donor

    dx0, dy0, dx1, dy1 = db
    bx0, by0, bx1, by1 = bb
    donor_h = max(1, dy1 - dy0)
    base_h = max(1, by1 - by0)
    donor_w = max(1, dx1 - dx0)
    base_w = max(1, bx1 - bx0)

    # Match height primarily; clamp extreme width stretch
    scale = base_h / donor_h
    scale = float(np.clip(scale, 0.75, 1.35))
    # Mild width correction toward base shoulder/body width
    width_ratio = (base_w / donor_w) / max(scale, 1e-6)
    width_ratio = float(np.clip(width_ratio, 0.85, 1.15))

    crop = np.array(donor)[dy0:dy1, dx0:dx1]
    new_h = max(1, int(donor_h * scale))
    new_w = max(1, int(donor_w * scale * width_ratio))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    # Place with same vertical centering bias as base person
    base_cy = (by0 + by1) / 2.0
    paste_y = int(np.clip(base_cy - new_h / 2.0, 0, h - new_h))
    paste_x = int(np.clip((w - new_w) / 2.0, 0, w - new_w))
    canvas[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = resized
    print(f"[body] Aligned pose donor scale={scale:.2f} width_adj={width_ratio:.2f}")
    return Image.fromarray(canvas)


def match_result_body_to_base(
    result: Image.Image,
    base_person: Image.Image,
    result_bbox: tuple[int, int, int, int] | None = None,
    base_bbox: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """
    After pose transfer, rescale the generated person to the base body height.

    Preserves the new pose/clothes while restoring overall body size proportions.
    """
    res = result.convert("RGB")
    base = base_person.convert("RGB")
    if res.size != base.size:
        res = res.resize(base.size, Image.BICUBIC)

    w, h = res.size
    rb = result_bbox or person_bbox_from_rgb(res)
    bb = base_bbox or person_bbox_from_rgb(base)
    if rb is None or bb is None:
        print("[body] Could not measure result/base; skipping body match")
        return res

    rx0, ry0, rx1, ry1 = rb
    bx0, by0, bx1, by1 = bb
    res_h = max(1, ry1 - ry0)
    base_h = max(1, by1 - by0)
    res_w = max(1, rx1 - rx0)
    base_w = max(1, bx1 - bx0)

    scale = base_h / res_h
    scale = float(np.clip(scale, 0.80, 1.25))
    if abs(scale - 1.0) < 0.03:
        return res

    width_ratio = (base_w / res_w) / max(scale, 1e-6)
    width_ratio = float(np.clip(width_ratio, 0.90, 1.10))

    arr = np.array(res)
    crop = arr[ry0:ry1, rx0:rx1]
    new_h = max(1, int(res_h * scale))
    new_w = max(1, int(res_w * scale * width_ratio))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Keep original background; paste scaled person centered on base body center
    out = arr.copy()
    # Soft-clear old person region toward background estimate
    bg = cv2.GaussianBlur(arr, (51, 51), 0)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[ry0:ry1, rx0:rx1] = 1.0
    mask = cv2.GaussianBlur(mask, (31, 31), 0)[..., None]
    out = (out * (1.0 - mask) + bg * mask).astype(np.uint8)

    base_cy = (by0 + by1) / 2.0
    base_cx = (bx0 + bx1) / 2.0
    paste_y = int(np.clip(base_cy - new_h / 2.0, 0, h - new_h))
    paste_x = int(np.clip(base_cx - new_w / 2.0, 0, w - new_w))
    out[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = resized
    print(f"[body] Matched result body scale={scale:.2f}")
    return Image.fromarray(out)


def soft_preserve_torso(
    posed: Image.Image,
    dressed_base: Image.Image,
    strength: float = 0.22,
) -> Image.Image:
    """
    Lightly blend the pre-pose (dressed base) torso into the posed result.

    Keeps body build / skin continuity from the base subject without fully
    undoing the new pose. Low strength by design.
    """
    a = np.array(posed.convert("RGB"), dtype=np.float32)
    b = np.array(dressed_base.convert("RGB").resize(posed.size, Image.BICUBIC), dtype=np.float32)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Ellipse covering upper/mid torso
    cx, cy = w * 0.5, h * 0.42
    rx, ry = w * 0.22, h * 0.20
    mask = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2
    mask = np.clip(1.0 - mask, 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=min(w, h) * 0.03)
    mask = (mask * float(np.clip(strength, 0.0, 0.45)))[..., None]
    out = a * (1.0 - mask) + b * mask
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
