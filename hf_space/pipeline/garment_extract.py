"""Extract clothing from a clothed-person photo onto a white background for VTON."""

from __future__ import annotations

import numpy as np
from PIL import Image

# SCHP / ATR-style labels used by Leffa Parsing
LABEL = {
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "scarf": 17,
}

GARMENT_LABELS = {
    "upper_body": (4, 7, 8, 17),  # tops + dress + belt/scarf accents
    "lower_body": (5, 6, 8),  # skirt / pants / belt
    "dresses": (4, 5, 6, 7, 8, 17),  # full outfit from a clothed person
}


def garment_labels_for(garment_type: str) -> tuple[int, ...]:
    return GARMENT_LABELS.get(garment_type, GARMENT_LABELS["upper_body"])


def extract_garment_from_person(
    person_rgb: Image.Image,
    parse_map: Image.Image | np.ndarray,
    garment_type: str = "upper_body",
    out_size: tuple[int, int] = (768, 1024),
    pad_ratio: float = 0.08,
) -> Image.Image:
    """
    Build a garment reference image from a person who is already wearing clothes.

    Uses human-parsing labels to keep only clothing pixels on a white canvas —
    the format Leffa VTON expects better than a full person photo.
    """
    person = person_rgb.convert("RGB").resize(out_size, Image.BICUBIC)
    parse = parse_map if isinstance(parse_map, Image.Image) else Image.fromarray(np.asarray(parse_map))
    parse = parse.resize(out_size, Image.NEAREST)

    arr = np.asarray(person)
    labels = np.asarray(parse)
    if labels.ndim == 3:
        labels = labels[..., 0]

    keep = np.isin(labels, garment_labels_for(garment_type))
    if not np.any(keep):
        # Fallback: try full clothing set if the chosen region was empty
        keep = np.isin(labels, GARMENT_LABELS["dresses"])
    if not np.any(keep):
        print("[garment] No clothing labels found; using full person image as VTON ref.")
        return person

    ys, xs = np.where(keep)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h, w = out_size[1], out_size[0]
    pad_y = int((y1 - y0) * pad_ratio)
    pad_x = int((x1 - x0) * pad_ratio)
    y0, x0 = max(0, y0 - pad_y), max(0, x0 - pad_x)
    y1, x1 = min(h, y1 + pad_y), min(w, x1 + pad_x)

    crop = arr[y0:y1, x0:x1].copy()
    crop_mask = keep[y0:y1, x0:x1]
    crop[~crop_mask] = 255

    # Center garment crop on white 768x1024 canvas
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    ch, cw = crop.shape[:2]
    # Fit inside canvas with margin
    scale = min((w * 0.92) / max(cw, 1), (h * 0.85) / max(ch, 1), 1.0)
    new_w, new_h = max(1, int(cw * scale)), max(1, int(ch * scale))
    crop_img = Image.fromarray(crop).resize((new_w, new_h), Image.LANCZOS)
    paste_x = (w - new_w) // 2
    paste_y = (h - new_h) // 2
    canvas[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = np.asarray(crop_img)
    return Image.fromarray(canvas)
