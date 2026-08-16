"""Garment detection/extraction for clothed-person Leffa references."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image

LABEL = {
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "scarf": 17,
}

GARMENT_LABELS = {
    "upper_body": (4, 7, 8, 17),
    "lower_body": (5, 6, 7, 8),
    "dresses": (4, 5, 6, 7, 8, 17),
}


class GarmentExtractionError(RuntimeError):
    """Raised when a clothed-person reference does not contain usable clothing."""


@dataclass(frozen=True)
class GarmentAnalysis:
    requested_type: str
    resolved_type: str
    confidence: float
    selected_coverage: float
    clothing_coverage: float
    bbox_coverage: float
    upper_pixels: int
    lower_pixels: int
    dress_pixels: int
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def garment_labels_for(garment_type: str) -> tuple[int, ...]:
    if garment_type not in GARMENT_LABELS:
        raise ValueError(f"Unknown garment type: {garment_type}")
    return GARMENT_LABELS[garment_type]


def _labels_array(
    parse_map: Image.Image | np.ndarray,
    out_size: tuple[int, int],
) -> np.ndarray:
    parse = parse_map if isinstance(parse_map, Image.Image) else Image.fromarray(np.asarray(parse_map))
    parse = parse.resize(out_size, Image.NEAREST)
    labels = np.asarray(parse)
    if labels.ndim == 3:
        labels = labels[..., 0]
    return labels.astype(np.int16, copy=False)


def _bbox_coverage(mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    ys, xs = np.where(mask)
    area = (int(ys.max()) - int(ys.min()) + 1) * (int(xs.max()) - int(xs.min()) + 1)
    return float(area) / float(mask.shape[0] * mask.shape[1])


def analyse_garment(
    parse_map: Image.Image | np.ndarray,
    requested_type: str = "auto",
    out_size: tuple[int, int] = (768, 1024),
) -> GarmentAnalysis:
    """Resolve `auto` and report whether the requested garment region is usable."""
    if requested_type not in {"auto", *GARMENT_LABELS.keys()}:
        raise ValueError(f"Unknown garment type: {requested_type}")

    labels = _labels_array(parse_map, out_size)
    image_pixels = int(labels.size)
    upper_pixels = int(np.isin(labels, (4, 17)).sum())
    lower_pixels = int(np.isin(labels, (5, 6)).sum())
    dress_pixels = int((labels == 7).sum())
    belt_pixels = int((labels == 8).sum())
    clothing_pixels = upper_pixels + lower_pixels + dress_pixels + belt_pixels
    clothing_coverage = clothing_pixels / float(max(image_pixels, 1))

    if clothing_pixels <= 0:
        return GarmentAnalysis(
            requested_type=requested_type,
            resolved_type="upper_body" if requested_type == "auto" else requested_type,
            confidence=0.0,
            selected_coverage=0.0,
            clothing_coverage=0.0,
            bbox_coverage=0.0,
            upper_pixels=0,
            lower_pixels=0,
            dress_pixels=0,
            reason="no clothing labels detected",
        )

    if requested_type == "auto":
        # A parser-labelled dress is the strongest signal. Otherwise, if both
        # upper and lower garments are materially visible, treat the reference
        # as a full outfit and route to DressCode. This avoids silently copying
        # only the shirt from an outfit reference.
        dress_share = dress_pixels / float(clothing_pixels)
        upper_share = upper_pixels / float(clothing_pixels)
        lower_share = lower_pixels / float(clothing_pixels)
        if dress_pixels >= 256 and dress_share >= 0.10:
            resolved = "dresses"
            confidence = min(1.0, 0.65 + dress_share)
            reason = "dress label detected"
        elif upper_pixels >= 256 and lower_pixels >= 256 and upper_share >= 0.12 and lower_share >= 0.12:
            resolved = "dresses"
            confidence = min(1.0, 0.55 + min(upper_share, lower_share))
            reason = "upper and lower garments both visible"
        elif lower_pixels > max(256, int(upper_pixels * 1.20)):
            resolved = "lower_body"
            confidence = min(1.0, 0.55 + lower_share)
            reason = "lower garment dominates"
        else:
            resolved = "upper_body"
            confidence = min(1.0, 0.55 + upper_share + dress_share * 0.25)
            reason = "upper garment dominates"
    else:
        resolved = requested_type
        selected_pixels = int(np.isin(labels, garment_labels_for(resolved)).sum())
        confidence = min(1.0, selected_pixels / float(max(clothing_pixels, 1)))
        reason = "explicit garment selection"

    selected = np.isin(labels, garment_labels_for(resolved))
    selected_pixels = int(selected.sum())
    selected_coverage = selected_pixels / float(max(image_pixels, 1))
    bbox_coverage = _bbox_coverage(selected)

    if selected_pixels < 256 or selected_coverage < 0.002:
        reason = f"{reason}; selected garment region is too small"
        confidence = min(confidence, 0.15)
    elif bbox_coverage > 0.92:
        reason = f"{reason}; garment region is implausibly large"
        confidence = min(confidence, 0.20)

    return GarmentAnalysis(
        requested_type=requested_type,
        resolved_type=resolved,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        selected_coverage=float(selected_coverage),
        clothing_coverage=float(clothing_coverage),
        bbox_coverage=float(bbox_coverage),
        upper_pixels=upper_pixels,
        lower_pixels=lower_pixels,
        dress_pixels=dress_pixels,
        reason=reason,
    )


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Remove tiny parser islands while preserving disconnected garment pieces."""
    hard = mask.astype(np.uint8)
    if not np.any(hard):
        return hard.astype(bool)

    count, component_map, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    if count <= 1:
        return hard.astype(bool)
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(component_areas.max(initial=0))
    min_area = max(24, int(largest * 0.012))
    cleaned = np.zeros_like(hard)
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[component_map == component_id] = 1
    return cleaned.astype(bool)


def _soft_mask(mask: np.ndarray, feather_px: int = 3) -> np.ndarray:
    hard = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, kernel, iterations=1)
    if feather_px > 0:
        sigma = max(0.65, feather_px / 2.2)
        hard = cv2.GaussianBlur(hard, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return hard.astype(np.float32) / 255.0


def extract_garment_from_person(
    person_rgb: Image.Image,
    parse_map: Image.Image | np.ndarray,
    garment_type: str = "auto",
    out_size: tuple[int, int] = (768, 1024),
    pad_ratio: float = 0.08,
    return_analysis: bool = False,
):
    """Create a clean garment reference and never fall back to the full person.

    Feeding the whole reference person to VTON after parser failure can leak the
    donor's body/skin/identity into the result. A failed or implausibly small
    extraction therefore stops with `GarmentExtractionError` instead.
    """
    person = person_rgb.convert("RGB").resize(out_size, Image.BICUBIC)
    labels = _labels_array(parse_map, out_size)
    analysis = analyse_garment(labels, requested_type=garment_type, out_size=out_size)

    if analysis.confidence < 0.20 or analysis.selected_coverage < 0.002:
        raise GarmentExtractionError(
            "Could not isolate a reliable garment from the reference person: "
            f"{analysis.reason}. Use a clearer/full-body reference or choose the garment region explicitly."
        )

    keep = np.isin(labels, garment_labels_for(analysis.resolved_type))
    keep = _clean_mask(keep)
    if not np.any(keep):
        raise GarmentExtractionError("Garment parser output disappeared after noise cleanup; refusing full-person fallback.")

    arr = np.asarray(person)
    ys, xs = np.where(keep)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h, w = out_size[1], out_size[0]
    pad_y = int((y1 - y0) * pad_ratio)
    pad_x = int((x1 - x0) * pad_ratio)
    y0, x0 = max(0, y0 - pad_y), max(0, x0 - pad_x)
    y1, x1 = min(h, y1 + pad_y), min(w, x1 + pad_x)

    crop = arr[y0:y1, x0:x1].copy()
    alpha = _soft_mask(keep[y0:y1, x0:x1], feather_px=3)[..., None]
    white = np.full_like(crop, 255)
    matted = np.clip(
        crop.astype(np.float32) * alpha + white.astype(np.float32) * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)

    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    ch, cw = matted.shape[:2]
    scale = min((w * 0.92) / max(cw, 1), (h * 0.88) / max(ch, 1), 1.0)
    new_w, new_h = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    crop_img = Image.fromarray(matted).resize((new_w, new_h), Image.LANCZOS)
    paste_x = (w - new_w) // 2
    paste_y = (h - new_h) // 2
    canvas[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = np.asarray(crop_img)
    result = Image.fromarray(canvas)
    return (result, analysis) if return_analysis else result
