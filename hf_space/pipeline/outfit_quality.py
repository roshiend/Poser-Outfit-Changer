"""Lightweight fidelity diagnostics for the outfit-transfer stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

HAIR = 2
FACE = 11
BACKGROUND = 0


@dataclass(frozen=True)
class OutfitQualityDiagnostics:
    face_hair_mae: float | None
    background_mae: float | None
    protected_pixel_fraction: float
    background_pixel_fraction: float
    warnings: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["warnings"] = list(self.warnings)
        return data


def _labels_array(parse_map: Image.Image | np.ndarray, size: tuple[int, int]) -> np.ndarray:
    parse = parse_map if isinstance(parse_map, Image.Image) else Image.fromarray(np.asarray(parse_map))
    parse = parse.resize(size, Image.NEAREST)
    labels = np.asarray(parse)
    if labels.ndim == 3:
        labels = labels[..., 0]
    return labels


def masked_rgb_mae(before: Image.Image | np.ndarray, after: Image.Image | np.ndarray, mask: np.ndarray) -> float | None:
    a = np.asarray(before.convert("RGB") if isinstance(before, Image.Image) else before, dtype=np.float32)
    b = np.asarray(after.convert("RGB") if isinstance(after, Image.Image) else after, dtype=np.float32)
    if a.shape != b.shape or mask.shape != a.shape[:2] or not np.any(mask):
        return None
    diff = np.abs(a - b).mean(axis=2) / 255.0
    return float(diff[mask].mean())


def assess_outfit_preservation(base_image: Image.Image, after_vton: Image.Image, base_parse: Image.Image | np.ndarray, face_hair_warn: float = 0.18, background_warn: float = 0.14) -> OutfitQualityDiagnostics:
    base = base_image.convert("RGB")
    result = after_vton.convert("RGB").resize(base.size, Image.BICUBIC)
    labels = _labels_array(base_parse, base.size)
    protected = np.isin(labels, (HAIR, FACE))
    background = labels == BACKGROUND
    face_hair_mae = masked_rgb_mae(base, result, protected)
    background_mae = masked_rgb_mae(base, result, background)
    warnings: list[str] = []
    if face_hair_mae is not None and face_hair_mae > face_hair_warn:
        warnings.append(f"VTON changed face/hair strongly (normalized MAE {face_hair_mae:.3f}); inspect identity before accepting.")
    if background_mae is not None and background_mae > background_warn:
        warnings.append(f"VTON changed background strongly (normalized MAE {background_mae:.3f}); reference/mask may be unstable.")
    return OutfitQualityDiagnostics(face_hair_mae, background_mae, float(protected.mean()), float(background.mean()), tuple(warnings))
