"""DensePose stand-in when detectron2/_C cannot be built on Colab."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _as_rgb(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    # DensePose predictor usually gets BGR from OpenCV; accept either
    return arr


class FallbackDensePosePredictor:
    """
    Approximate DensePose seg/IUV maps using human-parsing labels.

    Not as accurate as Detectron2 DensePose, but unblocks Colab when
    detectron2 cannot compile (_C missing).
    """

    def __init__(self, parsing_fn=None):
        self.parsing_fn = parsing_fn
        print("[densepose] Using FALLBACK predictor (no detectron2 _C)")

    def _parse_labels(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.parsing_fn is None:
            h, w = image_bgr.shape[:2]
            # crude foreground
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            labels = (th > 0).astype(np.uint8) * 4
            return labels

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((384, 512), Image.BICUBIC)
        parse_map, _ = self.parsing_fn(pil)
        labels = np.asarray(parse_map)
        if labels.ndim == 3:
            labels = labels[..., 0]
        h, w = image_bgr.shape[:2]
        return np.array(
            Image.fromarray(labels.astype(np.uint8)).resize((w, h), Image.NEAREST)
        )

    def predict_seg(self, image) -> np.ndarray:
        # Leffa expects a colorful segmentation visualization (H,W,3), often RGB-ish
        bgr = _as_rgb(image)
        if bgr.shape[2] == 3 and isinstance(image, np.ndarray):
            # assume BGR from caller (DensePosePredictor.predict uses cv2 images)
            image_bgr = image if image.shape[2] == 3 else cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)

        labels = self._parse_labels(image_bgr)
        # deterministic palette
        palette = np.array(
            [
                [0, 0, 0],
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 0],
                [255, 0, 255],
                [0, 255, 255],
                [128, 128, 0],
                [128, 0, 128],
                [0, 128, 128],
                [255, 128, 0],
                [255, 128, 128],
                [128, 255, 128],
                [128, 128, 255],
                [64, 128, 255],
                [255, 64, 128],
                [128, 255, 64],
                [64, 255, 128],
                [200, 200, 200],
            ],
            dtype=np.uint8,
        )
        idx = np.clip(labels.astype(np.int32), 0, len(palette) - 1)
        seg = palette[idx]
        return seg

    def predict_iuv(self, image) -> np.ndarray:
        bgr = image if isinstance(image, np.ndarray) else cv2.cvtColor(_as_rgb(image), cv2.COLOR_RGB2BGR)
        labels = self._parse_labels(bgr)
        h, w = labels.shape
        person = labels > 0
        iuv = np.zeros((h, w, 3), dtype=np.uint8)
        if not np.any(person):
            return iuv

        ys, xs = np.where(person)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        # I channel = part id scaled
        i_ch = (labels.astype(np.float32) * (255.0 / 18.0)).clip(0, 255).astype(np.uint8)
        yy, xx = np.mgrid[0:h, 0:w]
        u = ((xx - x0) / max(1, x1 - x0) * 255.0).clip(0, 255).astype(np.uint8)
        v = ((yy - y0) / max(1, y1 - y0) * 255.0).clip(0, 255).astype(np.uint8)
        iuv[..., 0] = np.where(person, i_ch, 0)
        iuv[..., 1] = np.where(person, u, 0)
        iuv[..., 2] = np.where(person, v, 0)
        return iuv
