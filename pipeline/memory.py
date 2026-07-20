"""VRAM helpers for sequential model loading on Colab T4."""

from __future__ import annotations

import gc


def free_vram() -> None:
    """Release unused CUDA memory as aggressively as possible."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
