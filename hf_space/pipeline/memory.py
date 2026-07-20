"""VRAM helpers for sequential model loading on Colab T4."""

from __future__ import annotations

import gc


def free_vram() -> None:
    """Release unused CUDA memory as aggressively as possible."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def cuda_mem_report(tag: str = "") -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[vram{(' ' + tag) if tag else ''}] allocated={alloc:.2f}GB reserved={reserved:.2f}GB")
    except Exception:
        pass
