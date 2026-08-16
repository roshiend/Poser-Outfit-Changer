"""Runtime readiness checks for local, Colab and Hugging Face deployments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    required: bool
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _detectron2_compiled() -> tuple[bool, str]:
    if not _module_available("detectron2"):
        return False, "detectron2 is not installed"
    try:
        from detectron2 import _C  # noqa: F401
        return True, "compiled detectron2._C is importable"
    except Exception as exc:
        return False, f"detectron2 installed but compiled _C is unavailable: {exc!r}"


def checkpoint_requirements(require_pose: bool = False) -> tuple[str, ...]:
    files = [
        "virtual_tryon.pth",
        "virtual_tryon_dc.pth",
        "humanparsing/parsing_atr.onnx",
        "humanparsing/parsing_lip.onnx",
        "openpose/body_pose_model.pth",
    ]
    if require_pose:
        files.extend(
            [
                "pose_transfer.pth",
                "densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
                "densepose/model_final_162be9.pkl",
            ]
        )
    return tuple(files)


def run_preflight(
    leffa_root: str | Path = "./Leffa",
    ckpt_dir: str | Path | None = None,
    require_pose: bool = False,
    require_gpu: bool = False,
    require_checkpoints: bool = False,
) -> dict:
    leffa = Path(leffa_root)
    ckpt = Path(ckpt_dir) if ckpt_dir else leffa / "ckpts"
    checks: list[Check] = []

    py_ok = sys.version_info >= (3, 10)
    checks.append(Check("python", py_ok, True, f"Python {sys.version.split()[0]} (requires >= 3.10)"))

    leffa_ok = (leffa / "leffa").exists() and (leffa / "preprocess").exists()
    checks.append(Check("leffa_source", leffa_ok, True, f"Leffa root: {leffa.resolve()}"))

    required_files = checkpoint_requirements(require_pose=require_pose)
    missing = [name for name in required_files if not (ckpt / name).exists()]
    checkpoints_ok = not missing
    detail = f"Checkpoint root: {ckpt.resolve()}"
    if missing:
        detail += "; missing: " + ", ".join(missing)
    checks.append(Check("checkpoints", checkpoints_ok, require_checkpoints, detail))

    for module in ("numpy", "PIL", "cv2", "huggingface_hub"):
        ok = _module_available(module)
        checks.append(Check(f"module:{module}", ok, True, "available" if ok else "not installed"))

    insightface_ok = _module_available("insightface")
    checks.append(Check("module:insightface", insightface_ok, False, "identity lock available" if insightface_ok else "optional identity lock unavailable"))

    densepose_ok, densepose_detail = _detectron2_compiled()
    checks.append(Check("detectron2_densepose", densepose_ok, require_pose, densepose_detail))

    torch_ok = _module_available("torch")
    cuda_ok = False
    cuda_detail = "torch not installed"
    if torch_ok:
        try:
            import torch

            cuda_ok = bool(torch.cuda.is_available())
            if cuda_ok:
                device = torch.cuda.get_device_name(0)
                cuda_detail = f"CUDA available: {device}"
            else:
                cuda_detail = "torch installed; CUDA is not available"
        except Exception as exc:
            cuda_detail = f"torch import failed: {exc!r}"
    checks.append(Check("cuda", cuda_ok, require_gpu, cuda_detail))

    blocking = [item for item in checks if item.required and not item.ok]
    warnings = [item for item in checks if not item.required and not item.ok]
    return {
        "ready": not blocking,
        "require_pose": bool(require_pose),
        "require_gpu": bool(require_gpu),
        "require_checkpoints": bool(require_checkpoints),
        "blocking": [item.name for item in blocking],
        "warnings": [item.name for item in warnings],
        "checks": [item.to_dict() for item in checks],
    }


def _print_human(report: dict) -> None:
    for item in report["checks"]:
        mark = "PASS" if item["ok"] else ("FAIL" if item["required"] else "WARN")
        print(f"[{mark:4}] {item['name']}: {item['detail']}")
    print("READY" if report["ready"] else "NOT READY")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Poser Outfit Changer runtime readiness")
    parser.add_argument("--leffa-root", type=Path, default=Path("./Leffa"))
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--require-pose", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-checkpoints", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_preflight(
        leffa_root=args.leffa_root,
        ckpt_dir=args.ckpt_dir,
        require_pose=args.require_pose,
        require_gpu=args.require_gpu,
        require_checkpoints=args.require_checkpoints,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_human(report)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
