#!/usr/bin/env python3
"""Benchmark pose-retarget strengths on representative base/reference pairs.

Example:
    python benchmark_fidelity.py benchmarks/example_manifest.json \
        --strengths 0,0.65,1 --output benchmark_runs/run1

This intentionally performs full Leffa inference and therefore requires the same
GPU/checkpoint/real-DensePose environment as normal pose transfer.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image, ImageDraw

from pipeline import PoseClothPipeline
from pipeline.benchmark_metrics import (
    body_proportion_error,
    composite_fidelity_score,
    pose_angle_error_deg,
)
from pipeline.face_lock import face_identity_similarity
from pipeline.memory import free_vram
from pipeline.pose_geometry import openpose_points


def _parse_strengths(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        value = float(item.strip())
        if not 0.0 <= value <= 1.0:
            raise argparse.ArgumentTypeError("retarget strengths must be between 0 and 1")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("provide at least one retarget strength")
    return values


def _load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest must contain a non-empty 'cases' list")
    required = {"name", "base", "reference"}
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or not required.issubset(case):
            raise ValueError(f"case {index} must include: {sorted(required)}")
    return cases


def _resolve_case_path(manifest_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (manifest_dir / path).resolve()


def _prepare_image(image: Image.Image) -> Image.Image:
    from leffa_utils.utils import resize_and_center

    return resize_and_center(image.convert("RGB"), 768, 1024)


def _measure_openpose(pipe: PoseClothPipeline, image: Image.Image):
    """Load only Leffa OpenPose for a metric pass, then release it before diffusion."""
    from preprocess.openpose.run_openpose import OpenPose

    detector = OpenPose(
        body_model_path=str(pipe.ckpt_dir / "openpose" / "body_pose_model.pth")
    )
    keypoints = detector(image.convert("RGB").resize((384, 512)))
    points, valid = openpose_points(keypoints)
    del detector
    gc.collect()
    free_vram()
    return points, valid


def _safe_float(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _save_debug_images(case_dir: Path, strength_label: str, debug: dict) -> None:
    debug_dir = case_dir / f"debug_strength_{strength_label}"
    for key, value in debug.items():
        if isinstance(value, Image.Image):
            debug_dir.mkdir(parents=True, exist_ok=True)
            value.convert("RGB").save(debug_dir / f"{key}.png")


def _contact_sheet(
    base: Image.Image,
    reference: Image.Image,
    variants: list[tuple[str, Image.Image, dict]],
    destination: Path,
) -> None:
    width, height = 320, 426
    entries = [("BASE", base), ("REFERENCE", reference)] + [(label, image) for label, image, _ in variants]
    canvas = Image.new("RGB", (width * len(entries), height + 64), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(entries):
        thumb = image.convert("RGB").resize((width, height), Image.LANCZOS)
        x = index * width
        canvas.paste(thumb, (x, 38))
        draw.text((x + 8, 10), label, fill="black")
        if index >= 2:
            metrics = variants[index - 2][2]
            score = metrics.get("composite_score")
            if score is not None:
                draw.text((x + 8, height + 42), f"score {score:.1f}", fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _aggregate(rows: list[dict]) -> dict:
    by_strength: dict[float, list[dict]] = defaultdict(list)
    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_strength[float(row["strength"])].append(row)
        by_case[str(row["case"])].append(row)

    def best(items: list[dict]):
        scored = [row for row in items if row.get("composite_score") is not None]
        return max(scored, key=lambda row: row["composite_score"]) if scored else None

    case_best = {}
    for name, items in by_case.items():
        winner = best(items)
        case_best[name] = None if winner is None else {
            "strength": winner["strength"],
            "composite_score": winner["composite_score"],
            "identity_similarity": winner["identity_similarity"],
            "pose_angle_error_deg": winner["pose_angle_error_deg"],
            "body_proportion_error": winner["body_proportion_error"],
        }

    strength_summary = {}
    for strength, items in sorted(by_strength.items()):
        scores = [row["composite_score"] for row in items if row.get("composite_score") is not None]
        identities = [row["identity_similarity"] for row in items if row.get("identity_similarity") is not None]
        poses = [row["pose_angle_error_deg"] for row in items if row.get("pose_angle_error_deg") is not None]
        bodies = [row["body_proportion_error"] for row in items if row.get("body_proportion_error") is not None]
        strength_summary[str(strength)] = {
            "cases": len(items),
            "mean_composite_score": mean(scores) if scores else None,
            "mean_identity_similarity": mean(identities) if identities else None,
            "mean_pose_angle_error_deg": mean(poses) if poses else None,
            "mean_body_proportion_error": mean(bodies) if bodies else None,
        }

    ranked = [
        (float(strength), summary["mean_composite_score"])
        for strength, summary in strength_summary.items()
        if summary["mean_composite_score"] is not None
    ]
    global_best = max(ranked, key=lambda item: item[1]) if ranked else None
    return {
        "best_by_case": case_best,
        "strength_summary": strength_summary,
        "recommended_default_strength": None if global_best is None else global_best[0],
    }


def run(args: argparse.Namespace) -> int:
    manifest = args.manifest.resolve()
    cases = _load_manifest(manifest)
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.force_pose:
        os.environ["LEFFA_ALLOW_POSE"] = "1"

    pipe = PoseClothPipeline(
        leffa_root=args.leffa_root,
        ckpt_dir=args.ckpt_dir,
        enable_face_lock=not args.disable_face_lock,
        preserve_body=True,
    )
    if args.download_checkpoints:
        pipe.download_checkpoints()

    rows: list[dict] = []
    manifest_dir = manifest.parent

    for case in cases:
        name = str(case["name"])
        mode = str(case.get("mode", "both"))
        if mode not in {"both", "pose_only"}:
            raise ValueError(f"benchmark case '{name}' uses mode={mode!r}; use both or pose_only")

        base_path = _resolve_case_path(manifest_dir, str(case["base"]))
        ref_path = _resolve_case_path(manifest_dir, str(case["reference"]))
        if not base_path.exists() or not ref_path.exists():
            raise FileNotFoundError(f"missing image for case '{name}': {base_path} / {ref_path}")

        base = _prepare_image(Image.open(base_path))
        reference = _prepare_image(Image.open(ref_path))
        case_dir = output_root / name
        case_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Benchmark case: {name} ===")
        print("Measuring base/reference skeletons...")
        base_points, base_valid = _measure_openpose(pipe, base)
        target_points, target_valid = _measure_openpose(pipe, reference)

        variants: list[tuple[str, Image.Image, dict]] = []
        for strength in args.strengths:
            label = f"{strength:.2f}".rstrip("0").rstrip(".")
            print(f"\n[{name}] retarget strength={strength:.2f}")
            result, debug = pipe.generate(
                base_image=base,
                ref_image=reference,
                mode=mode,
                garment_type=str(case.get("garment_type", "dresses")),
                vt_model_type="auto",
                steps=int(case.get("steps", args.steps)),
                guidance_scale=float(case.get("guidance_scale", args.guidance_scale)),
                seed=int(case.get("seed", args.seed)),
                ref_acceleration=not args.disable_ref_acceleration,
                face_lock=not args.disable_face_lock,
                ref_kind=str(case.get("ref_kind", "clothed_person")),
                preserve_body=True,
                pose_retarget_strength=strength,
                return_debug=True,
            )

            result_path = case_dir / f"result_strength_{label}.png"
            result.save(result_path)
            generated_points, generated_valid = _measure_openpose(pipe, result)

            identity = face_identity_similarity(base, result, pipe._get_face_app())
            pose_error = pose_angle_error_deg(
                generated_points, generated_valid, target_points, target_valid
            )
            proportion_error = body_proportion_error(
                generated_points, generated_valid, base_points, base_valid
            )
            score = composite_fidelity_score(identity, pose_error, proportion_error)
            retarget = debug.get("pose_retarget_diagnostics") or {}

            row = {
                "case": name,
                "strength": float(strength),
                "mode": mode,
                "garment_type": str(case.get("garment_type", "dresses")),
                "identity_similarity": _safe_float(identity),
                "pose_angle_error_deg": _safe_float(pose_error),
                "body_proportion_error": _safe_float(proportion_error),
                "composite_score": _safe_float(score),
                "retarget_applied": bool(retarget.get("safe_to_retarget", False)),
                "retarget_reason": str(retarget.get("reason", "n/a")),
                "retarget_proportion_gap": _safe_float(retarget.get("proportion_gap")),
                "result": str(result_path.relative_to(output_root)),
            }
            rows.append(row)
            variants.append((f"strength {label}", result.copy(), row))

            if args.save_debug:
                _save_debug_images(case_dir, label, debug)

            print(
                f"score={score if score is not None else 'n/a'} | "
                f"identity={identity if identity is not None else 'n/a'} | "
                f"pose_error={pose_error if pose_error is not None else 'n/a'}° | "
                f"body_error={proportion_error if proportion_error is not None else 'n/a'}"
            )

        _contact_sheet(base, reference, variants, case_dir / "comparison.png")

    fields = [
        "case", "strength", "mode", "garment_type", "identity_similarity",
        "pose_angle_error_deg", "body_proportion_error", "composite_score",
        "retarget_applied", "retarget_reason", "retarget_proportion_gap", "result",
    ]
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    (output_root / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    recommendations = _aggregate(rows)
    (output_root / "recommendations.json").write_text(
        json.dumps(recommendations, indent=2), encoding="utf-8"
    )

    print("\nBenchmark complete:", output_root)
    print("Recommended default strength:", recommendations["recommended_default_strength"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmark_runs/latest"))
    parser.add_argument("--strengths", type=_parse_strengths, default=_parse_strengths("0,0.65,1"))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leffa-root", type=Path, default=Path("./Leffa"))
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--download-checkpoints", action="store_true")
    parser.add_argument("--force-pose", action="store_true", help="set LEFFA_ALLOW_POSE=1 before pipeline creation")
    parser.add_argument("--disable-face-lock", action="store_true")
    parser.add_argument("--disable-ref-acceleration", action="store_true")
    parser.add_argument("--save-debug", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
