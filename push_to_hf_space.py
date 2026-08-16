"""Synchronize/deploy the canonical pipeline to the Hugging Face Space.

Usage:
  python push_to_hf_space.py --sync-only
  set HF_TOKEN=hf_xxx
  python push_to_hf_space.py

Optional environment:
  HF_SPACE_ID=roshiend/poser-outfit-changer
  HF_SPACE_HW=a10g-small   # informational; hardware is configured on HF
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPACE_DIR = ROOT / "hf_space"
PIPELINE_DIR = ROOT / "pipeline"
SPACE_PIPELINE_DIR = SPACE_DIR / "pipeline"
DEFAULT_SPACE = "roshiend/poser-outfit-changer"


def sync_pipeline() -> None:
    """Mirror the canonical root pipeline into the deployable Space package."""
    if not PIPELINE_DIR.exists():
        raise FileNotFoundError(f"Missing canonical pipeline directory: {PIPELINE_DIR}")

    SPACE_PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    source_names = {source.name for source in PIPELINE_DIR.glob("*.py")}
    for stale in SPACE_PIPELINE_DIR.glob("*.py"):
        if stale.name not in source_names:
            stale.unlink()
    for source in PIPELINE_DIR.glob("*.py"):
        shutil.copy2(source, SPACE_PIPELINE_DIR / source.name)
    print("Synced pipeline/ -> hf_space/pipeline/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="only mirror pipeline/ into hf_space/pipeline/; do not require HF credentials or upload",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sync_pipeline()
    if args.sync_only:
        return 0

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Missing HF_TOKEN.")
        print("1) Create a token with write access in Hugging Face settings")
        print("2) PowerShell:  $env:HF_TOKEN='hf_...'")
        print("3) Run:         python push_to_hf_space.py")
        return 1

    # Keep deployment-only dependencies out of --sync-only/CI paths.
    from huggingface_hub import HfApi, login, whoami

    login(token=token)
    info = whoami(token=token)
    username = info.get("name") or info.get("fullname", {}).get("name")
    print("Logged in as:", username)

    space_id = os.environ.get("HF_SPACE_ID", DEFAULT_SPACE)
    if "/" not in space_id:
        space_id = f"{username}/{space_id}"

    api = HfApi(token=token)
    print("Creating/using Space:", space_id)
    try:
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            exist_ok=True,
        )
    except Exception as exc:
        print("create_repo note:", exc)

    print("Uploading", SPACE_DIR, "...")
    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=space_id,
        repo_type="space",
        commit_message="Deploy Poser Outfit Changer Gradio Space",
        ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
    )

    print("\nSpace URL:", f"https://huggingface.co/spaces/{space_id}")
    print("Pose modes require real Detectron2 DensePose and a GPU large enough for Leffa SDXL pose transfer.")
    print("The Space builds vendored Detectron2 on demand when pose is first requested.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
