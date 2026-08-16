"""
Push hf_space/ to a Hugging Face Space.

Before upload, the canonical root pipeline/ directory is copied into
hf_space/pipeline/ so the deployed Space cannot silently drift from the source
used by local/Colab development.

Usage:
  set HF_TOKEN=hf_xxx
  python push_to_hf_space.py

Optional:
  set HF_SPACE_ID=roshiend/poser-outfit-changer
  set HF_SPACE_HW=a10g-small   # or t4-small / zerogpu
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, login, whoami

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
    for source in PIPELINE_DIR.glob("*.py"):
        target = SPACE_PIPELINE_DIR / source.name
        shutil.copy2(source, target)
    print("Synced pipeline/ -> hf_space/pipeline/")


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Missing HF_TOKEN.")
        print("1) Create a token with write access in Hugging Face settings")
        print("2) PowerShell:  $env:HF_TOKEN='hf_...'")
        print("3) Run:         python push_to_hf_space.py")
        return 1

    sync_pipeline()

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
