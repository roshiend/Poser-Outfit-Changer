"""
Push hf_space/ to a Hugging Face Space.

Usage:
  set HF_TOKEN=hf_xxx
  python push_to_hf_space.py

Optional:
  set HF_SPACE_ID=roshiend/poser-outfit-changer
  set HF_SPACE_HW=a10g-small   # or t4-small / zerogpu (zerogpu needs HF PRO to host)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, login, whoami

ROOT = Path(__file__).resolve().parent
SPACE_DIR = ROOT / "hf_space"
DEFAULT_SPACE = "roshiend/poser-outfit-changer"


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Missing HF_TOKEN.")
        print("1) Create a token (write access): https://huggingface.co/settings/tokens")
        print("2) PowerShell:  $env:HF_TOKEN='hf_...'")
        print("3) Run:         python push_to_hf_space.py")
        return 1

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

    # Upload Space folder as Space root
    print("Uploading", SPACE_DIR, "...")
    api.upload_folder(
        folder_path=str(SPACE_DIR),
        repo_id=space_id,
        repo_type="space",
        commit_message="Deploy Poser Outfit Changer Gradio Space",
    )

    url = f"https://huggingface.co/spaces/{space_id}"
    print("\nSpace URL:", url)
    print("In Space Settings → Hardware, pick a GPU (T4/A10G) or ZeroGPU if you have HF PRO.")
    print("First build downloads Leffa weights and can take several minutes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
