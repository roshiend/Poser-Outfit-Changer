"""
Hugging Face Spaces entrypoint for Pose & Cloth Swap.

Uses Leffa VTON (+ optional pose) with DensePose fallback (no detectron2 build).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

# Spaces marker used by pipeline for memory-safe defaults
os.environ.setdefault("SPACE_ID", os.environ.get("SPACE_ID", "local-space"))

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runtime"
LEFFA_ROOT = WORK / "Leffa"
CKPT_DIR = WORK / "ckpts"
PIPE_DIR = ROOT / "pipeline"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LEFFA_ROOT))

try:
    import spaces  # HF ZeroGPU / Spaces GPU decorator
except Exception:  # local / non-Spaces
    class _SpacesShim:
        @staticmethod
        def GPU(duration=120, size=None):
            def deco(fn):
                return fn

            return deco

    spaces = _SpacesShim()

_pipe = None
_setup_done = False


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def setup_runtime() -> None:
    """Clone Leffa + download checkpoints once per Space container."""
    global _setup_done
    if _setup_done:
        return

    WORK.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK)

    if not LEFFA_ROOT.exists():
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/franciszzj/Leffa.git",
                str(LEFFA_ROOT),
            ]
        )

    # Prefer Leffa 3rdparty SCHP/densepose trees for imports
    for name in ("SCHP", "densepose"):
        link = LEFFA_ROOT / name
        target = LEFFA_ROOT / "3rdparty" / name
        if target.exists() and not link.exists():
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError:
                shutil.copytree(target, link)

    # Remove vendored detectron2 symlink so we never require compiled _C
    d2 = LEFFA_ROOT / "detectron2"
    if d2.is_symlink() or d2.exists():
        if d2.is_symlink() or d2.is_file():
            d2.unlink()
        elif d2.is_dir():
            shutil.rmtree(d2, ignore_errors=True)

    if str(LEFFA_ROOT) not in sys.path:
        sys.path.insert(0, str(LEFFA_ROOT))

    from pipeline import PoseClothPipeline

    global _pipe
    _pipe = PoseClothPipeline(
        leffa_root=str(LEFFA_ROOT),
        ckpt_dir=str(CKPT_DIR),
        dtype="float16",
        enable_face_lock=True,
        default_ref_kind="clothed_person",
        preserve_body=True,
    )
    _pipe.download_checkpoints()
    _setup_done = True
    print("Runtime ready", flush=True)


def _side_by_side(base: Image.Image, ref: Image.Image, result: Image.Image) -> Image.Image:
    imgs = [im.convert("RGB").resize((384, 512)) for im in (base, ref, result)]
    canvas = Image.new("RGB", (384 * 3 + 20, 512 + 36), (245, 245, 245))
    labels = ["Base", "Pose & Outfit", "Result"]
    draw = ImageDraw.Draw(canvas)
    for i, (im, lab) in enumerate(zip(imgs, labels)):
        x = i * 384 + i * 10
        canvas.paste(im, (x, 28))
        draw.text((x + 8, 6), lab, fill=(20, 20, 20))
    return canvas


@spaces.GPU(duration=180)
def run_swap(
    base,
    ref,
    mode,
    garment_type,
    ref_kind,
    steps,
    guidance,
    seed,
    face_lock,
    preserve_body,
    progress=gr.Progress(track_tqdm=True),
):
    if base is None or ref is None:
        raise gr.Error("Upload both a Base image and a Pose & Outfit image.")

    setup_runtime()
    assert _pipe is not None

    mode_map = {
        "Outfit only (recommended)": "outfit_only",
        "Both (outfit + pose)": "both",
        "Pose only": "pose_only",
    }
    garment_map = {
        "Upper body": "upper_body",
        "Lower body": "lower_body",
        "Dress / full outfit": "dresses",
    }
    ref_kind_map = {
        "Clothed person (default)": "clothed_person",
        "Flat garment photo": "flat_garment",
    }

    progress(0.05, desc="Running swap...")
    result = _pipe.generate(
        base_image=base,
        ref_image=ref,
        mode=mode_map[mode],
        garment_type=garment_map[garment_type],
        steps=int(steps),
        guidance_scale=float(guidance),
        seed=int(seed),
        ref_acceleration=True,
        face_lock=bool(face_lock),
        ref_kind=ref_kind_map[ref_kind],
        preserve_body=bool(preserve_body),
    )
    compare = _side_by_side(base, ref, result)
    out = WORK / "outputs"
    out.mkdir(exist_ok=True)
    result_path = out / "result.png"
    result.save(result_path)
    progress(1.0, desc="Done")
    return result, compare, str(result_path)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Poser Outfit Changer") as demo:
        gr.Markdown(
            """
            # Poser Outfit Changer
            Keep the **base** person's face and body. Copy clothes (and optionally pose) from another clothed person.

            **Shared GPU tip:** start with **Outfit only**. Full pose uses SDXL and may fail on small free GPUs.
            """
        )
        with gr.Row():
            base_in = gr.Image(label="Base image (face + body to keep)", type="pil", height=360)
            ref_in = gr.Image(
                label="Pose & Outfit image (person wearing clothes)", type="pil", height=360
            )
        with gr.Row():
            mode = gr.Radio(
                [
                    "Outfit only (recommended)",
                    "Both (outfit + pose)",
                    "Pose only",
                ],
                value="Outfit only (recommended)",
                label="Mode",
            )
            garment = gr.Radio(
                ["Upper body", "Lower body", "Dress / full outfit"],
                value="Dress / full outfit",
                label="Which clothes to copy",
            )
            ref_kind = gr.Radio(
                ["Clothed person (default)", "Flat garment photo"],
                value="Clothed person (default)",
                label="Reference image type",
            )
        with gr.Accordion("Advanced", open=False):
            steps = gr.Slider(15, 40, value=20, step=1, label="Inference steps")
            guidance = gr.Slider(1.0, 5.0, value=2.5, step=0.1, label="Guidance scale")
            seed = gr.Number(value=42, precision=0, label="Seed")
            face_lock = gr.Checkbox(value=True, label="Face identity lock")
            preserve_body = gr.Checkbox(value=True, label="Preserve body proportions")
        btn = gr.Button("Generate", variant="primary")
        with gr.Row():
            result_out = gr.Image(label="Result", type="pil", height=420)
            compare_out = gr.Image(label="Base | Ref | Result", type="pil", height=420)
        file_out = gr.File(label="Download result PNG")

        btn.click(
            fn=run_swap,
            inputs=[
                base_in,
                ref_in,
                mode,
                garment,
                ref_kind,
                steps,
                guidance,
                seed,
                face_lock,
                preserve_body,
            ],
            outputs=[result_out, compare_out, file_out],
        )
    return demo


demo = build_ui()

if __name__ == "__main__":
    # Warm setup on dedicated GPU Spaces; on ZeroGPU setup runs inside @spaces.GPU
    if os.environ.get("SPACE_HW", "").lower() not in {"zerogpu", "zero-gpu"}:
        try:
            setup_runtime()
        except Exception as exc:
            print("Startup setup deferred:", exc, flush=True)
    demo.queue(default_concurrency_limit=1).launch()
