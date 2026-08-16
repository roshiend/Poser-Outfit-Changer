"""Hugging Face Spaces entrypoint for Poser Outfit Changer."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "runtime"
LEFFA_ROOT = WORK / "Leffa"
CKPT_DIR = WORK / "ckpts"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LEFFA_ROOT))

try:
    import spaces
except Exception:
    class _SpacesShim:
        @staticmethod
        def GPU(duration=120, size=None):
            def deco(fn):
                return fn
            return deco

    spaces = _SpacesShim()

_pipe = None
_setup_done = False
_densepose_runtime_ready = False


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def _ensure_vendor_links() -> None:
    for name in ("SCHP", "densepose", "detectron2"):
        link = LEFFA_ROOT / name
        target = LEFFA_ROOT / "3rdparty" / name
        if target.exists() and not link.exists():
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError:
                shutil.copytree(target, link)


def _detectron2_compiled() -> bool:
    try:
        import detectron2  # noqa: F401
        from detectron2 import _C  # noqa: F401
        return True
    except Exception as exc:
        print(f"[setup] Detectron2 extension unavailable: {exc!r}", flush=True)
        return False


def _ensure_real_densepose_runtime() -> None:
    global _densepose_runtime_ready
    if _densepose_runtime_ready and _detectron2_compiled():
        return
    if _detectron2_compiled():
        _densepose_runtime_ready = True
        return

    source = LEFFA_ROOT / "3rdparty" / "detectron2"
    if not source.exists():
        raise RuntimeError(f"Leffa Detectron2 source is missing: {source}")

    print("[setup] Building Detectron2 for accurate DensePose pose transfer...", flush=True)
    _run([sys.executable, "-m", "pip", "install", "-e", str(source)])

    for name in list(sys.modules):
        if name == "detectron2" or name.startswith("detectron2."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()

    if not _detectron2_compiled():
        raise RuntimeError(
            "Detectron2 installed but its compiled _C extension still cannot be imported. "
            "Pose transfer is stopped rather than using fake DensePose IUV."
        )
    _densepose_runtime_ready = True
    print("[setup] Real DensePose runtime ready", flush=True)


def setup_runtime(require_pose: bool = False) -> None:
    global _setup_done, _pipe

    if not _setup_done:
        WORK.mkdir(parents=True, exist_ok=True)
        os.chdir(WORK)

        if not LEFFA_ROOT.exists():
            _run([
                "git", "clone", "--depth", "1",
                "https://github.com/franciszzj/Leffa.git",
                str(LEFFA_ROOT),
            ])

        _ensure_vendor_links()
        if str(LEFFA_ROOT) not in sys.path:
            sys.path.insert(0, str(LEFFA_ROOT))

        from pipeline import PoseClothPipeline

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

    if require_pose:
        _ensure_real_densepose_runtime()


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


def _debug_gallery(debug: dict):
    preferred = [
        ("garment_ref", "Extracted garment"),
        ("outfit_mask", "VTON mask"),
        ("outfit_densepose", "VTON DensePose"),
        ("after_vton", "After outfit"),
        ("aligned_pose_donor", "Body-aligned pose donor"),
        ("pose_skeleton_before", "Target skeleton before anatomy retarget"),
        ("pose_skeleton_retargeted", "Skeleton after anatomy retarget"),
        ("pose_densepose_original", "Original real DensePose IUV"),
        ("pose_densepose_retargeted", "Anatomy-retargeted DensePose IUV"),
        ("after_pose", "After pose"),
        ("after_face_lock", "Final identity lock"),
    ]
    items = []
    for key, label in preferred:
        value = debug.get(key)
        if isinstance(value, Image.Image):
            items.append((value, label))
    return items


def _status_text(debug: dict, internal_mode: str, require_pose: bool) -> str:
    selected = debug.get("selected_vton_model", "n/a")
    parts = [
        f"**Completed:** `{internal_mode}`",
        f"VTON: `{selected}`",
        f"Pose DensePose: `{'real/required' if require_pose else 'not required'}`",
    ]

    retarget = debug.get("pose_retarget_diagnostics") or {}
    if require_pose:
        if retarget.get("safe_to_retarget"):
            gap = retarget.get("proportion_gap")
            gap_text = f"{float(gap):.2f}" if isinstance(gap, (int, float)) else "n/a"
            parts.append(f"Anatomy retarget: `applied` (gap {gap_text})")
        else:
            parts.append(f"Anatomy retarget: `skipped` ({retarget.get('reason', 'n/a')})")

    identity = debug.get("identity_info") or {}
    before = identity.get("identity_similarity_before")
    after = identity.get("identity_similarity_after")
    if isinstance(before, (int, float)):
        text = f"Identity similarity: `{before:.3f}`"
        if isinstance(after, (int, float)):
            text += f" → `{after:.3f}`"
        parts.append(text)

    return "  •  ".join(parts)


@spaces.GPU(duration=300)
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
    pose_retarget_strength,
    progress=gr.Progress(track_tqdm=True),
):
    if base is None or ref is None:
        raise gr.Error("Upload both a Base image and a Pose & Outfit image.")

    mode_map = {
        "Outfit only": "outfit_only",
        "Both (outfit + pose)": "both",
        "Pose only": "pose_only",
    }
    garment_map = {
        "Upper body": "upper_body",
        "Lower body": "lower_body",
        "Dress / full outfit": "dresses",
    }
    ref_kind_map = {
        "Clothed person": "clothed_person",
        "Flat garment photo": "flat_garment",
    }

    internal_mode = mode_map[mode]
    require_pose = internal_mode in {"both", "pose_only"}
    progress(0.03, desc="Preparing runtime...")

    try:
        setup_runtime(require_pose=require_pose)
        assert _pipe is not None

        progress(0.08, desc="Running transfer...")
        result, debug = _pipe.generate(
            base_image=base,
            ref_image=ref,
            mode=internal_mode,
            garment_type=garment_map[garment_type],
            vt_model_type="auto",
            steps=int(steps),
            guidance_scale=float(guidance),
            seed=int(seed),
            ref_acceleration=True,
            face_lock=bool(face_lock),
            ref_kind=ref_kind_map[ref_kind],
            preserve_body=bool(preserve_body),
            pose_retarget_strength=float(pose_retarget_strength),
            return_debug=True,
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    compare = _side_by_side(base, ref, result)
    out = WORK / "outputs"
    out.mkdir(exist_ok=True)
    result_path = out / "result.png"
    result.save(result_path)

    status = _status_text(debug, internal_mode, require_pose)
    progress(1.0, desc="Done")
    return result, compare, str(result_path), _debug_gallery(debug), status


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Poser Outfit Changer") as demo:
        gr.Markdown(
            """
            # Poser Outfit Changer
            Keep the **base person's identity/body appearance** and copy clothes and/or pose from a reference person.

            **Fidelity v2:** pose transfer uses real DensePose plus conservative OpenPose anatomy retargeting.
            The retargeter changes the **pose control**, never stretches the final generated body. Identity lock
            also reports InsightFace similarity and adapts correction strength while respecting head angle.
            """
        )
        with gr.Row():
            base_in = gr.Image(label="Base image — identity/body to keep", type="pil", height=360)
            ref_in = gr.Image(label="Reference — pose/outfit to copy", type="pil", height=360)

        with gr.Row():
            mode = gr.Radio(
                ["Both (outfit + pose)", "Outfit only", "Pose only"],
                value="Both (outfit + pose)",
                label="Mode",
            )
            garment = gr.Radio(
                ["Upper body", "Lower body", "Dress / full outfit"],
                value="Dress / full outfit",
                label="Clothes to copy",
            )
            ref_kind = gr.Radio(
                ["Clothed person", "Flat garment photo"],
                value="Clothed person",
                label="Reference type",
            )

        with gr.Accordion("Advanced", open=False):
            gr.Markdown(
                "VTON model is selected automatically: VITON-HD for upper-body, "
                "DressCode for lower/full outfit. Anatomy retargeting is safety-gated and is skipped "
                "automatically if too few joints are visible or the required warp would be excessive."
            )
            steps = gr.Slider(20, 50, value=30, step=1, label="Inference steps")
            guidance = gr.Slider(1.0, 5.0, value=2.5, step=0.1, label="Guidance scale")
            seed = gr.Number(value=42, precision=0, label="Seed")
            face_lock = gr.Checkbox(value=True, label="Adaptive face identity lock")
            preserve_body = gr.Checkbox(value=True, label="Preserve base body scale + anatomy")
            pose_retarget_strength = gr.Slider(
                0.0,
                1.0,
                value=0.65,
                step=0.05,
                label="Anatomy retarget strength",
                info="0 = original target DensePose; 1 = move safe bone lengths as far as allowed toward the base person.",
            )

        btn = gr.Button("Generate", variant="primary")
        status = gr.Markdown()
        with gr.Row():
            result_out = gr.Image(label="Result", type="pil", height=420)
            compare_out = gr.Image(label="Base | Reference | Result", type="pil", height=420)
        with gr.Accordion("Pipeline debug", open=False):
            debug_out = gr.Gallery(
                label="Intermediate controls/results",
                columns=4,
                height="auto",
            )
        file_out = gr.File(label="Result PNG")

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
                pose_retarget_strength,
            ],
            outputs=[result_out, compare_out, file_out, debug_out, status],
        )
    return demo


demo = build_ui()

if __name__ == "__main__":
    if os.environ.get("SPACE_HW", "").lower() not in {"zerogpu", "zero-gpu"}:
        try:
            setup_runtime(require_pose=False)
        except Exception as exc:
            print("Startup setup deferred:", exc, flush=True)
    demo.queue(default_concurrency_limit=1).launch()
