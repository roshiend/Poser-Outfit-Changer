# Pose & Outfit Changer — Leffa Fidelity v2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roshiend/Poser-Outfit-Changer/blob/main/Pose_Cloth_Changer.ipynb)

Transfer an **outfit and/or pose** from a reference image onto a **base person**, while keeping the base person's identity and body appearance as consistent as the available models allow.

## Fidelity v2 pipeline

```text
Base person + reference person
       │             │
       │             ├─ extract clothing
       │             └─ target pose
       ▼
  Leffa VTON
       │
       ▼
uniform body-scale alignment
       │
       ├─ OpenPose anatomy measurements
       ├─ safety gate
       └─ retarget real DensePose IUV control
       │
       ▼
  Leffa pose transfer
       │
       ▼
head-angle + identity-similarity-aware face lock
       │
       ▼
     result
```

The project deliberately separates clothing transfer, pose control and identity restoration instead of asking one diffusion pass to solve everything.

## Important accuracy rules

- **Pose transfer requires real Detectron2 DensePose.** The parsing-based fallback is allowed only for virtual try-on. It is never used as fake IUV conditioning for pose transfer.
- **No silent mode downgrade.** Selecting `Both` or `Pose only` either runs pose transfer or returns a clear error explaining what is missing.
- **No final-image body stretching.** The completed RGB person is never resized/pasted to force body proportions.
- **Body preservation happens before pose diffusion.** The reference is first uniformly scale-aligned, then the real DensePose IUV control can be anatomy-retargeted.
- **Anatomy retargeting is safety-gated.** It is skipped when core joints are missing, too few body landmarks are shared, the proportion gap is excessive, or the requested control warp is too large.
- **Reference pose directions are preserved.** Bone lengths move conservatively toward the base person's measured anatomy while reference joint directions remain the pose source.
- **Face lock is head-angle aware.** Strong frontal-face pasting is reduced or skipped for profile/turned heads.
- **Identity similarity is measured.** InsightFace normalized embeddings are compared before/after face correction and shown in debug/status output when available.
- **VTON model selection is automatic.** `VITON-HD` is used for upper-body clothing and `DressCode` for lower-body or full-outfit/dress transfer.

## Inputs

- **Base image** — identity/body appearance to keep.
- **Reference image** — pose and/or clothes to copy.

Clear, well-lit full-body images generally produce the best results. Extreme occlusion, back-facing heads, unusual crops and very different camera perspectives remain difficult for current diffusion models.

## Modes

| Mode | What happens |
|---|---|
| **Both** | Outfit transfer → real-DensePose pose transfer → adaptive identity lock |
| **Outfit only** | Change clothing while keeping the base pose |
| **Pose only** | Transfer the target pose while keeping the base appearance |

For garment regions, choose `Upper`, `Lower`, or `Dress / full outfit`.

## Anatomy retarget strength

The v2 pipeline exposes `pose_retarget_strength` from `0.0` to `1.0`.

- `0.0` — use the original target DensePose unchanged.
- `0.65` — default conservative retargeting.
- `1.0` — move safe measurable bone lengths as far as the configured clamps allow toward the base person's anatomy.

Even at `1.0`, safety checks and per-segment ratio clamps remain active. If the pair is not safe to warp, the original real DensePose control is used and the reason is reported.

## Hugging Face Space

The `hf_space/` app:

1. checks for a compiled Detectron2 `_C` extension when a pose mode is requested;
2. attempts to build Leffa's vendored Detectron2 package when needed;
3. stops pose generation if real DensePose cannot load;
4. exposes anatomy retarget strength;
5. shows original/retargeted skeletons and DensePose controls in **Pipeline debug**;
6. reports whether retargeting was applied or skipped and, when available, InsightFace identity similarity before/after correction.

Pose transfer uses Leffa's heavier SDXL checkpoint, so use a GPU with enough memory for the full pipeline.

## Google Colab

The one-click notebook imports the canonical repository `pipeline/` instead of embedding a second stale implementation.

Free Colab remains memory-constrained because Leffa pose transfer uses an SDXL-based checkpoint. Pose is disabled by default in a normal Colab runtime. To force pose on a suitable runtime, set this **before** creating `PoseClothPipeline`:

```python
import os
os.environ["LEFFA_ALLOW_POSE"] = "1"
```

You still need working real Detectron2 DensePose. If the GPU/RAM is too small, use `Outfit only` or a larger runtime rather than accepting an inaccurate pose fallback.

## Project layout

```text
Poser-Outfit-Changer/
├── Pose_Cloth_Changer.ipynb
├── README.md
├── push_to_hf_space.py
├── requirements.txt
├── pipeline/
│   ├── __init__.py
│   ├── body_lock.py
│   ├── densepose_fallback.py
│   ├── face_lock.py
│   ├── fidelity_v2.py
│   ├── garment_extract.py
│   ├── leffa_sequential.py
│   ├── memory.py
│   └── pose_geometry.py
├── hf_space/
│   ├── app.py
│   ├── README.md
│   ├── requirements.txt
│   └── pipeline/
└── tests/
    ├── test_fidelity_helpers.py
    └── test_pose_geometry.py
```

`pipeline/` is the canonical editable source. `push_to_hf_space.py` copies its Python files into `hf_space/pipeline/` before deployment so the deployed Space uses the same pipeline.

## Validation

Lightweight regression tests run on Python 3.10 and 3.11:

```bash
python -m unittest discover -s tests -v
```

They cover VTON model routing, legacy post-body-warp disabling, face pose safety, OpenPose coordinate conversion, anatomy pair safety, preservation of bone direction during retargeting and piecewise-warp identity behavior.

These tests do not run the multi-GB diffusion checkpoints. Full perceptual image-quality validation still requires a suitable GPU and representative base/reference image pairs.

## Requirements

- NVIDIA GPU for Leffa inference
- PyTorch / torchvision
- Leffa checkpoints
- SCHP/OpenPose preprocessing
- Detectron2 + DensePose for pose modes
- InsightFace for identity diagnostics/correction
- Gradio for the app

## Limits

No diffusion pipeline can guarantee literal pixel-perfect identity or body geometry from a single image. Fidelity v2 is designed to avoid known fidelity-destroying shortcuts, make safety fallbacks explicit, and modify pose controls rather than repairing the finished image afterward. Results still vary with pose extremity, clothing occlusion, face visibility, source resolution and model training distribution.

The upstream Leffa try-on/pose models are trained on academic fashion/person datasets. Check Leffa, Detectron2, InsightFace and checkpoint licenses before commercial deployment.

## Credits

- [Leffa](https://github.com/franciszzj/Leffa) — virtual try-on and pose transfer
- [InsightFace](https://github.com/deepinsight/insightface) — face detection, pose and identity embeddings
- Detectron2 DensePose — body-surface conditioning for pose transfer
- SCHP / OpenPose — human parsing and body landmarks
