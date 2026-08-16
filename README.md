# Pose & Outfit Changer — Leffa + Identity/Body Preservation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roshiend/Poser-Outfit-Changer/blob/main/Pose_Cloth_Changer.ipynb)

Transfer an **outfit and/or pose** from a reference image onto a **base person**, while keeping the base person's identity and body appearance as consistent as the available models allow.

## Fidelity-first pipeline

```text
Base person + reference person
       │             │
       │             ├─ extract clothing
       │             └─ target pose
       ▼
  Leffa VTON
       │
       ▼
body-scale-align target pose
       │
       ▼
real DensePose IUV + Leffa pose transfer
       │
       ▼
adaptive face identity lock
       │
       ▼
     result
```

The project deliberately separates clothing transfer, pose transfer and identity restoration instead of asking one diffusion pass to solve everything.

## Important accuracy rules

- **Pose transfer requires real Detectron2 DensePose.** The parsing-based fallback is allowed only for virtual try-on. It is not used as fake IUV conditioning for pose transfer.
- **No silent mode downgrade.** Selecting `Both` or `Pose only` either runs pose transfer or returns a clear error explaining what is missing.
- **No final-image body stretching.** The previous crop/resize/paste body correction and torso pixel blending are disabled because they could damage limbs, pose geometry and backgrounds.
- **Body preservation happens before pose diffusion.** The target-pose person is uniformly scaled/recentered to the base person's on-canvas body scale before DensePose extraction.
- **Face lock is head-angle aware.** Strong frontal-face pasting is reduced or skipped when the generated head is profile/turned enough that a 2-D affine face warp would look wrong.
- **VTON model selection is automatic.** `VITON-HD` is used for upper-body clothing and `DressCode` for lower-body or full-outfit/dress transfer.

## Inputs

- **Base image** — identity/body appearance to keep.
- **Reference image** — pose and/or clothes to copy.

Clear, well-lit full-body images generally produce the best results. Extreme occlusion, back-facing heads, unusual crops and very different camera perspectives remain difficult for current diffusion models.

## Modes

| Mode | What happens |
|---|---|
| **Both** | Outfit transfer, then real-DensePose pose transfer, then adaptive face lock |
| **Outfit only** | Change clothing while keeping the base pose |
| **Pose only** | Transfer the target pose while keeping the base appearance |

For garment regions, choose `Upper`, `Lower`, or `Dress / full outfit`.

## Hugging Face Space

The `hf_space/` app now treats pose fidelity as a hard requirement:

1. It does **not** fake `SPACE_ID` locally.
2. It does **not** remove Leffa's vendored Detectron2 source.
3. When a pose mode is requested, it checks for Detectron2's compiled `_C` extension.
4. If needed, it attempts to build/install Leffa's vendored Detectron2 package.
5. If real DensePose still cannot load, pose generation stops with a clear error instead of using the fallback predictor.

The Space UI also exposes intermediate debug images such as the extracted garment, VTON mask, DensePose control, body-aligned target pose and post-pose result.

## Google Colab

Free Colab remains memory-constrained because Leffa pose transfer uses an SDXL-based checkpoint. The Python pipeline therefore disables pose by default when it detects a normal Colab runtime.

To force pose on a suitable Colab GPU, set this **before** creating `PoseClothPipeline`:

```python
import os
os.environ["LEFFA_ALLOW_POSE"] = "1"
```

You still need a working real Detectron2 DensePose installation. If the GPU/RAM is too small, use `Outfit only` or a larger runtime rather than accepting an inaccurate fallback pose.

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
│   ├── garment_extract.py
│   ├── leffa_sequential.py
│   └── memory.py
├── hf_space/
│   ├── app.py
│   ├── README.md
│   ├── requirements.txt
│   └── pipeline/
└── tests/
    └── test_fidelity_helpers.py
```

`pipeline/` is the canonical editable source. `push_to_hf_space.py` now copies its Python files into `hf_space/pipeline/` before deployment so the Space cannot accidentally deploy stale pipeline code.

## Validation

Lightweight regression tests cover the new fidelity policies:

```bash
python -m unittest discover -s tests -v
```

These tests do not run the multi-GB diffusion checkpoints; full image-quality validation still requires a GPU and representative base/reference image pairs.

## Requirements

- NVIDIA GPU for Leffa inference
- PyTorch / torchvision
- Leffa checkpoints
- SCHP/OpenPose preprocessing
- Detectron2 + DensePose for pose modes
- InsightFace for adaptive identity lock
- Gradio for the app

## Limits

No diffusion pipeline can guarantee literal pixel-perfect identity or body geometry from a single image. The goal here is to avoid known fidelity-destroying shortcuts and make every fallback explicit. Results still vary with pose extremity, clothing occlusion, face visibility, source resolution and model training distribution.

The upstream Leffa try-on/pose models are trained on academic fashion/person datasets. Check Leffa, Detectron2, InsightFace and checkpoint licenses before commercial deployment.

## Credits

- [Leffa](https://github.com/franciszzj/Leffa) — virtual try-on and pose transfer
- [InsightFace](https://github.com/deepinsight/insightface) — face detection / pose-aware identity lock
- Detectron2 DensePose — real body-surface conditioning for pose transfer
- SCHP / OpenPose — human parsing and pose preprocessing
