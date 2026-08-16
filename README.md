# Pose & Outfit Changer — Fidelity v3

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roshiend/Poser-Outfit-Changer/blob/main/Pose_Cloth_Changer.ipynb)

Transfer an **outfit and/or pose** from a reference image onto a **base person**, while keeping the base person's identity and body appearance as consistent as the available models allow.

## Fidelity pipeline

```text
Base person + reference person
       │             │
       │             ├─ SCHP garment parsing
       │             ├─ auto garment routing + extraction quality gate
       │             └─ target pose
       ▼
  Leffa VTON
       │
       ├─ face/hair/background preservation diagnostics
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

The project deliberately separates garment extraction, virtual try-on, pose control, body-proportion preservation and identity restoration instead of asking one diffusion pass to solve everything.

## Important accuracy rules

- **Clothed-person references are reduced to garment-only inputs.** If parsing cannot isolate a reliable garment, generation stops. It never silently feeds the entire donor person into VTON.
- **Garment type can be detected automatically.** `Auto` distinguishes upper-only, lower-only and dress/full-outfit references from SCHP labels, then selects the matching Leffa VTON model.
- **Outfit transfer is quality-checked before pose diffusion.** Face/hair and background changes are measured while the base pose is still unchanged and large unintended changes are surfaced as warnings.
- **Pose transfer requires real Detectron2 DensePose.** The parsing-based fallback is allowed only for virtual try-on. It is never used as fake IUV conditioning for pose transfer.
- **No silent mode downgrade.** Selecting `Both` or `Pose only` either runs pose transfer or returns a clear error explaining what is missing.
- **No final-image body stretching.** The completed RGB person is never resized/pasted to force body proportions.
- **Body preservation happens before pose diffusion.** The reference is uniformly scale-aligned, then the real DensePose IUV control can be anatomy-retargeted.
- **Anatomy retargeting is safety-gated.** It is skipped when core joints are missing, too few body landmarks are shared, the proportion gap is excessive, or the requested control warp is too large.
- **Reference pose directions are preserved.** Bone lengths move conservatively toward the base person's measured anatomy while the reference remains the pose source.
- **Face lock is head-angle aware.** Strong frontal-face pasting is reduced or skipped for profile/turned heads.
- **Identity similarity is measured.** InsightFace normalized embeddings are compared before/after face correction and shown in debug/status output when available.
- **VTON model selection is automatic after garment resolution.** `VITON-HD` is used for upper-body clothing and `DressCode` for lower-body or full-outfit/dress transfer.

## Inputs

- **Base image** — identity/body appearance to keep.
- **Reference image** — pose and/or clothes to copy.

Clear, well-lit full-body images generally produce the best results. Extreme occlusion, back-facing heads, unusual crops and very different camera perspectives remain difficult for current diffusion models.

## Modes

| Mode | What happens |
|---|---|
| **Both** | garment extraction/VTON → real-DensePose pose transfer → adaptive identity lock |
| **Outfit only** | change clothing while keeping the base pose |
| **Pose only** | transfer the target pose while keeping the base appearance |

For a **clothed-person reference**, `Auto` is the recommended garment setting. For a **flat garment photo**, select `Upper body`, `Lower body`, or `Dress / full outfit` explicitly.

## Automatic garment routing

`garment_type="auto"` is the default in the fidelity pipeline. The parser evaluates visible upper-clothing, skirt/trouser and dress regions:

- upper garment only → `upper_body` → VITON-HD;
- lower garment dominates → `lower_body` → DressCode;
- dress label or material upper + lower regions → `dresses` / full outfit → DressCode.

The extraction also removes tiny parser islands, softly mattes the garment onto white, and rejects tiny/implausible masks. This is intentionally safer than using the whole reference person when parsing fails.

## Anatomy retarget strength

The pipeline exposes `pose_retarget_strength` from `0.0` to `1.0`.

- `0.0` — use the original target DensePose unchanged.
- `0.65` — default conservative retargeting.
- `1.0` — move safe measurable bone lengths as far as the configured clamps allow toward the base person's anatomy.

Even at `1.0`, safety checks and per-segment ratio clamps remain active. If the pair is not safe to warp, the original real DensePose control is used and the reason is reported.

## Fidelity benchmark and retarget tuning

`benchmark_fidelity.py` compares the same base/reference pair across multiple retarget strengths instead of tuning from a single image by eye.

The benchmark reports:

- **Pose angle error (degrees)** — difference in major arm/leg articulation angles against the reference pose. Lower is better.
- **Body proportion error** — RMS multiplicative mismatch between normalized generated and base-person body segments, so a locally stretched limb is not hidden by unchanged segments. Lower is better.
- **Identity similarity** — InsightFace cosine similarity between the base and final generated face. Higher is better. If the target face should be visible but the generated face disappears, identity is penalized; genuinely back-facing targets omit the identity term.

It also produces a bounded composite score for convenient ranking. This is a tuning aid, not a scientific identity/quality guarantee; always inspect the saved comparison sheet.

Create a manifest like `benchmarks/example_manifest.json`, then run:

```bash
python benchmark_fidelity.py benchmarks/my_manifest.json \
  --strengths 0,0.65,1 \
  --output benchmark_runs/run1 \
  --save-debug
```

If checkpoints are not present yet, add `--download-checkpoints`. On a suitable Colab runtime where pose has been explicitly enabled, add `--force-pose`.

Each run writes CSV/JSON metrics, `recommendations.json`, per-strength images, a comparison sheet and optional pipeline debug images. `benchmark_runs/` is ignored by Git.

## Runtime preflight

Before a local/Colab GPU run, inspect the environment instead of discovering missing components halfway through generation:

```bash
python -m pipeline.preflight
python -m pipeline.preflight --require-pose --require-gpu --require-checkpoints
```

The preflight reports Python compatibility, Leffa source presence, required checkpoint files, core Python modules, InsightFace availability, compiled Detectron2/DensePose and CUDA status. `--json` is available for automation.

## Hugging Face Space

The `hf_space/` app:

1. defaults clothed-person references to **Auto detect (recommended)**;
2. refuses unreliable garment extraction rather than leaking the donor person into VTON;
3. reports garment resolution/confidence and outfit fidelity warnings;
4. checks for compiled Detectron2 `_C` when pose is requested;
5. attempts to build Leffa's vendored Detectron2 package when needed;
6. stops pose generation if real DensePose cannot load;
7. exposes anatomy retarget strength and detailed pipeline debug views;
8. reports whether retargeting was applied/skipped and InsightFace identity similarity when available.

Pose transfer uses Leffa's heavier SDXL checkpoint, so use a GPU with enough memory for the full pipeline.

### Deployment synchronization

`pipeline/` is the single canonical implementation. Before any Space upload, synchronize it into the deployable package:

```bash
python push_to_hf_space.py --sync-only
```

The normal deployment command performs this synchronization automatically:

```bash
set HF_TOKEN=hf_...
python push_to_hf_space.py
```

With no `HF_SPACE_ID`, the script uses the **authenticated Hugging Face username** and deploys to `<authenticated-user>/poser-outfit-changer`. To target a different organization/account or Space name, set an explicit ID first:

```bash
set HF_SPACE_ID=my-org/my-space
python push_to_hf_space.py
```

CI also runs sync-only and fails if `hf_space/pipeline/` would change, preventing the Space copy from drifting away from the canonical source.

## Google Colab

The one-click notebook imports the canonical repository `pipeline/` instead of embedding a second implementation.

Free Colab remains memory-constrained because Leffa pose transfer uses an SDXL-based checkpoint. Pose is disabled by default in a normal Colab runtime. To force pose on a suitable runtime, set this **before** creating `PoseClothPipeline`:

```python
import os
os.environ["LEFFA_ALLOW_POSE"] = "1"
```

You still need working real Detectron2 DensePose. If GPU/RAM is too small, use `Outfit only` or a larger runtime rather than accepting an inaccurate pose fallback.

## Project layout

```text
Poser-Outfit-Changer/
├── Pose_Cloth_Changer.ipynb
├── README.md
├── benchmark_fidelity.py
├── benchmarks/
│   └── example_manifest.json
├── push_to_hf_space.py
├── requirements.txt
├── pipeline/
│   ├── __init__.py
│   ├── benchmark_metrics.py
│   ├── body_lock.py
│   ├── densepose_fallback.py
│   ├── face_lock.py
│   ├── fidelity_v2.py
│   ├── garment_extract.py
│   ├── leffa_sequential.py
│   ├── memory.py
│   ├── outfit_quality.py
│   ├── pose_geometry.py
│   └── preflight.py
├── hf_space/
│   ├── app.py
│   ├── README.md
│   ├── requirements.txt
│   └── pipeline/              # synchronized canonical copy
└── tests/
    ├── test_benchmark_metrics.py
    ├── test_fidelity_helpers.py
    ├── test_garment_fidelity.py
    ├── test_pose_geometry.py
    └── test_preflight.py
```

## Validation

Lightweight regression tests run on Python 3.10 and 3.11:

```bash
python -m unittest discover -s tests -v
```

CI synchronizes the deployable Space pipeline, verifies there is no drift, compiles both root and Space modules, and runs the full lightweight suite. Coverage includes VTON routing, garment auto-detection/extraction failure policy, outfit preservation diagnostics, legacy body-warp disabling, face pose safety, OpenPose conversion, anatomy retarget safety/direction, piecewise control warping, benchmark metrics/ranking and runtime preflight requirements.

These tests do **not** run the multi-GB diffusion checkpoints. Full perceptual image-quality validation still requires a suitable GPU and representative base/reference image pairs.

## Requirements

- Python 3.10+
- NVIDIA GPU for Leffa inference
- PyTorch / torchvision
- Leffa checkpoints
- SCHP/OpenPose preprocessing
- Detectron2 + DensePose for pose modes
- InsightFace for identity diagnostics/correction
- Gradio for the app

## Limits

No diffusion pipeline can guarantee literal pixel-perfect identity or body geometry from a single image. Fidelity v3 is designed to eliminate known fidelity-destroying shortcuts, make safety fallbacks explicit, and change generation controls rather than stretching/repairing the finished image afterward. Results still vary with pose extremity, clothing occlusion, face visibility, source resolution and model training distribution.

The upstream Leffa try-on/pose models are trained on academic fashion/person datasets. Check Leffa, Detectron2, InsightFace and checkpoint licenses before commercial deployment.

## Credits

- [Leffa](https://github.com/franciszzj/Leffa) — virtual try-on and pose transfer
- [InsightFace](https://github.com/deepinsight/insightface) — face detection, pose and identity embeddings
- Detectron2 DensePose — body-surface conditioning for pose transfer
- SCHP / OpenPose — human parsing and body landmarks
