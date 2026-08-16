# Pose & Outfit Changer — Fidelity v3

[![Open Latest Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roshiend/Poser-Outfit-Changer/blob/main/Pose_Cloth_Changer.ipynb)

**Latest Colab notebook:** https://colab.research.google.com/github/roshiend/Poser-Outfit-Changer/blob/main/Pose_Cloth_Changer.ipynb

The notebook always fetches/resets the project to the latest `main`, clears cached project modules, reloads `colab_runner`, and prints the repository version + commit before setup so an older Colab kernel cannot silently keep running stale bootstrap code.

Transfer an **outfit and/or pose** from a reference image onto a **base person**, while keeping the base person's identity and body appearance as consistent as the available models allow.

Current release: **1.0.3** — Fidelity v3 plus the notebook-native Google Colab low-memory runtime, corrected DensePose/Detectron2 bootstrap, and forced fresh-module reload after repository updates.

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
- **No silent mode downgrade.** Selecting `Both` or `Pose only` either runs pose transfer or returns a clear error.
- **No final-image body stretching.** The completed RGB person is never resized/pasted to force body proportions.
- **Body preservation happens before pose diffusion.** The reference is uniformly scale-aligned, then the real DensePose IUV control can be anatomy-retargeted.
- **Anatomy retargeting is safety-gated.** It is skipped when landmark coverage or required warping is unsafe.
- **Reference pose directions are preserved.** Bone lengths move conservatively toward the base person's measured anatomy while the reference remains the pose source.
- **Face lock is head-angle aware.** Strong frontal-face pasting is reduced or skipped for profile/turned heads.
- **Identity similarity is measured.** InsightFace normalized embeddings are compared before/after face correction when available.
- **VTON routing is automatic after garment resolution.** VITON-HD handles upper-body clothing; DressCode handles lower-body and full-outfit/dress transfer.

## Inputs and modes

- **Base image** — identity/body appearance to keep.
- **Reference image** — pose and/or clothes to copy.

| Mode | What happens |
|---|---|
| **Both** | garment extraction/VTON → real-DensePose pose transfer → adaptive identity lock |
| **Outfit only** | change clothing while keeping the base pose |
| **Pose only** | transfer the target pose while keeping the base appearance |

For a **clothed-person reference**, `garment_type="auto"` is recommended. For a **flat garment photo**, select `upper_body`, `lower_body`, or `dresses` explicitly.

## Automatic garment routing

The parser evaluates visible upper-clothing, skirt/trouser and dress regions:

- upper garment only → `upper_body` → VITON-HD;
- lower garment dominates → `lower_body` → DressCode;
- dress label or material upper + lower regions → `dresses` / full outfit → DressCode.

The extraction removes tiny parser islands, softly mattes the garment onto white, and rejects tiny/implausible masks rather than using the whole donor person.

## Anatomy retarget strength

`pose_retarget_strength` ranges from `0.0` to `1.0`.

- `0.0` — original target DensePose.
- `0.65` — default conservative retargeting.
- `1.0` — move safe measurable bone lengths as far as configured clamps allow toward the base person's anatomy.

Safety checks remain active at every strength.

## Google Colab low-memory mode

The one-click notebook is designed for **managed Colab notebooks**, including standard/free-style GPU assignments. Google does not guarantee a specific GPU model, GPU availability, fixed resource quota or fixed usage limit; the notebook therefore measures the assigned GPU at runtime instead of assuming a T4 or a fixed amount of VRAM.

The free-Colab workflow also stays inside normal notebook cells. It does **not** launch Gradio, `share=True`, a remote desktop or another web UI.

### What low-memory mode changes

On Colab, the public `PoseClothPipeline` automatically becomes `ColabAwareFidelityPipeline` unless `LEFFA_COLAB_LOW_MEMORY=0` is set.

Instead of upstream Leffa moving the complete model to CUDA at once, the staged path uses:

```text
CPU / mmap checkpoint
       │
       ▼
VAE encode on CUDA
       │   VAE back to CPU
       ▼
Reference UNet on CUDA
       │   reference UNet back to CPU
       ▼
Generative UNet on CUDA
       │   generative UNet back to CPU
       ▼
VAE decode on CUDA
```

Additional safeguards:

- preferred **meta-device + memory-mapped checkpoint loading** to reduce system-RAM peaks;
- float16 initialization fallback where meta/mmap assignment is unavailable;
- heavyweight modules are cast to **FP16 when moved to CUDA**;
- VAE slicing/tiling enabled when supported;
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce allocator fragmentation;
- Detectron2 build uses `MAX_JOBS=1` to reduce compile-time RAM pressure on ~13 GB Colab runtimes;
- the DensePose-facing Detectron2 0.6 `_C` extension is compiled in place from Leffa's matching source tree;
- real Detectron2 DensePose remains mandatory for pose transfer.

### Automatic VRAM policy

The default `LEFFA_COLAB_RESOLUTION=auto` uses measured GPU VRAM:

| Measured VRAM | Diffusion size | SDXL pose CFG |
|---|---:|---|
| **>= 20 GB** | 768×1024 when >=14 GB rule is also met | retained |
| **14–<20 GB** | 768×1024 | disabled to avoid batch doubling |
| **11–<14 GB** | 672×896 | disabled |
| **<11 GB** | 576×768 | disabled |

For unusually constrained runtimes, force the safe profile **before creating the pipeline**:

```python
import os
os.environ["LEFFA_COLAB_RESOLUTION"] = "safe"
```

The safe profile runs diffusion at 576×768 and returns the generated result on the canonical 768×1024 canvas. DensePose uses nearest-neighbour control resizing; it is not replaced by the approximate VTON fallback.

### Run the Colab notebook

1. Open the **Open Latest Colab** badge at the top of this README, or use the plain latest-notebook link directly below it.
2. Choose **Runtime → Change runtime type → GPU**.
3. Run the cells in order.
4. Confirm Cell 1 prints the current repository `VERSION` and short Git commit before continuing.
5. The notebook reports the actual GPU, VRAM and system RAM.
6. Upload the base image, then the reference image.
7. Run the Generate cell. Defaults: `both`, automatic garment routing, 25 steps and retarget strength `0.65`.
8. The notebook prints peak allocated CUDA memory and stage-specific memory policy details.

If CUDA still reports OOM, restart the runtime to clear fragmentation and use the notebook's `safe` recovery cell. Colab resource availability itself cannot be guaranteed by this repository.

## Runtime preflight

Before a local/Colab GPU run:

```bash
python -m pipeline.preflight
python -m pipeline.preflight --require-pose --require-gpu --require-checkpoints
```

The preflight reports Python compatibility, Leffa source presence, checkpoint files, core modules, InsightFace availability, compiled Detectron2/DensePose and CUDA status. `--json` is available for automation.

## Fidelity benchmark

`benchmark_fidelity.py` compares the same base/reference pair across retarget strengths and reports:

- pose angle error;
- normalized body-proportion error;
- InsightFace identity similarity;
- a bounded composite tuning score.

Example:

```bash
python benchmark_fidelity.py benchmarks/my_manifest.json \
  --strengths 0,0.65,1 \
  --output benchmark_runs/run1 \
  --save-debug
```

Generated `benchmark_runs/` data is ignored by Git.

## Hugging Face Space

The `hf_space/` app keeps the full Fidelity v3 UI and automatically uses the normal non-Colab inference path. The Colab-aware public wrapper detects that Hugging Face Space is not a managed Colab runtime, so the new low-memory behavior does not silently change Space inference.

The Space:

1. defaults clothed-person references to Auto detect;
2. refuses unreliable garment extraction;
3. reports garment confidence and outfit fidelity warnings;
4. requires real Detectron2 DensePose for pose;
5. exposes anatomy retarget strength and detailed debug views;
6. reports identity similarity when available.

### Deployment synchronization

`pipeline/` is the canonical source. Synchronize before deploying:

```bash
python push_to_hf_space.py --sync-only
```

Normal deployment performs synchronization automatically:

```bash
set HF_TOKEN=hf_...
python push_to_hf_space.py
```

Without `HF_SPACE_ID`, deployment targets `<authenticated-user>/poser-outfit-changer`. To target another owner/name:

```bash
set HF_SPACE_ID=my-org/my-space
python push_to_hf_space.py
```

CI fails if `hf_space/pipeline/` drifts from the canonical source.

## Project layout

```text
Poser-Outfit-Changer/
├── Pose_Cloth_Changer.ipynb
├── colab_runner.py
├── benchmark_fidelity.py
├── README.md
├── CHANGELOG.md
├── VERSION
├── push_to_hf_space.py
├── requirements.txt
├── pipeline/
│   ├── __init__.py
│   ├── benchmark_metrics.py
│   ├── body_lock.py
│   ├── colab_lowmem.py
│   ├── colab_pipeline.py
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
│   └── pipeline/              # exact synchronized copy
└── tests/
    ├── test_benchmark_metrics.py
    ├── test_colab_lowmem.py
    ├── test_fidelity_helpers.py
    ├── test_garment_fidelity.py
    ├── test_notebook_surface.py
    ├── test_pose_geometry.py
    └── test_preflight.py
```

## Validation

Lightweight regression tests run on Python 3.10 and 3.11:

```bash
python -m unittest discover -s tests -v
```

CI synchronizes the deployable Space pipeline, verifies no drift, compiles the canonical pipeline, Space pipeline and Colab runner, then runs the suite. Coverage includes garment routing/extraction, pose/body/identity policy, anatomy retargeting, benchmark metrics, preflight, Colab VRAM policy and the notebook's no-web-server surface.

**Validation boundary:** CI does not download or execute the multi-GB Leffa diffusion checkpoints. Actual Colab peak memory and perceptual quality must still be verified on a managed Colab GPU because Google's assigned resources vary dynamically.

## Requirements

- Python 3.10+
- NVIDIA GPU for Leffa inference
- PyTorch / torchvision
- Leffa checkpoints
- SCHP/OpenPose preprocessing
- Detectron2 + DensePose for pose modes
- InsightFace for identity diagnostics/correction
- Gradio only for the Hugging Face app; the free-Colab notebook does not use it as its interaction surface

## Limits

No diffusion pipeline can guarantee literal pixel-perfect identity or body geometry from a single image. Fidelity v3 is designed to eliminate known fidelity-destroying shortcuts and keep fallbacks explicit. The Colab low-memory mode additionally trades some compute configuration for memory safety; in particular, SDXL pose on sub-20-GB GPUs avoids classifier-free-guidance batch doubling. Results can therefore differ from large-GPU inference.

The upstream Leffa try-on/pose models are trained on academic fashion/person datasets. Check Leffa, Detectron2, InsightFace and checkpoint licenses before commercial deployment.

## Credits

- [Leffa](https://github.com/franciszzj/Leffa) — virtual try-on and pose transfer
- [InsightFace](https://github.com/deepinsight/insightface) — face detection, pose and identity embeddings
- Detectron2 DensePose — body-surface conditioning for pose transfer
- SCHP / OpenPose — human parsing and body landmarks
