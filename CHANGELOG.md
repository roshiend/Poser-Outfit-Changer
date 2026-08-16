# Changelog

## 1.0.1 — 2026-08-16

### Google Colab low-memory runtime

- Add a Colab-aware public pipeline that automatically selects staged inference on managed Colab runtimes while leaving local/Hugging Face behavior unchanged.
- Stage Leffa CUDA residency as VAE encode → reference UNet → generative UNet → VAE decode instead of placing the complete diffusion model on the GPU at once.
- Add a preferred meta-device + memory-mapped checkpoint load path to reduce system-RAM peaks, with a float16 initialization fallback for older Torch/Accelerate combinations.
- Cast heavyweight modules to float16 only when moved onto CUDA.
- Keep real Detectron2 DensePose mandatory for pose transfer; the low-memory path does not substitute approximate/fake pose conditioning.
- On GPUs below 20 GB VRAM, run SDXL pose transfer with a single denoising batch instead of classifier-free-guidance batch doubling.
- Choose diffusion resolution from measured VRAM: full 768×1024 for >=14 GB, balanced 672×896 for >=11 GB, and safe 576×768 below that; allow explicit `full`, `balanced`, or `safe` override.
- Enable VAE slicing/tiling when supported and use PyTorch expandable CUDA segments to reduce fragmentation.
- Limit Detectron2 compilation parallelism with `MAX_JOBS=2` in the Colab setup path to reduce setup RAM pressure.
- Add separate VTON and pose memory diagnostics plus peak CUDA allocation reporting.

### Colab notebook

- Replace the Gradio/share-server notebook flow with direct notebook-cell interaction and native Colab upload/download dialogs.
- Add `colab_runner.py` for environment setup, real-DensePose preparation, preflight, generation, memory reporting and result download.
- Add an explicit `safe` recovery path for unusually small assigned GPUs.
- Keep the notebook as a thin wrapper around the canonical repository pipeline so Colab cannot drift into a second implementation.

### Validation

- Add Colab memory-policy and notebook-surface regression tests.
- Keep canonical → Hugging Face pipeline drift verification and Python 3.10/3.11 compilation/testing.
- The lightweight CI suite does not run the multi-GB Leffa checkpoints, so actual peak memory and perceptual quality still require a managed Colab GPU run.

## 1.0.0 — 2026-08-16

### Fidelity

- Require real Detectron2 DensePose for pose transfer; VTON fallback is never used as fake pose IUV.
- Remove destructive post-generation body stretching and torso blending.
- Add uniform body-scale alignment and conservative OpenPose-guided DensePose anatomy retargeting.
- Add head-angle-aware face identity restoration and InsightFace embedding diagnostics.
- Add guarded garment-only extraction from clothed-person references with no full-person fallback.
- Add automatic upper/lower/full-outfit garment routing and matching VITON-HD/DressCode selection.
- Add face/hair/background preservation diagnostics after VTON.

### Evaluation

- Add a multi-strength fidelity benchmark with pose-angle, body-proportion and identity metrics.
- Add visibility-aware identity scoring and per-case/global retarget-strength recommendations.
- Add comparison sheets and optional intermediate debug exports.

### Runtime and deployment

- Add runtime preflight checks for Python, Leffa source, checkpoints, core modules, CUDA and Detectron2.
- Add synchronized canonical `pipeline/` → `hf_space/pipeline/` deployment flow and CI drift protection.
- Align the Colab notebook and Hugging Face UI with the same Fidelity v3 controls and diagnostics.
- Align local/Colab requirements with Leffa's current dependency surface without forcing a PyTorch reinstall.
- Add Python 3.10/3.11 CI compilation and lightweight regression coverage.

### Documentation

- Add deployment, benchmark, preflight and fidelity-policy documentation.
- Add MIT license file matching the Hugging Face Space metadata.

### Validation boundary

The repository-level implementation and lightweight test suite do not run the multi-GB Leffa diffusion checkpoints. Final perceptual quality still depends on the chosen GPU/runtime and representative base/reference image pairs; use `benchmark_fidelity.py` for that environment-specific validation.
