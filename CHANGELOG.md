# Changelog

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
