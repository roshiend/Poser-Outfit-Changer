# Pose & Cloth Swap — Free Google Colab App

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roshiend/Poser-Outfit-Changer/blob/main/Pose_Cloth_Changer.ipynb)

**[▶ Open in Google Colab (one click)](https://colab.research.google.com/github/roshiend/Poser-Outfit-Changer/blob/main/Pose_Cloth_Changer.ipynb)**

Transfer **pose + outfit** from a reference image onto a **base person**, while keeping face identity and body proportions as consistent as possible.

**Typical inputs:** base person (identity) + another **person wearing clothes** (pose + outfit). Flat garment photos are optional.

**Cost: $0** — free Google Colab T4 GPU + open-source [Leffa](https://github.com/franciszzj/Leffa) models + InsightFace face lock.

## How it works

```
Base person  +  Clothed person (pose + outfit)
       │                │
       │                ├── parse & extract clothes
       ▼                ▼
  Leffa VTON  ←——  garment from clothed ref
       │
       ▼
  Leffa Pose  ←——  pose from clothed ref, appearance from dressed base
       │
       ▼
  Face identity lock (InsightFace)
       │
       ▼
     Result
```

1. **Extract outfit** — isolate clothing from the clothed reference person (default).
2. **Outfit transfer** — put those clothes on the base person.
3. **Pose transfer** — move the dressed base into the reference pose.
4. **Face lock** — re-apply the base face after pose warping.

## Quick start (Google Colab)

1. Click **[Open in Colab](https://colab.research.google.com/github/roshiend/Poser-Outfit-Changer/blob/main/Pose_Cloth_Changer.ipynb)** (badge at the top of this README).
2. **Runtime → Change runtime type → T4 GPU** (required).
3. **Runtime → Run all** (section 1 installs packages *before* importing them — you should not need Restart session).
4. Wait for weight download on the first run (several GB).
5. Click the **public Gradio link** (`*.gradio.live`) when the last cell finishes.
6. Upload:
   - **Base image** — the person whose face/body you want to keep
   - **Pose & Outfit image** — the pose and clothes to copy
7. Choose mode (**Both** recommended) → **Generate** → download the PNG.

If Colab still asks to restart: **Runtime → Disconnect and delete runtime**, reopen the notebook, and run from the top again.

## UI options

| Control | Meaning |
|---------|---------|
| Mode: Both | Outfit then pose then face lock (full swap) |
| Mode: Outfit only | Change clothes only, keep base pose |
| Mode: Pose only | Change pose only, keep base clothes |
| Garment region | Upper / Lower / Dress — which area VTON replaces |
| Face identity lock | Paste/refine base face onto the result |
| Steps / seed | Quality vs speed; seed for reproducibility |

## Project layout

```
Pose&Cloth Changer/
├── Pose_Cloth_Changer.ipynb   # Main Colab notebook (self-contained)
├── README.md
└── pipeline/
    ├── __init__.py
    ├── face_lock.py           # InsightFace face identity lock
    ├── leffa_sequential.py    # VTON → pose → face, one model in VRAM at a time
    └── memory.py              # CUDA cache cleanup
```

The notebook writes the same `pipeline/` helpers into `/content` so you can run from Colab without uploading the whole repo. The local `pipeline/` folder is the editable source of truth.

## Free-tier tips

- **First run is slow** — models download from Hugging Face once per session.
- **Session disconnects** — Colab free runtimes time out; re-run install + download cells after reconnect.
- **CUDA out of memory** — switch to *Outfit only* or *Pose only*, or lower steps to ~20. The pipeline unloads each diffusion model between stages to fit ~15GB T4 VRAM.
- **Better face consistency** — use a clear, well-lit base photo with a visible face; keep Face lock enabled.
- **Better clothes transfer** — full-body shots work best; set garment region to match (upper / lower / dress).
- Optional: mount Google Drive and point `ckpt_dir` there so weights survive session resets.

## Requirements (Colab installs these for you)

- NVIDIA GPU (Colab T4 is enough)
- PyTorch (preinstalled on Colab)
- Leffa + detectron2 + InsightFace + Gradio

## Limits (honest)

Diffusion models cannot guarantee literal pixel-perfect “100%” identity. This app maximizes consistency with a staged pipeline and a face-lock pass. Results vary with lighting, occlusion, extreme poses, and resolution.

Models used for try-on/pose are trained on academic datasets (VITON-HD / DressCode / DeepFashion). Use for personal / research purposes; check upstream licenses before commercial use.

## Credits

- [Leffa](https://github.com/franciszzj/Leffa) — pose transfer & virtual try-on  
- [InsightFace](https://github.com/deepinsight/insightface) — face detection / identity lock  
- DensePose / SCHP / OpenPose — preprocessing (via Leffa)
