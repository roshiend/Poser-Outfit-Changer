---
title: Poser Outfit Changer
emoji: 👕
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: "5.49.1"
app_file: app.py
pinned: false
license: mit
short_description: Fidelity-first pose and outfit transfer with Leffa
tags:
  - virtual-try-on
  - pose-transfer
  - leffa
  - fashion
---

# Poser Outfit Changer

Upload a **base person** whose identity/body appearance should be kept and a **reference person** supplying the outfit and/or target pose.

## Accuracy behavior

- **Both** genuinely runs outfit + pose. It is never silently changed to outfit-only.
- Pose modes require **real Detectron2 DensePose**. The app will attempt to build Leffa's vendored Detectron2 package on first pose use.
- If real DensePose is unavailable, pose generation stops with a clear error rather than using approximate/fake IUV conditioning.
- Upper-body try-on uses **VITON-HD** automatically; lower-body and full-outfit/dress use **DressCode**.
- Final-image body stretching and old-torso blending are disabled.
- Face identity restoration reduces or skips 2-D face pasting when the generated head angle differs too much from the base face.
- Open **Pipeline debug** to inspect garment extraction, masks, DensePose controls and intermediate results.

Pose transfer uses Leffa's heavier SDXL checkpoint, so use a GPU with enough memory for the full pipeline.

Built with Leffa + Detectron2 DensePose + InsightFace.
