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

## Fidelity v2 behavior

- **Both** genuinely runs outfit + pose. It is never silently changed to outfit-only.
- Pose modes require **real Detectron2 DensePose**. If real DensePose cannot load, pose generation stops rather than using approximate/fake IUV conditioning.
- Upper-body try-on uses **VITON-HD** automatically; lower-body and full-outfit/dress use **DressCode**.
- The target person is uniformly body-scale aligned before pose diffusion.
- Leffa's OpenPose landmarks are used to measure anatomy and safety-gate a conservative DensePose-control retarget.
- Bone lengths can move toward the base person's proportions while target joint directions/pose are preserved.
- If too few joints are visible, proportions differ too much, or the requested warp is excessive, anatomy retargeting is skipped automatically and the original real DensePose control is used.
- The finished RGB person is never stretched to repair body proportions.
- Face identity correction remains head-angle aware and also reports InsightFace identity similarity when embeddings are available.
- Open **Pipeline debug** to inspect garment extraction, masks, original/retargeted skeletons, original/retargeted DensePose controls and intermediate results.

The **Anatomy retarget strength** slider defaults to `0.65`; set it to `0` to use the original target DensePose without anatomy retargeting.

Pose transfer uses Leffa's heavier SDXL checkpoint, so use a GPU with enough memory for the full pipeline.

Built with Leffa + Detectron2 DensePose + OpenPose + InsightFace.
