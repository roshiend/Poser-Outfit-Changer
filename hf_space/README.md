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

## Fidelity v3 behavior

- **Auto detect (recommended)** parses clothed-person references into upper, lower, or dress/full-outfit regions and routes to the matching Leffa VTON model.
- If a reliable garment cannot be isolated, generation stops. The app never silently uses the entire donor person as the garment reference.
- VTON reports garment confidence and warns when face/hair or background changed unusually strongly before pose transfer.
- **Both** genuinely runs outfit + pose. It is never silently changed to outfit-only.
- Pose modes require **real Detectron2 DensePose**. If real DensePose cannot load, pose generation stops rather than using approximate/fake IUV conditioning.
- Upper-body try-on uses **VITON-HD**; lower-body and full-outfit/dress use **DressCode**.
- The target person is uniformly body-scale aligned before pose diffusion.
- Leffa's OpenPose landmarks measure anatomy and safety-gate a conservative DensePose-control retarget.
- Bone lengths can move toward the base person's proportions while target joint directions/pose are preserved.
- If too few joints are visible, proportions differ too much, or the requested warp is excessive, anatomy retargeting is skipped automatically and the original real DensePose control is used.
- The finished RGB person is never stretched to repair body proportions.
- Face identity correction remains head-angle aware and reports InsightFace identity similarity when embeddings are available.
- Open **Pipeline debug** to inspect garment extraction, masks, original/retargeted skeletons, original/retargeted DensePose controls and intermediate results.

The **Anatomy retarget strength** slider defaults to `0.65`; set it to `0` to use the original target DensePose without anatomy retargeting.

For **flat garment photos**, choose upper/lower/full explicitly; parser-based Auto detection is only for clothed-person references.

Pose transfer uses Leffa's heavier SDXL checkpoint, so use a GPU with enough memory for the full pipeline.

Built with Leffa + Detectron2 DensePose + SCHP/OpenPose + InsightFace.
