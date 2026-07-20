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
short_description: Pose & cloth swap with face/body identity lock (Leffa)
suggested_hardware: a10g-small
tags:
  - virtual-try-on
  - pose-transfer
  - leffa
  - fashion
---

# Poser Outfit Changer

Upload a **base person** (identity to keep) and a **clothed person** (outfit / pose source).

On free / shared GPU hardware this Space defaults to **Outfit only** (clothes on your body/face). Full pose transfer uses an SDXL model and needs a larger GPU.

Built with [Leffa](https://github.com/franciszzj/Leffa) + InsightFace identity lock.

## Notes

- First run downloads multi-GB weights from `franciszzj/Leffa`.
- Prefer clear, full-body photos.
- For commercial use, check upstream model licenses.
