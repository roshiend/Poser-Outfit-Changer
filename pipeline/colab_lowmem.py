"""Staged Leffa inference for memory-constrained Colab GPUs.

Upstream Leffa's ``LeffaInference`` moves the VAE, reference UNet and generative
UNet to CUDA at the same time.  That is convenient on larger GPUs but is a poor
fit for a free/standard Colab runtime, especially for the SDXL pose checkpoint.

This module keeps the model in CPU RAM and moves only the component needed for
the current stage to CUDA:

    VAE encode -> reference UNet -> generative UNet -> VAE decode

For SDXL pose transfer on sub-20-GB GPUs it also disables classifier-free
guidance, avoiding the normal batch-size doubling.  Real DensePose conditioning
is unchanged; this is a memory policy, not a lower-fidelity pose fallback.

The denoising flow is adapted from Leffa's Apache-2.0 ``leffa/pipeline.py``.
"""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ColabMemoryPolicy:
    total_vram_gb: float
    control_type: str
    use_cfg: bool
    transform_width: int
    transform_height: int
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def choose_colab_policy(
    total_vram_gb: float,
    control_type: str,
    requested_cfg: bool = True,
    resolution_mode: str = "auto",
) -> ColabMemoryPolicy:
    """Choose a conservative inference policy from measured GPU VRAM.

    The thresholds intentionally use measured runtime VRAM instead of assuming
    a specific Colab GPU model because Colab hardware availability changes.
    """
    total = max(0.0, float(total_vram_gb))
    if control_type not in {"virtual_tryon", "pose_transfer"}:
        raise ValueError(f"Unknown control type: {control_type}")
    if resolution_mode not in {"auto", "full", "balanced", "safe"}:
        raise ValueError(f"Unknown Colab resolution mode: {resolution_mode}")

    if resolution_mode == "full":
        width, height = 768, 1024
    elif resolution_mode == "balanced":
        width, height = 672, 896
    elif resolution_mode == "safe":
        width, height = 576, 768
    elif total >= 14.0:
        width, height = 768, 1024
    elif total >= 11.0:
        width, height = 672, 896
    else:
        width, height = 576, 768

    # SDXL pose is the difficult case.  On sub-20-GB GPUs, avoiding CFG halves
    # the denoiser/reference batch and materially reduces activation memory.
    if control_type == "pose_transfer" and total < 20.0:
        use_cfg = False
        reason = "SDXL pose on <20GB VRAM: staged modules + single-batch denoising"
    elif control_type == "virtual_tryon" and total < 12.0:
        use_cfg = False
        reason = "low VRAM VTON: staged modules + single-batch denoising"
    else:
        use_cfg = bool(requested_cfg)
        reason = "staged modules; CFG retained"

    return ColabMemoryPolicy(
        total_vram_gb=total,
        control_type=control_type,
        use_cfg=use_cfg,
        transform_width=width,
        transform_height=height,
        reason=reason,
    )


def measured_cuda_vram_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.get_device_properties(0).total_memory) / 1024**3
    except Exception:
        pass
    return 0.0


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


class StagedLeffaInference:
    """Leffa-compatible inference object that never keeps all modules on CUDA."""

    def __init__(
        self,
        model,
        control_type: str,
        resolution_mode: str = "auto",
    ) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Staged Colab inference requires a CUDA GPU.")
        self.model = model.to("cpu")
        self.model.eval()
        self.device = "cuda"
        self.control_type = control_type
        self.resolution_mode = resolution_mode
        self.last_run_info: dict = {}

        # VAE tiling keeps encode/decode peaks predictable at 1024x768.  It does
        # not change the diffusion latent resolution or the DensePose control.
        try:
            self.model.vae.enable_tiling()
        except Exception:
            pass
        try:
            self.model.vae.enable_slicing()
        except Exception:
            pass
        _release_cuda()

    @staticmethod
    def _to_cpu(module) -> None:
        try:
            module.to("cpu")
        finally:
            _release_cuda()

    @staticmethod
    def _to_cuda(module) -> None:
        module.to("cuda")
        _release_cuda()

    def __call__(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        import numpy as np
        import torch
        import torch.nn.functional as F
        import tqdm
        from leffa.pipeline import do_repaint, numpy_to_pil, rescale_noise_cfg

        total_vram = measured_cuda_vram_gb()
        requested_cfg = bool(kwargs.get("do_classifier_free_guidance", True))
        policy = choose_colab_policy(
            total_vram,
            self.control_type,
            requested_cfg=requested_cfg,
            resolution_mode=self.resolution_mode,
        )
        use_cfg = policy.use_cfg
        ref_acceleration = bool(kwargs.get("ref_acceleration", True))
        num_inference_steps = int(kwargs.get("num_inference_steps", 30))
        guidance_scale = float(kwargs.get("guidance_scale", 2.5))
        seed = int(kwargs.get("seed", 42))
        repaint = bool(kwargs.get("repaint", False))

        self.last_run_info = {
            **policy.to_dict(),
            "staged_modules": True,
            "ref_acceleration": ref_acceleration,
            "requested_guidance_scale": guidance_scale,
            "effective_cfg": use_cfg,
        }
        print(
            "[colab-lowmem] "
            f"{policy.reason}; resolution={policy.transform_width}x{policy.transform_height}; "
            f"CFG={'on' if use_cfg else 'off'}"
        )

        src_cpu = data["src_image"].detach().cpu()
        ref_cpu = data["ref_image"].detach().cpu()
        mask_cpu = data["mask"].detach().cpu()
        densepose_cpu = data["densepose"].detach().cpu()

        vae = self.model.vae
        ref_unet = self.model.unet_encoder
        gen_unet = self.model.unet
        scheduler = self.model.noise_scheduler

        reference_features = None
        latent = None
        try:
            # Stage 1: VAE encode only.
            self._to_cuda(vae)
            dtype = vae.dtype
            src = src_cpu.to(self.device, dtype=dtype)
            ref = ref_cpu.to(self.device, dtype=dtype)
            mask = mask_cpu.to(self.device, dtype=dtype)
            densepose = densepose_cpu.to(self.device, dtype=dtype)
            masked = src * (mask < 0.5)
            with torch.inference_mode():
                masked_latent = vae.encode(masked).latent_dist.sample()
                ref_latent = vae.encode(ref).latent_dist.sample()
            masked_latent = masked_latent * vae.config.scaling_factor
            ref_latent = ref_latent * vae.config.scaling_factor
            mask_latent = F.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")
            densepose_latent = F.interpolate(densepose, size=masked_latent.shape[-2:], mode="nearest")
            del src, ref, mask, densepose, masked
            self._to_cpu(vae)

            noise = torch.randn(
                masked_latent.shape,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                device=self.device,
                dtype=masked_latent.dtype,
            )
            scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = scheduler.timesteps
            latent = noise * scheduler.init_noise_sigma
            del noise

            if use_cfg:
                masked_latent = torch.cat([masked_latent] * 2)
                ref_latent = torch.cat([torch.zeros_like(ref_latent), ref_latent])
                mask_latent = torch.cat([mask_latent] * 2)
                densepose_latent = torch.cat([densepose_latent] * 2)

            # Stage 2: reference UNet.  Ref acceleration is mandatory in the
            # staged path because recomputing the reference UNet every step would
            # require repeated CPU<->GPU module swaps and defeat the memory policy.
            self._to_cuda(ref_unet)
            mid_t = timesteps[max(0, num_inference_steps // 2)]
            with torch.inference_mode():
                _, reference_features = ref_unet(
                    ref_latent,
                    mid_t,
                    encoder_hidden_states=None,
                    return_dict=False,
                )
            reference_features = list(reference_features)
            del ref_latent
            self._to_cpu(ref_unet)

            # Stage 3: generative UNet.  Only this UNet plus the already computed
            # reference features and latent/control tensors occupy CUDA now.
            self._to_cuda(gen_unet)
            extra_step_kwargs = {}
            accepts_generator = "generator" in scheduler.step.__code__.co_varnames if hasattr(scheduler.step, "__code__") else False
            if accepts_generator:
                extra_step_kwargs["generator"] = torch.Generator(device=self.device).manual_seed(seed)

            with tqdm.tqdm(total=num_inference_steps) as progress_bar:
                for t in timesteps:
                    latent_in = torch.cat([latent] * 2) if use_cfg else latent
                    latent_in = scheduler.scale_model_input(latent_in, t)
                    model_input = torch.cat(
                        [latent_in, mask_latent, masked_latent, densepose_latent],
                        dim=1,
                    )
                    with torch.inference_mode():
                        noise_pred = gen_unet(
                            model_input,
                            t,
                            encoder_hidden_states=None,
                            cross_attention_kwargs=None,
                            added_cond_kwargs=None,
                            reference_features=reference_features,
                            return_dict=False,
                        )[0]
                    del model_input, latent_in

                    if use_cfg:
                        noise_uncond, noise_cond = noise_pred.chunk(2)
                        guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                        if guidance_scale > 0.0:
                            guided = rescale_noise_cfg(
                                guided,
                                noise_cond,
                                guidance_rescale=guidance_scale,
                            )
                        noise_pred = guided
                    latent = scheduler.step(
                        noise_pred,
                        t,
                        latent,
                        **extra_step_kwargs,
                        return_dict=False,
                    )[0]
                    del noise_pred
                    progress_bar.update()

            self._to_cpu(gen_unet)
            del reference_features, masked_latent, mask_latent, densepose_latent
            reference_features = None
            _release_cuda()

            # Stage 4: VAE decode only.
            self._to_cuda(vae)
            with torch.inference_mode():
                decoded = vae.decode(latent / vae.config.scaling_factor).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            decoded_np = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
            generated = numpy_to_pil(decoded_np)
            del decoded, decoded_np, latent
            latent = None
            self._to_cpu(vae)

            if repaint:
                src_np = (src_cpu / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().numpy()
                mask_np = mask_cpu.permute(0, 2, 3, 1).float().numpy()
                src_images = numpy_to_pil(src_np)
                masks = [image.convert("RGB") for image in numpy_to_pil(mask_np)]
                generated = [
                    do_repaint(source, this_mask, result)
                    for source, this_mask, result in zip(src_images, masks, generated)
                ]

            return {
                "src_image": (src_cpu + 1.0) / 2.0,
                "ref_image": (ref_cpu + 1.0) / 2.0,
                "generated_image": generated,
            }
        except torch.cuda.OutOfMemoryError as exc:
            self.last_run_info["oom"] = True
            raise RuntimeError(
                "CUDA ran out of memory even in staged Colab mode. Restart the runtime to clear fragmented VRAM, "
                "then use LEFFA_COLAB_RESOLUTION=safe (576x768)."
            ) from exc
        finally:
            # Always return heavyweight modules to CPU so an exception does not
            # leave the Colab runtime permanently full.
            for module in (vae, ref_unet, gen_unet):
                try:
                    module.to("cpu")
                except Exception:
                    pass
            reference_features = None
            latent = None
            _release_cuda()
