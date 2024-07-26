# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/pipelines/pipeline_animation.py
import inspect
import math
import numpy as np
from tqdm import tqdm
from einops import rearrange
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch
from transformers import CLIPImageProcessor

from ..diffusers import DiffusionPipeline
from ..diffusers.image_processor import VaeImageProcessor
from ..diffusers.utils import BaseOutput, is_accelerate_available
from ..diffusers.utils.torch_utils import randn_tensor

from ..models.utils.pipeline_context import get_context_scheduler
from ..models.utils.pipeline_utils import get_tensor_interpolation_method, set_tensor_interpolation_method

from comfy.utils import ProgressBar

class VideoPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        referencenet,
        unet,
        image_proj_model=None,
        tokenizer=None,
        text_encoder=None
    ):
        super().__init__()

        self.referencenet=referencenet
        self.unet=unet
        self.image_proj_model=image_proj_model
        self.tokenizer=tokenizer
        self.text_encoder=text_encoder

        self.vae_scale_factor = 8
        self.vae_scaling_factor = 0.18215
        self.clip_image_processor = CLIPImageProcessor()
        self.cond_image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device 

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def interpolate_latents(
        self, latents: torch.Tensor, interpolation_factor: int, device
    ):
        if interpolation_factor < 2:
            return latents

        new_latents = torch.zeros(
            (
                latents.shape[0],
                latents.shape[1],
                ((latents.shape[2] - 1) * interpolation_factor) + 1,
                latents.shape[3],
                latents.shape[4],
            ),
            device=latents.device,
            dtype=latents.dtype,
        )

        org_video_length = latents.shape[2]
        rate = [i / interpolation_factor for i in range(interpolation_factor)][1:]

        new_index = 0

        v0 = None
        v1 = None

        for i0, i1 in zip(range(org_video_length), range(org_video_length)[1:]):
            v0 = latents[:, :, i0, :, :]
            v1 = latents[:, :, i1, :, :]

            new_latents[:, :, new_index, :, :] = v0
            new_index += 1

            for f in rate:
                v = get_tensor_interpolation_method()(
                    v0.to(device=device), v1.to(device=device), f
                )
                new_latents[:, :, new_index, :, :] = v.to(latents.device)
                new_index += 1

        new_latents[:, :, new_index, :, :] = v1
        new_index += 1

        return new_latents

    @torch.no_grad()
    def __call__(
        self,
        ref_image_latents,
        landmark_features,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        scheduler,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        cond_images=None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        context_schedule="uniform",
        context_frames=24,
        context_stride=1,
        context_overlap=4,
        context_batch_size=1,
        interpolation_factor=1,
        ref_down_block_multiplier=1.0,
        ref_mid_block_multiplier=1.0,
        ref_up_block_multiplier=1.0,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler = scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1
        latent_timesteps = timesteps[0].repeat(batch_size)

        encoder_hidden_states = cond_images
        uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

        if do_classifier_free_guidance:
            encoder_hidden_states = torch.cat(
                [uncond_encoder_hidden_states, encoder_hidden_states], dim=0
            )

        num_channels_latents = self.unet.config.in_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            encoder_hidden_states.dtype,
            device,
            generator)
        
        set_tensor_interpolation_method(is_slerp=True)

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents

        ref_image_latents = ref_image_latents * self.vae_scaling_factor  # (b, 4, h, w)

        lmk_fea = landmark_features.to(self.unet.dtype)

        # context_schedule = uniform
        context_scheduler = get_context_scheduler(context_schedule)

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        comfy_pbar = ProgressBar(num_inference_steps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # do_classifier_free_guidance = True
                noise_pred = torch.zeros(
                    (
                        latents.shape[0] * (2 if do_classifier_free_guidance else 1),
                        *latents.shape[1:],
                    ),
                    device=latents.device,
                    dtype=latents.dtype,
                )

                counter = torch.zeros(
                    (1, 1, latents.shape[2], 1, 1),
                    device=latents.device,
                    dtype=latents.dtype,
                )

                # 1. Forward reference image
                if i == 0:
                    reference_down_block_res_samples, reference_mid_block_res_sample, reference_up_block_res_samples = \
                        self.referencenet(ref_image_latents.repeat((2 if do_classifier_free_guidance else 1), 1, 1, 1),
                                          torch.zeros_like(t),
                                          encoder_hidden_states=encoder_hidden_states,
                                          return_dict=False)
                     #adjust ref unet 
                    reference_down_block_res_samples = tuple(sample * ref_down_block_multiplier for sample in reference_down_block_res_samples)
                    reference_mid_block_res_sample = reference_mid_block_res_sample * ref_mid_block_multiplier
                    reference_up_block_res_samples = tuple(sample * ref_up_block_multiplier for sample in reference_up_block_res_samples)

                context_queue = list(
                    context_scheduler(
                        0,
                        num_inference_steps,
                        latents.shape[2],
                        context_frames,
                        context_stride,
                        context_overlap,
                        0,
                    )
                )

                num_context_batches = math.ceil(len(context_queue) / context_batch_size)

                global_context = []
                for i in range(num_context_batches):
                    global_context.append(
                        context_queue[
                            i * context_batch_size : (i + 1) * context_batch_size
                        ]
                    )

                for context in global_context:
                    print("latent_shape: ", latents.shape)
                    # 3.1 expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents[:, :, c] for c in context])
                        .to(device)
                        .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    )

                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    b, c, f, h, w = latent_model_input.shape

                    latent_lmk_input = torch.cat(
                        [lmk_fea[:, :, c] for c in context]
                    ).repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)

                    self.unet.set_do_classifier_free_guidance(do_classifier_free_guidance=do_classifier_free_guidance)
                    pred = self.unet(latent_model_input,
                                      t,
                                      lmk_cond_fea=latent_lmk_input,
                                      encoder_hidden_states=encoder_hidden_states[:b],
                                      reference_down_block_res_samples=reference_down_block_res_samples,
                                      reference_mid_block_res_sample=reference_mid_block_res_sample,
                                      reference_up_block_res_samples=reference_up_block_res_samples
                                      ).sample

                    for j, c in enumerate(context):
                        noise_pred[:, :, c] = noise_pred[:, :, c] + pred
                        counter[:, :, c] = counter[:, :, c] + 1

                # do_classifier_free_guidance = True
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)

                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample


                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    comfy_pbar.update(1)
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if interpolation_factor > 0:
            latents = self.interpolate_latents(latents, interpolation_factor, device)

        return latents

    def _gaussian_weights(self, t_tile_length, t_batch_size):
        from numpy import pi, exp, sqrt

        var = 0.01
        midpoint = (t_tile_length - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        t_probs = [exp(-(t-midpoint)*(t-midpoint)/(t_tile_length*t_tile_length)/(2*var)) / sqrt(2*pi*var) for t in range(t_tile_length)]
        weights = torch.tensor(t_probs)
        weights = weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, t_batch_size, 1, 1)
        return weights

    @torch.no_grad()
    def forward_long(
        self,
        ref_image_latents,
        landmark_features,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        scheduler,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        cond_images=None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        context_schedule="uniform",
        context_frames=24,
        context_stride=1,
        context_overlap=4,
        context_batch_size=1,
        interpolation_factor=1,
        t_tile_length=16,
        t_tile_overlap=4,
        pose_multiplier=1.0,
        ref_down_block_multiplier=1.0,
        ref_mid_block_multiplier=1.0,
        ref_up_block_multiplier=1.0,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler = scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1
        latent_timesteps = timesteps[0].repeat(batch_size)

        encoder_hidden_states = cond_images
        uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

        if do_classifier_free_guidance:
            encoder_hidden_states = torch.cat(
                [uncond_encoder_hidden_states, encoder_hidden_states], dim=0
            )

        num_channels_latents = self.unet.config.in_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            encoder_hidden_states.dtype,
            device,
            generator)

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        ref_image_latents = ref_image_latents * self.vae_scaling_factor  # (b, 4, h, w)
        lmk_fea = landmark_features.to(self.unet.dtype)

        # ---------------------------------------------

        t_tile_weights = self._gaussian_weights(t_tile_length=t_tile_length, t_batch_size=1).to(device=latents.device)
        t_tile_weights = t_tile_weights.to(dtype=lmk_fea.dtype)
        comfy_pbar = ProgressBar(num_inference_steps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # =====================================================
                grid_ts = 0
                cur_t = 0
                while cur_t < latents.shape[2]:
                    cur_t = max(grid_ts * t_tile_length - t_tile_overlap * grid_ts, 0) + t_tile_length
                    grid_ts += 1

                all_t = latents.shape[2]
                latents_all_list = []
                # =====================================================

                for t_i in range(grid_ts):
                    if t_i < grid_ts - 1:
                        ofs_t = max(t_i * t_tile_length - t_tile_overlap * t_i, 0)
                    if t_i == grid_ts - 1:
                        ofs_t = all_t - t_tile_length

                    input_start_t = ofs_t
                    input_end_t = ofs_t + t_tile_length

                    torch.cuda.empty_cache()

                    if i == 0:
                        reference_down_block_res_samples, reference_mid_block_res_sample, reference_up_block_res_samples = \
                            self.referencenet(
                                ref_image_latents.repeat((2 if do_classifier_free_guidance else 1), 1, 1, 1),
                                torch.zeros_like(t),
                                encoder_hidden_states=encoder_hidden_states,
                                return_dict=False)
                        
                        #adjust ref unet 
                        reference_down_block_res_samples = tuple(sample * ref_down_block_multiplier for sample in reference_down_block_res_samples)
                        reference_mid_block_res_sample = reference_mid_block_res_sample * ref_mid_block_multiplier
                        reference_up_block_res_samples = tuple(sample * ref_up_block_multiplier for sample in reference_up_block_res_samples)
                        
                    latents_tile = latents[:, :, input_start_t:input_end_t, :, :]
                    latent_model_input_tile = torch.cat([latents_tile] * 2) if do_classifier_free_guidance else latents_tile
                    # model_input_tile.shape = torch.Size([2, 4, 16, 32, 32])

                    latent_model_input_tile = self.scheduler.scale_model_input(latent_model_input_tile, t)
                    b, c, _, h, w = latent_model_input_tile.shape

                    lmk_fea_tile = lmk_fea[:, :, input_start_t:input_end_t, :, :]
                    latent_lmk_input_tile = torch.cat([lmk_fea_tile]).repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)

                    t_input = t[None].to(self.device)
                    t_input = t_input.expand(latent_model_input_tile.shape[0])

                    self.unet.set_do_classifier_free_guidance(do_classifier_free_guidance=do_classifier_free_guidance)
                    noises_pred = self.unet(latent_model_input_tile,
                                            t_input,
                                            lmk_cond_fea=latent_lmk_input_tile,
                                            encoder_hidden_states=encoder_hidden_states[:b],
                                            reference_down_block_res_samples=reference_down_block_res_samples,
                                            reference_mid_block_res_sample=reference_mid_block_res_sample,
                                            reference_up_block_res_samples=reference_up_block_res_samples
                                            ).sample

                    # perform guidance
                    # do_classifier_free_guidance = True/True
                    if do_classifier_free_guidance:
                        noises_pred_neg, noises_pred_pos = noises_pred.chunk(2)

                        noise_pred = noises_pred_neg + guidance_scale * (noises_pred_pos - noises_pred_neg)

                    latents_tile = self.scheduler.step(noise_pred, t, latents_tile, **extra_step_kwargs).prev_sample
                    latents_all_list.append(latents_tile)

                # ==========================================
                latents_all = torch.zeros(latents.shape, device=latents.device, dtype=latents.dtype)
                contributors = torch.zeros(latents.shape, device=latents.device, dtype=latents.dtype)
                # Add each tile contribution to overall latents
                for t_i in range(grid_ts):
                    if t_i < grid_ts - 1:
                        ofs_t = max(t_i * t_tile_length - t_tile_overlap * t_i, 0)
                    if t_i == grid_ts - 1:
                        ofs_t = all_t - t_tile_length

                    input_start_t = ofs_t
                    input_end_t = ofs_t + t_tile_length

                    latents_all[:, :, input_start_t:input_end_t, :, :] += latents_all_list[t_i] * t_tile_weights
                    contributors[:, :, input_start_t:input_end_t, :, :] += t_tile_weights

                latents_all /= contributors
                # latents_all /= torch.sqrt(contributors)
                latents = latents_all
                # ==========================================

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()
                    comfy_pbar.update(1)

        # ---------------------------------------------
        return latents
