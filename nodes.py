import os
import torch
import torchvision.transforms as T
import folder_paths
import comfy.model_management as mm
import comfy.utils
from comfy.clip_vision import clip_preprocess
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

from .models.guider import Guider
from .models.referencenet import ReferenceNet2DConditionModel
from .models.unet import UNet3DConditionModel
from .models.video_pipeline import VideoPipeline

from .media_pipe.mp_utils  import LMKExtractor
from .media_pipe.draw_util import FaceMeshVisualizer
from .media_pipe import FaceMeshAlign
import cv2

from transformers import CLIPVisionModelWithProjection
from .diffusers import DDIMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler, DEISMultistepScheduler, DDPMScheduler
  
from contextlib import nullcontext
try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except:
    pass

script_directory = os.path.dirname(os.path.abspath(__file__))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def load_model_state_dict(model, model_ckpt_path, name):
    ckpt = torch.load(model_ckpt_path, map_location="cpu")
    model_state_dict = model.state_dict()
    model_new_sd = {}
    count = 0
    for k, v in ckpt.items():
        if k in model_state_dict:
            count += 1
            model_new_sd[k] = v
    miss, _ = model.load_state_dict(model_new_sd, strict=False)
    print(f'load {name} from {model_ckpt_path}\n - load params: {count}\n - miss params: {miss}')

class DownloadAndLoadFYEModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                
            },
            "optional": {
                "precision": (
                    [
                        "fp16",
                        "fp32",
                        "bf16",
                    ],
                    {"default": "fp16"},
                ),
            },
        }

    RETURN_TYPES = ("FYEPIPE",)
    RETURN_NAMES = ("fye_pipe",)
    FUNCTION = "loadmodel"
    CATEGORY = "FollowYourEmojiWrapper"

    def loadmodel(self, precision):
        device = mm.get_torch_device()
        mm.soft_empty_cache()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        pbar = comfy.utils.ProgressBar(3)

        model_path = os.path.join(folder_paths.models_dir, "diffusers", "sd-image-variations-diffusers")

        if not os.path.exists(model_path):
            log.info(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id="lambdalabs/sd-image-variations-diffusers",
                allow_patterns=["*image_encoder*"],
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )

        ref_unet_config = OmegaConf.load(os.path.join(script_directory, "configs", "unet_config.json"))
        ad_unet_config = OmegaConf.load(os.path.join(script_directory, "configs", "3d_unet_config.json"))
        #image_encoder_config = OmegaConf.load(os.path.join(script_directory, "configs", "image_encoder_config.json"))

        fye_base_path = os.path.join(folder_paths.models_dir, "FYE")
        referencenet_path = os.path.join(fye_base_path, "FYE_referencenet-fp16.safetensors")
        unet_path = os.path.join(fye_base_path, "FYE_unet-fp16.safetensors")
        lmk_guider_path = os.path.join(fye_base_path, "FYE_lmk_guider.safetensors")
        #image_encoder_path = os.path.join(fye_base_path, "sd-image-variations-encoder-fp16.safetensors")

        pbar.update(1)

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_path, subfolder="image_encoder").to(dtype).to(device)

        pbar.update(1)

        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            referencenet = ReferenceNet2DConditionModel(**ref_unet_config)
            ad_unet = UNet3DConditionModel(**ad_unet_config)
           
        
        pbar.update(1)
       
       
        if not os.path.exists(fye_base_path):
            log.info(f"Downloading model to: {fye_base_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id="Kijai/FollowYourEmoji-safetensors",
                local_dir=fye_base_path,
                local_dir_use_symlinks=False,
            )
        #reference unet
        sd = comfy.utils.load_torch_file(referencenet_path)
        if is_accelerate_available:
            for key in sd:
                set_module_tensor_to_device(referencenet, key, device=device, dtype=dtype, value=sd[key])
        else:
            referencenet.load_state_dict(sd)
            referencenet.to(dtype).to(device)

        pbar.update(1)
        #3d unet
        sd = comfy.utils.load_torch_file(unet_path)
        if is_accelerate_available:
            for key in sd:
                set_module_tensor_to_device(ad_unet, key, device=device, dtype=dtype, value=sd[key])
        else:
            ad_unet.load_state_dict(sd, strict=False)
            ad_unet.to(dtype).to(device)

        pbar.update(1)
        # #image encoder
        # sd = comfy.utils.load_torch_file(image_encoder_path)
        # if is_accelerate_available:
        #     for key in sd:
        #         set_module_tensor_to_device(image_encoder, key, device=device, dtype=dtype, value=sd[key])
        # else:
        #     image_encoder.load_state_dict(sd, strict=False)
        #     image_encoder.to(dtype).to(device)

        #guider
        sd = comfy.utils.load_torch_file(lmk_guider_path)
        lmk_guider = Guider(conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)).to(dtype).to(device)
        lmk_guider.load_state_dict(sd)

        pbar.update(1)

        scheduler_config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "steps_offset": 1,
            "clip_sample": False,
            "rescale_betas_zero_snr":True,
            "prediction_type": "v_prediction",
            "timestep_spacing": "trailing",
        }

        noise_scheduler = DDIMScheduler(**scheduler_config)

        pipeline = VideoPipeline(
                             vae=None,
                             image_encoder=image_encoder,
                             referencenet=referencenet,
                             unet=ad_unet,
                             lmk_guider=lmk_guider)

        return (pipeline,)


class FYESampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipeline": ("FYEPIPE",),
            "ref_latent": ("LATENT", ),
            "clip_image": ("IMAGE",),
            "motions": ("IMAGE",),
            "steps": ("INT", {"default": 25, "min": 1}),
            "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 30.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "context_frames": ("INT", {"default": 24, "min": 8, "max": 48}),
            "context_overlap": ("INT", {"default": 4, "min": 1, "max": 24}),
            "context_stride": ("INT", {"default": 1, "min": 1, "max": 8}),
            "latent_interpolation_factor": ("INT", {"default": 1, "min": 1, "max": 10}),
            "pose_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "scheduler": (
                [
                    'DDIMScheduler',
                    'DDPMScheduler',
                    'DEISMultistepScheduler',
                    'DPMSolverMultistepScheduler',
                    'UniPCMultistepScheduler',
                ], {
                    "default": 'DDIMScheduler'
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "FollowYourEmojiWrapper"

    def process(self, pipeline, ref_latent, motions, steps, seed, clip_image, cfg, context_frames, context_overlap, context_stride, latent_interpolation_factor, pose_multiplier, scheduler):

        pipeline, ref_sample, lmk_images, H, W, steps, generator, clip_image, noise_scheduler = common_process(pipeline, ref_latent, motions, steps, seed, clip_image, scheduler)

        latents = pipeline(
                        ref_image=None,
                        ref_image_latents=ref_sample,
                        lmk_images=lmk_images,
                        width=W,
                        height=H,
                        video_length=len(motions),
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        generator=generator,
                        clip_image=clip_image,
                        context_frames=context_frames,
                        context_overlap=context_overlap,
                        context_stride=context_stride,
                        interpolation_factor=latent_interpolation_factor,
                        pose_multiplier=pose_multiplier,
                        scheduler=noise_scheduler
                        )

        latents = latents.squeeze(0).permute(1,0,2,3) / 0.18215
       
        return({"samples":latents},)
    
def common_process(pipeline, ref_latent, motions, steps, seed, clip_image, scheduler):

    scheduler_config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "steps_offset": 1,
            "clip_sample": False,
            "rescale_betas_zero_snr":True,
            "prediction_type": "v_prediction",
            "timestep_spacing": "trailing",
        }
     
    if scheduler == 'DDIMScheduler':
        noise_scheduler = DDIMScheduler(**scheduler_config)
    elif scheduler == 'DDPMScheduler':
        noise_scheduler = DDPMScheduler(**scheduler_config)
    elif scheduler == 'DEISMultistepScheduler':
        scheduler_config.pop("clip_sample", None)
        scheduler_config.pop("rescale_betas_zero_snr", None)
        noise_scheduler = DEISMultistepScheduler(**scheduler_config)
    elif scheduler == 'DPMSolverMultistepScheduler':
        scheduler_config.pop("clip_sample", None)
        scheduler_config.pop("rescale_betas_zero_snr", None)
        scheduler_config.update({"algorithm_type": "sde-dpmsolver++"})
        scheduler_config.update({"use_karras_sigmas": "True"})
        noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
    elif scheduler == 'UniPCMultistepScheduler':
        scheduler_config.pop("clip_sample", None)
        scheduler_config.pop("rescale_betas_zero_snr", None)
        noise_scheduler = UniPCMultistepScheduler(**scheduler_config)
        
    device = mm.get_torch_device()
    dtype = pipeline.unet.dtype

    ref_sample = ref_latent["samples"]
    ref_sample = ref_sample.to(dtype).to(device)
    H = int(ref_sample.shape[2] * 8)
    W = int(ref_sample.shape[3] * 8)

    motions = (motions * 255).cpu().numpy().astype(np.uint8)
    lmk_images = [Image.fromarray(motion) for motion in motions]

    clip_image = clip_preprocess(clip_image.clone(), 224)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return (pipeline, ref_sample, lmk_images, H, W, steps, generator, clip_image[0], noise_scheduler)

class FYESamplerLong:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipeline": ("FYEPIPE",),
            "ref_latent": ("LATENT", ),
            "clip_image": ("IMAGE",),
            "motions": ("IMAGE",),
            "steps": ("INT", {"default": 25, "min": 1}),
            "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 30.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "t_tile_length": ("INT", {"default": 16, "min": 8, "max": 256}),
            "t_tile_overlap": ("INT", {"default": 4, "min": 1, "max": 24}),
            "pose_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "scheduler": (
                [
                    'DDIMScheduler',
                    'DDPMScheduler',
                    'DEISMultistepScheduler',
                    'DPMSolverMultistepScheduler',
                    'UniPCMultistepScheduler',
                ], {
                    "default": 'DDIMScheduler'
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "FollowYourEmojiWrapper"

    def process(self, pipeline, ref_latent, motions, steps, seed, clip_image, cfg, t_tile_length, t_tile_overlap, pose_multiplier, scheduler):
      
        pipeline, ref_sample, lmk_images, H, W, steps, generator, clip_image, noise_scheduler = common_process(pipeline, ref_latent, motions, steps, seed, clip_image, scheduler)

        latents = pipeline.forward_long(
                        ref_image=None,
                        ref_image_latents=ref_sample,
                        lmk_images=lmk_images,
                        width=W,
                        height=H,
                        video_length=len(motions),
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        generator=generator,
                        clip_image=clip_image,
                        t_tile_length=t_tile_length,
                        t_tile_overlap=t_tile_overlap,
                        pose_multiplier=pose_multiplier,
                        scheduler=noise_scheduler
                        )

        latents = latents.squeeze(0).permute(1,0,2,3) / 0.18215
       
        return({"samples":latents},)
    
class FYEMediaPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {

            "images": ("IMAGE",),
            "draw_outer_lips": ("BOOLEAN", {"default": False}),
            "draw_iris_points": ("BOOLEAN", {"default": True}),
            },
           
            "optional": {
                "align_to_face_results": ("FACERESULTS", {"default": None}),

            }
        }

    RETURN_TYPES = ("IMAGE", "FACERESULTS",)
    RETURN_NAMES = ("images", "face_results",)
    FUNCTION = "process"
    CATEGORY = "FollowYourEmojiWrapper"

    def process(self, images, draw_outer_lips, draw_iris_points, align_to_face_results=None):
        device = mm.get_torch_device()

        B, H, W, C = images.shape
       
        # tensor to pil image
        #ref_frame = torch.clamp((images + 1.0) / 2.0, min=0, max=1)
        #ref_frame = ref_frame[0]
        images_np = (images * 255).cpu().numpy().astype(np.uint8)
       
        lmk_extractor = LMKExtractor()
        vis = FaceMeshVisualizer(forehead_edge=False, iris_point=draw_iris_points, draw_outer_lips=draw_outer_lips)
        aligner = FaceMeshAlign()

        to_tensor = T.ToTensor()
        
        face_results = []
        motions = []
        for frame in tqdm(images_np):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            face_result = lmk_extractor(frame_bgr)
            
            assert face_result is not None, "Can not detect a face in the reference image."
            face_result['width'] = frame_bgr.shape[1]
            face_result['height'] = frame_bgr.shape[0]
            
            face_results.append(face_result)
            lmks = face_result['lmks'].astype(np.float32)
            motion = vis.draw_landmarks((frame_bgr.shape[1], frame_bgr.shape[0]), lmks, normed=True)
            motions.append(motion)
        #print("face_results:", face_results)
        
        if align_to_face_results is not None:
            aligned_face_results = aligner(align_to_face_results[0], face_results)
            motions = [to_tensor(motion) for motion in aligned_face_results]
            motion_tensors_out = torch.stack(motions, dim=0).permute((0, 2, 3, 1))
            print(motion_tensors_out.shape)
        else:
            motion_tensors_out = (
                torch.stack([torch.from_numpy(np_array) for np_array in motions])
                / 255
            )


        return(motion_tensors_out, face_results)
    


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadFYEModel": DownloadAndLoadFYEModel,
    "FYESampler": FYESampler,
    "FYEMediaPipe": FYEMediaPipe,
    "FYESamplerLong": FYESamplerLong
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadFYEModel": "(Down)LoadFYE Model",
    "FYESampler": "FYESampler",
    "FYEMediaPipe": "MediaPipe",
    "FYESamplerLong": "FYESamplerLong"
    }