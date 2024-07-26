import os
import torch
import torch.nn as nn
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

from .diffusers import DDIMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler, DEISMultistepScheduler, DDPMScheduler
from .diffusers.image_processor import VaeImageProcessor
  
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

    RETURN_TYPES = ("FYEPIPE", "CLIP_VISION",)
    RETURN_NAMES = ("fye_pipe", "clip_vision",)
    FUNCTION = "loadmodel"
    CATEGORY = "FollowYourEmojiWrapper"

    def loadmodel(self, precision):
        device = mm.get_torch_device()
        mm.soft_empty_cache()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        pbar = comfy.utils.ProgressBar(3)

        ref_unet_config = OmegaConf.load(os.path.join(script_directory, "configs", "unet_config.json"))
        ad_unet_config = OmegaConf.load(os.path.join(script_directory, "configs", "3d_unet_config.json"))        

        fye_base_path = os.path.join(folder_paths.models_dir, "FYE")
        referencenet_path = os.path.join(fye_base_path, "FYE_referencenet-fp16.safetensors")
        unet_path = os.path.join(fye_base_path, "FYE_unet-fp16.safetensors")
        #lmk_guider_path = os.path.join(fye_base_path, "FYE_lmk_guider.safetensors")

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
                ignore_patterns=["*sd-image-variations-encoder-fp16.safetensors", "fye_motion_module-fp16.safetensors"],
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

        clip_vision_model_path = os.path.join(folder_paths.models_dir, "clip_vision", "sd-image-variations-encoder-fp16.safetensors")
        if not os.path.exists(clip_vision_model_path):
            print(f"Downloading model to: {clip_vision_model_path}")
            from huggingface_hub import hf_hub_download             
            hf_hub_download(repo_id="Kijai/FollowYourEmoji-safetensors", 
                                filename = "sd-image-variations-encoder-fp16.safetensors",
                                local_dir = os.path.join(folder_paths.models_dir, "clip_vision"), 
                                local_dir_use_symlinks=False)

        clip_vision = comfy.clip_vision.load(clip_vision_model_path)

        print(f"Loading model from: {clip_vision_model_path}")

        pipeline = VideoPipeline(
                             referencenet=referencenet,
                             unet=ad_unet)

        return (pipeline, clip_vision,)

class FYECLIPEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip_vision": ("CLIP_VISION", ),
            "clip_image": ("IMAGE",),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FYECLIPEMBED",)
    RETURN_NAMES = ("clip_embeds",)
    FUNCTION = "encode"
    CATEGORY = "FollowYourEmojiWrapper"

    def encode(self, clip_vision, clip_image, strength):
        dtype=clip_vision.dtype
        device=mm.get_torch_device()
        clip_image = clip_preprocess(clip_image.clone(), 224)
        clip_embeds = clip_vision.encode_image(clip_image.permute(0, 2, 3, 1))["last_hidden_state"].to(dtype).to(device)
        clip_embeds = clip_embeds * strength
        return(clip_embeds,)
    
class FYEClipEmbedToComfy:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "clip_embeds": ("FYECLIPEMBED",),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                    }
                }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning", )
    FUNCTION = "embed"
    CATEGORY = "FollowYourEmojiWrapper"

    def embed(self, clip_embeds, strength):
        clip_fc_path = os.path.join(script_directory, "models","FYE_clip_fc.safetensors")
        sd = comfy.utils.load_torch_file(clip_fc_path)
        self.clip_fc = nn.Linear(1024, 768, bias=True).to(clip_embeds.dtype).to(clip_embeds.device)
        self.clip_fc.load_state_dict(sd)

        clip_in = clip_embeds
        clip_out = self.clip_fc(clip_in) * strength
        clip_out = clip_out.to('cpu')
           
        return ([[clip_out, {"pooled_output": clip_out}]], )

class FYELandmarkEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "motions": ("IMAGE",),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LMKFEAT",)
    RETURN_NAMES = ("landmark_features",)
    FUNCTION = "encode"
    CATEGORY = "FollowYourEmojiWrapper"

    def encode(self, motions, strength):
        B, H, W, C = motions.shape
        device=mm.get_torch_device()

        self.cond_image_processor = VaeImageProcessor(vae_scale_factor = 8, do_convert_rgb=True, do_normalize=False)

        lmk_guider_path = os.path.join(script_directory, "models","FYE_lmk_guider.safetensors")
        sd = comfy.utils.load_torch_file(lmk_guider_path)
        self.lmk_guider = Guider(conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)).to(device)
        self.lmk_guider.load_state_dict(sd)

        motions = (motions * 255).cpu().numpy().astype(np.uint8)
        lmk_images = [Image.fromarray(motion) for motion in motions]

        lmk_cond_tensor_list = []
        for lmk_image in lmk_images:
            lmk_cond_tensor = self.cond_image_processor.preprocess(
                lmk_image, height=H, width=W
            )

            lmk_cond_tensor = lmk_cond_tensor.unsqueeze(2)  # (bs, c, 1, h, w)
            lmk_cond_tensor_list.append(lmk_cond_tensor)
        lmk_cond_tensor = torch.cat(lmk_cond_tensor_list, dim=2)  # (bs, c, t, h, w)
        lmk_cond_tensor = lmk_cond_tensor.to(device=device, dtype=self.lmk_guider.dtype)
        print("lmk_cond_tensor.shape", lmk_cond_tensor.shape)

        lmk_fea = self.lmk_guider(lmk_cond_tensor)
        print("lmk_fea.shape", lmk_fea.shape)
        lmk_fea = lmk_fea * strength
        return(lmk_fea,)
       
class FYELandmarkToComfy:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "landmark_features": ("LMKFEAT",),
                    }
                }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "patch"
    CATEGORY = "FollowYourEmojiWrapper"

    def patch(self, model, landmark_features):        
        def input_block_patch(h, transformer_options):
            if transformer_options['block'][1] == 0:
                lmk_fea = landmark_features.squeeze(0).permute(1, 0, 2, 3).to(h.dtype).to(h.device)
                if "ad_params" in transformer_options and transformer_options["ad_params"]['sub_idxs'] is not None:
                    sub_idxs = transformer_options['ad_params']['sub_idxs']
                    lmk_fea = lmk_fea[sub_idxs].repeat(2, 1, 1, 1).to(h.dtype).to(h.device)
                else:
                    lmk_fea = lmk_fea.repeat(2, 1, 1, 1)
                h = h + lmk_fea
            return h
        model_clone = model.clone()
        model_clone.set_model_input_block_patch(input_block_patch)
           
        return (model_clone, )
    

class FYESampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipeline": ("FYEPIPE",),
            "clip_embeds": ("FYECLIPEMBED", ),
            "ref_latent": ("LATENT", ),
            "landmark_features": ("LMKFEAT",),
            "steps": ("INT", {"default": 25, "min": 1}),
            "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 30.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "context_frames": ("INT", {"default": 24, "min": 8, "max": 48}),
            "context_overlap": ("INT", {"default": 4, "min": 1, "max": 24}),
            "context_stride": ("INT", {"default": 1, "min": 1, "max": 8}),
            "latent_interpolation_factor": ("INT", {"default": 1, "min": 1, "max": 10}),
            "ref_down_block_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "ref_mid_block_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "ref_up_block_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
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

    def process(self, pipeline, clip_embeds, ref_latent, landmark_features, steps, seed, cfg, context_frames, context_overlap, 
                context_stride, latent_interpolation_factor, scheduler, ref_down_block_multiplier, ref_mid_block_multiplier, ref_up_block_multiplier):

        ref_sample, H, W, generator, noise_scheduler = common_process(pipeline, ref_latent, seed, scheduler)

        latents = pipeline(
                        ref_image_latents=ref_sample,
                        cond_images=clip_embeds,
                        landmark_features=landmark_features,
                        width=W,
                        height=H,
                        video_length=landmark_features.shape[2],
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        generator=generator,
                        context_frames=context_frames,
                        context_overlap=context_overlap,
                        context_stride=context_stride,
                        interpolation_factor=latent_interpolation_factor,
                        scheduler=noise_scheduler,
                        ref_down_block_multiplier=ref_down_block_multiplier,
                        ref_mid_block_multiplier=ref_mid_block_multiplier,
                        ref_up_block_multiplier=ref_up_block_multiplier
                        )

        latents = latents.squeeze(0).permute(1,0,2,3) / 0.18215
       
        return({"samples":latents},)
    
def common_process(pipeline, ref_latent, seed, scheduler):

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
        scheduler_config.pop("rescale_betas_zero_snr", None)
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

    #motions = (motions * 255).cpu().numpy().astype(np.uint8)
    #lmk_images = [Image.fromarray(motion) for motion in motions]

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return (ref_sample, H, W, generator, noise_scheduler)

class FYESamplerLong:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipeline": ("FYEPIPE",),
            "clip_embeds": ("FYECLIPEMBED", ),
            "ref_latent": ("LATENT", ),
            "landmark_features": ("LMKFEAT",),
            "steps": ("INT", {"default": 25, "min": 1}),
            "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 30.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "t_tile_length": ("INT", {"default": 16, "min": 8, "max": 256}),
            "t_tile_overlap": ("INT", {"default": 4, "min": 1, "max": 24}),
            "ref_down_block_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "ref_mid_block_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "ref_up_block_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
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

    def process(self, pipeline, clip_embeds, ref_latent, landmark_features, steps, seed, cfg, t_tile_length, t_tile_overlap, 
                scheduler, ref_down_block_multiplier, ref_mid_block_multiplier, ref_up_block_multiplier):
      
        ref_sample, H, W, generator, noise_scheduler = common_process(pipeline, ref_latent, seed, scheduler)

        latents = pipeline.forward_long(
                        ref_image_latents=ref_sample,
                        landmark_features=landmark_features,
                        cond_images=clip_embeds,
                        width=W,
                        height=H,
                        video_length=landmark_features.shape[2],
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        generator=generator,
                        t_tile_length=t_tile_length,
                        t_tile_overlap=t_tile_overlap,
                        scheduler=noise_scheduler,
                        ref_down_block_multiplier=ref_down_block_multiplier,
                        ref_mid_block_multiplier=ref_mid_block_multiplier,
                        ref_up_block_multiplier=ref_up_block_multiplier
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
       
        images_np = (images * 255).cpu().numpy().astype(np.uint8)
       
        lmk_extractor = LMKExtractor()
        vis = FaceMeshVisualizer(forehead_edge=False, iris_point=draw_iris_points, draw_outer_lips=draw_outer_lips)
        aligner = FaceMeshAlign(vis)

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
    "FYESamplerLong": FYESamplerLong,
    "FYECLIPEncode": FYECLIPEncode,
    "FYELandmarkEncode": FYELandmarkEncode,
    "FYELandmarkToComfy": FYELandmarkToComfy,
    "FYEClipEmbedToComfy": FYEClipEmbedToComfy
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadFYEModel": "(Down)LoadFYE Model",
    "FYESampler": "FYESampler",
    "FYEMediaPipe": "MediaPipe Landmarks",
    "FYESamplerLong": "FYESamplerLong",
    "FYECLIPEncode": "FYECLIPEncode",
    "FYELandmarkEncode": "FYELandmarkEncode",
    "FYELandmarkToComfy": "FYELandmarkToComfy",
    "FYEClipEmbedToComfy": "FYEClipEmbedToComfy"
    }