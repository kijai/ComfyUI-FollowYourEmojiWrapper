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
        lmk_guider_path = os.path.join(fye_base_path, "FYE_lmk_guider.safetensors")

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

        #guider
        sd = comfy.utils.load_torch_file(lmk_guider_path)
        lmk_guider = Guider(conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)).to(dtype).to(device)
        lmk_guider.load_state_dict(sd)

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
                             vae=None,
                             image_encoder=None,
                             referencenet=referencenet,
                             unet=ad_unet,
                             lmk_guider=lmk_guider)

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
    
class FYESampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipeline": ("FYEPIPE",),
            "clip_embeds": ("FYECLIPEMBED", ),
            "ref_latent": ("LATENT", ),
            "motions": ("IMAGE",),
            "steps": ("INT", {"default": 25, "min": 1}),
            "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 30.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "context_frames": ("INT", {"default": 24, "min": 8, "max": 48}),
            "context_overlap": ("INT", {"default": 4, "min": 1, "max": 24}),
            "context_stride": ("INT", {"default": 1, "min": 1, "max": 8}),
            "latent_interpolation_factor": ("INT", {"default": 1, "min": 1, "max": 10}),
            "pose_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
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

    def process(self, pipeline, clip_embeds, ref_latent, motions, steps, seed, cfg, context_frames, context_overlap, 
                context_stride, latent_interpolation_factor, pose_multiplier, scheduler, ref_down_block_multiplier, ref_mid_block_multiplier, ref_up_block_multiplier):

        ref_sample, lmk_images, H, W, generator, noise_scheduler = common_process(pipeline, ref_latent, motions, seed, scheduler)

        latents = pipeline(
                        ref_image=None,
                        ref_image_latents=ref_sample,
                        cond_images=clip_embeds,
                        lmk_images=lmk_images,
                        width=W,
                        height=H,
                        video_length=len(motions),
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        generator=generator,
                        context_frames=context_frames,
                        context_overlap=context_overlap,
                        context_stride=context_stride,
                        interpolation_factor=latent_interpolation_factor,
                        pose_multiplier=pose_multiplier,
                        scheduler=noise_scheduler,
                        ref_down_block_multiplier=ref_down_block_multiplier,
                        ref_mid_block_multiplier=ref_mid_block_multiplier,
                        ref_up_block_multiplier=ref_up_block_multiplier
                        )

        latents = latents.squeeze(0).permute(1,0,2,3) / 0.18215
       
        return({"samples":latents},)
    
def common_process(pipeline, ref_latent, motions, seed, scheduler):

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

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return (ref_sample, lmk_images, H, W, generator, noise_scheduler)

class FYESamplerLong:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipeline": ("FYEPIPE",),
            "clip_embeds": ("FYECLIPEMBED", ),
            "ref_latent": ("LATENT", ),
            "motions": ("IMAGE",),
            "steps": ("INT", {"default": 25, "min": 1}),
            "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 30.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "t_tile_length": ("INT", {"default": 16, "min": 8, "max": 256}),
            "t_tile_overlap": ("INT", {"default": 4, "min": 1, "max": 24}),
            "pose_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
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

    def process(self, pipeline, clip_embeds, ref_latent, motions, steps, seed, cfg, t_tile_length, t_tile_overlap, 
                pose_multiplier, scheduler, ref_down_block_multiplier, ref_mid_block_multiplier, ref_up_block_multiplier):
      
        ref_sample, lmk_images, H, W, generator, noise_scheduler = common_process(pipeline, ref_latent, motions, seed, scheduler)

        latents = pipeline.forward_long(
                        ref_image=None,
                        ref_image_latents=ref_sample,
                        lmk_images=lmk_images,
                        cond_images=clip_embeds,
                        width=W,
                        height=H,
                        video_length=len(motions),
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        generator=generator,
                        t_tile_length=t_tile_length,
                        t_tile_overlap=t_tile_overlap,
                        pose_multiplier=pose_multiplier,
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
    "FYESamplerLong": FYESamplerLong,
    "FYECLIPEncode": FYECLIPEncode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadFYEModel": "(Down)LoadFYE Model",
    "FYESampler": "FYESampler",
    "FYEMediaPipe": "MediaPipe",
    "FYESamplerLong": "FYESamplerLong",
    "FYECLIPEncode": "FYECLIPEncode"
    }

def split_tiles(x, num_split):
    _, H, W, _ = x.shape
    h, w = H // num_split, W // num_split
    x_split = torch.cat([x[:, i*h:(i+1)*h, j*w:(j+1)*w, :] for i in range(num_split) for j in range(num_split)], dim=0)    

    return x_split

def merge_hiddenstates(embeds):
    num_tiles = embeds.shape[0]
    tile_size = int((embeds.shape[1]-1) ** 0.5)
    grid_size = int(num_tiles ** 0.5)

    # Extract class tokens
    class_tokens = embeds[:, 0, :]  # Save class tokens: [num_tiles, embeds[-1]]
    avg_class_token = class_tokens.mean(dim=0, keepdim=True).unsqueeze(0)  # Average token, shape: [1, 1, embeds[-1]]

    patch_embeds = embeds[:, 1:, :]  # Shape: [num_tiles, tile_size^2, embeds[-1]]
    reshaped = patch_embeds.reshape(grid_size, grid_size, tile_size, tile_size, embeds.shape[-1])

    merged = torch.cat([torch.cat([reshaped[i, j] for j in range(grid_size)], dim=1) 
                        for i in range(grid_size)], dim=0)

    merged = merged.unsqueeze(0)  # Shape: [1, grid_size*tile_size, grid_size*tile_size, embeds[-1]]

    # Pool to original size
    pooled = torch.nn.functional.adaptive_avg_pool2d(merged.permute(0, 3, 1, 2), (tile_size, tile_size)).permute(0, 2, 3, 1)
    flattened = pooled.reshape(1, tile_size*tile_size, embeds.shape[-1])

    # Add back the class token
    with_class = torch.cat([avg_class_token, flattened], dim=1)  # Shape: original shape

    return with_class

def merge_embeddings(embeds): # TODO: this needs so much testing that I don't even
    num_tiles = embeds.shape[0]
    grid_size = int(num_tiles ** 0.5)
    tile_size = int(embeds.shape[1] ** 0.5)
    reshaped = embeds.reshape(grid_size, grid_size, tile_size, tile_size)

    # Merge the tiles
    merged = torch.cat([torch.cat([reshaped[i, j] for j in range(grid_size)], dim=1) 
                        for i in range(grid_size)], dim=0)

    merged = merged.unsqueeze(0)  # Shape: [1, grid_size*tile_size, grid_size*tile_size]

    # Pool to original size
    pooled = torch.nn.functional.adaptive_avg_pool2d(merged, (tile_size, tile_size))  # pool to [1, tile_size, tile_size]
    pooled = pooled.flatten(1)  # flatten to [1, tile_size^2]

    return pooled

def encode_image_masked(clip_vision, image, mask=None, batch_size=0, tiles=1, ratio=1.0):
    # full image embeds
    embeds = encode_image_masked_(clip_vision, image, mask, batch_size)
    tiles = min(tiles, 16)

    if tiles > 1:
        # split in tiles
        image_split = split_tiles(image, tiles)

        # get the embeds for each tile
        embeds_split = encode_image_masked_(clip_vision, image_split, mask, batch_size)

        embeds_split['last_hidden_state'] = merge_hiddenstates(embeds_split['last_hidden_state'])
        #embeds_split["image_embeds"] = merge_embeddings(embeds_split["image_embeds"])
        #embeds_split["penultimate_hidden_states"] = merge_hiddenstates(embeds_split["penultimate_hidden_states"])

        embeds['last_hidden_state'] = torch.cat([embeds['last_hidden_state']*ratio, embeds_split['last_hidden_state']])
        #embeds['image_embeds'] = torch.cat([embeds['image_embeds']*ratio, embeds_split['image_embeds']])
        #embeds['penultimate_hidden_states'] = torch.cat([embeds['penultimate_hidden_states']*ratio, embeds_split['penultimate_hidden_states']])

    #del embeds_split

    return embeds

from comfy.clip_vision import clip_preprocess, Output

def encode_image_masked_(clip_vision, image, mask=None, batch_size=0):

    outputs = Output()

    if batch_size == 0:
        batch_size = image.shape[0]
    elif batch_size > image.shape[0]:
        batch_size = image.shape[0]

    image_batch = torch.split(image, batch_size, dim=0)

    for img in image_batch:
        img = img.to(clip_vision.load_device)
        pixel_values = clip_preprocess(img).float()

        # TODO: support for multiple masks
        if mask is not None:
            pixel_values = pixel_values * mask.to(clip_vision.load_device)

        out = clip_vision.model(pixel_values=pixel_values, intermediate_output=-2)

        if not hasattr(outputs, "last_hidden_state"):
            outputs["last_hidden_state"] = out[0].to(mm.intermediate_device())
            outputs["image_embeds"] = out[2].to(mm.intermediate_device())
            outputs["penultimate_hidden_states"] = out[1].to(mm.intermediate_device())
        else:
            outputs["last_hidden_state"] = torch.cat((outputs["last_hidden_state"], out[0].to(mm.intermediate_device())), dim=0)
            outputs["image_embeds"] = torch.cat((outputs["image_embeds"], out[2].to(mm.intermediate_device())), dim=0)
            outputs["penultimate_hidden_states"] = torch.cat((outputs["penultimate_hidden_states"], out[1].to(mm.intermediate_device())), dim=0)

    del img, pixel_values, out
    torch.cuda.empty_cache()

    return outputs