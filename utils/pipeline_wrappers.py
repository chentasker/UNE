"""
Pipeline wrappers for diffusion models and encoders.

SDPipelineWrapper, SDXLPipelineWrapper: Stable Diffusion 1.5/2.1/XL with DDIM inversion.
DDPMPipelineWrapper: DDPM (e.g. CelebA-HQ) without VAE.
CLIPPipelineWrapper, EncoderPipelineWrapper: CLIP and MAE/Swin/ConvNeXt image encoders.
"""
from transformers import logging as hf_logging

# Disable ALL transformer warnings
hf_logging.set_verbosity_error()

from torchvision import transforms
import torch
from tqdm import tqdm
import numpy as np
from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline, DDPMPipeline, DDIMScheduler
from transformers import CLIPProcessor, CLIPModel, ViTMAEModel, ViTImageProcessor, SwinModel, AutoImageProcessor, ConvNextModel
from huggingface_hub.utils import EntryNotFoundError
from PIL import Image
import os

from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

def preprocess_img(imgs, resolution=None, return_type='pil', dtype=None, device=None):
    """
    Preprocesses images by center-cropping to a square and optionally resizing.
    If return_type is 'tensor', the output is stacked into a single 4D batch and normalized to the range [-1, 1].
    Grayscale images are automatically expanded to 3 channels.

    Args:
        imgs (PIL.Image or list): A PIL image or a list of images
        resolution (int, optional): Target size for square resizing.
        return_type (str, optional): _description_. Defaults to 'pil'.
        dtype (_type_, optional): _description_. Defaults to None.
        device (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    assert return_type.lower() in ['pil', 'tensor', 'list']

    if not isinstance(imgs, list):
        imgs = [imgs]
    
    for i, img in enumerate(imgs):
        # Apply center crop
        img = transforms.functional.center_crop(img, min(img.size))
        # Apply resize
        if resolution is not None:
            img = transforms.functional.resize(img, (resolution, resolution))
        # Convert to tensor
        if return_type == 'tensor':
            img = transforms.functional.to_tensor(img)
            # Deal with grey scale images
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
        imgs[i] = img

    if return_type == 'tensor':
        return torch.stack([img.to(dtype).to(device) * 2 - 1 for img in imgs])
    if len(imgs) == 1 and return_type=='pil':
        return imgs[0]
    return imgs

def ddim_step(scheduler, latents, noise_pred, t_curr, t_next):
    alpha_t = scheduler.alphas_cumprod[t_curr.long()]
    alpha_t_next = scheduler.alphas_cumprod[t_next.long()]

    latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (
        alpha_t_next.sqrt() / alpha_t.sqrt()
    ) + (1 - alpha_t_next).sqrt() * noise_pred

    return latents

@torch.no_grad()
def disable_safety_checker(pipe):
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

# -----------------------
class SDPipelineWrapper:
    def __init__(self, device="cuda", dtype=torch.float16, model_name="sd15", do_compile=True):
        self.model_name = model_name.lower()
        self.known_models = ['sd15',
                             'sd21',
                             'lcmv7',]
        assert self.model_name in self.known_models, f"Unknown model name: {self.model_name}"

        self.device = device
        self.dtype = dtype

        self.pipe = self._load_pipe(do_compile)
        disable_safety_checker(self.pipe)

        self.resolution = 512
        self.latent_shape = (4, self.resolution // 8, self.resolution // 8)  # assuming VAE downscales by factor of 8
        self.latent_size = self.latent_shape[0] * self.latent_shape[1] * self.latent_shape[2]

        self.num_inference_steps = 150  # default value

    def _load_pipe(self, do_compile):
        model_id = {
            'sd15': "runwayml/stable-diffusion-v1-5",
            'lcmv7': "SimianLuo/LCM_Dreamshaper_v7",
            # 'sd21': "stabilityai/stable-diffusion-2-1-base", # Original model from Huggingface, depracted
            'sd21': "/home1/chent/UNETests/my_sd21_model",
        }[self.model_name]

        variant = {
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
            torch.float32: None
        }[self.dtype]

        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                variant=variant,
                use_safetensors=True
            ).to(self.device)

        except (EntryNotFoundError, OSError, ValueError):
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                variant=None,              # Download the default file
                use_safetensors=True
            ).to(self.device)
        
        pipe.enable_xformers_memory_efficient_attention()
        
        if do_compile:
            # Note: torch.compile can be slow on first run; set to False if you want instant startup
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        return pipe
      
    @torch.no_grad()
    def encode_images(self, imgs,
                      prompt="",
                      show_tqdm=True):
        # prepare images
        processed_imgs = preprocess_img(imgs, return_type='tensor',
                                        device=self.device,
                                        dtype=self.dtype,
                                        resolution=self.resolution)
        
        # Encode pixels using VAE
        vae_output = self.pipe.vae.encode(processed_imgs)
        latents = vae_output.latent_dist.mode() * self.pipe.vae.config.scaling_factor
        latents = latents.to(dtype=self.dtype)

        # Prepare timesteps
        self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps.flip(0)

        # Prepare step inputs
        batch_size = len(processed_imgs)
        if isinstance(prompt, str):
            prompt = [prompt]*batch_size
        text_embeddings = self.pipe.encode_prompt(
            prompt, self.device, 1, True, [""]*len(prompt)
        )[0].to(device=self.device, dtype=self.dtype)

        # Inversion loop
        for i in tqdm(range(1, len(timesteps)), initial=1, disable=not show_tqdm):
            t_curr = timesteps[i-1]
            t_next = timesteps[i]

            latent_input = self.pipe.scheduler.scale_model_input(latents, t_curr)
            noise_pred = self.pipe.unet(
                latent_input, t_curr,
                encoder_hidden_states=text_embeddings
            ).sample
            
            latents = ddim_step(
                self.pipe.scheduler, latents, noise_pred, t_curr, t_next
            )

        return latents.flatten(1)
       
    @torch.no_grad()
    def decode_latent(self, start_latents,
                      prompt="",
                      num_inference_steps=None,
                      guidance_scale=1.0,
                      show_tqdm=True):
        # prepare latents
        if isinstance(start_latents, np.ndarray):
            start_latents = torch.from_numpy(start_latents)
        start_latents = start_latents.to(device=self.device, dtype=self.dtype)
        start_latents = start_latents.reshape(-1, *self.latent_shape)

        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps

        # prepare prompt embeddings
        batch_size=start_latents.shape[0]
        if isinstance(prompt, str):
            prompt = [prompt]*batch_size
        else:
            assert len(prompt) == batch_size, "Prompt length must match batch size"
        uncond_embeds, cond_embeds = self.pipe.encode_prompt(
            prompt, self.device, 1, True, [""]*len(prompt)
        )
        prompt_embeds = uncond_embeds.to(device=self.device, dtype=self.dtype)
        negative_prompt_embeds = cond_embeds.to(device=self.device, dtype=self.dtype)

        # set progress bar config
        self.pipe.set_progress_bar_config()
        prev_progress_bar_config = self.pipe._progress_bar_config.get('disable', False)
        self.pipe.set_progress_bar_config(disable=not show_tqdm)

        # generation
        res = self.pipe(
            latents=start_latents,
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        ).images
        
        # restore progress bar config
        self.pipe.set_progress_bar_config(disable=prev_progress_bar_config)

        return res[0] if len(res) == 1 else res


class SDXLPipelineWrapper:
    def __init__(self, device=None, dtype=torch.float16, do_compile=True):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype
        self.pipe = self._load_pipe(do_compile)
        disable_safety_checker(self.pipe)
        
        # SDXL specific defaults
        self.num_inference_steps = 300 
        self.resolution = 1024
        self.latent_shape = (4, self.resolution // 8, self.resolution // 8)  # assuming VAE downscales by factor of 8
        self.latent_size = self.latent_shape[0] * self.latent_shape[1] * self.latent_shape[2]

    def _load_pipe(self, do_compile):
        variant = {
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
            torch.float32: None
        }[self.dtype]
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=self.dtype,
            variant=variant,
            use_safetensors=True
        ).to(self.device)
        
        pipe.enable_xformers_memory_efficient_attention()
        
        if do_compile:
            # Note: torch.compile can be slow on first run; remove if you want instant startup
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        return pipe
    
    def _get_add_time_ids(self, batch_size, dtype):
        # SDXL expects: original_size, crops_coords_top_left, target_size
        # Defaulting to 1024x1024 without cropping
        base_ids = [1024, 1024, 0, 0, 1024, 1024]
        add_time_ids = torch.tensor([base_ids] * batch_size, dtype=dtype, device=self.device)
        return add_time_ids
    
    @torch.no_grad()
    def encode_images(self, imgs,
                      prompt="",
                      show_tqdm=True):
        # prepare images
        processed_imgs = preprocess_img(imgs, return_type='tensor',
                                        device=self.device,
                                        dtype=self.dtype,
                                        resolution=self.resolution)
        
        # encode pixels with VAE
        # Move to float32 to avoid nan issues
        self.pipe.vae.to(dtype=torch.float32)
        vae_output = self.pipe.vae.encode(processed_imgs.to(dtype=torch.float32))
        latents = vae_output.latent_dist.mode() * self.pipe.vae.config.scaling_factor
        self.pipe.vae.to(dtype=self.dtype)
        latents = latents.to(dtype=self.dtype)

        # prepare timesteps
        self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps.flip(0)

        #step_kwargs = self.get_step_inputs(prompt=prompt, batch_size=len(processed_imgs))
        batch_size=len(processed_imgs)
        if isinstance(prompt, str):
            prompt = [prompt]*batch_size
        
        # SDXL encode_prompt returns a tuple of 4: 
        # (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        prompt_embeds, _, pooled_embeds, _ = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=[""]*batch_size
        )
        encoder_hidden_states = prompt_embeds
        added_cond_kwargs = {
            "text_embeds": pooled_embeds,
            "time_ids": self._get_add_time_ids(batch_size, prompt_embeds.dtype)
        }

        for i in tqdm(range(1, len(timesteps)), initial=1, disable=not show_tqdm):
            t_curr = timesteps[i-1]
            t_next = timesteps[i]

            latent_input = self.pipe.scheduler.scale_model_input(latents, t_curr)
            noise_pred = self.pipe.unet(
                latent_input, t_curr,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs
                ).sample
            
            latents = ddim_step(
                self.pipe.scheduler, latents, noise_pred, t_curr, t_next
            )

        return latents.flatten(1)
    
    @torch.no_grad()
    def decode_latent(self, start_latents,
                      prompt="",
                      num_inference_steps=None,
                      guidance_scale=1.0,
                      show_tqdm=True):
        # prepare latents
        if isinstance(start_latents, np.ndarray):
            start_latents = torch.from_numpy(start_latents)
        start_latents = start_latents.to(device=self.device, dtype=self.dtype)
        start_latents = start_latents.reshape(-1, *self.latent_shape)
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps

        # prepare prompt embeddings
        batch_size=start_latents.shape[0]
        if isinstance(prompt, str):
            prompt = [prompt]*batch_size

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=[""]*batch_size
        )
        

        self.pipe.set_progress_bar_config()
        prev_progress_bar_config = self.pipe._progress_bar_config.get('disable', False)
        self.pipe.set_progress_bar_config(disable=not show_tqdm)

        res = self.pipe(
            latents=start_latents,
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds
        ).images
        
        self.pipe.set_progress_bar_config(disable=prev_progress_bar_config)
        return res[0] if len(res) == 1 else res

class EncoderPipelineWrapper:
    def __init__(self, device=None, dtype=torch.float16,
                 model_name='l14', load_components=False):
        self.model_name = model_name.lower()
        implemented_model_names = [
            'mae', 'swin', 'convnext'
        ]
        assert self.model_name in implemented_model_names

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None

        if load_components:
            self._load_components()

    # -----------------------
    # Model loading
    # -----------------------

    def _load_components(self):
        if (self.model is not None) and (self.processor is not None):
            return
        
        model_id = {
            'mae': "facebook/vit-mae-base",
            'swin': "microsoft/swin-tiny-patch4-window7-224",
            'convnext': "facebook/convnext-tiny-224"
        }[self.model_name]

        model_obj = {
            'mae': ViTMAEModel,
            'swin': SwinModel,
            'convnext': ConvNextModel
        }[self.model_name]

        processor_obj = {
            'mae': ViTImageProcessor,
            'swin': AutoImageProcessor,
            'convnext': AutoImageProcessor
        }[self.model_name]

        self.model = model_obj.from_pretrained(
            model_id
        ).to(self.device)
        self.processor = processor_obj.from_pretrained(
            model_id
        )

    # -----------------------
    # Encoding (CLIP image → embedding)
    # -----------------------
    @torch.no_grad()
    def encode_images(self, image, prompt, show_tqdm=False):
        if show_tqdm:
            raise NotImplementedError("TQDM not implemented for CLIP encoding.")
        
        self._load_components()

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        inputs = self.processor(
            images=image, return_tensors="pt", padding=True
        ).to(self.device)

        model_outputs = self.model(inputs["pixel_values"])

        if self.model_name == 'mae':
            features = model_outputs.last_hidden_state.mean(dim=1)
        else:
            features = model_outputs.pooler_output

        return features.half()

    # Encoding (Text → embedding)
    # -----------------------
    @torch.no_grad()
    def encode_text(self, text, normalize=False):
        """
        Encodes a string or list of strings into CLIP text embeddings.
        """
        self._load_components()

        # Tokenize and process text
        inputs = self.processor(
            text=text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        # Get text features (projections)
        text_features = self.model.get_text_features(**inputs).half()

        if normalize:
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-12)

        return text_features
    

class CLIPPipelineWrapper:
    def __init__(self, device=None, dtype=torch.float16,
                 model_name='l14',
                 load_unclip_pipe=False, load_clip_components=False):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype
        self.model_name = model_name.lower()
        self.pipe = None
        self.clip_model = None
        self.clip_processor = None

        if load_unclip_pipe:
            self._load_pipe()
        if load_clip_components:
            self._load_clip()

    # -----------------------
    # Model loading
    # -----------------------
    def _load_pipe(self):
        if self.pipe is None:
            self.pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1-unclip-small",
                torch_dtype=self.dtype
            ).to(self.device)

    def _load_clip(self):
        if self.clip_model is None or self.clip_processor is None:
            model_id = {
                'l14': "openai/clip-vit-large-patch14",
                'b16': "openai/clip-vit-base-patch16",
                'oc-l14': "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
                'oc-b16': "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
            }[self.model_name]
            self.clip_model = CLIPModel.from_pretrained(
                model_id
            ).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(
                model_id
            )

    # -----------------------
    # Encoding (CLIP image → embedding)
    # -----------------------
    @torch.no_grad()
    def encode_images(self, image, prompt, normalize=False, show_tqdm=False):
        if show_tqdm:
            raise NotImplementedError("TQDM not implemented for CLIP encoding.")
        
        self._load_clip()

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        inputs = self.clip_processor(
            text=prompt, images=image, return_tensors="pt", padding=True
        ).to(self.device)

        image_features = self.clip_model.get_image_features(inputs["pixel_values"]).half()

        if normalize:
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-12)

        return image_features

    # Encoding (Text → embedding)
    # -----------------------
    @torch.no_grad()
    def encode_text(self, text, normalize=False):
        """
        Encodes a string or list of strings into CLIP text embeddings.
        """
        self._load_clip()

        # Tokenize and process text
        inputs = self.clip_processor(
            text=text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        # Get text features (projections)
        text_features = self.clip_model.get_text_features(**inputs).half()

        if normalize:
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-12)

        return text_features
    
    # -----------------------
    # Decoding (embedding → image) vis UnCLIP
    # -----------------------
    @torch.no_grad()
    def decode_latent(self, image_embeds,
                       guidance_scale=1.0,
                       num_inference_steps=25,
                       prompt="",
                       latents=None):
        self._load_pipe()

        if isinstance(image_embeds, np.ndarray):
            image_embeds = torch.from_numpy(image_embeds)
        image_embeds = image_embeds.to(self.device, dtype=self.dtype)
            
        if image_embeds.dim() == 1:
            image_embeds = image_embeds.reshape(1,-1)
        else:
            image_embeds = image_embeds.flatten(1)

        result = self.pipe(
            image_embeds=image_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            prompt=prompt,
            latents=latents
        )

        images = result.images
        return images[0] if len(images) == 1 else images


class DINOPipelineWrapper:
    def __init__(self, device=None, dtype=torch.float16):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype
        self.pipe = self._load_pipe()

    def _load_pipe(self):
        # TODO: ???
        # Your existing setup
        REPO_DIR = '/home/omerben/github_download/dinov3/'
        weights = os.path.join(REPO_DIR, 'weights', 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')

        # Load DINOv3 model
        dinov3_vitl16 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=weights)
        dinov3_vitl16.eval()  # Set to evaluation mode
        dinov3_vitl16 = dinov3_vitl16.to(self.device)
        return dinov3_vitl16
    
        
    @torch.no_grad()
    def get_text_embeddings(self, prompt, batch_size=1):
        return self.pipe._encode_prompt(
                prompt, self.device, batch_size, True, ''
        )

    ## Inversion
    @torch.no_grad()
    def encode_images(self, imgs):
        imgs = preprocess_img(imgs, return_type='tensor',
                              device=self.device,
                              dtype=self.dtype,
                              resolution=224)
        features = self.pipe(imgs)
        return features
