# Additional ComfyUI Workflow Templates
"""
Extended ComfyUI workflow templates for generative AI.
Milestone 1: More comprehensive image generation patterns.
"""

from datetime import datetime
from typing import Dict, Any


def get_comfyui_templates() -> Dict[str, callable]:
    """Return all available ComfyUI templates."""
    return {
        "text_to_image": text_to_image_workflow,
        "image_to_image": image_to_image_workflow,
        "inpainting": inpainting_workflow,
        "upscale": upscale_workflow,
        "controlnet": controlnet_workflow,
        "batch_generation": batch_generation_workflow,
        "style_transfer": style_transfer_workflow,
        "lora_generation": lora_generation_workflow,
    }


def text_to_image_workflow(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate basic text-to-image workflow."""
    prompt = params.get("prompt", "a beautiful landscape")
    negative = params.get("negative_prompt", "blurry, low quality, distorted")
    model = params.get("model", "sd_xl_base_1.0.safetensors")
    width = params.get("width", 1024)
    height = params.get("height", 1024)
    steps = params.get("steps", 25)
    cfg = params.get("cfg", 7.0)
    seed = params.get("seed", -1)
    
    return {
        "3": {
            "inputs": {
                "seed": seed if seed > 0 else 12345,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"}
        },
        "4": {
            "inputs": {"ckpt_name": model},
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"}
        },
        "5": {
            "inputs": {"width": width, "height": height, "batch_size": 1},
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Empty Latent"}
        },
        "6": {
            "inputs": {"text": prompt, "clip": ["4", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive Prompt"}
        },
        "7": {
            "inputs": {"text": negative, "clip": ["4", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Negative Prompt"}
        },
        "8": {
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"}
        },
        "9": {
            "inputs": {"filename_prefix": "txt2img", "images": ["8", 0]},
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"}
        }
    }


def image_to_image_workflow(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate image-to-image transformation workflow."""
    prompt = params.get("prompt", "enhance this image")
    negative = params.get("negative_prompt", "blurry, low quality")
    model = params.get("model", "sd_xl_base_1.0.safetensors")
    denoise = params.get("denoise", 0.75)
    steps = params.get("steps", 30)
    
    return {
        "1": {
            "inputs": {"image": "input.png", "upload": "image"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Input Image"}
        },
        "2": {
            "inputs": {"ckpt_name": model},
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"}
        },
        "3": {
            "inputs": {"pixels": ["1", 0], "vae": ["2", 2]},
            "class_type": "VAEEncode",
            "_meta": {"title": "VAE Encode"}
        },
        "4": {
            "inputs": {"text": prompt, "clip": ["2", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive Prompt"}
        },
        "5": {
            "inputs": {"text": negative, "clip": ["2", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Negative Prompt"}
        },
        "6": {
            "inputs": {
                "seed": 12345,
                "steps": steps,
                "cfg": 7.0,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": denoise,
                "model": ["2", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["3", 0]
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"}
        },
        "7": {
            "inputs": {"samples": ["6", 0], "vae": ["2", 2]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"}
        },
        "8": {
            "inputs": {"filename_prefix": "img2img", "images": ["7", 0]},
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"}
        }
    }


def inpainting_workflow(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate inpainting workflow."""
    prompt = params.get("prompt", "fill in the masked area")
    negative = params.get("negative_prompt", "blurry, distorted")
    model = params.get("model", "sd_xl_base_1.0.safetensors")
    
    return {
        "1": {
            "inputs": {"image": "input.png", "upload": "image"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"}
        },
        "2": {
            "inputs": {"image": "mask.png", "upload": "image"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Mask"}
        },
        "3": {
            "inputs": {"ckpt_name": model},
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"}
        },
        "4": {
            "inputs": {
                "pixels": ["1", 0],
                "vae": ["3", 2],
                "mask": ["2", 0],
                "grow_mask_by": 6
            },
            "class_type": "VAEEncodeForInpaint",
            "_meta": {"title": "VAE Encode (Inpaint)"}
        },
        "5": {
            "inputs": {"text": prompt, "clip": ["3", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive"}
        },
        "6": {
            "inputs": {"text": negative, "clip": ["3", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Negative"}
        },
        "7": {
            "inputs": {
                "seed": 12345,
                "steps": 30,
                "cfg": 8.0,
                "sampler_name": "dpmpp_2m_sde",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["3", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["4", 0]
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"}
        },
        "8": {
            "inputs": {"samples": ["7", 0], "vae": ["3", 2]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"}
        },
        "9": {
            "inputs": {"filename_prefix": "inpaint", "images": ["8", 0]},
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"}
        }
    }


def upscale_workflow(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate image upscaling workflow."""
    upscale_model = params.get("upscale_model", "RealESRGAN_x4plus.pth")
    
    return {
        "1": {
            "inputs": {"image": "input.png", "upload": "image"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"}
        },
        "2": {
            "inputs": {"model_name": upscale_model},
            "class_type": "UpscaleModelLoader",
            "_meta": {"title": "Load Upscale Model"}
        },
        "3": {
            "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]},
            "class_type": "ImageUpscaleWithModel",
            "_meta": {"title": "Upscale Image"}
        },
        "4": {
            "inputs": {"filename_prefix": "upscaled", "images": ["3", 0]},
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"}
        }
    }


def controlnet_workflow(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate ControlNet-guided workflow."""
    prompt = params.get("prompt", "a detailed illustration")
    negative = params.get("negative_prompt", "blurry, low quality")
    model = params.get("model", "sd_xl_base_1.0.safetensors")
    controlnet = params.get("controlnet", "controlnet-canny-sdxl-1.0.safetensors")
    strength = params.get("strength", 1.0)
    
    return {
        "1": {
            "inputs": {"image": "control_image.png", "upload": "image"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Control Image"}
        },
        "2": {
            "inputs": {"ckpt_name": model},
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"}
        },
        "3": {
            "inputs": {"control_net_name": controlnet},
            "class_type": "ControlNetLoader",
            "_meta": {"title": "Load ControlNet"}
        },
        "4": {
            "inputs": {"text": prompt, "clip": ["2", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive"}
        },
        "5": {
            "inputs": {"text": negative, "clip": ["2", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Negative"}
        },
        "6": {
            "inputs": {
                "strength": strength,
                "conditioning": ["4", 0],
                "control_net": ["3", 0],
                "image": ["1", 0]
            },
            "class_type": "ControlNetApply",
            "_meta": {"title": "Apply ControlNet"}
        },
        "7": {
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Empty Latent"}
        },
        "8": {
            "inputs": {
                "seed": 12345,
                "steps": 30,
                "cfg": 7.5,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["2", 0],
                "positive": ["6", 0],
                "negative": ["5", 0],
                "latent_image": ["7", 0]
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"}
        },
        "9": {
            "inputs": {"samples": ["8", 0], "vae": ["2", 2]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"}
        },
        "10": {
            "inputs": {"filename_prefix": "controlnet", "images": ["9", 0]},
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"}
        }
    }


def batch_generation_workflow(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate batch image generation workflow."""
    prompt = params.get("prompt", "artistic image")
    negative = params.get("negative_prompt", "blurry")
    model = params.get("model", "sd_xl_base_1.0.safetensors")
    batch_size = params.get("batch_size", 4)
    
    return {
        "1": {
            "inputs": {"ckpt_name": model},
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"}
        },
        "2": {
            "inputs": {"width": 1024, "height": 1024, "batch_size": batch_size},
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Empty Latent (Batch)"}
        },
        "3": {
            "inputs": {"text": prompt, "clip": ["1", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive"}
        },
        "4": {
            "inputs": {"text": negative, "clip": ["1", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Negative"}
        },
        "5": {
            "inputs": {
                "seed": 12345,
                "steps": 25,
                "cfg": 7.0,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["2", 0]
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"}
        },
        "6": {
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"}
        },
        "7": {
            "inputs": {"filename_prefix": "batch", "images": ["6", 0]},
            "class_type": "SaveImage",
            "_meta": {"title": "Save Images"}
        }
    }


def style_transfer_workflow(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate style transfer workflow using IPAdapter."""
    prompt = params.get("prompt", "in the style of the reference")
    negative = params.get("negative_prompt", "blurry, low quality")
    model = params.get("model", "sd_xl_base_1.0.safetensors")
    weight = params.get("style_weight", 0.8)
    
    return {
        "1": {
            "inputs": {"image": "content.png", "upload": "image"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Content Image"}
        },
        "2": {
            "inputs": {"image": "style.png", "upload": "image"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Style Image"}
        },
        "3": {
            "inputs": {"ckpt_name": model},
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"}
        },
        "4": {
            "inputs": {"ipadapter_file": "ip-adapter_sdxl.safetensors"},
            "class_type": "IPAdapterModelLoader",
            "_meta": {"title": "Load IPAdapter"}
        },
        "5": {
            "inputs": {"clip_name": "clip_vision_g.safetensors"},
            "class_type": "CLIPVisionLoader",
            "_meta": {"title": "Load CLIP Vision"}
        },
        "6": {
            "inputs": {
                "weight": weight,
                "model": ["3", 0],
                "ipadapter": ["4", 0],
                "image": ["2", 0],
                "clip_vision": ["5", 0]
            },
            "class_type": "IPAdapterApply",
            "_meta": {"title": "Apply IPAdapter"}
        },
        "7": {
            "inputs": {"text": prompt, "clip": ["3", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive"}
        },
        "8": {
            "inputs": {"text": negative, "clip": ["3", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Negative"}
        },
        "9": {
            "inputs": {"pixels": ["1", 0], "vae": ["3", 2]},
            "class_type": "VAEEncode",
            "_meta": {"title": "VAE Encode"}
        },
        "10": {
            "inputs": {
                "seed": 12345,
                "steps": 30,
                "cfg": 7.0,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 0.7,
                "model": ["6", 0],
                "positive": ["7", 0],
                "negative": ["8", 0],
                "latent_image": ["9", 0]
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"}
        },
        "11": {
            "inputs": {"samples": ["10", 0], "vae": ["3", 2]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"}
        },
        "12": {
            "inputs": {"filename_prefix": "style_transfer", "images": ["11", 0]},
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"}
        }
    }


def lora_generation_workflow(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate LoRA-enhanced workflow."""
    prompt = params.get("prompt", "high quality artwork")
    negative = params.get("negative_prompt", "blurry, low quality")
    model = params.get("model", "sd_xl_base_1.0.safetensors")
    lora_name = params.get("lora", "detail_enhancer.safetensors")
    lora_strength = params.get("lora_strength", 0.8)
    
    return {
        "1": {
            "inputs": {"ckpt_name": model},
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"}
        },
        "2": {
            "inputs": {
                "lora_name": lora_name,
                "strength_model": lora_strength,
                "strength_clip": lora_strength,
                "model": ["1", 0],
                "clip": ["1", 1]
            },
            "class_type": "LoraLoader",
            "_meta": {"title": "Load LoRA"}
        },
        "3": {
            "inputs": {"text": prompt, "clip": ["2", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive"}
        },
        "4": {
            "inputs": {"text": negative, "clip": ["2", 1]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Negative"}
        },
        "5": {
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Empty Latent"}
        },
        "6": {
            "inputs": {
                "seed": 12345,
                "steps": 25,
                "cfg": 7.0,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["2", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"}
        },
        "7": {
            "inputs": {"samples": ["6", 0], "vae": ["1", 2]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"}
        },
        "8": {
            "inputs": {"filename_prefix": "lora_gen", "images": ["7", 0]},
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"}
        }
    }
