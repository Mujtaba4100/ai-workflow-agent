# ComfyUI Workflow Builder Tool
"""
Generate and execute ComfyUI workflow JSON templates.
Supports common generative AI patterns.
"""

import httpx
import json
import logging
import uuid
from typing import Dict, Any, List, Optional

from config import settings

logger = logging.getLogger(__name__)


class ComfyUIWorkflowBuilder:
    """
    ComfyUI workflow generator and executor.
    Creates JSON workflow graphs and executes via ComfyUI API.
    """
    
    def __init__(self):
        self.comfyui_host = settings.COMFYUI_HOST
        self.client = httpx.AsyncClient(timeout=300.0)  # Long timeout for image generation
    
    async def check_health(self) -> str:
        """Check if ComfyUI is running and responsive."""
        try:
            response = await self.client.get(f"{self.comfyui_host}/system_stats")
            if response.status_code == 200:
                return "healthy"
            return "unhealthy"
        except Exception as e:
            logger.error(f"ComfyUI health check failed: {e}")
            return "unreachable"
    
    async def generate_workflow(self, query: str) -> Dict[str, Any]:
        """
        Generate ComfyUI workflow JSON based on user query.
        
        Args:
            query: User's natural language request
            
        Returns:
            ComfyUI workflow JSON structure
        """
        # Detect workflow type from query
        workflow_type = self._detect_workflow_type(query)
        
        # Extract parameters from query
        params = self._extract_params(query)
        
        # Generate appropriate template
        templates = {
            "txt2img": self._generate_txt2img_workflow,
            "img2img": self._generate_img2img_workflow,
            "upscale": self._generate_upscale_workflow,
            "inpaint": self._generate_inpaint_workflow,
            "controlnet": self._generate_controlnet_workflow,
            "generic": self._generate_generic_workflow
        }
        
        generator = templates.get(workflow_type, self._generate_generic_workflow)
        workflow = generator(params)
        
        return workflow
    
    def _detect_workflow_type(self, query: str) -> str:
        """Detect the type of ComfyUI workflow needed."""
        query_lower = query.lower()
        
        if any(w in query_lower for w in ["upscale", "enhance", "higher resolution", "4x", "2x"]):
            return "upscale"
        elif any(w in query_lower for w in ["inpaint", "edit", "remove", "fill", "mask"]):
            return "inpaint"
        elif any(w in query_lower for w in ["controlnet", "pose", "depth", "canny", "edge"]):
            return "controlnet"
        elif any(w in query_lower for w in ["img2img", "transform", "style transfer", "from image"]):
            return "img2img"
        else:
            return "txt2img"
    
    def _extract_params(self, query: str) -> Dict[str, Any]:
        """Extract generation parameters from query."""
        # Default parameters
        params = {
            "prompt": query,
            "negative_prompt": "bad quality, blurry, deformed",
            "width": 512,
            "height": 512,
            "steps": 20,
            "cfg": 7.0,
            "seed": -1,  # Random
            "checkpoint": "v1-5-pruned-emaonly.safetensors"
        }
        
        query_lower = query.lower()
        
        # Detect resolution
        if "portrait" in query_lower or "vertical" in query_lower:
            params["width"] = 512
            params["height"] = 768
        elif "landscape" in query_lower or "horizontal" in query_lower:
            params["width"] = 768
            params["height"] = 512
        elif "square" in query_lower:
            params["width"] = 512
            params["height"] = 512
        elif "hd" in query_lower or "1024" in query_lower:
            params["width"] = 1024
            params["height"] = 1024
        
        # Detect model
        if "sdxl" in query_lower:
            params["checkpoint"] = "sd_xl_base_1.0.safetensors"
            params["width"] = 1024
            params["height"] = 1024
        elif "flux" in query_lower:
            params["checkpoint"] = "flux1-dev.safetensors"
        
        # Detect quality settings
        if "high quality" in query_lower or "detailed" in query_lower:
            params["steps"] = 30
            params["cfg"] = 8.0
        elif "fast" in query_lower or "quick" in query_lower:
            params["steps"] = 15
            params["cfg"] = 6.0
        
        return params
    
    def _generate_txt2img_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text-to-image workflow."""
        return {
            "prompt": {
                "3": {
                    "inputs": {
                        "seed": params.get("seed", -1),
                        "steps": params.get("steps", 20),
                        "cfg": params.get("cfg", 7.0),
                        "sampler_name": "euler",
                        "scheduler": "normal",
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
                    "inputs": {
                        "ckpt_name": params.get("checkpoint", "v1-5-pruned-emaonly.safetensors")
                    },
                    "class_type": "CheckpointLoaderSimple",
                    "_meta": {"title": "Load Checkpoint"}
                },
                "5": {
                    "inputs": {
                        "width": params.get("width", 512),
                        "height": params.get("height", 512),
                        "batch_size": 1
                    },
                    "class_type": "EmptyLatentImage",
                    "_meta": {"title": "Empty Latent Image"}
                },
                "6": {
                    "inputs": {
                        "text": params.get("prompt", "beautiful landscape"),
                        "clip": ["4", 1]
                    },
                    "class_type": "CLIPTextEncode",
                    "_meta": {"title": "CLIP Text Encode (Prompt)"}
                },
                "7": {
                    "inputs": {
                        "text": params.get("negative_prompt", "bad quality, blurry"),
                        "clip": ["4", 1]
                    },
                    "class_type": "CLIPTextEncode",
                    "_meta": {"title": "CLIP Text Encode (Negative)"}
                },
                "8": {
                    "inputs": {
                        "samples": ["3", 0],
                        "vae": ["4", 2]
                    },
                    "class_type": "VAEDecode",
                    "_meta": {"title": "VAE Decode"}
                },
                "9": {
                    "inputs": {
                        "filename_prefix": "ComfyUI",
                        "images": ["8", 0]
                    },
                    "class_type": "SaveImage",
                    "_meta": {"title": "Save Image"}
                }
            },
            "meta": {
                "generated_by": "AI Workflow Agent",
                "type": "txt2img",
                "params": params
            }
        }
    
    def _generate_img2img_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image-to-image workflow."""
        workflow = self._generate_txt2img_workflow(params)
        
        # Modify for img2img
        workflow["prompt"]["5"] = {
            "inputs": {
                "image": "INPUT_IMAGE_PATH",
                "upload": "image"
            },
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"}
        }
        
        # Add VAE encode for input
        workflow["prompt"]["10"] = {
            "inputs": {
                "pixels": ["5", 0],
                "vae": ["4", 2]
            },
            "class_type": "VAEEncode",
            "_meta": {"title": "VAE Encode"}
        }
        
        # Update sampler to use encoded image
        workflow["prompt"]["3"]["inputs"]["latent_image"] = ["10", 0]
        workflow["prompt"]["3"]["inputs"]["denoise"] = 0.75
        
        workflow["meta"]["type"] = "img2img"
        
        return workflow
    
    def _generate_upscale_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate upscale workflow."""
        return {
            "prompt": {
                "1": {
                    "inputs": {
                        "image": "INPUT_IMAGE_PATH",
                        "upload": "image"
                    },
                    "class_type": "LoadImage",
                    "_meta": {"title": "Load Image"}
                },
                "2": {
                    "inputs": {
                        "model_name": "RealESRGAN_x4plus.pth"
                    },
                    "class_type": "UpscaleModelLoader",
                    "_meta": {"title": "Load Upscale Model"}
                },
                "3": {
                    "inputs": {
                        "upscale_model": ["2", 0],
                        "image": ["1", 0]
                    },
                    "class_type": "ImageUpscaleWithModel",
                    "_meta": {"title": "Upscale Image"}
                },
                "4": {
                    "inputs": {
                        "filename_prefix": "Upscaled",
                        "images": ["3", 0]
                    },
                    "class_type": "SaveImage",
                    "_meta": {"title": "Save Image"}
                }
            },
            "meta": {
                "generated_by": "AI Workflow Agent",
                "type": "upscale",
                "params": params
            }
        }
    
    def _generate_inpaint_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate inpainting workflow."""
        workflow = self._generate_txt2img_workflow(params)
        
        # Add mask loading
        workflow["prompt"]["10"] = {
            "inputs": {
                "image": "INPUT_IMAGE_PATH",
                "upload": "image"
            },
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"}
        }
        
        workflow["prompt"]["11"] = {
            "inputs": {
                "image": "MASK_IMAGE_PATH",
                "upload": "image"
            },
            "class_type": "LoadImage",
            "_meta": {"title": "Load Mask"}
        }
        
        # Replace empty latent with masked image
        workflow["prompt"]["5"] = {
            "inputs": {
                "grow_mask_by": 6,
                "pixels": ["10", 0],
                "vae": ["4", 2],
                "mask": ["11", 0]
            },
            "class_type": "VAEEncodeForInpaint",
            "_meta": {"title": "VAE Encode (Inpaint)"}
        }
        
        workflow["meta"]["type"] = "inpaint"
        
        return workflow
    
    def _generate_controlnet_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ControlNet workflow."""
        workflow = self._generate_txt2img_workflow(params)
        
        # Add ControlNet
        workflow["prompt"]["10"] = {
            "inputs": {
                "control_net_name": "control_v11p_sd15_canny.pth"
            },
            "class_type": "ControlNetLoader",
            "_meta": {"title": "Load ControlNet"}
        }
        
        workflow["prompt"]["11"] = {
            "inputs": {
                "image": "CONTROL_IMAGE_PATH",
                "upload": "image"
            },
            "class_type": "LoadImage",
            "_meta": {"title": "Load Control Image"}
        }
        
        workflow["prompt"]["12"] = {
            "inputs": {
                "strength": 1.0,
                "conditioning": ["6", 0],
                "control_net": ["10", 0],
                "image": ["11", 0]
            },
            "class_type": "ControlNetApply",
            "_meta": {"title": "Apply ControlNet"}
        }
        
        # Update sampler to use ControlNet conditioning
        workflow["prompt"]["3"]["inputs"]["positive"] = ["12", 0]
        
        workflow["meta"]["type"] = "controlnet"
        
        return workflow
    
    def _generate_generic_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate generic workflow (defaults to txt2img)."""
        return self._generate_txt2img_workflow(params)
    
    async def execute_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute workflow in ComfyUI.
        
        Args:
            workflow: ComfyUI workflow JSON
            
        Returns:
            Execution result with output paths
        """
        try:
            # Get the prompt part of workflow
            prompt = workflow.get("prompt", workflow)
            
            # Generate client ID
            client_id = str(uuid.uuid4())
            
            # Queue the prompt
            response = await self.client.post(
                f"{self.comfyui_host}/prompt",
                json={
                    "prompt": prompt,
                    "client_id": client_id
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                prompt_id = result.get("prompt_id")
                
                logger.info(f"ComfyUI prompt queued: {prompt_id}")
                
                # Wait for completion (poll history)
                output = await self._wait_for_completion(prompt_id)
                
                return {
                    "success": True,
                    "prompt_id": prompt_id,
                    "output": output
                }
            else:
                logger.error(f"ComfyUI queue failed: {response.status_code}")
                return {
                    "success": False,
                    "error": f"Queue failed: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"ComfyUI execute error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _wait_for_completion(
        self,
        prompt_id: str,
        timeout: int = 300,
        poll_interval: int = 2
    ) -> Dict[str, Any]:
        """Wait for ComfyUI prompt to complete."""
        import asyncio
        
        elapsed = 0
        while elapsed < timeout:
            try:
                response = await self.client.get(
                    f"{self.comfyui_host}/history/{prompt_id}"
                )
                
                if response.status_code == 200:
                    history = response.json()
                    if prompt_id in history:
                        return history[prompt_id]
                
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                
            except Exception as e:
                logger.warning(f"Poll error: {e}")
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
        
        return {"status": "timeout", "elapsed": elapsed}
    
    async def get_models(self) -> List[str]:
        """Get available models in ComfyUI."""
        try:
            response = await self.client.get(
                f"{self.comfyui_host}/object_info/CheckpointLoaderSimple"
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("CheckpointLoaderSimple", {}).get(
                    "input", {}
                ).get("required", {}).get("ckpt_name", [[]])[0]
                return models
            return []
            
        except Exception as e:
            logger.error(f"Get models error: {e}")
            return []
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current ComfyUI queue status."""
        try:
            response = await self.client.get(f"{self.comfyui_host}/queue")
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            logger.error(f"Queue status error: {e}")
            return {}
