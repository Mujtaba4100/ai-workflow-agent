# Colab Integration - Basic Tunnel and Remote Execution
"""
Basic Colab integration for Phase 0.
Full implementation in Milestone 2.
"""

import httpx
import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


class ColabConnector:
    """
    Basic Colab connector for detecting heavy tasks and establishing tunnels.
    Phase 0: Simple implementation
    Milestone 2: Full ColabCode + pyngrok integration
    """
    
    def __init__(self):
        self.ngrok_token = settings.NGROK_AUTH_TOKEN
        self.client = httpx.AsyncClient(timeout=60.0)
        self.is_connected = False
        self.tunnel_url: Optional[str] = None
    
    async def should_offload(self, task_description: str) -> Dict[str, Any]:
        """
        Decide if task should be offloaded to Colab.
        
        Criteria for offloading:
        - Large model inference (>7B parameters)
        - Batch image generation (>5 images)
        - Video generation
        - Training tasks
        - Explicit user request
        
        Args:
            task_description: Description of the task
            
        Returns:
            Dict with offload decision and reason
        """
        task_lower = task_description.lower()
        
        # Keywords indicating heavy computation
        heavy_keywords = [
            # Large models
            "70b", "40b", "13b", "llama 2", "mixtral", "flux",
            # Batch operations
            "batch", "multiple images", "generate 10", "generate 20",
            # Video
            "video", "animation", "animatediff",
            # Training
            "train", "fine-tune", "finetune", "lora training",
            # Explicit
            "use colab", "offload", "remote", "not local", "save gpu"
        ]
        
        # Keywords indicating light computation (should stay local)
        light_keywords = [
            "single image", "quick", "fast", "small model", "7b", "3b",
            "test", "preview"
        ]
        
        # Check for heavy keywords
        heavy_score = sum(1 for kw in heavy_keywords if kw in task_lower)
        light_score = sum(1 for kw in light_keywords if kw in task_lower)
        
        should_offload = heavy_score > light_score and heavy_score >= 1
        
        if should_offload:
            reason = f"Task appears heavy (score: {heavy_score}). Detected: "
            detected = [kw for kw in heavy_keywords if kw in task_lower]
            reason += ", ".join(detected[:3])
        else:
            reason = "Task can run locally"
        
        return {
            "should_offload": should_offload,
            "confidence": min(heavy_score / 3, 1.0),
            "reason": reason,
            "heavy_score": heavy_score,
            "light_score": light_score
        }
    
    async def setup_tunnel(self) -> Dict[str, Any]:
        """
        Setup ngrok tunnel for Colab communication.
        
        NOTE: Requires ngrok to be installed and configured.
        Full implementation in Milestone 2.
        """
        if not self.ngrok_token:
            return {
                "success": False,
                "message": "NGROK_AUTH_TOKEN not configured"
            }
        
        try:
            # Import pyngrok
            from pyngrok import ngrok, conf
            
            # Configure ngrok
            conf.get_default().auth_token = self.ngrok_token
            
            # Start tunnel to local agent
            tunnel = ngrok.connect(8000, "http")
            self.tunnel_url = tunnel.public_url
            self.is_connected = True
            
            logger.info(f"Ngrok tunnel established: {self.tunnel_url}")
            
            return {
                "success": True,
                "tunnel_url": self.tunnel_url,
                "message": "Tunnel established successfully"
            }
            
        except ImportError:
            return {
                "success": False,
                "message": "pyngrok not installed. Run: pip install pyngrok"
            }
        except Exception as e:
            logger.error(f"Tunnel setup error: {e}")
            return {
                "success": False,
                "message": f"Tunnel setup failed: {str(e)}"
            }
    
    async def close_tunnel(self) -> bool:
        """Close the ngrok tunnel."""
        try:
            from pyngrok import ngrok
            ngrok.disconnect(self.tunnel_url)
            ngrok.kill()
            self.is_connected = False
            self.tunnel_url = None
            return True
        except Exception as e:
            logger.error(f"Tunnel close error: {e}")
            return False
    
    def generate_colab_notebook(
        self,
        task_type: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a Colab notebook for remote execution.
        
        Args:
            task_type: Type of task (comfyui, llm, training)
            params: Task parameters
            
        Returns:
            Notebook JSON (ipynb format)
        """
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 0,
            "metadata": {
                "colab": {
                    "name": f"AI_Agent_{task_type}",
                    "provenance": []
                },
                "kernelspec": {
                    "name": "python3",
                    "display_name": "Python 3"
                },
                "accelerator": "GPU"
            },
            "cells": []
        }
        
        # Add setup cell
        setup_code = """# Auto-generated by AI Workflow Agent
# Setup and dependencies

!pip install -q torch torchvision
!pip install -q pyngrok httpx

# Connect to local agent
import os
from pyngrok import ngrok

# Your tunnel URL will be used to send results back
AGENT_URL = "{tunnel_url}"
"""
        
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": setup_code.format(tunnel_url=self.tunnel_url or "http://localhost:8000")
        })
        
        # Add task-specific cells
        if task_type == "comfyui":
            notebook["cells"].extend(self._generate_comfyui_cells(params))
        elif task_type == "llm":
            notebook["cells"].extend(self._generate_llm_cells(params))
        else:
            notebook["cells"].extend(self._generate_generic_cells(params))
        
        return notebook
    
    def _generate_comfyui_cells(self, params: Dict[str, Any]) -> list:
        """Generate ComfyUI execution cells for Colab."""
        cells = []
        
        # Install ComfyUI
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": """# Install ComfyUI
%cd /content
!git clone https://github.com/comfyanonymous/ComfyUI.git
%cd ComfyUI
!pip install -q -r requirements.txt
"""
        })
        
        # Download models if specified
        checkpoint = params.get("checkpoint", "v1-5-pruned-emaonly.safetensors")
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": f"""# Download model
!wget -q -O models/checkpoints/{checkpoint} \\
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/{checkpoint}"
"""
        })
        
        # Execute workflow
        workflow_json = json.dumps(params.get("workflow", {}), indent=2)
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": f"""# Execute workflow
import json
import httpx

workflow = {workflow_json}

# Start ComfyUI server (background)
import subprocess
import time
proc = subprocess.Popen(['python', 'main.py', '--listen', '0.0.0.0'])
time.sleep(30)  # Wait for server to start

# Queue prompt
response = httpx.post('http://localhost:8188/prompt', json={{"prompt": workflow}})
print(response.json())
"""
        })
        
        return cells
    
    def _generate_llm_cells(self, params: Dict[str, Any]) -> list:
        """Generate LLM inference cells for Colab."""
        cells = []
        
        model = params.get("model", "meta-llama/Llama-2-7b-hf")
        prompt = params.get("prompt", "Hello, how are you?")
        
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": f"""# Install transformers
!pip install -q transformers accelerate bitsandbytes

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model (4-bit quantization for memory efficiency)
model_name = "{model}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True
)

# Generate
prompt = "{prompt}"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
"""
        })
        
        return cells
    
    def _generate_generic_cells(self, params: Dict[str, Any]) -> list:
        """Generate generic execution cells."""
        code = params.get("code", "print('Hello from Colab!')")
        
        return [{
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code
        }]
    
    async def check_colab_availability(self) -> Dict[str, Any]:
        """
        Check if Colab is available and has GPU.
        
        NOTE: This is a placeholder - actual implementation
        requires Google OAuth and Colab API access.
        """
        return {
            "available": True,
            "gpu_type": "T4",  # Typical free tier
            "note": "Actual availability check requires Colab runtime"
        }


# Singleton instance
colab_connector = ColabConnector()
