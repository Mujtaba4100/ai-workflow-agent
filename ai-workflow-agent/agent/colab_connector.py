"""
Google Colab Integration - M3 Feature
Offload heavy computation tasks to Google Colab notebooks
"""

import asyncio
import uuid
import json
import logging
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ColabTaskType(str, Enum):
    """Types of tasks that can be offloaded to Colab"""
    IMAGE_GENERATION = "image_generation"
    MODEL_TRAINING = "model_training"
    DATA_PROCESSING = "data_processing"
    VIDEO_GENERATION = "video_generation"
    AUDIO_PROCESSING = "audio_processing"
    LLM_INFERENCE = "llm_inference"
    CUSTOM = "custom"


class ColabStatus(str, Enum):
    """Status of Colab task"""
    PENDING = "pending"
    CONNECTING = "connecting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ColabTask:
    """A task to be executed on Google Colab"""
    task_id: str
    task_type: ColabTaskType
    status: ColabStatus
    notebook_code: str
    parameters: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    runtime_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "runtime_info": self.runtime_info
        }


class ColabNotebookGenerator:
    """Generate Colab notebooks for various tasks"""
    
    # Base notebook template
    BASE_TEMPLATE = '''
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": ["# AI Workflow Agent - Colab Task\\n", "Auto-generated notebook for: {task_name}"],
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": {setup_code},
      "metadata": {},
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": {main_code},
      "metadata": {},
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": {callback_code},
      "metadata": {},
      "execution_count": null,
      "outputs": []
    }
  ]
}
'''
    
    @classmethod
    def generate_image_generation_notebook(
        cls,
        prompt: str,
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        callback_url: Optional[str] = None
    ) -> str:
        """Generate notebook for Stable Diffusion image generation"""
        
        setup_code = [
            "# Install dependencies",
            "!pip install -q diffusers transformers accelerate torch",
            "!pip install -q requests pillow",
            "",
            "import torch",
            "from diffusers import StableDiffusionXLPipeline",
            "from PIL import Image",
            "import requests",
            "import base64",
            "from io import BytesIO",
            "",
            f'MODEL_ID = "{model}"',
            "",
            "# Load model with GPU",
            "pipe = StableDiffusionXLPipeline.from_pretrained(",
            '    MODEL_ID,',
            '    torch_dtype=torch.float16,',
            '    use_safetensors=True,',
            '    variant="fp16"',
            ")",
            "pipe.to('cuda')",
            "print('Model loaded successfully!')"
        ]
        
        main_code = [
            "# Generate image",
            f'prompt = """{prompt}"""',
            f"width = {width}",
            f"height = {height}",
            f"steps = {steps}",
            "",
            "image = pipe(",
            "    prompt=prompt,",
            "    width=width,",
            "    height=height,",
            "    num_inference_steps=steps",
            ").images[0]",
            "",
            "# Save image",
            "image.save('generated_image.png')",
            "print('Image generated successfully!')",
            "",
            "# Display",
            "from IPython.display import display",
            "display(image)"
        ]
        
        callback_code = cls._generate_callback_code(callback_url, "image")
        
        return cls._build_notebook("Image Generation", setup_code, main_code, callback_code)
    
    @classmethod
    def generate_llm_inference_notebook(
        cls,
        model: str = "microsoft/phi-2",
        prompt: str = "",
        max_tokens: int = 512,
        callback_url: Optional[str] = None
    ) -> str:
        """Generate notebook for LLM inference"""
        
        setup_code = [
            "# Install dependencies",
            "!pip install -q transformers accelerate torch bitsandbytes",
            "!pip install -q requests",
            "",
            "import torch",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "import requests",
            "",
            f'MODEL_ID = "{model}"',
            "",
            "# Load model",
            "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)",
            "model = AutoModelForCausalLM.from_pretrained(",
            "    MODEL_ID,",
            "    torch_dtype=torch.float16,",
            "    device_map='auto',",
            "    trust_remote_code=True",
            ")",
            "print('Model loaded!')"
        ]
        
        main_code = [
            "# Run inference",
            f'prompt = """{prompt}"""',
            f"max_tokens = {max_tokens}",
            "",
            "inputs = tokenizer(prompt, return_tensors='pt').to('cuda')",
            "outputs = model.generate(",
            "    **inputs,",
            "    max_new_tokens=max_tokens,",
            "    do_sample=True,",
            "    temperature=0.7",
            ")",
            "",
            "response = tokenizer.decode(outputs[0], skip_special_tokens=True)",
            "print('Response:')",
            "print(response)"
        ]
        
        callback_code = cls._generate_callback_code(callback_url, "text")
        
        return cls._build_notebook("LLM Inference", setup_code, main_code, callback_code)
    
    @classmethod
    def generate_training_notebook(
        cls,
        model_type: str = "image_classifier",
        dataset_url: Optional[str] = None,
        epochs: int = 10,
        callback_url: Optional[str] = None
    ) -> str:
        """Generate notebook for model training"""
        
        setup_code = [
            "# Install dependencies",
            "!pip install -q torch torchvision",
            "!pip install -q requests tqdm",
            "",
            "import torch",
            "import torch.nn as nn",
            "from torch.utils.data import DataLoader",
            "from torchvision import datasets, transforms, models",
            "from tqdm import tqdm",
            "import requests",
            "",
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
            "print(f'Using device: {device}')"
        ]
        
        main_code = [
            f"# Training configuration",
            f"epochs = {epochs}",
            f'model_type = "{model_type}"',
            "",
            "# Load pretrained model",
            "model = models.resnet18(pretrained=True)",
            "model.fc = nn.Linear(512, 10)  # Adjust for your classes",
            "model = model.to(device)",
            "",
            "# Training setup",
            "criterion = nn.CrossEntropyLoss()",
            "optimizer = torch.optim.Adam(model.parameters(), lr=0.001)",
            "",
            "# Training loop (placeholder - add your data)",
            "print(f'Model ready for training on {device}')",
            "print(f'Epochs: {epochs}')",
            "",
            "# Save model",
            "torch.save(model.state_dict(), 'trained_model.pth')",
            "print('Model saved!')"
        ]
        
        callback_code = cls._generate_callback_code(callback_url, "model")
        
        return cls._build_notebook("Model Training", setup_code, main_code, callback_code)
    
    @classmethod
    def generate_video_generation_notebook(
        cls,
        prompt: str,
        num_frames: int = 16,
        callback_url: Optional[str] = None
    ) -> str:
        """Generate notebook for video generation"""
        
        setup_code = [
            "# Install dependencies",
            "!pip install -q diffusers transformers accelerate torch",
            "!pip install -q imageio requests",
            "",
            "import torch",
            "from diffusers import DiffusionPipeline",
            "from diffusers.utils import export_to_video",
            "import requests",
            "",
            "# Load video generation model",
            "pipe = DiffusionPipeline.from_pretrained(",
            '    "damo-vilab/text-to-video-ms-1.7b",',
            "    torch_dtype=torch.float16,",
            "    variant='fp16'",
            ")",
            "pipe.to('cuda')",
            "print('Video model loaded!')"
        ]
        
        main_code = [
            "# Generate video",
            f'prompt = """{prompt}"""',
            f"num_frames = {num_frames}",
            "",
            "video_frames = pipe(",
            "    prompt,",
            "    num_inference_steps=25,",
            "    num_frames=num_frames",
            ").frames",
            "",
            "# Export video",
            "video_path = export_to_video(video_frames)",
            "print(f'Video saved to: {video_path}')"
        ]
        
        callback_code = cls._generate_callback_code(callback_url, "video")
        
        return cls._build_notebook("Video Generation", setup_code, main_code, callback_code)
    
    @classmethod
    def generate_custom_notebook(
        cls,
        task_name: str,
        setup_code: List[str],
        main_code: List[str],
        callback_url: Optional[str] = None
    ) -> str:
        """Generate custom notebook with provided code"""
        callback_code = cls._generate_callback_code(callback_url, "custom")
        return cls._build_notebook(task_name, setup_code, main_code, callback_code)
    
    @classmethod
    def _generate_callback_code(cls, callback_url: Optional[str], result_type: str) -> List[str]:
        """Generate callback code to send results back"""
        if not callback_url:
            return ["# No callback configured", "print('Task complete!')"]
        
        return [
            "# Send results back to agent",
            f'callback_url = "{callback_url}"',
            "",
            "try:",
            "    result_data = {",
            f'        "type": "{result_type}",',
            '        "status": "completed",',
            '        "message": "Task completed successfully"',
            "    }",
            "",
            f'    if "{result_type}" == "image":',
            "        # Encode image as base64",
            "        from io import BytesIO",
            "        import base64",
            "        buffer = BytesIO()",
            "        image.save(buffer, format='PNG')",
            "        result_data['image_base64'] = base64.b64encode(buffer.getvalue()).decode()",
            "",
            f'    elif "{result_type}" == "text":',
            "        result_data['response'] = response",
            "",
            "    requests.post(callback_url, json=result_data, timeout=30)",
            "    print('Results sent to callback!')",
            "except Exception as e:",
            "    print(f'Callback failed: {e}')"
        ]
    
    @classmethod
    def _build_notebook(
        cls,
        task_name: str,
        setup_code: List[str],
        main_code: List[str],
        callback_code: List[str]
    ) -> str:
        """Build the complete notebook JSON"""
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 0,
            "metadata": {
                "colab": {
                    "provenance": [],
                    "gpuType": "T4"
                },
                "kernelspec": {
                    "name": "python3",
                    "display_name": "Python 3"
                },
                "accelerator": "GPU"
            },
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": [f"# AI Workflow Agent - {task_name}\n", "Auto-generated notebook"],
                    "metadata": {}
                },
                {
                    "cell_type": "code",
                    "source": setup_code,
                    "metadata": {},
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "source": main_code,
                    "metadata": {},
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "source": callback_code,
                    "metadata": {},
                    "execution_count": None,
                    "outputs": []
                }
            ]
        }
        return json.dumps(notebook, indent=2)


class ColabConnector:
    """
    Connect to Google Colab for offloading heavy tasks.
    Manages task creation, notebook generation, and result handling.
    """
    
    def __init__(self, callback_base_url: str = "http://localhost:8000"):
        self.callback_base_url = callback_base_url
        self.notebook_generator = ColabNotebookGenerator()
        self._tasks: Dict[str, ColabTask] = {}
        self._results: Dict[str, Any] = {}
        
    def should_offload(self, task_type: str, parameters: Dict[str, Any]) -> bool:
        """
        Determine if a task should be offloaded to Colab.
        
        Decision factors:
        - Local GPU availability
        - Task complexity (VRAM requirements)
        - Model size
        """
        # Tasks that typically need offloading
        heavy_tasks = {
            "sdxl": True,  # SDXL needs >8GB VRAM
            "video_generation": True,  # Video gen is heavy
            "llm_70b": True,  # Large LLMs
            "training": True,  # Training usually needs more
        }
        
        # Check task type
        if task_type in heavy_tasks:
            return True
            
        # Check VRAM requirements
        vram_required = parameters.get("vram_gb", 0)
        if vram_required > 8:  # Typical consumer GPU limit
            return True
            
        # Check model size
        model_params = parameters.get("model_params_b", 0)
        if model_params > 7:  # >7B parameters
            return True
            
        return False
    
    def create_task(
        self,
        task_type: ColabTaskType,
        parameters: Dict[str, Any]
    ) -> ColabTask:
        """Create a new Colab task"""
        task_id = str(uuid.uuid4())
        
        # Generate notebook based on task type
        notebook_code = self._generate_notebook(task_type, parameters, task_id)
        
        task = ColabTask(
            task_id=task_id,
            task_type=task_type,
            status=ColabStatus.PENDING,
            notebook_code=notebook_code,
            parameters=parameters,
            created_at=datetime.now()
        )
        
        self._tasks[task_id] = task
        logger.info(f"Created Colab task: {task_id} ({task_type.value})")
        
        return task
    
    def _generate_notebook(
        self,
        task_type: ColabTaskType,
        parameters: Dict[str, Any],
        task_id: str
    ) -> str:
        """Generate appropriate notebook for task type"""
        callback_url = f"{self.callback_base_url}/colab/callback/{task_id}"
        
        if task_type == ColabTaskType.IMAGE_GENERATION:
            return ColabNotebookGenerator.generate_image_generation_notebook(
                prompt=parameters.get("prompt", ""),
                model=parameters.get("model", "stabilityai/stable-diffusion-xl-base-1.0"),
                width=parameters.get("width", 1024),
                height=parameters.get("height", 1024),
                steps=parameters.get("steps", 30),
                callback_url=callback_url
            )
            
        elif task_type == ColabTaskType.LLM_INFERENCE:
            return ColabNotebookGenerator.generate_llm_inference_notebook(
                model=parameters.get("model", "microsoft/phi-2"),
                prompt=parameters.get("prompt", ""),
                max_tokens=parameters.get("max_tokens", 512),
                callback_url=callback_url
            )
            
        elif task_type == ColabTaskType.MODEL_TRAINING:
            return ColabNotebookGenerator.generate_training_notebook(
                model_type=parameters.get("model_type", "image_classifier"),
                epochs=parameters.get("epochs", 10),
                callback_url=callback_url
            )
            
        elif task_type == ColabTaskType.VIDEO_GENERATION:
            return ColabNotebookGenerator.generate_video_generation_notebook(
                prompt=parameters.get("prompt", ""),
                num_frames=parameters.get("num_frames", 16),
                callback_url=callback_url
            )
            
        else:
            # Custom task
            return ColabNotebookGenerator.generate_custom_notebook(
                task_name=parameters.get("task_name", "Custom Task"),
                setup_code=parameters.get("setup_code", ["# Setup"]),
                main_code=parameters.get("main_code", ["# Main code"]),
                callback_url=callback_url
            )
    
    def get_task(self, task_id: str) -> Optional[ColabTask]:
        """Get task by ID"""
        return self._tasks.get(task_id)
    
    def get_tasks(self, status: Optional[ColabStatus] = None) -> List[ColabTask]:
        """Get all tasks, optionally filtered by status"""
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)
    
    def update_task_status(
        self,
        task_id: str,
        status: ColabStatus,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> Optional[ColabTask]:
        """Update task status (called by callback)"""
        task = self._tasks.get(task_id)
        if not task:
            return None
            
        task.status = status
        
        if status == ColabStatus.RUNNING and not task.started_at:
            task.started_at = datetime.now()
            
        if status in [ColabStatus.COMPLETED, ColabStatus.FAILED, ColabStatus.TIMEOUT]:
            task.completed_at = datetime.now()
            
        if result:
            task.result = result
            self._results[task_id] = result
            
        if error:
            task.error = error
            
        logger.info(f"Task {task_id} status: {status.value}")
        return task
    
    def receive_callback(
        self,
        task_id: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process callback from Colab notebook"""
        status_str = payload.get("status", "completed")
        
        if status_str == "completed":
            status = ColabStatus.COMPLETED
        elif status_str == "failed":
            status = ColabStatus.FAILED
        else:
            status = ColabStatus.RUNNING
            
        self.update_task_status(
            task_id=task_id,
            status=status,
            result=payload,
            error=payload.get("error")
        )
        
        return {"received": True, "task_id": task_id}
    
    def get_notebook_download_url(self, task_id: str) -> Optional[str]:
        """Get URL to download notebook (for manual upload to Colab)"""
        task = self._tasks.get(task_id)
        if not task:
            return None
        return f"/colab/notebook/{task_id}"
    
    def get_notebook_content(self, task_id: str) -> Optional[str]:
        """Get notebook content for download"""
        task = self._tasks.get(task_id)
        if task:
            return task.notebook_code
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        task = self._tasks.get(task_id)
        if not task:
            return False
            
        if task.status in [ColabStatus.PENDING, ColabStatus.RUNNING]:
            task.status = ColabStatus.CANCELLED
            task.completed_at = datetime.now()
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Colab task statistics"""
        tasks = list(self._tasks.values())
        
        by_status = {}
        by_type = {}
        total_duration = 0
        completed_count = 0
        
        for task in tasks:
            # By status
            status = task.status.value
            by_status[status] = by_status.get(status, 0) + 1
            
            # By type
            ttype = task.task_type.value
            by_type[ttype] = by_type.get(ttype, 0) + 1
            
            # Duration
            if task.started_at and task.completed_at:
                duration = (task.completed_at - task.started_at).total_seconds()
                total_duration += duration
                completed_count += 1
                
        return {
            "total_tasks": len(tasks),
            "by_status": by_status,
            "by_type": by_type,
            "average_duration_seconds": total_duration / completed_count if completed_count > 0 else 0,
            "pending_count": by_status.get("pending", 0),
            "running_count": by_status.get("running", 0)
        }


# Singleton instance
_connector: Optional[ColabConnector] = None


def get_colab_connector() -> ColabConnector:
    """Get the global Colab connector instance"""
    global _connector
    if _connector is None:
        _connector = ColabConnector()
    return _connector
