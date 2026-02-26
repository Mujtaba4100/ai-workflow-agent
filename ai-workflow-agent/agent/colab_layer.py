"""
Milestone 2: Full Colab Layer
- ColabCode Integration
- pyngrok Tunnel System
- Auto-Offload Logic
- Fallback Logic
"""

import asyncio
import os
import json
import logging
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================
# Enums and Data Classes
# ============================================================

class TaskComplexity(str, Enum):
    """Task complexity levels for auto-offload decision"""
    LIGHT = "light"           # Run locally
    MEDIUM = "medium"         # Can run either
    HEAVY = "heavy"           # Should offload to Colab
    GPU_REQUIRED = "gpu_required"  # Must use Colab


class ExecutionTarget(str, Enum):
    """Where to execute the task"""
    LOCAL = "local"
    COLAB = "colab"
    AUTO = "auto"  # Let system decide


class TaskStatus(str, Enum):
    """Status of task execution"""
    PENDING = "pending"
    CONNECTING = "connecting"
    TUNNELING = "tunneling"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    FALLBACK = "fallback"  # Running locally after Colab failure


class TunnelStatus(str, Enum):
    """Status of ngrok tunnel"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class TunnelInfo:
    """Information about an ngrok tunnel"""
    tunnel_id: str
    public_url: str
    local_port: int
    status: TunnelStatus
    created_at: datetime
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tunnel_id": self.tunnel_id,
            "public_url": self.public_url,
            "local_port": self.local_port,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }


@dataclass
class ColabTask:
    """A task to execute on Colab or locally"""
    task_id: str
    name: str
    code: str
    complexity: TaskComplexity
    status: TaskStatus
    target: ExecutionTarget
    actual_target: Optional[str] = None  # Where it actually ran
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    tunnel_info: Optional[TunnelInfo] = None
    fallback_used: bool = False
    execution_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "complexity": self.complexity.value,
            "status": self.status.value,
            "target": self.target.value,
            "actual_target": self.actual_target,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "tunnel_info": self.tunnel_info.to_dict() if self.tunnel_info else None,
            "fallback_used": self.fallback_used,
            "execution_time_seconds": self.execution_time_seconds
        }


# ============================================================
# pyngrok Tunnel System
# ============================================================

class TunnelManager:
    """
    Manage ngrok tunnels for Colab communication.
    Uses pyngrok for tunnel creation and management.
    """
    
    def __init__(self, auth_token: Optional[str] = None):
        self.auth_token = auth_token or os.getenv("NGROK_AUTH_TOKEN")
        self.tunnels: Dict[str, TunnelInfo] = {}
        self._ngrok_process = None
        self._initialized = False
        
    def _ensure_pyngrok(self) -> bool:
        """Ensure pyngrok is available and configured"""
        try:
            from pyngrok import ngrok, conf
            
            if self.auth_token:
                conf.get_default().auth_token = self.auth_token
            
            self._initialized = True
            return True
        except ImportError:
            logger.warning("pyngrok not installed. Tunnel features unavailable.")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize pyngrok: {e}")
            return False
    
    def create_tunnel(self, local_port: int = 8000, protocol: str = "http") -> Optional[TunnelInfo]:
        """
        Create an ngrok tunnel to expose local port.
        
        Args:
            local_port: Local port to tunnel
            protocol: Protocol (http, tcp)
            
        Returns:
            TunnelInfo if successful, None otherwise
        """
        if not self._ensure_pyngrok():
            return None
            
        try:
            from pyngrok import ngrok
            
            # Create tunnel
            if protocol == "tcp":
                tunnel = ngrok.connect(local_port, "tcp")
            else:
                tunnel = ngrok.connect(local_port, "http")
            
            tunnel_id = f"tunnel_{uuid.uuid4().hex[:8]}"
            
            info = TunnelInfo(
                tunnel_id=tunnel_id,
                public_url=tunnel.public_url,
                local_port=local_port,
                status=TunnelStatus.CONNECTED,
                created_at=datetime.now()
            )
            
            self.tunnels[tunnel_id] = info
            logger.info(f"Tunnel created: {info.public_url} -> localhost:{local_port}")
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to create tunnel: {e}")
            return None
    
    def close_tunnel(self, tunnel_id: str) -> bool:
        """Close a specific tunnel"""
        if tunnel_id not in self.tunnels:
            return False
            
        try:
            from pyngrok import ngrok
            
            tunnel_info = self.tunnels[tunnel_id]
            ngrok.disconnect(tunnel_info.public_url)
            
            tunnel_info.status = TunnelStatus.DISCONNECTED
            del self.tunnels[tunnel_id]
            
            logger.info(f"Tunnel {tunnel_id} closed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close tunnel: {e}")
            return False
    
    def close_all_tunnels(self) -> int:
        """Close all active tunnels"""
        closed = 0
        tunnel_ids = list(self.tunnels.keys())
        
        for tunnel_id in tunnel_ids:
            if self.close_tunnel(tunnel_id):
                closed += 1
                
        return closed
    
    def get_tunnel(self, tunnel_id: str) -> Optional[TunnelInfo]:
        """Get tunnel info by ID"""
        return self.tunnels.get(tunnel_id)
    
    def list_tunnels(self) -> List[TunnelInfo]:
        """List all active tunnels"""
        return list(self.tunnels.values())
    
    def get_status(self) -> Dict[str, Any]:
        """Get tunnel manager status"""
        return {
            "initialized": self._initialized,
            "auth_configured": bool(self.auth_token),
            "active_tunnels": len(self.tunnels),
            "tunnels": [t.to_dict() for t in self.tunnels.values()]
        }


# ============================================================
# ColabCode Integration
# ============================================================

class ColabCodeManager:
    """
    Integration with ColabCode for remote notebook execution.
    ColabCode allows running VS Code server on Google Colab.
    """
    
    def __init__(self, tunnel_manager: TunnelManager):
        self.tunnel_manager = tunnel_manager
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
    def generate_colabcode_notebook(
        self,
        port: int = 10000,
        password: Optional[str] = None,
        mount_drive: bool = False
    ) -> str:
        """
        Generate a Colab notebook that starts ColabCode server.
        
        Args:
            port: Port for VS Code server
            password: Optional password
            mount_drive: Whether to mount Google Drive
            
        Returns:
            Notebook JSON content
        """
        password_str = f'password="{password}"' if password else 'password=None'
        
        mount_code = """
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
""" if mount_drive else ""
        
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
                    "source": [
                        "# ColabCode - VS Code Server on Colab\n",
                        "This notebook starts a VS Code server for remote development."
                    ],
                    "metadata": {}
                },
                {
                    "cell_type": "code",
                    "source": [
                        "# Install ColabCode\n",
                        "!pip install colabcode -q\n"
                    ],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "source": mount_code.split('\n') if mount_drive else ["# Drive not mounted"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "source": [
                        "# Start ColabCode Server\n",
                        "from colabcode import ColabCode\n",
                        f"ColabCode(port={port}, {password_str})\n"
                    ],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": []
                }
            ]
        }
        
        return json.dumps(notebook, indent=2)
    
    def generate_task_notebook(
        self,
        task_code: str,
        task_name: str = "AI Task",
        callback_url: Optional[str] = None,
        requirements: Optional[List[str]] = None
    ) -> str:
        """
        Generate a notebook for running a specific task with callback.
        
        Args:
            task_code: Python code to execute
            task_name: Name of the task
            callback_url: URL to POST results to
            requirements: Additional pip packages
            
        Returns:
            Notebook JSON content
        """
        install_code = ""
        if requirements:
            install_code = "!pip install " + " ".join(requirements) + " -q"
        
        callback_code = ""
        if callback_url:
            callback_code = f'''
# Send results back
import requests
try:
    requests.post("{callback_url}", json={{"status": "completed", "result": result}})
except Exception as e:
    print(f"Callback failed: {{e}}")
'''
        
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
                    "source": [
                        f"# {task_name}\n",
                        "Auto-generated by AI Workflow Agent"
                    ],
                    "metadata": {}
                },
                {
                    "cell_type": "code",
                    "source": [
                        "# Check GPU availability\n",
                        "import torch\n",
                        "print(f'GPU Available: {torch.cuda.is_available()}')\n",
                        "if torch.cuda.is_available():\n",
                        "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n"
                    ],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "source": [install_code] if install_code else ["# No additional requirements"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "source": task_code.split('\n'),
                    "metadata": {},
                    "execution_count": None,
                    "outputs": []
                },
                {
                    "cell_type": "code",
                    "source": callback_code.split('\n') if callback_code else ["# No callback configured"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": []
                }
            ]
        }
        
        return json.dumps(notebook, indent=2)
    
    def create_session(
        self,
        session_name: str,
        port: int = 10000,
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a ColabCode session configuration"""
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        session = {
            "session_id": session_id,
            "name": session_name,
            "port": port,
            "password": password,
            "notebook": self.generate_colabcode_notebook(port, password),
            "created_at": datetime.now().isoformat(),
            "status": "ready"
        }
        
        self.active_sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        return self.active_sessions.get(session_id)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions"""
        return list(self.active_sessions.values())


# ============================================================
# Auto-Offload Logic
# ============================================================

class OffloadDecider:
    """
    Decides whether to run tasks locally or on Colab.
    Uses task complexity, resource requirements, and system state.
    """
    
    # Thresholds for auto-offload
    MEMORY_THRESHOLD_GB = 4  # Tasks needing > 4GB should go to Colab
    GPU_KEYWORDS = ["cuda", "torch", "tensorflow", "gpu", "model", "neural", "train"]
    HEAVY_TASK_KEYWORDS = ["training", "fine-tune", "finetune", "large", "batch", "video"]
    
    def __init__(self, local_gpu_available: bool = False):
        self.local_gpu_available = local_gpu_available
        self.colab_available = True  # Assume Colab is available
        self.decision_history: List[Dict[str, Any]] = []
        
    def analyze_task(self, code: str, metadata: Optional[Dict[str, Any]] = None) -> TaskComplexity:
        """
        Analyze task code to determine complexity.
        
        Args:
            code: Python code to analyze
            metadata: Optional metadata about the task
            
        Returns:
            TaskComplexity level
        """
        code_lower = code.lower()
        
        # Check for GPU requirements
        gpu_score = sum(1 for kw in self.GPU_KEYWORDS if kw in code_lower)
        
        # Check for heavy task indicators
        heavy_score = sum(1 for kw in self.HEAVY_TASK_KEYWORDS if kw in code_lower)
        
        # Check metadata for explicit requirements
        if metadata:
            if metadata.get("requires_gpu"):
                return TaskComplexity.GPU_REQUIRED
            if metadata.get("memory_gb", 0) > self.MEMORY_THRESHOLD_GB:
                return TaskComplexity.HEAVY
        
        # Determine complexity
        if gpu_score >= 2 or "cuda" in code_lower:
            return TaskComplexity.GPU_REQUIRED
        elif heavy_score >= 2 or gpu_score >= 1:
            return TaskComplexity.HEAVY
        elif heavy_score >= 1:
            return TaskComplexity.MEDIUM
        else:
            return TaskComplexity.LIGHT
    
    def decide_target(
        self,
        complexity: TaskComplexity,
        prefer_local: bool = False
    ) -> ExecutionTarget:
        """
        Decide where to execute based on complexity.
        
        Args:
            complexity: Task complexity level
            prefer_local: User preference for local execution
            
        Returns:
            ExecutionTarget
        """
        decision = {
            "complexity": complexity.value,
            "prefer_local": prefer_local,
            "local_gpu": self.local_gpu_available,
            "colab_available": self.colab_available,
            "timestamp": datetime.now().isoformat()
        }
        
        # GPU required - must use Colab unless local GPU available
        if complexity == TaskComplexity.GPU_REQUIRED:
            if self.local_gpu_available:
                target = ExecutionTarget.LOCAL
            else:
                target = ExecutionTarget.COLAB
                
        # Heavy tasks - prefer Colab
        elif complexity == TaskComplexity.HEAVY:
            if prefer_local and self.local_gpu_available:
                target = ExecutionTarget.LOCAL
            else:
                target = ExecutionTarget.COLAB
                
        # Medium tasks - based on preference
        elif complexity == TaskComplexity.MEDIUM:
            target = ExecutionTarget.LOCAL if prefer_local else ExecutionTarget.AUTO
            
        # Light tasks - always local
        else:
            target = ExecutionTarget.LOCAL
            
        decision["target"] = target.value
        self.decision_history.append(decision)
        
        return target
    
    def should_offload(self, code: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Simple check if task should be offloaded to Colab"""
        complexity = self.analyze_task(code, metadata)
        target = self.decide_target(complexity)
        return target == ExecutionTarget.COLAB
    
    def get_decision_stats(self) -> Dict[str, Any]:
        """Get statistics about offload decisions"""
        if not self.decision_history:
            return {"total_decisions": 0}
            
        local_count = sum(1 for d in self.decision_history if d["target"] == "local")
        colab_count = sum(1 for d in self.decision_history if d["target"] == "colab")
        
        return {
            "total_decisions": len(self.decision_history),
            "local_executions": local_count,
            "colab_executions": colab_count,
            "offload_rate": colab_count / len(self.decision_history) if self.decision_history else 0
        }


# ============================================================
# Fallback Logic
# ============================================================

class FallbackHandler:
    """
    Handles fallback to local execution when Colab fails.
    """
    
    MAX_COLAB_RETRIES = 2
    COLAB_TIMEOUT_SECONDS = 300  # 5 minutes
    
    def __init__(self):
        self.fallback_history: List[Dict[str, Any]] = []
        
    async def execute_with_fallback(
        self,
        task: ColabTask,
        colab_executor: Callable,
        local_executor: Callable,
        timeout: int = COLAB_TIMEOUT_SECONDS
    ) -> ColabTask:
        """
        Try to execute on Colab, fall back to local if it fails.
        
        Args:
            task: Task to execute
            colab_executor: Async function to run on Colab
            local_executor: Async function to run locally
            timeout: Timeout in seconds for Colab execution
            
        Returns:
            Updated task with results
        """
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        fallback_record = {
            "task_id": task.task_id,
            "started_at": task.started_at.isoformat(),
            "attempts": []
        }
        
        # Try Colab first if target is COLAB or AUTO
        if task.target in [ExecutionTarget.COLAB, ExecutionTarget.AUTO]:
            for attempt in range(self.MAX_COLAB_RETRIES):
                try:
                    fallback_record["attempts"].append({
                        "attempt": attempt + 1,
                        "target": "colab",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    task.status = TaskStatus.CONNECTING
                    
                    # Execute on Colab with timeout
                    result = await asyncio.wait_for(
                        colab_executor(task),
                        timeout=timeout
                    )
                    
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.actual_target = "colab"
                    task.completed_at = datetime.now()
                    
                    fallback_record["attempts"][-1]["success"] = True
                    self.fallback_history.append(fallback_record)
                    
                    return task
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Colab execution timed out (attempt {attempt + 1})")
                    fallback_record["attempts"][-1]["error"] = "timeout"
                    
                except Exception as e:
                    logger.warning(f"Colab execution failed: {e}")
                    fallback_record["attempts"][-1]["error"] = str(e)
        
        # Fallback to local execution
        logger.info("Falling back to local execution")
        task.status = TaskStatus.FALLBACK
        task.fallback_used = True
        
        try:
            fallback_record["attempts"].append({
                "attempt": len(fallback_record["attempts"]) + 1,
                "target": "local",
                "timestamp": datetime.now().isoformat()
            })
            
            result = await local_executor(task)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.actual_target = "local"
            task.completed_at = datetime.now()
            
            fallback_record["attempts"][-1]["success"] = True
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            fallback_record["attempts"][-1]["error"] = str(e)
            
        self.fallback_history.append(fallback_record)
        
        # Calculate execution time
        if task.completed_at and task.started_at:
            task.execution_time_seconds = (task.completed_at - task.started_at).total_seconds()
            
        return task
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """Get fallback statistics"""
        if not self.fallback_history:
            return {"total_tasks": 0}
            
        fallback_used = sum(1 for r in self.fallback_history 
                          if any(a.get("target") == "local" for a in r["attempts"]))
        
        return {
            "total_tasks": len(self.fallback_history),
            "fallback_used_count": fallback_used,
            "fallback_rate": fallback_used / len(self.fallback_history)
        }


# ============================================================
# Main Colab Layer Manager
# ============================================================

class ColabLayer:
    """
    Main coordinator for the Colab Layer.
    Integrates tunnel, ColabCode, offload decisions, and fallback.
    """
    
    def __init__(self, ngrok_token: Optional[str] = None, local_gpu: bool = False):
        self.tunnel_manager = TunnelManager(ngrok_token)
        self.colabcode = ColabCodeManager(self.tunnel_manager)
        self.offload_decider = OffloadDecider(local_gpu)
        self.fallback_handler = FallbackHandler()
        self.tasks: Dict[str, ColabTask] = {}
        
    def create_task(
        self,
        name: str,
        code: str,
        target: ExecutionTarget = ExecutionTarget.AUTO,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ColabTask:
        """
        Create a new task with automatic complexity analysis.
        
        Args:
            name: Task name
            code: Python code to execute
            target: Where to execute (AUTO for automatic decision)
            metadata: Optional metadata about requirements
            
        Returns:
            ColabTask object
        """
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        # Analyze complexity
        complexity = self.offload_decider.analyze_task(code, metadata)
        
        # Decide target if AUTO
        if target == ExecutionTarget.AUTO:
            target = self.offload_decider.decide_target(complexity)
        
        task = ColabTask(
            task_id=task_id,
            name=name,
            code=code,
            complexity=complexity,
            status=TaskStatus.PENDING,
            target=target
        )
        
        self.tasks[task_id] = task
        logger.info(f"Task created: {task_id}, complexity={complexity.value}, target={target.value}")
        
        return task
    
    async def execute_task(self, task_id: str) -> ColabTask:
        """
        Execute a task with automatic offload/fallback.
        
        Args:
            task_id: ID of task to execute
            
        Returns:
            Updated task with results
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        
        async def colab_executor(t: ColabTask) -> Any:
            """Execute on Colab (generates notebook for manual execution)"""
            # In a full implementation, this would:
            # 1. Upload notebook to Colab via API
            # 2. Execute and wait for callback
            # For now, generate notebook and return instructions
            
            notebook = self.colabcode.generate_task_notebook(
                t.code,
                t.name,
                callback_url=None  # Would be tunnel URL
            )
            
            return {
                "execution_type": "colab_notebook",
                "notebook": notebook,
                "instructions": "Upload this notebook to Google Colab and run all cells"
            }
        
        async def local_executor(t: ColabTask) -> Any:
            """Execute locally"""
            # Create temp file and execute
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(t.code)
                temp_path = f.name
            
            try:
                # Execute the code
                result = subprocess.run(
                    ["python", temp_path],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                return {
                    "execution_type": "local",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                }
            finally:
                os.unlink(temp_path)
        
        # Execute with fallback
        return await self.fallback_handler.execute_with_fallback(
            task,
            colab_executor,
            local_executor
        )
    
    def get_task(self, task_id: str) -> Optional[ColabTask]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[ColabTask]:
        """List all tasks, optionally filtered by status"""
        tasks = list(self.tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return tasks
    
    def generate_colab_notebook(
        self,
        task_id: str,
        callback_url: Optional[str] = None
    ) -> Optional[str]:
        """Generate a Colab notebook for a task"""
        task = self.tasks.get(task_id)
        if not task:
            return None
            
        return self.colabcode.generate_task_notebook(
            task.code,
            task.name,
            callback_url
        )
    
    def setup_tunnel(self, port: int = 8000) -> Optional[TunnelInfo]:
        """Set up ngrok tunnel for Colab callbacks"""
        return self.tunnel_manager.create_tunnel(port)
    
    def get_tunnel_url(self) -> Optional[str]:
        """Get the first active tunnel URL"""
        tunnels = self.tunnel_manager.list_tunnels()
        return tunnels[0].public_url if tunnels else None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            "tasks": {
                "total": len(self.tasks),
                "pending": len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
                "completed": len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]),
                "failed": len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED])
            },
            "offload": self.offload_decider.get_decision_stats(),
            "fallback": self.fallback_handler.get_fallback_stats(),
            "tunnels": self.tunnel_manager.get_status()
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.tunnel_manager.close_all_tunnels()


# ============================================================
# Singleton Instance
# ============================================================

_colab_layer: Optional[ColabLayer] = None


def get_colab_layer(ngrok_token: Optional[str] = None, local_gpu: bool = False) -> ColabLayer:
    """Get or create the Colab Layer singleton"""
    global _colab_layer
    if _colab_layer is None:
        _colab_layer = ColabLayer(ngrok_token, local_gpu)
    return _colab_layer
