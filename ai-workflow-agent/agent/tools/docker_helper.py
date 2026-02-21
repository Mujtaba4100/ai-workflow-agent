# Docker Helper Tool
"""
Clone repositories and manage Docker builds.
Includes error analysis and fix suggestions.
"""

import httpx
import subprocess
import os
import shutil
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


class DockerHelper:
    """
    Docker automation helper.
    Clones repos, builds containers, analyzes errors, suggests fixes.
    """
    
    def __init__(self):
        self.projects_dir = Path(settings.PROJECTS_DIR)
        self.ollama_host = settings.OLLAMA_HOST
        self.ollama_model = settings.OLLAMA_MODEL
        self.client = httpx.AsyncClient(timeout=120.0)
        
        # Ensure projects directory exists
        self.projects_dir.mkdir(parents=True, exist_ok=True)
    
    async def clone_and_build(
        self,
        repo_url: str,
        branch: str = "main"
    ) -> Dict[str, Any]:
        """
        Clone repository and attempt Docker build.
        
        Args:
            repo_url: GitHub repository URL
            branch: Branch to clone (default: main)
            
        Returns:
            Dict with success status, logs, and fix suggestions
        """
        project_name = self._extract_project_name(repo_url)
        project_path = self.projects_dir / project_name
        
        try:
            # Step 1: Clone repository
            clone_result = await self._clone_repo(repo_url, project_path, branch)
            if not clone_result["success"]:
                return clone_result
            
            # Step 2: Detect project structure
            structure = await self._detect_structure(project_path)
            
            # Step 3: Attempt Docker build
            build_result = await self._docker_build(project_path, structure)
            
            if build_result["success"]:
                return {
                    "success": True,
                    "message": f"Project {project_name} built successfully",
                    "container_id": build_result.get("container_id"),
                    "logs": build_result.get("logs", "")
                }
            else:
                # Step 4: Analyze error and suggest fix
                fix = await self._analyze_and_suggest_fix(
                    build_result.get("logs", ""),
                    project_path
                )
                
                return {
                    "success": False,
                    "message": f"Build failed for {project_name}",
                    "logs": build_result.get("logs", ""),
                    "fix_suggestion": fix
                }
                
        except Exception as e:
            logger.error(f"Clone and build error: {e}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "logs": str(e)
            }
    
    def _extract_project_name(self, repo_url: str) -> str:
        """Extract project name from repository URL."""
        # Handle various URL formats
        url = repo_url.rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]
        return url.split("/")[-1]
    
    async def _clone_repo(
        self,
        repo_url: str,
        project_path: Path,
        branch: str
    ) -> Dict[str, Any]:
        """Clone repository to local directory."""
        try:
            # Remove existing directory if present
            if project_path.exists():
                shutil.rmtree(project_path)
            
            # Clone repository
            result = subprocess.run(
                ["git", "clone", "--depth", "1", "-b", branch, repo_url, str(project_path)],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                logger.info(f"Cloned {repo_url} to {project_path}")
                return {"success": True, "message": "Repository cloned"}
            else:
                # Try without branch specification (use default)
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", repo_url, str(project_path)],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    return {"success": True, "message": "Repository cloned (default branch)"}
                else:
                    return {
                        "success": False,
                        "message": f"Clone failed: {result.stderr}",
                        "logs": result.stderr
                    }
                    
        except subprocess.TimeoutExpired:
            return {"success": False, "message": "Clone timed out"}
        except Exception as e:
            return {"success": False, "message": f"Clone error: {str(e)}"}
    
    async def _detect_structure(self, project_path: Path) -> Dict[str, Any]:
        """Detect project structure and configuration files."""
        structure = {
            "has_dockerfile": False,
            "has_compose": False,
            "has_requirements": False,
            "has_package_json": False,
            "has_makefile": False,
            "dockerfile_path": None,
            "compose_path": None,
            "language": "unknown"
        }
        
        files_to_check = {
            "Dockerfile": ("has_dockerfile", "dockerfile_path"),
            "docker-compose.yml": ("has_compose", "compose_path"),
            "docker-compose.yaml": ("has_compose", "compose_path"),
            "compose.yml": ("has_compose", "compose_path"),
            "compose.yaml": ("has_compose", "compose_path"),
            "requirements.txt": ("has_requirements", None),
            "package.json": ("has_package_json", None),
            "Makefile": ("has_makefile", None)
        }
        
        for filename, (flag, path_key) in files_to_check.items():
            file_path = project_path / filename
            if file_path.exists():
                structure[flag] = True
                if path_key:
                    structure[path_key] = str(file_path)
        
        # Detect language
        if structure["has_requirements"]:
            structure["language"] = "python"
        elif structure["has_package_json"]:
            structure["language"] = "javascript"
        
        return structure
    
    async def _docker_build(
        self,
        project_path: Path,
        structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Attempt to build Docker container."""
        try:
            project_name = project_path.name.lower().replace("_", "-").replace(".", "-")
            
            # Prefer docker-compose if available
            if structure["has_compose"]:
                compose_path = structure["compose_path"]
                
                result = subprocess.run(
                    ["docker", "compose", "-f", compose_path, "build"],
                    capture_output=True,
                    text=True,
                    cwd=str(project_path),
                    timeout=600  # 10 minute timeout
                )
                
                if result.returncode == 0:
                    # Start containers
                    start_result = subprocess.run(
                        ["docker", "compose", "-f", compose_path, "up", "-d"],
                        capture_output=True,
                        text=True,
                        cwd=str(project_path),
                        timeout=300
                    )
                    
                    return {
                        "success": start_result.returncode == 0,
                        "logs": result.stdout + start_result.stdout,
                        "method": "docker-compose"
                    }
                else:
                    return {
                        "success": False,
                        "logs": result.stderr,
                        "method": "docker-compose"
                    }
            
            # Fall back to Dockerfile
            elif structure["has_dockerfile"]:
                result = subprocess.run(
                    ["docker", "build", "-t", project_name, "."],
                    capture_output=True,
                    text=True,
                    cwd=str(project_path),
                    timeout=600
                )
                
                if result.returncode == 0:
                    # Run container
                    run_result = subprocess.run(
                        ["docker", "run", "-d", "--name", f"{project_name}-container", project_name],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    return {
                        "success": run_result.returncode == 0,
                        "container_id": run_result.stdout.strip()[:12] if run_result.returncode == 0 else None,
                        "logs": result.stdout + run_result.stdout,
                        "method": "dockerfile"
                    }
                else:
                    return {
                        "success": False,
                        "logs": result.stderr,
                        "method": "dockerfile"
                    }
            
            # No Docker configuration found - generate Dockerfile
            else:
                generated = await self._generate_dockerfile(project_path, structure)
                if generated:
                    # Retry build with generated Dockerfile
                    structure["has_dockerfile"] = True
                    structure["dockerfile_path"] = str(project_path / "Dockerfile")
                    return await self._docker_build(project_path, structure)
                else:
                    return {
                        "success": False,
                        "logs": "No Dockerfile found and auto-generation failed",
                        "method": "none"
                    }
                    
        except subprocess.TimeoutExpired:
            return {"success": False, "logs": "Build timed out (>10 minutes)"}
        except Exception as e:
            return {"success": False, "logs": f"Build error: {str(e)}"}
    
    async def _generate_dockerfile(
        self,
        project_path: Path,
        structure: Dict[str, Any]
    ) -> bool:
        """Generate a Dockerfile based on project structure."""
        try:
            dockerfile_content = ""
            
            if structure["language"] == "python":
                dockerfile_content = """# Auto-generated Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
"""
            elif structure["language"] == "javascript":
                dockerfile_content = """# Auto-generated Dockerfile
FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 3000

CMD ["npm", "start"]
"""
            else:
                # Generic fallback
                dockerfile_content = """# Auto-generated Dockerfile
FROM ubuntu:22.04

WORKDIR /app

COPY . .

CMD ["bash"]
"""
            
            dockerfile_path = project_path / "Dockerfile"
            dockerfile_path.write_text(dockerfile_content)
            
            logger.info(f"Generated Dockerfile for {project_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Dockerfile generation error: {e}")
            return False
    
    async def _analyze_and_suggest_fix(
        self,
        error_logs: str,
        project_path: Path
    ) -> str:
        """
        Analyze build error and suggest fix using LLM.
        
        NOTE: This only suggests ONE fix, no infinite loops.
        """
        try:
            prompt = f"""Analyze this Docker build error and suggest ONE specific fix.

Error logs:
```
{error_logs[:2000]}
```

Project: {project_path.name}

Provide a concise fix suggestion. If multiple issues, focus on the first/most critical one.
Format:
1. Problem: [what went wrong]
2. Fix: [specific action to take]
3. Command: [if applicable, the command to run]"""

            response = await self.client.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "Unable to analyze error")
            else:
                return "Error analysis unavailable (LLM request failed)"
                
        except Exception as e:
            logger.error(f"Error analysis failed: {e}")
            return f"Error analysis failed: {str(e)}"
    
    async def get_container_logs(self, container_id: str, lines: int = 100) -> str:
        """Get logs from a running container."""
        try:
            result = subprocess.run(
                ["docker", "logs", "--tail", str(lines), container_id],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout + result.stderr
        except Exception as e:
            return f"Failed to get logs: {str(e)}"
    
    async def list_containers(self, all_containers: bool = False) -> List[Dict[str, str]]:
        """List Docker containers."""
        try:
            cmd = ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}"]
            if all_containers:
                cmd.append("-a")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            containers = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split("\t")
                    if len(parts) >= 4:
                        containers.append({
                            "id": parts[0],
                            "name": parts[1],
                            "status": parts[2],
                            "image": parts[3]
                        })
            return containers
            
        except Exception as e:
            logger.error(f"List containers error: {e}")
            return []
    
    async def stop_container(self, container_id: str) -> bool:
        """Stop a running container."""
        try:
            result = subprocess.run(
                ["docker", "stop", container_id],
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def remove_container(self, container_id: str) -> bool:
        """Remove a container."""
        try:
            result = subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode == 0
        except Exception:
            return False
