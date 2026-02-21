# AI Workflow Agent - Configuration
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Ollama LLM Configuration
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    
    # n8n Configuration
    N8N_HOST: str = os.getenv("N8N_HOST", "http://localhost:5678")
    N8N_API_KEY: Optional[str] = os.getenv("N8N_API_KEY", None)
    
    # ComfyUI Configuration
    COMFYUI_HOST: str = os.getenv("COMFYUI_HOST", "http://localhost:8188")
    
    # PostgreSQL Configuration
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "agent")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "agent_secret_2026")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "workflow_agent")
    
    # GitHub Configuration
    GITHUB_TOKEN: Optional[str] = os.getenv("GITHUB_TOKEN", None)
    
    # Colab Configuration (Milestone 2)
    NGROK_AUTH_TOKEN: Optional[str] = os.getenv("NGROK_AUTH_TOKEN", None)
    
    # Project Directories
    PROJECTS_DIR: str = os.getenv("PROJECTS_DIR", "/app/projects")
    WORKFLOWS_DIR: str = os.getenv("WORKFLOWS_DIR", "/app/workflows")
    
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Project Type Definitions
class ProjectType:
    N8N = "n8n"
    COMFYUI = "comfyui"
    HYBRID = "hybrid"
    EXTERNAL_REPO = "external_repo"
    UNKNOWN = "unknown"


# Keywords for project classification
CLASSIFICATION_KEYWORDS = {
    ProjectType.N8N: [
        "automation", "workflow", "integrate", "api", "webhook", "schedule",
        "email", "slack", "telegram", "notification", "trigger", "connect",
        "sync", "transfer", "backup", "monitor", "alert", "scrape", "fetch"
    ],
    ProjectType.COMFYUI: [
        "image", "generate", "ai art", "stable diffusion", "flux", "sdxl",
        "inpaint", "upscale", "controlnet", "lora", "checkpoint", "model",
        "txt2img", "img2img", "video", "animation", "diffusion", "generative"
    ],
    ProjectType.HYBRID: [
        "generate and send", "create image and", "automation with ai",
        "workflow with image", "process and automate", "ai image generation",
        "image and automation", "generate image and email", "ai and workflow"
    ],
    ProjectType.EXTERNAL_REPO: [
        "github", "repository", "repo", "clone", "download project",
        "install", "setup project", "deploy", "docker project"
    ]
}
