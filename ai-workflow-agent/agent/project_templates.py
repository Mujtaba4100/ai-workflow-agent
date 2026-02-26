"""
Project Templates - M3 Feature
Quick-start templates for common project types
Provides scaffolding, configurations, and ready-to-use setups
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ProjectCategory(str, Enum):
    """Categories of project templates"""
    AUTOMATION = "automation"
    AI_IMAGE = "ai_image"
    AI_TEXT = "ai_text"
    DATA_PIPELINE = "data_pipeline"
    WEB_SCRAPING = "web_scraping"
    CHATBOT = "chatbot"
    MONITORING = "monitoring"
    INTEGRATION = "integration"


class Difficulty(str, Enum):
    """Template difficulty level"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class ProjectFile:
    """A file in a project template"""
    path: str
    content: str
    is_template: bool = True  # Has placeholders to fill
    description: str = ""


@dataclass
class ProjectTemplate:
    """A complete project template"""
    template_id: str
    name: str
    description: str
    category: ProjectCategory
    difficulty: Difficulty
    tags: List[str]
    files: List[ProjectFile]
    dependencies: List[str]
    environment_vars: Dict[str, str]
    docker_services: List[str]
    estimated_setup_time: str
    documentation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "tags": self.tags,
            "file_count": len(self.files),
            "dependencies": self.dependencies,
            "docker_services": self.docker_services,
            "estimated_setup_time": self.estimated_setup_time
        }


class ProjectTemplateRegistry:
    """Registry of project templates"""
    
    def __init__(self):
        self._templates: Dict[str, ProjectTemplate] = {}
        self._register_builtin_templates()
    
    def _register_builtin_templates(self):
        """Register built-in project templates"""
        
        # ============== AUTOMATION TEMPLATES ==============
        self.register(ProjectTemplate(
            template_id="email-automation",
            name="Email Automation Hub",
            description="Automated email processing with AI classification and responses",
            category=ProjectCategory.AUTOMATION,
            difficulty=Difficulty.INTERMEDIATE,
            tags=["email", "automation", "ai", "classification"],
            files=[
                ProjectFile(
                    path="n8n-workflow.json",
                    content=json.dumps({
                        "name": "Email Automation",
                        "nodes": [
                            {
                                "name": "Email Trigger",
                                "type": "n8n-nodes-base.emailTrigger",
                                "parameters": {"pollInterval": 60}
                            },
                            {
                                "name": "Classify Email",
                                "type": "n8n-nodes-base.httpRequest",
                                "parameters": {
                                    "url": "{{LLM_ENDPOINT}}/classify",
                                    "method": "POST"
                                }
                            },
                            {
                                "name": "Route by Category",
                                "type": "n8n-nodes-base.switch"
                            }
                        ],
                        "connections": {}
                    }, indent=2),
                    description="Main n8n workflow for email processing"
                ),
                ProjectFile(
                    path=".env.example",
                    content="""# Email Automation Configuration
EMAIL_HOST={{EMAIL_HOST}}
EMAIL_USER={{EMAIL_USER}}
EMAIL_PASSWORD={{EMAIL_PASSWORD}}
LLM_ENDPOINT=http://localhost:11434
SLACK_WEBHOOK={{SLACK_WEBHOOK}}
""",
                    description="Environment variables template"
                ),
                ProjectFile(
                    path="README.md",
                    content="""# Email Automation Hub

Automated email processing with AI classification.

## Features
- Auto-classify incoming emails
- Route to appropriate handlers
- Generate AI responses for common queries
- Slack notifications for urgent emails

## Setup
1. Configure .env from .env.example
2. Import n8n-workflow.json to n8n
3. Start the workflow

## Configuration
- Set email credentials in .env
- Configure LLM endpoint for classification
""",
                    description="Project documentation"
                )
            ],
            dependencies=["n8n", "ollama"],
            environment_vars={
                "EMAIL_HOST": "imap.gmail.com",
                "EMAIL_USER": "",
                "EMAIL_PASSWORD": "",
                "LLM_ENDPOINT": "http://localhost:11434"
            },
            docker_services=["n8n", "ollama"],
            estimated_setup_time="15 minutes",
            documentation="Processes incoming emails using AI classification"
        ))
        
        self.register(ProjectTemplate(
            template_id="slack-bot-automation",
            name="Slack Bot Automation",
            description="Multi-purpose Slack bot with custom commands and AI responses",
            category=ProjectCategory.CHATBOT,
            difficulty=Difficulty.BEGINNER,
            tags=["slack", "bot", "automation", "commands"],
            files=[
                ProjectFile(
                    path="n8n-workflow.json",
                    content=json.dumps({
                        "name": "Slack Bot",
                        "nodes": [
                            {
                                "name": "Slack Trigger",
                                "type": "n8n-nodes-base.slackTrigger",
                                "parameters": {"events": ["message"]}
                            },
                            {
                                "name": "Parse Command",
                                "type": "n8n-nodes-base.function",
                                "parameters": {}
                            },
                            {
                                "name": "Execute Command",
                                "type": "n8n-nodes-base.switch"
                            },
                            {
                                "name": "Send Response",
                                "type": "n8n-nodes-base.slack"
                            }
                        ],
                        "connections": {}
                    }, indent=2),
                    description="Slack bot n8n workflow"
                ),
                ProjectFile(
                    path=".env.example",
                    content="""# Slack Bot Configuration
SLACK_BOT_TOKEN={{SLACK_BOT_TOKEN}}
SLACK_SIGNING_SECRET={{SLACK_SIGNING_SECRET}}
""",
                    description="Environment template"
                ),
                ProjectFile(
                    path="commands.json",
                    content=json.dumps({
                        "commands": [
                            {"name": "help", "description": "Show available commands"},
                            {"name": "status", "description": "Check system status"},
                            {"name": "ask", "description": "Ask AI a question"}
                        ]
                    }, indent=2),
                    description="Bot commands configuration"
                )
            ],
            dependencies=["n8n"],
            environment_vars={
                "SLACK_BOT_TOKEN": "",
                "SLACK_SIGNING_SECRET": ""
            },
            docker_services=["n8n"],
            estimated_setup_time="10 minutes",
            documentation="Slack bot with customizable commands"
        ))
        
        # ============== AI IMAGE TEMPLATES ==============
        self.register(ProjectTemplate(
            template_id="ai-image-generator",
            name="AI Image Generation Pipeline",
            description="Complete image generation pipeline with SDXL, upscaling, and batch processing",
            category=ProjectCategory.AI_IMAGE,
            difficulty=Difficulty.INTERMEDIATE,
            tags=["ai", "image", "stable-diffusion", "sdxl", "comfyui"],
            files=[
                ProjectFile(
                    path="comfyui-workflow.json",
                    content=json.dumps({
                        "1": {
                            "class_type": "CheckpointLoaderSimple",
                            "inputs": {"ckpt_name": "{{MODEL_NAME}}"}
                        },
                        "2": {
                            "class_type": "CLIPTextEncode",
                            "inputs": {"text": "{{PROMPT}}", "clip": ["1", 1]}
                        },
                        "3": {
                            "class_type": "CLIPTextEncode",
                            "inputs": {"text": "{{NEGATIVE_PROMPT}}", "clip": ["1", 1]}
                        },
                        "4": {
                            "class_type": "EmptyLatentImage",
                            "inputs": {"width": 1024, "height": 1024, "batch_size": 1}
                        },
                        "5": {
                            "class_type": "KSampler",
                            "inputs": {
                                "model": ["1", 0],
                                "positive": ["2", 0],
                                "negative": ["3", 0],
                                "latent_image": ["4", 0],
                                "seed": "{{SEED}}",
                                "steps": 30,
                                "cfg": 7.5,
                                "sampler_name": "euler_ancestral"
                            }
                        },
                        "6": {
                            "class_type": "VAEDecode",
                            "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
                        },
                        "7": {
                            "class_type": "SaveImage",
                            "inputs": {"images": ["6", 0], "filename_prefix": "generated"}
                        }
                    }, indent=2),
                    description="ComfyUI workflow for image generation"
                ),
                ProjectFile(
                    path="api-server.py",
                    content='''"""
Simple API server for image generation
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import json

app = FastAPI(title="Image Generation API")

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    seed: int = -1

COMFYUI_URL = "http://localhost:8188"

@app.post("/generate")
async def generate_image(request: ImageRequest):
    # Load and modify workflow
    with open("comfyui-workflow.json") as f:
        workflow = json.load(f)
    
    workflow["2"]["inputs"]["text"] = request.prompt
    workflow["3"]["inputs"]["text"] = request.negative_prompt
    workflow["4"]["inputs"]["width"] = request.width
    workflow["4"]["inputs"]["height"] = request.height
    
    # Queue in ComfyUI
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow}
        )
        return response.json()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
''',
                    description="FastAPI server for image generation"
                ),
                ProjectFile(
                    path=".env.example",
                    content="""# Image Generation Configuration
COMFYUI_URL=http://localhost:8188
MODEL_NAME=sd_xl_base_1.0.safetensors
OUTPUT_DIR=./outputs
""",
                    description="Environment template"
                ),
                ProjectFile(
                    path="requirements.txt",
                    content="fastapi\nuvicorn\nhttpx\npydantic",
                    description="Python dependencies"
                )
            ],
            dependencies=["comfyui", "python"],
            environment_vars={
                "COMFYUI_URL": "http://localhost:8188",
                "MODEL_NAME": "sd_xl_base_1.0.safetensors"
            },
            docker_services=["comfyui"],
            estimated_setup_time="20 minutes",
            documentation="Complete SDXL image generation pipeline"
        ))
        
        self.register(ProjectTemplate(
            template_id="image-batch-processor",
            name="Batch Image Processor",
            description="Process multiple images with AI upscaling, watermarking, and optimization",
            category=ProjectCategory.AI_IMAGE,
            difficulty=Difficulty.INTERMEDIATE,
            tags=["batch", "image", "processing", "upscale"],
            files=[
                ProjectFile(
                    path="processor.py",
                    content='''"""
Batch Image Processor
"""
import os
import asyncio
from pathlib import Path
from typing import List
import httpx

INPUT_DIR = "./input"
OUTPUT_DIR = "./output"
COMFYUI_URL = "http://localhost:8188"

async def upscale_image(image_path: str, scale: int = 2) -> dict:
    """Upscale single image"""
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_path}},
        "2": {"class_type": "UpscaleImage", "inputs": {"image": ["1", 0], "scale": scale}},
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}}
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow})
        return response.json()

async def process_batch(input_dir: str, output_dir: str):
    """Process all images in directory"""
    Path(output_dir).mkdir(exist_ok=True)
    
    images = list(Path(input_dir).glob("*.png")) + list(Path(input_dir).glob("*.jpg"))
    
    for img_path in images:
        print(f"Processing: {img_path}")
        result = await upscale_image(str(img_path))
        print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(process_batch(INPUT_DIR, OUTPUT_DIR))
''',
                    description="Batch processing script"
                ),
                ProjectFile(
                    path="requirements.txt",
                    content="httpx\nPillow",
                    description="Dependencies"
                )
            ],
            dependencies=["comfyui", "python"],
            environment_vars={
                "INPUT_DIR": "./input",
                "OUTPUT_DIR": "./output"
            },
            docker_services=["comfyui"],
            estimated_setup_time="10 minutes",
            documentation="Process multiple images with AI enhancements"
        ))
        
        # ============== DATA PIPELINE TEMPLATES ==============
        self.register(ProjectTemplate(
            template_id="data-sync-pipeline",
            name="Database Sync Pipeline",
            description="Sync data between databases with transformation and validation",
            category=ProjectCategory.DATA_PIPELINE,
            difficulty=Difficulty.ADVANCED,
            tags=["database", "sync", "etl", "pipeline"],
            files=[
                ProjectFile(
                    path="n8n-workflow.json",
                    content=json.dumps({
                        "name": "Database Sync",
                        "nodes": [
                            {
                                "name": "Schedule",
                                "type": "n8n-nodes-base.scheduleTrigger",
                                "parameters": {"cron": "0 */6 * * *"}
                            },
                            {
                                "name": "Read Source",
                                "type": "n8n-nodes-base.postgres",
                                "parameters": {"operation": "select"}
                            },
                            {
                                "name": "Transform",
                                "type": "n8n-nodes-base.function"
                            },
                            {
                                "name": "Write Target",
                                "type": "n8n-nodes-base.postgres",
                                "parameters": {"operation": "upsert"}
                            },
                            {
                                "name": "Log Results",
                                "type": "n8n-nodes-base.slack"
                            }
                        ],
                        "connections": {}
                    }, indent=2),
                    description="n8n sync workflow"
                ),
                ProjectFile(
                    path=".env.example",
                    content="""# Database Sync Configuration
SOURCE_DB_HOST={{SOURCE_HOST}}
SOURCE_DB_NAME={{SOURCE_DB}}
SOURCE_DB_USER={{SOURCE_USER}}
SOURCE_DB_PASSWORD={{SOURCE_PASS}}

TARGET_DB_HOST={{TARGET_HOST}}
TARGET_DB_NAME={{TARGET_DB}}
TARGET_DB_USER={{TARGET_USER}}
TARGET_DB_PASSWORD={{TARGET_PASS}}

SLACK_WEBHOOK={{SLACK_WEBHOOK}}
""",
                    description="Environment template"
                )
            ],
            dependencies=["n8n", "postgresql"],
            environment_vars={
                "SOURCE_DB_HOST": "",
                "TARGET_DB_HOST": ""
            },
            docker_services=["n8n", "postgres"],
            estimated_setup_time="30 minutes",
            documentation="ETL pipeline for database synchronization"
        ))
        
        # ============== WEB SCRAPING TEMPLATES ==============
        self.register(ProjectTemplate(
            template_id="web-scraper",
            name="Smart Web Scraper",
            description="Intelligent web scraping with AI extraction and structured output",
            category=ProjectCategory.WEB_SCRAPING,
            difficulty=Difficulty.INTERMEDIATE,
            tags=["scraping", "web", "ai", "extraction"],
            files=[
                ProjectFile(
                    path="scraper.py",
                    content='''"""
Smart Web Scraper with AI Extraction
"""
import httpx
from bs4 import BeautifulSoup
from typing import Dict, Any, List
import json

LLM_ENDPOINT = "http://localhost:11434/api/generate"

async def scrape_page(url: str) -> str:
    """Scrape webpage content"""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.decompose()
            
        return soup.get_text(separator="\\n", strip=True)

async def extract_with_ai(content: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structured data using AI"""
    prompt = f"""Extract the following information from this text:
Schema: {json.dumps(schema)}

Text:
{content[:4000]}

Return as JSON:"""

    async with httpx.AsyncClient() as client:
        response = await client.post(
            LLM_ENDPOINT,
            json={"model": "qwen2.5:7b", "prompt": prompt, "stream": False}
        )
        return response.json()

async def scrape_and_extract(url: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Complete scrape and extract pipeline"""
    content = await scrape_page(url)
    extracted = await extract_with_ai(content, schema)
    return {
        "url": url,
        "raw_length": len(content),
        "extracted": extracted
    }

if __name__ == "__main__":
    import asyncio
    
    schema = {
        "title": "string",
        "author": "string",
        "date": "string",
        "summary": "string"
    }
    
    result = asyncio.run(scrape_and_extract("https://example.com", schema))
    print(json.dumps(result, indent=2))
''',
                    description="AI-powered web scraper"
                ),
                ProjectFile(
                    path="requirements.txt",
                    content="httpx\nbeautifulsoup4\nlxml",
                    description="Dependencies"
                ),
                ProjectFile(
                    path="schemas/article.json",
                    content=json.dumps({
                        "title": "string",
                        "author": "string",
                        "date": "string",
                        "content": "string",
                        "tags": "array"
                    }, indent=2),
                    description="Article extraction schema"
                )
            ],
            dependencies=["python", "ollama"],
            environment_vars={
                "LLM_ENDPOINT": "http://localhost:11434"
            },
            docker_services=["ollama"],
            estimated_setup_time="10 minutes",
            documentation="Web scraping with AI-powered data extraction"
        ))
        
        # ============== MONITORING TEMPLATES ==============
        self.register(ProjectTemplate(
            template_id="api-monitor",
            name="API Health Monitor",
            description="Monitor APIs and services with alerting and dashboards",
            category=ProjectCategory.MONITORING,
            difficulty=Difficulty.BEGINNER,
            tags=["monitoring", "api", "health", "alerts"],
            files=[
                ProjectFile(
                    path="n8n-workflow.json",
                    content=json.dumps({
                        "name": "API Monitor",
                        "nodes": [
                            {
                                "name": "Every 5 Minutes",
                                "type": "n8n-nodes-base.scheduleTrigger",
                                "parameters": {"cron": "*/5 * * * *"}
                            },
                            {
                                "name": "Check APIs",
                                "type": "n8n-nodes-base.httpRequest",
                                "parameters": {"url": "{{API_URL}}/health"}
                            },
                            {
                                "name": "Check Status",
                                "type": "n8n-nodes-base.if",
                                "parameters": {"conditions": {"number": [{"value1": "={{$json.statusCode}}", "value2": 200}]}}
                            },
                            {
                                "name": "Alert on Failure",
                                "type": "n8n-nodes-base.slack",
                                "parameters": {"text": "🚨 API Down: {{$json.url}}"}
                            }
                        ],
                        "connections": {}
                    }, indent=2),
                    description="n8n monitoring workflow"
                ),
                ProjectFile(
                    path="endpoints.json",
                    content=json.dumps({
                        "endpoints": [
                            {"name": "Main API", "url": "https://api.example.com/health"},
                            {"name": "Database", "url": "https://db.example.com/status"},
                            {"name": "Auth Service", "url": "https://auth.example.com/ping"}
                        ]
                    }, indent=2),
                    description="Endpoints to monitor"
                ),
                ProjectFile(
                    path=".env.example",
                    content="""# API Monitor Configuration
SLACK_WEBHOOK={{SLACK_WEBHOOK}}
CHECK_INTERVAL=5
ALERT_THRESHOLD=2
""",
                    description="Environment template"
                )
            ],
            dependencies=["n8n"],
            environment_vars={
                "SLACK_WEBHOOK": "",
                "CHECK_INTERVAL": "5"
            },
            docker_services=["n8n"],
            estimated_setup_time="10 minutes",
            documentation="Monitor your APIs with automatic alerting"
        ))
        
        # ============== CHATBOT TEMPLATES ==============
        self.register(ProjectTemplate(
            template_id="ai-chatbot",
            name="AI Chatbot with Memory",
            description="Conversational AI chatbot with context memory and knowledge base",
            category=ProjectCategory.CHATBOT,
            difficulty=Difficulty.ADVANCED,
            tags=["chatbot", "ai", "llm", "memory", "rag"],
            files=[
                ProjectFile(
                    path="chatbot.py",
                    content='''"""
AI Chatbot with Memory
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import httpx
import json
from datetime import datetime

app = FastAPI(title="AI Chatbot")

# In-memory conversation storage
conversations: Dict[str, List[Dict]] = {}

LLM_ENDPOINT = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

def get_context(session_id: str, max_turns: int = 5) -> str:
    """Get conversation context"""
    if session_id not in conversations:
        return ""
    
    history = conversations[session_id][-max_turns:]
    context = "\\n".join([
        f"User: {turn['user']}\\nAssistant: {turn['assistant']}"
        for turn in history
    ])
    return context

async def generate_response(message: str, context: str) -> str:
    """Generate AI response"""
    prompt = f"""You are a helpful AI assistant. Use the conversation history for context.

Previous conversation:
{context}

User: {message}
Assistant:"""

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            LLM_ENDPOINT,
            json={"model": MODEL, "prompt": prompt, "stream": False}
        )
        data = response.json()
        return data.get("response", "I'm sorry, I couldn't process that.")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatMessage):
    """Chat endpoint with memory"""
    import uuid
    
    session_id = request.session_id or str(uuid.uuid4())
    
    if session_id not in conversations:
        conversations[session_id] = []
    
    context = get_context(session_id)
    response = await generate_response(request.message, context)
    
    # Store in memory
    conversations[session_id].append({
        "user": request.message,
        "assistant": response,
        "timestamp": datetime.now().isoformat()
    })
    
    return ChatResponse(response=response, session_id=session_id)

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """Get conversation history"""
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"history": conversations[session_id]}

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history"""
    if session_id in conversations:
        del conversations[session_id]
    return {"message": "Session cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
''',
                    description="Chatbot with conversation memory"
                ),
                ProjectFile(
                    path="requirements.txt",
                    content="fastapi\nuvicorn\nhttpx\npydantic",
                    description="Dependencies"
                ),
                ProjectFile(
                    path=".env.example",
                    content="""# Chatbot Configuration
LLM_ENDPOINT=http://localhost:11434
MODEL=qwen2.5:7b
MAX_CONTEXT_TURNS=10
""",
                    description="Environment template"
                )
            ],
            dependencies=["python", "ollama"],
            environment_vars={
                "LLM_ENDPOINT": "http://localhost:11434",
                "MODEL": "qwen2.5:7b"
            },
            docker_services=["ollama"],
            estimated_setup_time="15 minutes",
            documentation="AI chatbot with conversation memory and context"
        ))
    
    def register(self, template: ProjectTemplate):
        """Register a template"""
        self._templates[template.template_id] = template
        
    def get(self, template_id: str) -> Optional[ProjectTemplate]:
        """Get template by ID"""
        return self._templates.get(template_id)
    
    def list_all(self) -> List[ProjectTemplate]:
        """List all templates"""
        return list(self._templates.values())
    
    def list_by_category(self, category: ProjectCategory) -> List[ProjectTemplate]:
        """List templates by category"""
        return [t for t in self._templates.values() if t.category == category]
    
    def list_by_difficulty(self, difficulty: Difficulty) -> List[ProjectTemplate]:
        """List templates by difficulty"""
        return [t for t in self._templates.values() if t.difficulty == difficulty]
    
    def search(self, query: str) -> List[ProjectTemplate]:
        """Search templates"""
        query = query.lower()
        return [
            t for t in self._templates.values()
            if query in t.name.lower() 
            or query in t.description.lower()
            or any(query in tag.lower() for tag in t.tags)
        ]


class ProjectScaffolder:
    """Generate project from template"""
    
    def __init__(self, registry: Optional[ProjectTemplateRegistry] = None):
        self.registry = registry or ProjectTemplateRegistry()
        
    def scaffold(
        self,
        template_id: str,
        project_name: str,
        output_dir: str,
        variables: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Generate project from template
        
        Args:
            template_id: Template to use
            project_name: Name for the new project
            output_dir: Directory to create project in
            variables: Template variables to substitute
        """
        template = self.registry.get(template_id)
        if not template:
            return {"success": False, "error": f"Template not found: {template_id}"}
            
        variables = variables or {}
        project_path = os.path.join(output_dir, project_name)
        
        files_created = []
        
        try:
            # Create project directory
            os.makedirs(project_path, exist_ok=True)
            
            # Generate files
            for file in template.files:
                file_path = os.path.join(project_path, file.path)
                
                # Create subdirectories if needed
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Process template variables
                content = file.content
                if file.is_template:
                    for key, value in variables.items():
                        content = content.replace(f"{{{{{key}}}}}", value)
                
                # Write file
                with open(file_path, 'w') as f:
                    f.write(content)
                    
                files_created.append(file.path)
                
            logger.info(f"Created project: {project_name} from {template_id}")
            
            return {
                "success": True,
                "project_name": project_name,
                "project_path": project_path,
                "template": template_id,
                "files_created": files_created,
                "next_steps": [
                    f"cd {project_path}",
                    "Configure .env from .env.example",
                    f"Start services: {', '.join(template.docker_services)}"
                ]
            }
            
        except Exception as e:
            logger.error(f"Scaffold error: {e}")
            return {"success": False, "error": str(e)}
    
    def preview(self, template_id: str) -> Dict[str, Any]:
        """Preview template files without creating"""
        template = self.registry.get(template_id)
        if not template:
            return {"success": False, "error": f"Template not found: {template_id}"}
            
        return {
            "success": True,
            "template": template.to_dict(),
            "files": [
                {
                    "path": f.path,
                    "description": f.description,
                    "is_template": f.is_template,
                    "preview": f.content[:500] + "..." if len(f.content) > 500 else f.content
                }
                for f in template.files
            ],
            "required_variables": list(template.environment_vars.keys())
        }


# Singleton instances
_registry: Optional[ProjectTemplateRegistry] = None
_scaffolder: Optional[ProjectScaffolder] = None


def get_template_registry() -> ProjectTemplateRegistry:
    """Get the global template registry"""
    global _registry
    if _registry is None:
        _registry = ProjectTemplateRegistry()
    return _registry


def get_scaffolder() -> ProjectScaffolder:
    """Get the global project scaffolder"""
    global _scaffolder
    if _scaffolder is None:
        _scaffolder = ProjectScaffolder()
    return _scaffolder
