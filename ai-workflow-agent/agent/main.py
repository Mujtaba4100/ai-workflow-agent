# AI Workflow Agent - Main API
"""
Main FastAPI application for the AI Workflow Agent.
Provides endpoints for natural language workflow generation.
Milestone 1: Added chat interface and session management.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging

from config import settings, ProjectType
from decision_agent import DecisionAgent
from tools.github_search import GitHubSearchTool
from tools.n8n_builder import N8NWorkflowBuilder
from tools.comfyui_builder import ComfyUIWorkflowBuilder
from tools.docker_helper import DockerHelper
from tools.web_search import WebSearchTool
from chat_handler import chat_handler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Workflow Agent",
    description="Intelligent assistant for creating n8n, ComfyUI, and hybrid workflows",
    version="1.0.0"  # Milestone 1
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
decision_agent = DecisionAgent()
github_tool = GitHubSearchTool()
n8n_builder = N8NWorkflowBuilder()
comfyui_builder = ComfyUIWorkflowBuilder()
docker_helper = DockerHelper()
web_search = WebSearchTool()


# ============================================
# Request/Response Models
# ============================================

class QueryRequest(BaseModel):
    """User query request model."""
    query: str
    context: Optional[Dict[str, Any]] = None


class ProjectDecision(BaseModel):
    """Decision agent response model."""
    project_type: str
    confidence: float
    explanation: str
    suggested_tools: List[str]
    next_steps: List[str]


class WorkflowResponse(BaseModel):
    """Workflow generation response model."""
    success: bool
    project_type: str
    workflow: Optional[Dict[str, Any]] = None
    message: str
    errors: Optional[List[str]] = None


class GitHubSearchRequest(BaseModel):
    """GitHub search request model."""
    keywords: str
    max_results: int = 3


class GitHubSearchResponse(BaseModel):
    """GitHub search response model."""
    success: bool
    repositories: List[Dict[str, Any]]
    recommendation: str


class DockerBuildRequest(BaseModel):
    """Docker build request model."""
    repo_url: str
    branch: Optional[str] = "main"


class DockerBuildResponse(BaseModel):
    """Docker build response model."""
    success: bool
    message: str
    container_id: Optional[str] = None
    logs: Optional[str] = None
    fix_suggestion: Optional[str] = None


# ============================================
# API Endpoints
# ============================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "AI Workflow Agent",
        "version": "0.1.0",
        "components": {
            "ollama": settings.OLLAMA_HOST,
            "n8n": settings.N8N_HOST,
            "comfyui": settings.COMFYUI_HOST
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check for all services."""
    health_status = {
        "agent": "healthy",
        "ollama": await decision_agent.check_ollama_health(),
        "n8n": await n8n_builder.check_health(),
        "comfyui": await comfyui_builder.check_health()
    }
    return health_status


@app.post("/analyze", response_model=ProjectDecision)
async def analyze_query(request: QueryRequest):
    """
    Analyze user query and decide project type.
    
    This is the core decision endpoint that determines:
    - n8n automation
    - ComfyUI generative workflow
    - Hybrid (n8n + ComfyUI)
    - External repo project
    """
    try:
        decision = await decision_agent.analyze(request.query, request.context)
        return ProjectDecision(**decision)
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/build", response_model=WorkflowResponse)
async def build_workflow(request: QueryRequest):
    """
    Full pipeline: Analyze query → Generate workflow → Return result.
    
    This endpoint handles the complete workflow generation process.
    """
    try:
        # Step 1: Analyze query
        decision = await decision_agent.analyze(request.query, request.context)
        project_type = decision["project_type"]
        
        # Step 2: Generate workflow based on type
        workflow = None
        message = ""
        
        if project_type == ProjectType.N8N:
            workflow = await n8n_builder.generate_workflow(request.query)
            message = "n8n workflow generated successfully"
            
        elif project_type == ProjectType.COMFYUI:
            workflow = await comfyui_builder.generate_workflow(request.query)
            message = "ComfyUI workflow generated successfully"
            
        elif project_type == ProjectType.HYBRID:
            # Generate both and combine
            n8n_wf = await n8n_builder.generate_workflow(request.query)
            comfyui_wf = await comfyui_builder.generate_workflow(request.query)
            workflow = {
                "n8n": n8n_wf,
                "comfyui": comfyui_wf,
                "integration": "n8n triggers ComfyUI via HTTP Request node"
            }
            message = "Hybrid workflow generated (n8n + ComfyUI)"
            
        elif project_type == ProjectType.EXTERNAL_REPO:
            # Search for relevant repos
            repos = await github_tool.search(request.query)
            workflow = {
                "type": "external_repo",
                "suggested_repos": repos,
                "next_step": "Select a repo to clone and configure"
            }
            message = "Found relevant GitHub repositories"
            
        else:
            message = "Could not determine project type. Please be more specific."
        
        return WorkflowResponse(
            success=workflow is not None,
            project_type=project_type,
            workflow=workflow,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Build error: {str(e)}")
        return WorkflowResponse(
            success=False,
            project_type=ProjectType.UNKNOWN,
            message=f"Error: {str(e)}",
            errors=[str(e)]
        )


@app.post("/github/search", response_model=GitHubSearchResponse)
async def search_github(request: GitHubSearchRequest):
    """
    Search GitHub for relevant repositories.
    Returns top 3 results with recommendations.
    """
    try:
        results = await github_tool.search(
            request.keywords,
            max_results=request.max_results
        )
        recommendation = await github_tool.generate_recommendation(results)
        
        return GitHubSearchResponse(
            success=True,
            repositories=results,
            recommendation=recommendation
        )
    except Exception as e:
        logger.error(f"GitHub search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/docker/build", response_model=DockerBuildResponse)
async def docker_build(request: DockerBuildRequest):
    """
    Clone repo and attempt Docker build.
    If build fails, analyze logs and suggest fix.
    """
    try:
        result = await docker_helper.clone_and_build(
            request.repo_url,
            request.branch
        )
        return DockerBuildResponse(**result)
    except Exception as e:
        logger.error(f"Docker build error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/n8n/deploy")
async def deploy_n8n_workflow(workflow: Dict[str, Any]):
    """Deploy a workflow to n8n via API."""
    try:
        result = await n8n_builder.deploy_workflow(workflow)
        return {"success": True, "workflow_id": result}
    except Exception as e:
        logger.error(f"n8n deploy error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/comfyui/execute")
async def execute_comfyui(workflow: Dict[str, Any]):
    """Execute a workflow in ComfyUI."""
    try:
        result = await comfyui_builder.execute_workflow(workflow)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"ComfyUI execute error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Chat Endpoints (Milestone 1)
# ============================================

class ChatRequest(BaseModel):
    """Chat request model."""
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response model."""
    success: bool
    session_id: str
    response: str
    state: str
    project_type: Optional[str] = None
    workflow: Optional[Dict[str, Any]] = None
    needs_clarification: Optional[bool] = False
    questions: Optional[List[str]] = None


class WebSearchRequest(BaseModel):
    """Web search request model."""
    query: str
    max_results: int = 5
    site_filter: Optional[str] = None


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Conversational interface for workflow generation.
    
    Supports multi-turn conversations with session management.
    The agent will ask clarifying questions if needed.
    
    Example:
    ```
    POST /chat
    {"message": "Create an automation that sends Slack notifications"}
    ```
    """
    try:
        result = await chat_handler.chat(
            message=request.message,
            session_id=request.session_id
        )
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/sessions")
async def list_chat_sessions():
    """List all active chat sessions."""
    return {
        "sessions": chat_handler.list_sessions()
    }


@app.get("/chat/session/{session_id}")
async def get_chat_session(session_id: str):
    """Get details of a specific chat session."""
    session = chat_handler.get_session_info(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.delete("/chat/session/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session."""
    if chat_handler.clear_session(session_id):
        return {"success": True, "message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


# ============================================
# Web Search Endpoints (Milestone 1)
# ============================================

@app.post("/search/web")
async def search_web(request: WebSearchRequest):
    """
    Search the web for relevant information.
    
    Useful for finding tools, documentation, and alternatives.
    """
    try:
        result = await web_search.search_with_summary(
            query=request.query,
            max_results=request.max_results
        )
        return {"success": True, **result}
    except Exception as e:
        logger.error(f"Web search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/projects")
async def search_projects(request: WebSearchRequest):
    """
    Search for GitHub projects related to a query.
    
    Combines GitHub API search with web search for better coverage.
    """
    try:
        # Search GitHub API
        github_results = await github_tool.search(
            request.query,
            max_results=request.max_results
        )
        
        # Generate recommendation
        recommendation = await github_tool.generate_recommendation(github_results)
        
        return {
            "success": True,
            "repositories": github_results,
            "recommendation": recommendation
        }
    except Exception as e:
        logger.error(f"Project search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/alternatives")
async def search_alternatives(tool: str, purpose: str):
    """
    Find alternative tools for a specific purpose.
    
    Example: Find alternatives to Zapier for automation
    """
    try:
        results = await web_search.find_alternatives(tool, purpose)
        return {
            "success": True,
            "tool": tool,
            "purpose": purpose,
            "alternatives": results
        }
    except Exception as e:
        logger.error(f"Alternatives search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Enhanced Docker Endpoints (Milestone 1)
# ============================================

@app.get("/docker/containers")
async def list_docker_containers(all_containers: bool = False):
    """List Docker containers."""
    try:
        containers = await docker_helper.list_containers(all_containers)
        return {"success": True, "containers": containers}
    except Exception as e:
        logger.error(f"List containers error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/docker/logs/{container_id}")
async def get_container_logs(container_id: str, lines: int = 100):
    """Get logs from a Docker container."""
    try:
        logs = await docker_helper.get_container_logs(container_id, lines)
        return {"success": True, "container_id": container_id, "logs": logs}
    except Exception as e:
        logger.error(f"Get logs error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/docker/stop/{container_id}")
async def stop_docker_container(container_id: str):
    """Stop a running Docker container."""
    try:
        success = await docker_helper.stop_container(container_id)
        return {"success": success, "container_id": container_id}
    except Exception as e:
        logger.error(f"Stop container error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/docker/container/{container_id}")
async def remove_docker_container(container_id: str):
    """Remove a Docker container."""
    try:
        success = await docker_helper.remove_container(container_id)
        return {"success": success, "container_id": container_id}
    except Exception as e:
        logger.error(f"Remove container error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Startup/Shutdown Events
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting AI Workflow Agent...")
    logger.info(f"Ollama: {settings.OLLAMA_HOST}")
    logger.info(f"n8n: {settings.N8N_HOST}")
    logger.info(f"ComfyUI: {settings.COMFYUI_HOST}")
    
    # Pull Qwen2.5 model if not present
    await decision_agent.ensure_model_available()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down AI Workflow Agent...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
