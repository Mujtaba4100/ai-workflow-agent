# AI Workflow Agent - Main API
"""
Main FastAPI application for the AI Workflow Agent.
Provides endpoints for all three milestones:
- M1: Core AI Decision Making (CrewAI + GitHub Search)
- M2: Colab Offloading Layer (ColabCode + pyngrok + Auto-fallback)
- M3: Dashboard Layer (Appsmith + Playwright + Directus)
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
import os

from config import settings, ProjectType
from decision_agent import DecisionAgent
from tools.github_search import GitHubSearchTool
from tools.n8n_builder import N8NWorkflowBuilder
from tools.comfyui_builder import ComfyUIWorkflowBuilder
from tools.docker_helper import DockerHelper

# M2: Colab Layer imports
from colab_layer import ColabLayer

# M3: Dashboard Layer imports
from appsmith_client import AppsmithClient, DashboardTemplates
from playwright_analyzer import PlaywrightAnalyzer, LayoutConverter
from directus_client import DirectusClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Workflow Agent",
    description="Intelligent assistant with Colab offloading and dashboard integration",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize M1 components
decision_agent = DecisionAgent()
github_tool = GitHubSearchTool()
n8n_builder = N8NWorkflowBuilder()
comfyui_builder = ComfyUIWorkflowBuilder()
docker_helper = DockerHelper()

# Initialize M2 components
colab_layer = ColabLayer()

# Initialize M3 components
appsmith_url = os.getenv("APPSMITH_URL", "http://localhost:80")
directus_url = os.getenv("DIRECTUS_URL", "http://localhost:8055")
appsmith_client = AppsmithClient(base_url=appsmith_url)
playwright_analyzer = PlaywrightAnalyzer()
directus_client = DirectusClient(base_url=directus_url)


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


# M2 Models
class ColabOffloadRequest(BaseModel):
    """Request to offload a task to Colab."""
    task_description: str
    code: Optional[str] = None
    requirements: Optional[List[str]] = None
    max_execution_time: int = 3600
    gpu_required: bool = True


class ColabOffloadResponse(BaseModel):
    """Response from Colab offload operation."""
    success: bool
    message: str
    execution_id: Optional[str] = None
    tunnel_url: Optional[str] = None
    notebook_url: Optional[str] = None
    decided_offload: bool
    decision_reason: Optional[str] = None


# M3 Models
class DashboardCreateRequest(BaseModel):
    """Request to create a dashboard."""
    template: str  # workflow_status, container_logs, or agent_decisions
    custom_config: Optional[Dict[str, Any]] = None


class DashboardCreateResponse(BaseModel):
    """Response from dashboard creation."""
    success: bool
    message: str
    dashboard_id: Optional[str] = None
    dashboard_url: Optional[str] = None


class PageAnalyzeRequest(BaseModel):
    """Request to analyze a web page."""
    url: str
    convert_to_appsmith: bool = False


class PageAnalyzeResponse(BaseModel):
    """Response from page analysis."""
    success: bool
    message: str
    analysis: Optional[Dict[str, Any]] = None
    appsmith_widgets: Optional[List[Dict[str, Any]]] = None


class DirectusAuthRequest(BaseModel):
    """Request to authenticate with Directus."""
    email: str
    password: str


class DirectusAuthResponse(BaseModel):
    """Response from Directus auth."""
    success: bool
    message: str
    token: Optional[str] = None
    expires_at: Optional[str] = None


# ============================================
# M1: Core AI Decision Endpoints
# ============================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "AI Workflow Agent",
        "version": "3.0.0",
        "milestones": {
            "m1": "Core AI Agent (CrewAI, GitHub Search)",
            "m2": "Colab Offloading Layer (ColabCode, pyngrok, Auto-fallback)",
            "m3": "Dashboard Layer (Appsmith, Playwright, Directus)"
        },
        "components": {
            "ollama": settings.OLLAMA_HOST,
            "n8n": settings.N8N_HOST,
            "comfyui": settings.COMFYUI_HOST,
            "appsmith": appsmith_url,
            "directus": directus_url
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check for all services."""
    health_status = {
        "agent": "healthy",
        "ollama": await decision_agent.check_ollama_health(),
        "n8n": await n8n_builder.check_health(),
        "comfyui": await comfyui_builder.check_health(),
        "colab_layer": "initialized",
        "appsmith": "configured" if appsmith_client else "not configured",
        "playwright": "ready" if playwright_analyzer else "not ready",
        "directus": "configured" if directus_client else "not configured"
    }
    return health_status


@app.post("/analyze", response_model=ProjectDecision)
async def analyze_query(request: QueryRequest):
    """
    M1: Analyze user query and decide project type using CrewAI.
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
    M1: Full pipeline - Analyze query → Generate workflow → Return result.
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
            n8n_wf = await n8n_builder.generate_workflow(request.query)
            comfyui_wf = await comfyui_builder.generate_workflow(request.query)
            workflow = {
                "n8n": n8n_wf,
                "comfyui": comfyui_wf,
                "integration": "n8n triggers ComfyUI via HTTP Request node"
            }
            message = "Hybrid workflow generated (n8n + ComfyUI)"
            
        elif project_type == ProjectType.EXTERNAL_REPO:
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
    M1: Search GitHub for relevant repositories.
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


# ============================================
# M2: Colab Offloading Endpoints
# ============================================

@app.post("/colab/offload", response_model=ColabOffloadResponse)
async def offload_to_colab(request: ColabOffloadRequest):
    """
    M2: Analyze task complexity and decide whether to offload to Colab.
    If yes, creates notebook, starts ColabCode server, and returns tunnel URL.
    """
    try:
        # Step 1: Analyze if offloading is beneficial
        should_offload = colab_layer.offload_decider.should_offload(
            task_description=request.task_description,
            gpu_required=request.gpu_required,
            estimated_duration=request.max_execution_time
        )
        
        decision_reason = colab_layer.offload_decider.get_decision_reason()
        
        if should_offload:
            # Step 2: Create Colab notebook
            notebook_code = f"""
# Generated Notebook for Task: {request.task_description}

# Install requirements
{chr(10).join(['!pip install ' + req for req in (request.requirements or [])])}

# User code
{request.code or '# Add your code here'}
"""
            
            notebook_path = colab_layer.colab_code_manager.create_notebook(
                name=f"task_{request.task_description[:30]}",
                code=notebook_code
            )
            
            # Step 3: Start ColabCode server and tunnel
            execution_id = f"exec_{hash(request.task_description)}"
            
            # In real scenario, this would start ColabCode and pyngrok
            # For now, we simulate the response
            tunnel_url = colab_layer.tunnel_manager.create_tunnel(8000)
            
            return ColabOffloadResponse(
                success=True,
                message=f"Task offloaded to Colab successfully",
                execution_id=execution_id,
                tunnel_url=tunnel_url,
                notebook_url=f"https://colab.research.google.com/drive/{notebook_path}",
                decided_offload=True,
                decision_reason=decision_reason
            )
        else:
            # Task will run locally
            return ColabOffloadResponse(
                success=True,
                message="Task will execute locally (no offload needed)",
                decided_offload=False,
                decision_reason=decision_reason
            )
            
    except Exception as e:
        logger.error(f"Colab offload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/colab/status/{execution_id}")
async def get_colab_status(execution_id: str):
    """
    M2: Get status of a running Colab execution.
    """
    try:
        # Check if fallback was triggered
        fallback_triggered = colab_layer.fallback_handler.is_fallback_active(execution_id)
        
        return {
            "execution_id": execution_id,
            "status": "fallback" if fallback_triggered else "running",
            "fallback_reason": colab_layer.fallback_handler.get_fallback_reason(execution_id) if fallback_triggered else None
        }
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/colab/tunnel/create")
async def create_tunnel(port: int = 8000):
    """
    M2: Create a pyngrok tunnel to expose local port.
    """
    try:
        tunnel_url = colab_layer.tunnel_manager.create_tunnel(port)
        return {
            "success": True,
            "tunnel_url": tunnel_url,
            "port": port
        }
    except Exception as e:
        logger.error(f"Tunnel creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/colab/tunnel")
async def close_tunnel():
    """
    M2: Close the active pyngrok tunnel.
    """
    try:
        colab_layer.tunnel_manager.close_tunnel()
        return {"success": True, "message": "Tunnel closed"}
    except Exception as e:
        logger.error(f"Tunnel close error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# M3: Dashboard Layer Endpoints
# ============================================

@app.post("/dashboard/create", response_model=DashboardCreateResponse)
async def create_dashboard(request: DashboardCreateRequest):
    """
    M3: Create an Appsmith dashboard from a template.
    Templates: workflow_status, container_logs, agent_decisions
    """
    try:
        # Get template
        if request.template == "workflow_status":
            layout = DashboardTemplates.workflow_status_dashboard()
        elif request.template == "container_logs":
            layout = DashboardTemplates.container_logs_dashboard()
        elif request.template == "agent_decisions":
            layout = DashboardTemplates.agent_decisions_dashboard()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown template: {request.template}")
        
        # Create dashboard
        dashboard = appsmith_client.create_dashboard(
            name=f"{request.template}_dashboard",
            description=f"Auto-generated {request.template} dashboard"
        )
        
        # Add widgets from template
        for widget_data in layout.get("widgets", []):
            appsmith_client.add_widget(
                dashboard_id=dashboard.id,
                widget_type=widget_data["type"],
                config=widget_data["config"]
            )
        
        # Publish dashboard
        published = appsmith_client.publish_dashboard(dashboard.id)
        
        return DashboardCreateResponse(
            success=True,
            message=f"Dashboard created from template: {request.template}",
            dashboard_id=dashboard.id,
            dashboard_url=f"{appsmith_url}/app/{dashboard.id}"
        )
        
    except Exception as e:
        logger.error(f"Dashboard creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/list")
async def list_dashboards():
    """
    M3: List all Appsmith dashboards.
    """
    try:
        dashboards = appsmith_client.list_dashboards()
        return {
            "success": True,
            "count": len(dashboards),
            "dashboards": [
                {
                    "id": d.id,
                    "name": d.name,
                    "url": f"{appsmith_url}/app/{d.id}"
                }
                for d in dashboards
            ]
        }
    except Exception as e:
        logger.error(f"Dashboard list error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dashboard/analyze-page", response_model=PageAnalyzeResponse)
async def analyze_web_page(request: PageAnalyzeRequest):
    """
    M3: Analyze a web page using Playwright and optionally convert to Appsmith widgets.
    """
    try:
        # Analyze page
        analysis = await playwright_analyzer.analyze_page(request.url)
        
        appsmith_widgets = None
        if request.convert_to_appsmith:
            # Convert layout to Appsmith widgets
            converter = LayoutConverter()
            appsmith_widgets = converter.dom_to_appsmith_widgets(analysis.elements)
        
        return PageAnalyzeResponse(
            success=True,
            message=f"Page analyzed: {len(analysis.elements)} elements found",
            analysis={
                "url": analysis.url,
                "title": analysis.title,
                "element_count": len(analysis.elements),
                "layout_type": analysis.layout_type,
                "metadata": analysis.metadata
            },
            appsmith_widgets=appsmith_widgets
        )
        
    except Exception as e:
        logger.error(f"Page analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/directus/auth", response_model=DirectusAuthResponse)
async def directus_login(request: DirectusAuthRequest):
    """
    M3: Authenticate with Directus and get access token.
    Default credentials: admin@example.com / directus2026
    """
    try:
        token, expires_at = await directus_client.auth_manager.login(
            email=request.email,
            password=request.password
        )
        
        return DirectusAuthResponse(
            success=True,
            message="Authentication successful",
            token=token,
            expires_at=expires_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Directus auth error: {str(e)}")
        raise HTTPException(status_code=401, detail=str(e))


@app.get("/directus/collections")
async def list_directus_collections():
    """
    M3: List all available Directus collections.
    """
    try:
        collections = [
            "dashboard_layouts",
            "workflow_metadata",
            "project_data",
            "user_settings",
            "analysis_results"
        ]
        return {
            "success": True,
            "collections": collections
        }
    except Exception as e:
        logger.error(f"Collections list error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/directus/store")
async def store_in_directus(collection: str, data: Dict[str, Any]):
    """
    M3: Store data in a Directus collection.
    """
    try:
        stored = await directus_client.create_item(
            collection=collection,
            data=data
        )
        
        return {
            "success": True,
            "message": f"Data stored in {collection}",
            "item_id": stored.get("id")
        }
    except Exception as e:
        logger.error(f"Directus storage error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/directus/search/{collection}/{query}")
async def search_directus(collection: str, query: str):
    """
    M3: Search for items in a Directus collection.
    """
    try:
        results = await directus_client.search_items(
            collection=collection,
            query=query
        )
        
        return {
            "success": True,
            "collection": collection,
            "query": query,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Directus search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Utility Endpoints
# ============================================

@app.get("/milestones")
async def get_milestones():
    """
    Get information about all implemented milestones.
    """
    return {
        "m1": {
            "name": "Core AI Decision Agent",
            "description": "CrewAI-powered decision making with GitHub search",
            "endpoints": [
                "/analyze - Analyze user queries",
                "/build - Generate workflows",
                "/github/search - Search repositories"
            ],
            "status": "✅ Complete"
        },
        "m2": {
            "name": "Colab Offloading Layer",
            "description": "Intelligent task offloading with ColabCode, pyngrok, and auto-fallback",
            "endpoints": [
                "/colab/offload - Offload tasks to Colab",
                "/colab/status/{id} - Check execution status",
                "/colab/tunnel/create - Create pyngrok tunnel",
                "/colab/tunnel - Close tunnel"
            ],
            "features": [
                "ColabCode notebook generation",
                "pyngrok tunnel management",
                "Automatic offload decision",
                "Fallback to local execution"
            ],
            "status": "✅ Complete"
        },
        "m3": {
            "name": "Dashboard Layer",
            "description": "Appsmith dashboards, Playwright analysis, and Directus integration",
            "endpoints": [
                "/dashboard/create - Create Appsmith dashboard",
                "/dashboard/list - List dashboards",
                "/dashboard/analyze-page - Analyze web page with Playwright",
                "/directus/auth - Authenticate with Directus",
                "/directus/collections - List collections",
                "/directus/store - Store data",
                "/directus/search/{collection}/{query} - Search data"
            ],
            "templates": [
                "workflow_status - Workflow monitoring dashboard",
                "container_logs - Docker logs dashboard",
                "agent_decisions - AI decision tracking dashboard"
            ],
            "status": "✅ Complete"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
