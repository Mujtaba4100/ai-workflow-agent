# AI Workflow Agent - Main API
"""
Main FastAPI application for the AI Workflow Agent.
Provides endpoints for natural language workflow generation.
Milestone 1: Added chat interface and session management.
Milestone 2: Added workflow execution, monitoring, webhooks, and notifications.
"""

from fastapi import FastAPI, HTTPException, Header, Request
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

# M2 imports
from workflow_executor import get_executor
from workflow_monitor import get_monitor
from webhook_receiver import get_webhook_receiver, WebhookSource
from notifications import get_notification_manager, NotificationType, NotificationPriority
from storage import get_storage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Workflow Agent",
    description="Intelligent assistant for creating n8n, ComfyUI, and hybrid workflows",
    version="2.0.0"  # Milestone 2
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

# M2 components (lazy-loaded singletons)
workflow_executor = get_executor()
workflow_monitor = get_monitor()
webhook_receiver = get_webhook_receiver()
notification_manager = get_notification_manager()
storage = get_storage()


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
# Workflow Execution Endpoints (Milestone 2)
# ============================================

class ExecuteWorkflowRequest(BaseModel):
    """Request to execute a workflow."""
    workflow_json: Dict[str, Any]
    workflow_id: Optional[str] = None
    workflow_name: str = "Unnamed"
    test_mode: bool = False
    wait_for_completion: bool = True


class ExecuteHybridRequest(BaseModel):
    """Request to execute a hybrid workflow."""
    n8n_workflow: Dict[str, Any]
    comfyui_workflow: Dict[str, Any]
    workflow_id: Optional[str] = None
    workflow_name: str = "Unnamed Hybrid"


@app.post("/execute/n8n")
async def execute_n8n_workflow(request: ExecuteWorkflowRequest):
    """
    Execute an n8n workflow with monitoring.
    
    Returns execution result with status, output, and duration.
    Execution is tracked in history and triggers notifications.
    """
    try:
        result = await workflow_monitor.execute_n8n(
            workflow_json=request.workflow_json,
            workflow_id=request.workflow_id,
            workflow_name=request.workflow_name,
            test_mode=request.test_mode
        )
        return {
            "success": result.status.value == "completed",
            "execution": result.to_dict()
        }
    except Exception as e:
        logger.error(f"n8n execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute/comfyui")
async def execute_comfyui_workflow(request: ExecuteWorkflowRequest):
    """
    Execute a ComfyUI workflow with monitoring.
    
    Queues the prompt and optionally waits for generation to complete.
    """
    try:
        result = await workflow_monitor.execute_comfyui(
            workflow_json=request.workflow_json,
            workflow_id=request.workflow_id,
            workflow_name=request.workflow_name,
            wait_for_completion=request.wait_for_completion
        )
        return {
            "success": result.status.value == "completed",
            "execution": result.to_dict()
        }
    except Exception as e:
        logger.error(f"ComfyUI execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute/hybrid")
async def execute_hybrid_workflow(request: ExecuteHybridRequest):
    """
    Execute a hybrid workflow (n8n + ComfyUI).
    
    Executes n8n first, then ComfyUI. Both results are combined.
    """
    try:
        result = await workflow_monitor.execute_hybrid(
            n8n_workflow=request.n8n_workflow,
            comfyui_workflow=request.comfyui_workflow,
            workflow_id=request.workflow_id,
            workflow_name=request.workflow_name
        )
        return {
            "success": result.status.value == "completed",
            "execution": result.to_dict()
        }
    except Exception as e:
        logger.error(f"Hybrid execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute/cancel/{execution_id}")
async def cancel_execution(execution_id: str):
    """Cancel a running workflow execution."""
    success = workflow_monitor.cancel_execution(execution_id)
    if not success:
        raise HTTPException(status_code=404, detail="Execution not found or not running")
    return {"success": True, "message": f"Execution {execution_id} cancelled"}


@app.get("/execute/status/{execution_id}")
async def get_execution_status(execution_id: str):
    """Get status of a specific workflow execution."""
    status = workflow_monitor.get_execution_status(execution_id)
    if not status:
        raise HTTPException(status_code=404, detail="Execution not found")
    return {"success": True, "execution": status}


@app.get("/execute/running")
async def get_running_executions():
    """Get all currently running workflow executions."""
    return {
        "success": True,
        "running": workflow_monitor.get_running_executions()
    }


# ============================================
# Monitoring & Dashboard Endpoints (Milestone 2)
# ============================================

@app.get("/monitor/stats")
async def get_monitoring_stats():
    """
    Get workflow execution statistics.
    
    Returns totals, success rates, average durations, and errors.
    """
    stats = workflow_monitor.get_stats()
    return {
        "success": True,
        "stats": stats.to_dict()
    }


@app.get("/monitor/dashboard")
async def get_dashboard():
    """
    Get complete dashboard data.
    
    Combines stats, running executions, recent history, and notifications.
    """
    return {
        "success": True,
        "dashboard": workflow_monitor.get_dashboard_data()
    }


@app.get("/monitor/history")
async def get_execution_history(
    limit: int = 20,
    workflow_id: Optional[str] = None
):
    """
    Get workflow execution history.
    
    Optionally filter by workflow_id.
    """
    if workflow_id:
        history = workflow_monitor.get_executions_by_workflow(workflow_id, limit)
    else:
        history = workflow_monitor.get_recent_executions(limit)
    
    return {
        "success": True,
        "count": len(history),
        "history": history
    }


# ============================================
# Webhook Endpoints (Milestone 2)
# ============================================

class CreateWebhookRequest(BaseModel):
    """Request to create a webhook endpoint."""
    name: str
    source: str = "custom"  # n8n, comfyui, github, custom
    secret: Optional[str] = None
    allowed_events: Optional[List[str]] = None


@app.post("/webhook/create")
async def create_webhook(request: CreateWebhookRequest):
    """
    Create a new webhook endpoint.
    
    Returns webhook_id to use for receiving webhooks.
    """
    try:
        source = WebhookSource(request.source) if request.source in [s.value for s in WebhookSource] else WebhookSource.CUSTOM
        config = webhook_receiver.create_webhook(
            name=request.name,
            source=source,
            secret=request.secret,
            allowed_events=request.allowed_events
        )
        return {
            "success": True,
            "webhook_id": config.webhook_id,
            "name": config.name,
            "source": config.source.value,
            "endpoint": f"/webhook/receive/{config.webhook_id}"
        }
    except Exception as e:
        logger.error(f"Create webhook error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/webhook/list")
async def list_webhooks(source: Optional[str] = None):
    """List all webhook endpoints."""
    source_filter = WebhookSource(source) if source else None
    webhooks = webhook_receiver.list_webhooks(source_filter)
    return {
        "success": True,
        "webhooks": [{
            "webhook_id": w.webhook_id,
            "name": w.name,
            "source": w.source.value,
            "is_active": w.is_active,
            "endpoint": f"/webhook/receive/{w.webhook_id}"
        } for w in webhooks]
    }


@app.post("/webhook/receive/{webhook_id}")
async def receive_webhook(
    webhook_id: str,
    request: Request,
    x_github_event: Optional[str] = Header(None),
    x_hub_signature_256: Optional[str] = Header(None),
    x_webhook_signature: Optional[str] = Header(None)
):
    """
    Receive a webhook payload.
    
    Supports GitHub, n8n, and custom webhooks.
    Signature verification if secret is configured.
    """
    try:
        payload = await request.json()
        headers = dict(request.headers)
        
        event_type = x_github_event  # GitHub event type
        
        event = await webhook_receiver.receive_webhook(
            webhook_id=webhook_id,
            payload=payload,
            headers=headers,
            event_type=event_type
        )
        
        # Notify about webhook
        await notification_manager.notify_webhook_received(
            webhook_id=webhook_id,
            source=event.source.value,
            event_type=event.event_type
        )
        
        # Store in persistent storage
        storage.save_webhook_event(
            event_id=event.event_id,
            webhook_id=webhook_id,
            source=event.source.value,
            payload=payload,
            event_type=event.event_type
        )
        
        return {
            "success": True,
            "event_id": event.event_id,
            "processed": event.processed,
            "result": event.result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Webhook receive error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook/raw/{source}")
async def receive_raw_webhook(source: str, request: Request):
    """
    Receive a webhook without pre-configured endpoint.
    
    Use for testing or dynamic webhook handling.
    """
    try:
        payload = await request.json()
        headers = dict(request.headers)
        
        event = await webhook_receiver.receive_raw_webhook(
            source=source,
            payload=payload,
            headers=headers
        )
        
        return {
            "success": True,
            "event_id": event.event_id,
            "source": event.source.value,
            "event_type": event.event_type,
            "processed": event.processed
        }
    except Exception as e:
        logger.error(f"Raw webhook error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/webhook/events")
async def get_webhook_events(
    webhook_id: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 50
):
    """Get recent webhook events."""
    source_filter = WebhookSource(source) if source else None
    events = webhook_receiver.get_events(
        webhook_id=webhook_id,
        source=source_filter,
        limit=limit
    )
    return {
        "success": True,
        "count": len(events),
        "events": [{
            "event_id": e.event_id,
            "webhook_id": e.webhook_id,
            "source": e.source.value,
            "event_type": e.event_type,
            "processed": e.processed,
            "received_at": e.received_at.isoformat()
        } for e in events]
    }


@app.delete("/webhook/{webhook_id}")
async def delete_webhook(webhook_id: str):
    """Delete a webhook endpoint."""
    if not webhook_receiver.delete_webhook(webhook_id):
        raise HTTPException(status_code=404, detail="Webhook not found")
    return {"success": True, "message": f"Webhook {webhook_id} deleted"}


# ============================================
# Notification Endpoints (Milestone 2)
# ============================================

class CreateNotificationRequest(BaseModel):
    """Request to create a notification."""
    title: str
    message: str = ""
    notification_type: str = "info"  # info, success, warning, error
    priority: str = "normal"  # low, normal, high, urgent
    data: Optional[Dict[str, Any]] = None


@app.post("/notifications/create")
async def create_notification(request: CreateNotificationRequest):
    """Create a new notification."""
    try:
        ntype = NotificationType(request.notification_type) if request.notification_type in [t.value for t in NotificationType] else NotificationType.INFO
        priority = NotificationPriority(request.priority) if request.priority in [p.value for p in NotificationPriority] else NotificationPriority.NORMAL
        
        notification = await notification_manager.notify(
            title=request.title,
            message=request.message,
            notification_type=ntype,
            priority=priority,
            data=request.data
        )
        return {
            "success": True,
            "notification": notification.to_dict()
        }
    except Exception as e:
        logger.error(f"Create notification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/notifications")
async def get_notifications(
    unread_only: bool = False,
    notification_type: Optional[str] = None,
    limit: int = 50
):
    """Get notifications with optional filters."""
    ntype = NotificationType(notification_type) if notification_type else None
    notifications = notification_manager.get_notifications(
        notification_type=ntype,
        unread_only=unread_only,
        limit=limit
    )
    return {
        "success": True,
        "count": len(notifications),
        "unread_count": notification_manager.get_unread_count(),
        "notifications": [n.to_dict() for n in notifications]
    }


@app.get("/notifications/stats")
async def get_notification_stats():
    """Get notification statistics."""
    return {
        "success": True,
        "stats": notification_manager.get_stats()
    }


@app.post("/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str):
    """Mark a notification as read."""
    if not notification_manager.mark_read(notification_id):
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"success": True, "message": "Notification marked as read"}


@app.post("/notifications/read-all")
async def mark_all_notifications_read():
    """Mark all notifications as read."""
    count = notification_manager.mark_all_read()
    return {"success": True, "marked_read": count}


@app.delete("/notifications/{notification_id}")
async def delete_notification(notification_id: str):
    """Delete a notification."""
    if not notification_manager.delete_notification(notification_id):
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"success": True, "message": "Notification deleted"}


@app.delete("/notifications/clear")
async def clear_all_notifications():
    """Clear all notifications."""
    count = notification_manager.clear_all()
    return {"success": True, "cleared": count}


# ============================================
# Storage/History Endpoints (Milestone 2)
# ============================================

class SaveWorkflowRequest(BaseModel):
    """Request to save a workflow definition."""
    workflow_id: str
    name: str
    workflow_type: str  # n8n, comfyui, hybrid
    workflow_json: Dict[str, Any]
    description: str = ""


@app.post("/workflows/save")
async def save_workflow(request: SaveWorkflowRequest):
    """Save a workflow definition to storage."""
    success = storage.save_workflow(
        workflow_id=request.workflow_id,
        name=request.name,
        workflow_type=request.workflow_type,
        workflow_json=request.workflow_json,
        description=request.description
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save workflow")
    return {"success": True, "message": f"Workflow {request.workflow_id} saved"}


@app.get("/workflows/saved")
async def list_saved_workflows(workflow_type: Optional[str] = None):
    """List all saved workflows."""
    workflows = storage.list_workflows(workflow_type)
    return {
        "success": True,
        "count": len(workflows),
        "workflows": workflows
    }


@app.get("/workflows/saved/{workflow_id}")
async def get_saved_workflow(workflow_id: str):
    """Get a saved workflow by ID."""
    workflow = storage.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return {"success": True, "workflow": workflow}


@app.delete("/workflows/saved/{workflow_id}")
async def delete_saved_workflow(workflow_id: str):
    """Delete a saved workflow."""
    if not storage.delete_workflow(workflow_id):
        raise HTTPException(status_code=404, detail="Workflow not found")
    return {"success": True, "message": f"Workflow {workflow_id} deleted"}


@app.get("/settings/{key}")
async def get_setting(key: str):
    """Get a stored setting value."""
    value = storage.get_setting(key)
    return {"success": True, "key": key, "value": value}


@app.post("/settings/{key}")
async def set_setting(key: str, value: Any):
    """Set a setting value."""
    success = storage.set_setting(key, value)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save setting")
    return {"success": True, "key": key, "message": "Setting saved"}


@app.get("/settings")
async def get_all_settings():
    """Get all stored settings."""
    return {"success": True, "settings": storage.get_all_settings()}


# ============================================
# Startup/Shutdown Events
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting AI Workflow Agent v2.0.0 (M2)...")
    logger.info(f"Ollama: {settings.OLLAMA_HOST}")
    logger.info(f"n8n: {settings.N8N_HOST}")
    logger.info(f"ComfyUI: {settings.COMFYUI_HOST}")
    
    # Pull Qwen2.5 model if not present
    await decision_agent.ensure_model_available()
    
    # Start workflow monitoring
    await workflow_monitor.start_monitoring(interval=60)
    logger.info("Workflow monitoring started")
    
    # M2 initialization notification
    await notification_manager.notify(
        title="System Started",
        message="AI Workflow Agent v2.0.0 is ready",
        notification_type=NotificationType.SYSTEM
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down AI Workflow Agent...")
    
    # Stop monitoring
    await workflow_monitor.stop_monitoring()
    
    # Close executor HTTP client
    await workflow_executor.close()
    
    logger.info("Shutdown complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
