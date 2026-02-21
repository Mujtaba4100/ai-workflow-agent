# AI Workflow Agent - Milestone 1 Tests
"""
Test script for Milestone 1 features.
"""

import pytest
import sys
import os

# Add agent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent"))


def test_chat_handler_session_management():
    """Test chat handler session creation and management."""
    print("\n" + "="*50)
    print("TEST: Chat Handler Session Management")
    print("="*50)
    
    from chat_handler import SessionManager, ConversationState
    
    # Create session manager
    manager = SessionManager()
    
    # Create a session
    session = manager.create_session()
    assert session is not None
    assert session.session_id is not None
    print(f"✅ Created session: {session.session_id}...")
    
    # Verify initial state (state is stored as string)
    assert session.state == ConversationState.INITIAL.value
    print(f"✅ Initial state: {session.state}")
    
    # Get session
    retrieved = manager.get_session(session.session_id)
    assert retrieved is not None
    assert retrieved.session_id == session.session_id
    print(f"✅ Retrieved session successfully")
    
    # Delete session
    deleted = manager.delete_session(session.session_id)
    assert deleted is True
    assert manager.get_session(session.session_id) is None
    print(f"✅ Deleted session successfully")


def test_conversation_states():
    """Test conversation state transitions."""
    print("\n" + "="*50)
    print("TEST: Conversation States")
    print("="*50)
    
    from chat_handler import ConversationState
    
    states = [
        ConversationState.INITIAL,
        ConversationState.ANALYZING,
        ConversationState.CLARIFYING,
        ConversationState.PLANNING,
        ConversationState.BUILDING,
        ConversationState.COMPLETE,
        ConversationState.ERROR
    ]
    
    for state in states:
        print(f"✅ State available: {state.value}")
    
    assert len(states) == 7
    print(f"✅ All 7 conversation states verified")


def test_workflow_templates():
    """Test n8n workflow templates."""
    print("\n" + "="*50)
    print("TEST: N8N Workflow Templates")
    print("="*50)
    
    from tools.workflow_templates import get_workflow_templates
    
    templates = get_workflow_templates()
    expected_templates = [
        "database_sync",
        "file_processor",
        "social_media",
        "crm_integration",
        "monitoring",
        "data_pipeline",
        "chatbot",
        "report_generator"
    ]
    
    for template_name in expected_templates:
        assert template_name in templates
        # Generate template
        workflow = templates[template_name]({})
        assert "name" in workflow or isinstance(workflow, dict)
        print(f"✅ Template '{template_name}' generates valid workflow")
    
    print(f"✅ All {len(expected_templates)} templates verified")


def test_comfyui_templates():
    """Test ComfyUI workflow templates."""
    print("\n" + "="*50)
    print("TEST: ComfyUI Templates")
    print("="*50)
    
    from tools.comfyui_templates import get_comfyui_templates
    
    templates = get_comfyui_templates()
    expected_templates = [
        "text_to_image",
        "image_to_image",
        "inpainting",
        "upscale",
        "controlnet",
        "batch_generation",
        "style_transfer",
        "lora_generation"
    ]
    
    for template_name in expected_templates:
        assert template_name in templates
        # Generate template
        workflow = templates[template_name]({"prompt": "test"})
        assert isinstance(workflow, dict)
        print(f"✅ Template '{template_name}' generates valid workflow")
    
    print(f"✅ All {len(expected_templates)} templates verified")


def test_web_search_tool_initialization():
    """Test web search tool can be initialized."""
    print("\n" + "="*50)
    print("TEST: Web Search Tool")
    print("="*50)
    
    from tools.web_search import WebSearchTool
    
    tool = WebSearchTool()
    assert tool is not None
    print(f"✅ WebSearchTool initialized successfully")


def test_crewai_imports():
    """Test CrewAI components can be imported."""
    print("\n" + "="*50)
    print("TEST: CrewAI Imports")
    print("="*50)
    
    from crew_agents import (
        CrewAIAgentSystem,
        AgentRole,
        ConversationContext,
        get_crew_agent_system
    )
    
    # Test AgentRole enum
    roles = [AgentRole.ANALYZER, AgentRole.PLANNER, AgentRole.BUILDER, AgentRole.VALIDATOR]
    for role in roles:
        print(f"✅ Role: {role.value}")
    
    # Test ConversationContext
    ctx = ConversationContext(user_query="test query")
    assert ctx.user_query == "test query"
    assert ctx.clarifications == []
    assert ctx.requirements == {}
    print(f"✅ ConversationContext works correctly")


def test_fastapi_app():
    """Test FastAPI app has all expected routes."""
    print("\n" + "="*50)
    print("TEST: FastAPI Routes")
    print("="*50)
    
    from main import app
    
    # Get all route paths
    routes = [route.path for route in app.routes if hasattr(route, 'path')]
    
    expected_routes = [
        "/",
        "/health",
        "/analyze",
        "/build",
        "/chat",
        "/github/search"
    ]
    
    for route in expected_routes:
        assert route in routes, f"Missing route: {route}"
        print(f"✅ Route available: {route}")
    
    print(f"✅ App has {len(routes)} routes total")
    print(f"✅ App version: {app.version}")


def main():
    """Run all M1 tests."""
    print("="*50)
    print("MILESTONE 1 - TEST SUITE")
    print("="*50)
    
    test_chat_handler_session_management()
    test_conversation_states()
    test_workflow_templates()
    test_comfyui_templates()
    test_web_search_tool_initialization()
    test_crewai_imports()
    test_fastapi_app()
    
    print("\n" + "="*50)
    print("ALL MILESTONE 1 TESTS PASSED ✅")
    print("="*50)


if __name__ == "__main__":
    main()
