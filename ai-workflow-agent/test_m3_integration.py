"""
Milestone 3 Integration Tests
Tests for Colab Integration, Visual Workflow Builder, Project Templates, and Enhanced Assistant
"""

import pytest
import os
import sys
import json
import tempfile
import shutil
from datetime import datetime

# Add agent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agent'))

# ============================================
# Colab Connector Tests
# ============================================

class TestColabConnector:
    """Test Colab integration functionality"""
    
    def test_import_colab_connector(self):
        """Test that colab_connector module imports correctly"""
        from agent.colab_connector import (
            ColabTaskType, ColabStatus, ColabTask,
            ColabNotebookGenerator, ColabConnector, get_colab_connector
        )
        
        assert ColabTaskType is not None
        assert ColabStatus is not None
        assert ColabConnector is not None
    
    def test_colab_task_types(self):
        """Test task type enumeration"""
        from agent.colab_connector import ColabTaskType
        
        assert ColabTaskType.IMAGE_GENERATION.value == "image_generation"
        assert ColabTaskType.LLM_INFERENCE.value == "llm_inference"
        assert ColabTaskType.MODEL_TRAINING.value == "model_training"
        assert ColabTaskType.VIDEO_GENERATION.value == "video_generation"
        assert ColabTaskType.DATA_PROCESSING.value == "data_processing"
        assert ColabTaskType.AUDIO_PROCESSING.value == "audio_processing"
    
    def test_create_image_generation_task(self):
        """Test creating image generation task"""
        from agent.colab_connector import ColabConnector, ColabTaskType
        
        connector = ColabConnector()
        task = connector.create_task(
            task_type=ColabTaskType.IMAGE_GENERATION,
            parameters={
                "prompt": "A beautiful sunset over mountains",
                "model": "stabilityai/sdxl-turbo",
                "width": 1024,
                "height": 1024
            }
        )
        
        assert task is not None
        assert task.task_type == ColabTaskType.IMAGE_GENERATION
        assert task.parameters.get("prompt") == "A beautiful sunset over mountains"
        assert "width" in task.parameters
    
    def test_create_llm_inference_task(self):
        """Test creating LLM inference task"""
        from agent.colab_connector import ColabConnector, ColabTaskType
        
        connector = ColabConnector()
        task = connector.create_task(
            task_type=ColabTaskType.LLM_INFERENCE,
            parameters={
                "prompt": "Explain quantum computing",
                "model": "meta-llama/Llama-2-7b-chat-hf"
            }
        )
        
        assert task is not None
        assert task.task_type == ColabTaskType.LLM_INFERENCE
    
    def test_generate_notebook(self):
        """Test notebook generation"""
        from agent.colab_connector import ColabConnector, ColabTaskType
        
        connector = ColabConnector()
        task = connector.create_task(
            task_type=ColabTaskType.IMAGE_GENERATION,
            parameters={
                "prompt": "Test prompt",
                "steps": 30
            }
        )
        
        # The notebook code is stored in the task
        assert task.notebook_code is not None
        assert len(task.notebook_code) > 0
        
        # Can also get the full notebook
        notebook_content = connector.get_notebook_content(task.task_id)
        assert notebook_content is not None
    
    def test_task_list_and_stats(self):
        """Test task listing and statistics"""
        from agent.colab_connector import ColabConnector, ColabTaskType
        
        connector = ColabConnector()
        
        # Create a few tasks
        for i in range(3):
            connector.create_task(
                task_type=ColabTaskType.IMAGE_GENERATION,
                parameters={"prompt": f"Test prompt {i}"}
            )
        
        tasks = connector.get_tasks()
        assert len(tasks) >= 3
        
        stats = connector.get_stats()
        assert "total_tasks" in stats
        assert stats["total_tasks"] >= 3


# ============================================
# Visual Workflow Builder Tests
# ============================================

class TestWorkflowBuilder:
    """Test visual workflow builder functionality"""
    
    def test_import_workflow_builder(self):
        """Test that workflow_builder module imports correctly"""
        from agent.workflow_builder import (
            NodeCategory, DataType, NodePort, NodeDefinition,
            WorkflowNode, WorkflowConnection, VisualWorkflow,
            NodeRegistry, WorkflowBuilder, get_workflow_builder, get_node_registry
        )
        
        assert NodeCategory is not None
        assert WorkflowBuilder is not None
        assert NodeRegistry is not None
    
    def test_node_categories(self):
        """Test node category enumeration"""
        from agent.workflow_builder import NodeCategory
        
        assert NodeCategory.TRIGGER.value == "trigger"
        assert NodeCategory.ACTION.value == "action"
        assert NodeCategory.LOGIC.value == "logic"
        assert NodeCategory.DATA.value == "data"
        assert NodeCategory.AI.value == "ai"
        assert NodeCategory.INTEGRATION.value == "integration"
        assert NodeCategory.OUTPUT.value == "output"
    
    def test_node_registry_builtin_nodes(self):
        """Test that builtin nodes are registered"""
        from agent.workflow_builder import get_node_registry
        
        registry = get_node_registry()
        all_nodes = registry.list_all()
        
        assert len(all_nodes) >= 15  # Should have at least 15 builtin nodes
        
        # Check specific nodes exist (using correct node_type format)
        webhook_trigger = registry.get("trigger.webhook")
        assert webhook_trigger is not None
        assert webhook_trigger.name == "Webhook Trigger"
        
        http_request = registry.get("action.http")
        assert http_request is not None
        
        generate_image = registry.get("ai.generate_image")
        assert generate_image is not None
    
    def test_create_visual_workflow(self):
        """Test creating a visual workflow"""
        from agent.workflow_builder import get_workflow_builder
        
        builder = get_workflow_builder()
        workflow = builder.create_workflow(
            name="Test Workflow",
            description="A test workflow"
        )
        
        assert workflow is not None
        assert workflow.name == "Test Workflow"
        assert workflow.description == "A test workflow"
        assert len(workflow.nodes) == 0
        assert len(workflow.connections) == 0
    
    def test_add_nodes_to_workflow(self):
        """Test adding nodes to a workflow"""
        from agent.workflow_builder import get_workflow_builder
        
        builder = get_workflow_builder()
        workflow = builder.create_workflow(name="Node Test")
        
        # Add trigger node
        trigger = builder.add_node(
            workflow_id=workflow.workflow_id,
            node_type="trigger.webhook",
            position={"x": 100, "y": 100}
        )
        assert trigger is not None
        assert trigger.node_type == "trigger.webhook"
        
        # Add action node
        action = builder.add_node(
            workflow_id=workflow.workflow_id,
            node_type="action.http",
            position={"x": 300, "y": 100},
            properties={"url": "https://api.example.com"}
        )
        assert action is not None
        assert "url" in action.properties
        
        assert len(workflow.nodes) == 2
    
    def test_connect_nodes(self):
        """Test connecting nodes in a workflow"""
        from agent.workflow_builder import get_workflow_builder
        
        builder = get_workflow_builder()
        workflow = builder.create_workflow(name="Connection Test")
        
        trigger = builder.add_node(
            workflow_id=workflow.workflow_id,
            node_type="trigger.webhook",
            position={"x": 0, "y": 0}
        )
        action = builder.add_node(
            workflow_id=workflow.workflow_id,
            node_type="action.http",
            position={"x": 200, "y": 0}
        )
        
        # Connect webhook body output to HTTP body input
        connection = builder.add_connection(
            workflow_id=workflow.workflow_id,
            source_node=trigger.node_id,
            source_port="body",
            target_node=action.node_id,
            target_port="body"
        )
        
        assert connection is not None
        assert connection.source_node == trigger.node_id
        assert connection.target_node == action.node_id
        assert len(workflow.connections) == 1
    
    def test_workflow_validation(self):
        """Test workflow validation"""
        from agent.workflow_builder import get_workflow_builder
        
        builder = get_workflow_builder()
        workflow = builder.create_workflow(name="Validation Test")
        
        # Empty workflow - validation returns dict with warnings
        result = builder.validate_workflow(workflow.workflow_id)
        assert "valid" in result
        # may have warnings about no trigger
        
        # Add a trigger
        trigger = builder.add_node(
            workflow_id=workflow.workflow_id,
            node_type="trigger.webhook",
            position={"x": 0, "y": 0}
        )
        result = builder.validate_workflow(workflow.workflow_id)
        assert "valid" in result
    
    def test_export_to_n8n_format(self):
        """Test exporting workflow to n8n format"""
        from agent.workflow_builder import get_workflow_builder
        
        builder = get_workflow_builder()
        workflow = builder.create_workflow(name="Export Test")
        
        trigger = builder.add_node(
            workflow_id=workflow.workflow_id,
            node_type="trigger.webhook",
            position={"x": 0, "y": 0}
        )
        action = builder.add_node(
            workflow_id=workflow.workflow_id,
            node_type="action.http",
            position={"x": 200, "y": 0}
        )
        builder.add_connection(
            workflow_id=workflow.workflow_id,
            source_node=trigger.node_id,
            source_port="body",
            target_node=action.node_id,
            target_port="body"
        )
        
        exported = builder.export_workflow(workflow.workflow_id, format="n8n")
        
        assert exported is not None
        assert "nodes" in exported
        assert "connections" in exported
    
    def test_export_to_comfyui_format(self):
        """Test exporting workflow to ComfyUI format"""
        from agent.workflow_builder import get_workflow_builder
        
        builder = get_workflow_builder()
        workflow = builder.create_workflow(name="ComfyUI Export Test")
        
        gen_image = builder.add_node(
            workflow_id=workflow.workflow_id,
            node_type="ai.generate_image",
            position={"x": 0, "y": 0}
        )
        save_image = builder.add_node(
            workflow_id=workflow.workflow_id,
            node_type="output.save_image",
            position={"x": 200, "y": 0}
        )
        # Connect image output to image input
        builder.add_connection(
            workflow_id=workflow.workflow_id,
            source_node=gen_image.node_id,
            source_port="image",
            target_node=save_image.node_id,
            target_port="images"
        )
        
        exported = builder.export_workflow(workflow.workflow_id, format="comfyui")
        
        assert exported is not None


# ============================================
# Project Templates Tests
# ============================================

class TestProjectTemplates:
    """Test project templates functionality"""
    
    def test_import_project_templates(self):
        """Test that project_templates module imports correctly"""
        from agent.project_templates import (
            ProjectCategory, Difficulty, ProjectFile, ProjectTemplate,
            ProjectTemplateRegistry, ProjectScaffolder,
            get_template_registry, get_scaffolder
        )
        
        assert ProjectCategory is not None
        assert ProjectTemplate is not None
        assert ProjectTemplateRegistry is not None
    
    def test_project_categories(self):
        """Test project category enumeration"""
        from agent.project_templates import ProjectCategory
        
        assert ProjectCategory.AUTOMATION.value == "automation"
        assert ProjectCategory.AI_IMAGE.value == "ai_image"
        assert ProjectCategory.AI_TEXT.value == "ai_text"
        assert ProjectCategory.DATA_PIPELINE.value == "data_pipeline"
        assert ProjectCategory.WEB_SCRAPING.value == "web_scraping"
        assert ProjectCategory.CHATBOT.value == "chatbot"
        assert ProjectCategory.MONITORING.value == "monitoring"
    
    def test_builtin_templates_exist(self):
        """Test that builtin templates are registered"""
        from agent.project_templates import get_template_registry
        
        registry = get_template_registry()
        all_templates = registry.list_all()
        
        assert len(all_templates) >= 5  # Should have at least 5 builtin templates
        
        # Check specific templates exist
        email_template = registry.get("email-automation")
        assert email_template is not None
        assert email_template.name == "Email Automation Hub"
        
        image_template = registry.get("ai-image-generator")
        assert image_template is not None
    
    def test_list_templates_by_category(self):
        """Test listing templates by category"""
        from agent.project_templates import get_template_registry, ProjectCategory
        
        registry = get_template_registry()
        
        automation_templates = registry.list_by_category(ProjectCategory.AUTOMATION)
        assert len(automation_templates) >= 1
        
        ai_image_templates = registry.list_by_category(ProjectCategory.AI_IMAGE)
        assert len(ai_image_templates) >= 1
    
    def test_search_templates(self):
        """Test searching templates"""
        from agent.project_templates import get_template_registry
        
        registry = get_template_registry()
        
        results = registry.search("image")
        assert len(results) >= 1
        
        results = registry.search("automation")
        assert len(results) >= 1
    
    def test_preview_template(self):
        """Test template preview"""
        from agent.project_templates import get_scaffolder
        
        scaffolder = get_scaffolder()
        preview = scaffolder.preview("email-automation")
        
        assert preview["success"]
        assert "template" in preview
        assert "files" in preview
        assert len(preview["files"]) > 0
    
    def test_scaffold_project(self):
        """Test scaffolding a project"""
        from agent.project_templates import get_scaffolder
        
        scaffolder = get_scaffolder()
        
        # Use temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            result = scaffolder.scaffold(
                template_id="email-automation",
                project_name="test-project",
                output_dir=tmpdir,
                variables={"EMAIL_HOST": "imap.test.com"}
            )
            
            assert result["success"]
            assert "project_path" in result
            assert "files_created" in result
            assert len(result["files_created"]) > 0
            
            # Verify files were created
            project_path = result["project_path"]
            assert os.path.exists(project_path)
            assert os.path.exists(os.path.join(project_path, "README.md"))
    
    def test_scaffold_with_variables(self):
        """Test variable substitution in scaffolding"""
        from agent.project_templates import get_scaffolder
        
        scaffolder = get_scaffolder()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = scaffolder.scaffold(
                template_id="ai-image-generator",
                project_name="my-image-project",
                output_dir=tmpdir,
                variables={
                    "MODEL_NAME": "custom-model.safetensors",
                    "COMFYUI_URL": "http://localhost:9999"
                }
            )
            
            assert result["success"]


# ============================================
# Enhanced Assistant Tests
# ============================================

class TestEnhancedAssistant:
    """Test enhanced assistant functionality"""
    
    def test_import_assistant(self):
        """Test that assistant_enhanced module imports correctly"""
        from agent.assistant_enhanced import (
            ConversationRole, MessageIntent, ActionType,
            ConversationMessage, ActionSuggestion,
            ConversationMemory, IntentClassifier, ActionSuggester,
            EnhancedAssistant, QuickAction, get_assistant
        )
        
        assert EnhancedAssistant is not None
        assert IntentClassifier is not None
        assert QuickAction is not None
    
    def test_message_intents(self):
        """Test message intent enumeration"""
        from agent.assistant_enhanced import MessageIntent
        
        assert MessageIntent.CREATE_WORKFLOW.value == "create_workflow"
        assert MessageIntent.GENERATE_IMAGE.value == "generate_image"
        assert MessageIntent.QUERY_STATUS.value == "query_status"
        assert MessageIntent.GET_HELP.value == "get_help"
        assert MessageIntent.CHAT.value == "chat"
    
    def test_intent_classification_workflow(self):
        """Test intent classification for workflow creation"""
        from agent.assistant_enhanced import IntentClassifier, MessageIntent
        
        # Workflow creation intents
        workflow_messages = [
            "Create a workflow to process emails",
            "I want to automate my Slack notifications",
            "Build an n8n automation for data sync"
        ]
        
        for msg in workflow_messages:
            intent = IntentClassifier.classify(msg)
            assert intent == MessageIntent.CREATE_WORKFLOW, f"Failed for: {msg}"
    
    def test_intent_classification_image(self):
        """Test intent classification for image generation"""
        from agent.assistant_enhanced import IntentClassifier, MessageIntent
        
        image_messages = [
            "Generate an image of a sunset",
            "Create a picture of a cat",
            "Draw an illustration of mountains"
        ]
        
        for msg in image_messages:
            intent = IntentClassifier.classify(msg)
            assert intent == MessageIntent.GENERATE_IMAGE, f"Failed for: {msg}"
    
    def test_intent_classification_help(self):
        """Test intent classification for help requests"""
        from agent.assistant_enhanced import IntentClassifier, MessageIntent
        
        help_messages = [
            "How do I create a workflow?",
            "Help me understand the system",
            "What is n8n?"
        ]
        
        for msg in help_messages:
            intent = IntentClassifier.classify(msg)
            assert intent == MessageIntent.GET_HELP, f"Failed for: {msg}"
    
    def test_conversation_memory(self):
        """Test conversation memory functionality"""
        from agent.assistant_enhanced import ConversationMemory, ConversationRole
        
        memory = ConversationMemory()
        
        memory.add_message(ConversationRole.USER, "Hello")
        memory.add_message(ConversationRole.ASSISTANT, "Hi there!")
        memory.add_message(ConversationRole.USER, "How are you?")
        
        assert len(memory.messages) == 3
        
        context = memory.get_context()
        assert "Hello" in context
        assert "Hi there" in context
        
        summary = memory.get_summary()
        assert summary["message_count"] == 3
    
    def test_enhanced_assistant_process_message(self):
        """Test processing a message through the assistant"""
        from agent.assistant_enhanced import EnhancedAssistant
        
        assistant = EnhancedAssistant()
        
        result = assistant.process_message("Create a workflow to send notifications")
        
        assert "intent" in result
        assert "response" in result
        assert "actions" in result
        assert "context" in result
        
        assert result["intent"] == "create_workflow"
        assert len(result["response"]) > 0
    
    def test_action_suggestions(self):
        """Test action suggestion generation"""
        from agent.assistant_enhanced import ActionSuggester, MessageIntent
        
        suggester = ActionSuggester()
        
        suggestions = suggester.suggest(
            text="Generate an image of mountains at sunset",
            intent=MessageIntent.GENERATE_IMAGE
        )
        
        assert len(suggestions) >= 1
        assert suggestions[0].action_type.value == "queue_comfyui"
    
    def test_quick_actions(self):
        """Test quick actions"""
        from agent.assistant_enhanced import QuickAction
        
        all_actions = QuickAction.get_all()
        assert len(all_actions) >= 4
        
        portrait_action = QuickAction.get_action("generate_portrait")
        assert portrait_action is not None
        assert "default_params" in portrait_action
        
        email_action = QuickAction.get_action("email_workflow")
        assert email_action is not None
    
    def test_assistant_state_management(self):
        """Test assistant state management"""
        from agent.assistant_enhanced import EnhancedAssistant
        
        assistant = EnhancedAssistant()
        
        # Process some messages
        assistant.process_message("Hello")
        assistant.process_message("Create a workflow")
        
        state = assistant.get_state()
        assert state["message_count"] == 4  # 2 user + 2 assistant
        
        # Clear context
        assistant.clear_context()
        state = assistant.get_state()
        assert state["message_count"] == 0


# ============================================
# API Endpoint Tests (using TestClient)
# ============================================

class TestM3APIEndpoints:
    """Test M3 API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        # Only run if fastapi and pytest-asyncio are available
        try:
            from fastapi.testclient import TestClient
            # Add agent to path
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agent'))
            from agent.main import app
            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI TestClient not available")
    
    def test_colab_task_endpoint(self, client):
        """Test Colab task creation endpoint"""
        response = client.post("/colab/task", json={
            "task_type": "image_generation",
            "prompt": "Test prompt",
            "parameters": {"width": 512}
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "task" in data
    
    def test_builder_workflow_endpoint(self, client):
        """Test visual workflow builder endpoint"""
        # Create workflow
        response = client.post("/builder/workflow?name=Test%20API%20Workflow")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "workflow" in data
    
    def test_builder_nodes_endpoint(self, client):
        """Test listing available nodes"""
        response = client.get("/builder/nodes")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "nodes" in data
        assert data["count"] >= 15
    
    def test_templates_endpoint(self, client):
        """Test templates listing endpoint"""
        response = client.get("/templates")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "templates" in data
        assert data["count"] >= 5
    
    def test_template_preview_endpoint(self, client):
        """Test template preview endpoint"""
        response = client.get("/templates/email-automation/preview")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "files" in data
    
    def test_assistant_chat_endpoint(self, client):
        """Test assistant chat endpoint"""
        response = client.post("/assistant/chat", json={
            "message": "Help me create a workflow"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "intent" in data
        assert "response" in data
    
    def test_assistant_quick_actions_endpoint(self, client):
        """Test quick actions endpoint"""
        response = client.get("/assistant/quick-actions")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "actions" in data
        assert len(data["actions"]) >= 4
    
    def test_m3_dashboard_endpoint(self, client):
        """Test M3 dashboard endpoint"""
        response = client.get("/m3/dashboard")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "colab" in data
        assert "builder" in data
        assert "templates" in data
        assert "assistant" in data


# ============================================
# Run Tests
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
