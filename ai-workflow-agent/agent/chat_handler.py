# AI Workflow Agent - Chat API
"""
Conversational interface for the AI Workflow Agent.
Supports multi-turn conversations with session management.
"""

import uuid
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """Message roles in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationState(Enum):
    """Current state of conversation."""
    INITIAL = "initial"
    ANALYZING = "analyzing"
    CLARIFYING = "clarifying"
    PLANNING = "planning"
    BUILDING = "building"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class Message:
    """Single message in conversation."""
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """Conversation session."""
    session_id: str
    created_at: str
    state: str = ConversationState.INITIAL.value
    messages: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    project_type: Optional[str] = None
    workflow: Optional[Dict[str, Any]] = None
    pending_questions: List[str] = field(default_factory=list)
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add a message to the conversation."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
    
    def get_history_text(self, limit: int = 10) -> str:
        """Get conversation history as text for LLM context."""
        recent = self.messages[-limit:]
        lines = []
        for msg in recent:
            role = msg["role"].upper()
            content = msg["content"]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class SessionManager:
    """Manages conversation sessions."""
    
    def __init__(self, max_sessions: int = 100):
        self.sessions: Dict[str, Session] = {}
        self.max_sessions = max_sessions
    
    def create_session(self) -> Session:
        """Create a new conversation session."""
        # Cleanup old sessions if limit reached
        if len(self.sessions) >= self.max_sessions:
            self._cleanup_old_sessions()
        
        session_id = str(uuid.uuid4())[:8]
        session = Session(
            session_id=session_id,
            created_at=datetime.now().isoformat()
        )
        
        # Add system message
        session.add_message(
            MessageRole.SYSTEM.value,
            "AI Workflow Agent initialized. Ready to help build n8n, ComfyUI, "
            "or hybrid workflows. Describe what you want to create."
        )
        
        self.sessions[session_id] = session
        logger.info(f"Created session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get existing session by ID."""
        return self.sessions.get(session_id)
    
    def get_or_create(self, session_id: Optional[str] = None) -> Session:
        """Get existing session or create new one."""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        return self.create_session()
    
    def update_state(self, session_id: str, state: ConversationState):
        """Update session state."""
        if session_id in self.sessions:
            self.sessions[session_id].state = state.value
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        return [
            {
                "session_id": s.session_id,
                "created_at": s.created_at,
                "state": s.state,
                "message_count": len(s.messages),
                "project_type": s.project_type
            }
            for s in self.sessions.values()
        ]
    
    def _cleanup_old_sessions(self):
        """Remove oldest sessions to make room."""
        if not self.sessions:
            return
        
        # Sort by creation time and remove oldest 20%
        sorted_sessions = sorted(
            self.sessions.items(),
            key=lambda x: x[1].created_at
        )
        
        to_remove = len(sorted_sessions) // 5
        for session_id, _ in sorted_sessions[:to_remove]:
            del self.sessions[session_id]
        
        logger.info(f"Cleaned up {to_remove} old sessions")


class ChatHandler:
    """Handles chat interactions with the agent system."""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self._agent_system = None  # Lazy load
    
    @property
    def agent_system(self):
        """Lazy load agent system to avoid circular imports."""
        if self._agent_system is None:
            from crew_agents import crew_agent_system
            self._agent_system = crew_agent_system
        return self._agent_system
    
    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message and return response.
        
        Args:
            message: User message
            session_id: Optional existing session ID
            
        Returns:
            Dict with response, session_id, state, and optionally questions/workflow
        """
        # Get or create session
        session = self.session_manager.get_or_create(session_id)
        
        # Add user message
        session.add_message(MessageRole.USER.value, message)
        
        try:
            # Handle based on current state
            if session.state == ConversationState.CLARIFYING.value:
                # User is answering clarifying questions
                return await self._handle_clarification(session, message)
            else:
                # New request or continuation
                return await self._handle_request(session, message)
                
        except Exception as e:
            logger.error(f"Chat error: {e}")
            session.state = ConversationState.ERROR.value
            session.add_message(
                MessageRole.ASSISTANT.value,
                f"Sorry, I encountered an error: {str(e)}. Please try again."
            )
            return {
                "success": False,
                "session_id": session.session_id,
                "response": f"Error: {str(e)}",
                "state": session.state
            }
    
    async def _handle_request(self, session: Session, message: str) -> Dict[str, Any]:
        """Handle a new or continuing request."""
        session.state = ConversationState.ANALYZING.value
        
        # Analyze the request
        analysis = await self.agent_system.analyze_request(
            query=message,
            session_id=session.session_id,
            context={"history": session.get_history_text()}
        )
        
        if not analysis.get("success"):
            error_msg = analysis.get("error", "Analysis failed")
            session.add_message(MessageRole.ASSISTANT.value, f"Error: {error_msg}")
            return {
                "success": False,
                "session_id": session.session_id,
                "response": error_msg,
                "state": session.state
            }
        
        # Check if clarification needed
        if analysis.get("needs_clarification") and analysis.get("confidence", 0) < 0.7:
            session.state = ConversationState.CLARIFYING.value
            questions = analysis.get("questions", [])
            session.pending_questions = questions
            
            # Build response with questions
            response_parts = [analysis.get("analysis", "I need some clarification:")]
            for i, q in enumerate(questions, 1):
                response_parts.append(f"\n{i}. {q}")
            
            response = "\n".join(response_parts)
            session.add_message(MessageRole.ASSISTANT.value, response)
            
            return {
                "success": True,
                "session_id": session.session_id,
                "response": response,
                "state": session.state,
                "needs_clarification": True,
                "questions": questions,
                "project_type": analysis.get("project_type")
            }
        
        # Proceed to build
        return await self._build_workflow(session, analysis)
    
    async def _handle_clarification(self, session: Session, answer: str) -> Dict[str, Any]:
        """Handle user's answer to clarifying questions."""
        # Store the clarification
        if session.pending_questions:
            question = session.pending_questions[0]
            self.agent_system.add_clarification(
                session.session_id,
                question,
                answer
            )
            session.pending_questions = session.pending_questions[1:]
        
        # If more questions pending, ask next one
        if session.pending_questions:
            next_question = session.pending_questions[0]
            response = f"Thanks! Next question: {next_question}"
            session.add_message(MessageRole.ASSISTANT.value, response)
            
            return {
                "success": True,
                "session_id": session.session_id,
                "response": response,
                "state": session.state,
                "needs_clarification": True,
                "questions": session.pending_questions
            }
        
        # All questions answered, proceed to build
        session.add_message(
            MessageRole.ASSISTANT.value,
            "Great, I have all the information I need. Building your workflow..."
        )
        
        # Re-analyze with new information
        conv_context = self.agent_system.get_session(session.session_id)
        if conv_context:
            analysis = {
                "project_type": conv_context.project_type,
                "confidence": 0.9,
                "requirements": conv_context.requirements
            }
            return await self._build_workflow(session, analysis)
        
        return await self._handle_request(session, session.messages[-2]["content"])
    
    async def _build_workflow(self, session: Session, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Build the workflow based on analysis."""
        session.state = ConversationState.PLANNING.value
        session.project_type = analysis.get("project_type")
        
        # Use the simple builders for reliability (CrewAI for complex cases)
        from tools.n8n_builder import N8NWorkflowBuilder
        from tools.comfyui_builder import ComfyUIWorkflowBuilder
        from tools.github_search import GitHubSearchTool
        
        project_type = analysis.get("project_type", "unknown")
        original_query = session.messages[1]["content"] if len(session.messages) > 1 else ""
        
        session.state = ConversationState.BUILDING.value
        
        try:
            if project_type == "n8n":
                builder = N8NWorkflowBuilder()
                workflow = await builder.generate_workflow(original_query)
                response = "I've generated an n8n workflow for you. Here's the configuration:"
                
            elif project_type == "comfyui":
                builder = ComfyUIWorkflowBuilder()
                workflow = await builder.generate_workflow(original_query)
                response = "I've generated a ComfyUI workflow. Here's the configuration:"
                
            elif project_type == "hybrid":
                n8n_builder = N8NWorkflowBuilder()
                comfyui_builder = ComfyUIWorkflowBuilder()
                
                n8n_wf = await n8n_builder.generate_workflow(original_query)
                comfyui_wf = await comfyui_builder.generate_workflow(original_query)
                
                workflow = {
                    "type": "hybrid",
                    "n8n_workflow": n8n_wf,
                    "comfyui_workflow": comfyui_wf,
                    "integration_note": "n8n can call ComfyUI via HTTP Request node to /prompt endpoint"
                }
                response = "I've generated a hybrid workflow combining n8n automation with ComfyUI for AI generation."
                
            elif project_type == "external_repo":
                github = GitHubSearchTool()
                repos = await github.search(original_query, max_results=3)
                recommendation = await github.generate_recommendation(repos)
                
                workflow = {
                    "type": "external_repo",
                    "repositories": repos,
                    "recommendation": recommendation
                }
                response = f"I found some relevant repositories:\n\n{recommendation}"
                
            else:
                workflow = None
                response = "I couldn't determine the project type. Could you provide more details?"
            
            session.workflow = workflow
            session.state = ConversationState.COMPLETE.value
            
            if workflow and project_type not in ["external_repo"]:
                response += f"\n\n```json\n{json.dumps(workflow, indent=2)[:2000]}\n```"
            
            session.add_message(MessageRole.ASSISTANT.value, response[:500] + "..." if len(response) > 500 else response)
            
            return {
                "success": True,
                "session_id": session.session_id,
                "response": response,
                "state": session.state,
                "project_type": project_type,
                "workflow": workflow
            }
            
        except Exception as e:
            logger.error(f"Build error: {e}")
            session.state = ConversationState.ERROR.value
            response = f"Error building workflow: {str(e)}"
            session.add_message(MessageRole.ASSISTANT.value, response)
            
            return {
                "success": False,
                "session_id": session.session_id,
                "response": response,
                "state": session.state,
                "error": str(e)
            }
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information."""
        session = self.session_manager.get_session(session_id)
        if session:
            return session.to_dict()
        return None
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions."""
        return self.session_manager.list_sessions()
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a session."""
        # Also clear from agent system
        if hasattr(self, '_agent_system') and self._agent_system:
            self._agent_system.clear_session(session_id)
        return self.session_manager.delete_session(session_id)


# Singleton instance
chat_handler = ChatHandler()
