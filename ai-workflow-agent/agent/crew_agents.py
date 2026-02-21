# AI Workflow Agent - CrewAI Agent System
"""
Advanced multi-agent system using CrewAI for intelligent workflow generation.
Milestone 1: Full CrewAI integration with specialized agents.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

# Set environment variable before importing crewai to avoid OpenAI key requirement
os.environ.setdefault("OPENAI_API_KEY", "not-needed-for-ollama")

from crewai import Agent, Task, Crew, Process, LLM

from config import settings, ProjectType

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent roles in the system."""
    ANALYZER = "analyzer"
    PLANNER = "planner"
    BUILDER = "builder"
    VALIDATOR = "validator"


@dataclass
class ConversationContext:
    """Tracks conversation state and gathered information."""
    user_query: str
    project_type: Optional[str] = None
    clarifications: List[Dict[str, str]] = None
    requirements: Dict[str, Any] = None
    workflow_generated: bool = False
    
    def __post_init__(self):
        if self.clarifications is None:
            self.clarifications = []
        if self.requirements is None:
            self.requirements = {}


class CrewAIAgentSystem:
    """
    Multi-agent system for intelligent workflow generation.
    
    Agents:
    - Analyzer: Understands user intent, asks clarifying questions
    - Planner: Decides project type and creates execution plan
    - Builder: Generates workflows and configurations
    - Validator: Checks outputs and suggests improvements
    """
    
    def __init__(self):
        self.llm = self._setup_llm()
        self.agents = self._create_agents()
        self.conversation_history: Dict[str, ConversationContext] = {}
    
    def _setup_llm(self) -> LLM:
        """Initialize Ollama LLM for agents."""
        # Use CrewAI's native LLM class with Ollama provider
        return LLM(
            model=f"ollama/{settings.OLLAMA_MODEL}",
            base_url=settings.OLLAMA_HOST,
            temperature=0.7
        )
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Create specialized agents for different tasks."""
        
        analyzer = Agent(
            role="Request Analyzer",
            goal="Understand user requests and gather necessary information",
            backstory="""You are an expert at understanding user needs for automation 
            and AI workflows. You analyze requests, identify ambiguities, and ask 
            clarifying questions when needed. You're familiar with n8n, ComfyUI, 
            Docker, and various automation tools.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        planner = Agent(
            role="Project Planner",
            goal="Create detailed execution plans for workflow projects",
            backstory="""You are a senior architect who designs automation systems.
            You decide whether to use n8n (for general automation), ComfyUI (for AI 
            generation), hybrid approaches, or external tools. You create step-by-step
            plans that are clear and actionable.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        builder = Agent(
            role="Workflow Builder",
            goal="Generate working workflow configurations",
            backstory="""You are an expert developer who creates n8n workflows, 
            ComfyUI graphs, Docker configurations, and automation scripts. You 
            write clean, working JSON/YAML configurations. You know the exact 
            node structures and API formats for n8n and ComfyUI.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        validator = Agent(
            role="Quality Validator",
            goal="Validate outputs and ensure they meet requirements",
            backstory="""You are a QA specialist who reviews workflow configurations
            for completeness, correctness, and best practices. You identify potential
            issues and suggest improvements. You ensure all user requirements are met.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        return {
            AgentRole.ANALYZER: analyzer,
            AgentRole.PLANNER: planner,
            AgentRole.BUILDER: builder,
            AgentRole.VALIDATOR: validator
        }
    
    async def analyze_request(
        self,
        query: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze user request and determine if clarification is needed.
        
        Returns:
            Dict with:
            - needs_clarification: bool
            - questions: List of clarifying questions (if needed)
            - project_type: Detected type (if confident)
            - analysis: Detailed analysis
        """
        # Create or get conversation context
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = ConversationContext(user_query=query)
        
        conv_context = self.conversation_history[session_id]
        
        # Create analysis task
        analysis_task = Task(
            description=f"""Analyze this user request for a workflow/automation project:

User Request: "{query}"

{f"Additional Context: {json.dumps(context)}" if context else ""}

Previous clarifications gathered: {json.dumps(conv_context.clarifications)}

Your task:
1. Determine what the user wants to build
2. Identify the project type:
   - "n8n": General automation, API integrations, scheduled tasks
   - "comfyui": AI image/video generation, Stable Diffusion workflows
   - "hybrid": Combination of automation + AI generation
   - "external_repo": Clone and setup existing GitHub project
   
3. Check if you have enough information to proceed
4. If not, list 1-3 specific clarifying questions

Respond in this JSON format:
{{
    "project_type": "n8n" | "comfyui" | "hybrid" | "external_repo" | "unknown",
    "confidence": 0.0 to 1.0,
    "needs_clarification": true | false,
    "questions": ["question1", "question2"],
    "analysis": "Brief analysis of the request",
    "key_requirements": ["req1", "req2"]
}}""",
            agent=self.agents[AgentRole.ANALYZER],
            expected_output="JSON analysis of the user request"
        )
        
        # Run single-agent task
        crew = Crew(
            agents=[self.agents[AgentRole.ANALYZER]],
            tasks=[analysis_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            result = crew.kickoff()
            
            # Parse result
            result_str = str(result)
            
            # Try to extract JSON from result
            try:
                # Find JSON in response
                start = result_str.find('{')
                end = result_str.rfind('}') + 1
                if start >= 0 and end > start:
                    parsed = json.loads(result_str[start:end])
                    
                    # Update conversation context
                    conv_context.project_type = parsed.get("project_type")
                    conv_context.requirements["key_requirements"] = parsed.get("key_requirements", [])
                    
                    return {
                        "success": True,
                        "session_id": session_id,
                        **parsed
                    }
            except json.JSONDecodeError:
                pass
            
            # Fallback: return raw analysis
            return {
                "success": True,
                "session_id": session_id,
                "project_type": "unknown",
                "confidence": 0.5,
                "needs_clarification": True,
                "questions": ["Could you provide more details about what you want to build?"],
                "analysis": result_str[:500]
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    async def plan_project(
        self,
        session_id: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create execution plan based on gathered information.
        """
        if session_id not in self.conversation_history:
            return {"success": False, "error": "Session not found"}
        
        conv_context = self.conversation_history[session_id]
        
        planning_task = Task(
            description=f"""Create an execution plan for this project:

Original Request: "{conv_context.user_query}"
Project Type: {conv_context.project_type}
Requirements: {json.dumps(conv_context.requirements)}
Clarifications: {json.dumps(conv_context.clarifications)}
{f"Additional Info: {json.dumps(additional_info)}" if additional_info else ""}

Create a detailed plan with:
1. Step-by-step actions to take
2. Required tools/services
3. Configuration details
4. Potential challenges and solutions

Respond in JSON format:
{{
    "plan_name": "Descriptive name",
    "steps": [
        {{"step": 1, "action": "...", "tool": "...", "details": "..."}},
        ...
    ],
    "required_services": ["n8n", "comfyui", etc],
    "estimated_complexity": "low" | "medium" | "high",
    "notes": "Any important notes"
}}""",
            agent=self.agents[AgentRole.PLANNER],
            expected_output="JSON execution plan"
        )
        
        crew = Crew(
            agents=[self.agents[AgentRole.PLANNER]],
            tasks=[planning_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            result = crew.kickoff()
            result_str = str(result)
            
            # Parse JSON
            try:
                start = result_str.find('{')
                end = result_str.rfind('}') + 1
                if start >= 0 and end > start:
                    parsed = json.loads(result_str[start:end])
                    return {"success": True, "session_id": session_id, "plan": parsed}
            except json.JSONDecodeError:
                pass
            
            return {
                "success": True,
                "session_id": session_id,
                "plan": {"raw_plan": result_str[:1000]}
            }
            
        except Exception as e:
            logger.error(f"Planning error: {e}")
            return {"success": False, "error": str(e)}
    
    async def build_workflow(
        self,
        session_id: str,
        plan: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate actual workflow based on plan.
        """
        if session_id not in self.conversation_history:
            return {"success": False, "error": "Session not found"}
        
        conv_context = self.conversation_history[session_id]
        
        build_task = Task(
            description=f"""Generate the workflow configuration:

Project Type: {conv_context.project_type}
Original Request: "{conv_context.user_query}"
Requirements: {json.dumps(conv_context.requirements)}
{f"Plan: {json.dumps(plan)}" if plan else ""}

Generate a complete, working workflow configuration.

For n8n: Generate valid n8n JSON workflow with proper node structure.
For ComfyUI: Generate valid ComfyUI prompt/workflow JSON.
For Hybrid: Generate both and explain how they connect.
For External Repo: Provide clone and setup instructions.

Output the workflow JSON directly, ready to deploy.""",
            agent=self.agents[AgentRole.BUILDER],
            expected_output="Complete workflow JSON configuration"
        )
        
        crew = Crew(
            agents=[self.agents[AgentRole.BUILDER]],
            tasks=[build_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            result = crew.kickoff()
            result_str = str(result)
            
            # Try to extract JSON workflow
            try:
                start = result_str.find('{')
                end = result_str.rfind('}') + 1
                if start >= 0 and end > start:
                    workflow = json.loads(result_str[start:end])
                    conv_context.workflow_generated = True
                    return {
                        "success": True,
                        "session_id": session_id,
                        "project_type": conv_context.project_type,
                        "workflow": workflow
                    }
            except json.JSONDecodeError:
                pass
            
            return {
                "success": True,
                "session_id": session_id,
                "project_type": conv_context.project_type,
                "raw_output": result_str
            }
            
        except Exception as e:
            logger.error(f"Build error: {e}")
            return {"success": False, "error": str(e)}
    
    async def full_pipeline(
        self,
        query: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run full pipeline: Analyze → Plan → Build → Validate
        
        This runs all agents in sequence for a complete workflow generation.
        """
        # Step 1: Analyze
        analysis = await self.analyze_request(query, session_id, context)
        
        if not analysis.get("success"):
            return analysis
        
        if analysis.get("needs_clarification") and analysis.get("confidence", 0) < 0.7:
            return {
                "success": True,
                "stage": "clarification_needed",
                "questions": analysis.get("questions", []),
                "analysis": analysis.get("analysis"),
                "session_id": session_id
            }
        
        # Step 2: Plan
        plan_result = await self.plan_project(session_id)
        
        if not plan_result.get("success"):
            return plan_result
        
        # Step 3: Build
        build_result = await self.build_workflow(session_id, plan_result.get("plan"))
        
        return {
            "success": build_result.get("success", False),
            "stage": "complete",
            "analysis": analysis,
            "plan": plan_result.get("plan"),
            "workflow": build_result.get("workflow"),
            "project_type": analysis.get("project_type"),
            "session_id": session_id
        }
    
    def add_clarification(
        self,
        session_id: str,
        question: str,
        answer: str
    ) -> bool:
        """Add user's answer to a clarifying question."""
        if session_id not in self.conversation_history:
            return False
        
        self.conversation_history[session_id].clarifications.append({
            "question": question,
            "answer": answer
        })
        return True
    
    def get_session(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context for a session."""
        return self.conversation_history.get(session_id)
    
    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            return True
        return False


# Lazy-loaded singleton instance
_crew_agent_system = None

def get_crew_agent_system() -> CrewAIAgentSystem:
    """Get or create the CrewAI agent system singleton."""
    global _crew_agent_system
    if _crew_agent_system is None:
        _crew_agent_system = CrewAIAgentSystem()
    return _crew_agent_system

# For backward compatibility (lazy-loaded)
crew_agent_system = None  # Will be set on first access
