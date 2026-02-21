# AI Workflow Agent - Decision Agent (Core Brain)
"""
Decision Agent using CrewAI + Ollama (Qwen2.5)
Analyzes user queries and decides:
- n8n automation
- ComfyUI generative workflow
- Hybrid (n8n + ComfyUI)
- External repo project
"""

import httpx
import json
import logging
from typing import Dict, Any, Optional, List

from config import settings, ProjectType, CLASSIFICATION_KEYWORDS

logger = logging.getLogger(__name__)


class DecisionAgent:
    """
    Core decision-making agent that analyzes user queries
    and determines the appropriate workflow type.
    """
    
    def __init__(self):
        self.ollama_host = settings.OLLAMA_HOST
        self.model = settings.OLLAMA_MODEL
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def check_ollama_health(self) -> str:
        """Check if Ollama is running and responsive."""
        try:
            response = await self.client.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                return "healthy"
            return "unhealthy"
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return "unreachable"
    
    async def ensure_model_available(self) -> bool:
        """Ensure the required model is available in Ollama."""
        try:
            # Check if model exists
            response = await self.client.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                if self.model not in model_names and f"{self.model}:latest" not in model_names:
                    logger.info(f"Pulling model {self.model}...")
                    # Pull the model
                    pull_response = await self.client.post(
                        f"{self.ollama_host}/api/pull",
                        json={"name": self.model},
                        timeout=600.0  # 10 minutes for large models
                    )
                    return pull_response.status_code == 200
                return True
            return False
        except Exception as e:
            logger.error(f"Model check failed: {e}")
            return False
    
    async def analyze(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze user query and determine project type.
        
        Returns:
            Dict with project_type, confidence, explanation, suggested_tools, next_steps
        """
        # Step 1: Quick keyword classification
        keyword_result = self._keyword_classify(query)
        
        # Step 2: LLM-based deeper analysis
        llm_result = await self._llm_analyze(query, context)
        
        # Step 3: Combine results (LLM takes priority if confident)
        if llm_result["confidence"] > 0.7:
            final_result = llm_result
        elif keyword_result["confidence"] > llm_result["confidence"]:
            final_result = keyword_result
        else:
            final_result = llm_result
        
        # Add suggested tools and next steps
        final_result["suggested_tools"] = self._get_suggested_tools(final_result["project_type"])
        final_result["next_steps"] = self._get_next_steps(final_result["project_type"], query)
        
        logger.info(f"Decision: {final_result['project_type']} (confidence: {final_result['confidence']:.2f})")
        return final_result
    
    def _keyword_classify(self, query: str) -> Dict[str, Any]:
        """Fast keyword-based classification."""
        query_lower = query.lower()
        scores = {}
        
        for project_type, keywords in CLASSIFICATION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            scores[project_type] = score
        
        if not scores or max(scores.values()) == 0:
            return {
                "project_type": ProjectType.UNKNOWN,
                "confidence": 0.0,
                "explanation": "No matching keywords found"
            }
        
        best_type = max(scores, key=scores.get)
        max_score = scores[best_type]
        total_keywords = len(CLASSIFICATION_KEYWORDS[best_type])
        confidence = min(max_score / 3, 1.0)  # 3+ keywords = 100% confidence
        
        return {
            "project_type": best_type,
            "confidence": confidence,
            "explanation": f"Matched {max_score} keywords for {best_type}"
        }
    
    async def _llm_analyze(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """LLM-based analysis using Ollama."""
        
        system_prompt = """You are an AI assistant that analyzes user requests and classifies them into project types.

Your task is to determine which type of project the user wants:

1. **n8n** - Workflow automation, API integrations, data syncing, notifications, scheduled tasks
2. **comfyui** - Image generation, AI art, Stable Diffusion workflows, generative AI
3. **hybrid** - Combines automation with generative AI (e.g., "generate image and send via email")
4. **external_repo** - User wants to download/setup an existing GitHub project

Respond ONLY with valid JSON in this exact format:
{
    "project_type": "n8n" | "comfyui" | "hybrid" | "external_repo" | "unknown",
    "confidence": 0.0 to 1.0,
    "explanation": "Brief explanation of why you chose this type"
}"""

        user_prompt = f"""Analyze this user request and classify the project type:

User Request: "{query}"

{f"Additional Context: {json.dumps(context)}" if context else ""}

Remember: Respond ONLY with valid JSON."""

        try:
            response = await self.client.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{system_prompt}\n\n{user_prompt}",
                    "stream": False,
                    "format": "json"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                llm_response = result.get("response", "{}")
                
                # Parse JSON response
                try:
                    parsed = json.loads(llm_response)
                    return {
                        "project_type": parsed.get("project_type", ProjectType.UNKNOWN),
                        "confidence": float(parsed.get("confidence", 0.5)),
                        "explanation": parsed.get("explanation", "LLM analysis")
                    }
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse LLM response: {llm_response}")
                    return {
                        "project_type": ProjectType.UNKNOWN,
                        "confidence": 0.3,
                        "explanation": "LLM response parsing failed"
                    }
            else:
                logger.error(f"Ollama request failed: {response.status_code}")
                return {
                    "project_type": ProjectType.UNKNOWN,
                    "confidence": 0.0,
                    "explanation": "LLM request failed"
                }
                
        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return {
                "project_type": ProjectType.UNKNOWN,
                "confidence": 0.0,
                "explanation": f"LLM error: {str(e)}"
            }
    
    def _get_suggested_tools(self, project_type: str) -> List[str]:
        """Get suggested tools based on project type."""
        tools_map = {
            ProjectType.N8N: [
                "n8n_builder - Generate workflow JSON",
                "n8n_deploy - Deploy to n8n instance",
                "webhook_trigger - Setup webhook triggers"
            ],
            ProjectType.COMFYUI: [
                "comfyui_builder - Generate workflow graph",
                "comfyui_execute - Run workflow",
                "model_download - Download required models"
            ],
            ProjectType.HYBRID: [
                "n8n_builder - Generate automation workflow",
                "comfyui_builder - Generate AI workflow",
                "api_connector - Connect n8n to ComfyUI"
            ],
            ProjectType.EXTERNAL_REPO: [
                "github_search - Find relevant repositories",
                "docker_helper - Clone and build",
                "error_analyzer - Fix build issues"
            ]
        }
        return tools_map.get(project_type, ["unknown_tool"])
    
    def _get_next_steps(self, project_type: str, query: str) -> List[str]:
        """Get recommended next steps."""
        steps_map = {
            ProjectType.N8N: [
                "1. Generate workflow JSON template",
                "2. Customize nodes and connections",
                "3. Deploy to n8n instance",
                "4. Test with sample data"
            ],
            ProjectType.COMFYUI: [
                "1. Generate ComfyUI workflow graph",
                "2. Check required models are installed",
                "3. Execute workflow",
                "4. Review generated output"
            ],
            ProjectType.HYBRID: [
                "1. Create ComfyUI workflow for AI task",
                "2. Create n8n workflow for automation",
                "3. Connect n8n → ComfyUI via HTTP",
                "4. Test end-to-end pipeline"
            ],
            ProjectType.EXTERNAL_REPO: [
                "1. Search GitHub for relevant projects",
                "2. Select best matching repository",
                "3. Clone and configure with Docker",
                "4. Validate and fix any errors"
            ]
        }
        return steps_map.get(project_type, ["Please provide more details"])


# ============================================
# CrewAI Integration (Advanced Mode)
# ============================================

class CrewAIDecisionAgent:
    """
    Advanced decision agent using CrewAI framework.
    Used for more complex multi-step reasoning.
    """
    
    def __init__(self):
        self.simple_agent = DecisionAgent()
        # CrewAI setup would go here
        # We use simple agent for Phase 0, upgrade to CrewAI in Milestone 1
    
    async def analyze_complex(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complex analysis using CrewAI agents.
        Reserved for Milestone 1 implementation.
        """
        # For now, delegate to simple agent
        return await self.simple_agent.analyze(query, context)
