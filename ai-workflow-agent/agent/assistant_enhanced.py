"""
Enhanced Assistant - M3 Feature
Advanced conversational AI with memory, context, and intelligent routing
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import re

logger = logging.getLogger(__name__)


class ConversationRole(str, Enum):
    """Roles in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageIntent(str, Enum):
    """Detected user intent"""
    CREATE_WORKFLOW = "create_workflow"
    GENERATE_IMAGE = "generate_image"
    QUERY_STATUS = "query_status"
    GET_HELP = "get_help"
    CONFIGURE_SETTINGS = "configure_settings"
    RUN_TASK = "run_task"
    CHAT = "chat"
    UNKNOWN = "unknown"


class ActionType(str, Enum):
    """Types of actions the assistant can take"""
    CREATE_N8N_WORKFLOW = "create_n8n_workflow"
    CREATE_COMFYUI_WORKFLOW = "create_comfyui_workflow"
    QUEUE_COMFYUI = "queue_comfyui"
    START_COLAB_TASK = "start_colab_task"
    SEND_NOTIFICATION = "send_notification"
    SAVE_FILE = "save_file"
    EXECUTE_COMMAND = "execute_command"
    RESPOND_TEXT = "respond_text"


@dataclass
class ConversationMessage:
    """A message in conversation history"""
    role: ConversationRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    intent: Optional[MessageIntent] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "intent": self.intent.value if self.intent else None
        }


@dataclass 
class ActionSuggestion:
    """A suggested action based on conversation"""
    action_type: ActionType
    confidence: float  # 0-1
    description: str
    parameters: Dict[str, Any]
    requires_confirmation: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "confidence": self.confidence,
            "description": self.description,
            "parameters": self.parameters,
            "requires_confirmation": self.requires_confirmation
        }


class ConversationMemory:
    """Manages conversation history and context"""
    
    def __init__(self, max_messages: int = 100, max_context_tokens: int = 4000):
        self.messages: List[ConversationMessage] = []
        self.max_messages = max_messages
        self.max_context_tokens = max_context_tokens
        self.session_start = datetime.now()
        self.metadata: Dict[str, Any] = {}
        self.extracted_entities: Dict[str, List[str]] = {
            "workflows": [],
            "images": [],
            "urls": [],
            "files": []
        }
        
    def add_message(self, role: ConversationRole, content: str, 
                    metadata: Optional[Dict] = None, intent: Optional[MessageIntent] = None):
        """Add message to history"""
        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {},
            intent=intent
        )
        self.messages.append(message)
        
        # Extract entities
        self._extract_entities(content)
        
        # Trim if needed
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
            
    def _extract_entities(self, content: str):
        """Extract useful entities from text"""
        # URLs
        urls = re.findall(r'https?://[^\s]+', content)
        self.extracted_entities["urls"].extend(urls)
        
        # File paths
        files = re.findall(r'[\w\-./]+\.(py|json|yaml|yml|js|ts|png|jpg|jpeg)', content)
        self.extracted_entities["files"].extend(files)
        
    def get_context(self, max_messages: int = 10) -> str:
        """Get formatted context for LLM"""
        recent = self.messages[-max_messages:]
        context_parts = []
        
        for msg in recent:
            role = msg.role.value.capitalize()
            context_parts.append(f"{role}: {msg.content}")
            
        return "\n".join(context_parts)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get conversation summary"""
        return {
            "message_count": len(self.messages),
            "session_duration_minutes": (datetime.now() - self.session_start).seconds // 60,
            "intents_detected": list(set(m.intent.value for m in self.messages if m.intent)),
            "extracted_entities": {k: list(set(v)) for k, v in self.extracted_entities.items()}
        }
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []
        self.extracted_entities = {k: [] for k in self.extracted_entities}
        self.session_start = datetime.now()


class IntentClassifier:
    """Classify user intent from text"""
    
    # Keyword patterns for intent detection
    INTENT_PATTERNS = {
        MessageIntent.CREATE_WORKFLOW: [
            r'\b(create|make|build|design|set\s*up)\b.*\b(workflow|automation|pipeline)\b',
            r'\b(workflow|automation)\b.*\b(for|to|that)\b',
            r'\bautomate\b',
            r'\bn8n\b.*\b(create|make|build)\b'
        ],
        MessageIntent.GENERATE_IMAGE: [
            r'\b(generate|create|make|draw|render)\b.*\b(image|picture|photo|art)\b',
            r'\b(image|picture)\b.*\b(of|with|showing)\b',
            r'\bstable\s*diffusion\b',
            r'\bsdxl\b',
            r'\bcomfyui\b.*\b(generate|run)\b',
            r'\b(portrait|landscape|artwork|illustration)\b.*\b(of|with)\b'
        ],
        MessageIntent.QUERY_STATUS: [
            r'\b(status|state|progress|check)\b.*\b(workflow|task|job|generation)\b',
            r'\b(how|what)\b.*\b(doing|going|status)\b',
            r'\bis\b.*\b(running|complete|done|finished)\b',
            r'\bshow\b.*\b(status|tasks|queue)\b'
        ],
        MessageIntent.GET_HELP: [
            r'\b(help|how\s+to|how\s+do|explain|what\s+is)\b',
            r'\b(guide|tutorial|documentation)\b',
            r'\?\s*$'
        ],
        MessageIntent.CONFIGURE_SETTINGS: [
            r'\b(configure|config|settings?|setup)\b',
            r'\b(change|set|update|modify)\b.*\b(setting|option|parameter)\b',
            r'\b(enable|disable)\b.*\b(feature|option)\b'
        ],
        MessageIntent.RUN_TASK: [
            r'\b(run|execute|start|trigger|launch)\b.*\b(task|workflow|job|script)\b',
            r'\b(process|handle)\b.*\b(file|data|batch)\b',
            r'\bcall\b.*\b(api|endpoint|webhook)\b'
        ]
    }
    
    @classmethod
    def classify(cls, text: str) -> MessageIntent:
        """Classify intent from text"""
        text_lower = text.lower()
        
        scores = {}
        for intent, patterns in cls.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            scores[intent] = score
            
        # Find best match
        best_intent = max(scores, key=scores.get)
        if scores[best_intent] > 0:
            return best_intent
            
        # Default to chat if has text, unknown otherwise
        return MessageIntent.CHAT if len(text.strip()) > 3 else MessageIntent.UNKNOWN
    
    @classmethod
    def get_confidence(cls, text: str, intent: MessageIntent) -> float:
        """Get confidence score for an intent"""
        if intent not in cls.INTENT_PATTERNS:
            return 0.0
            
        text_lower = text.lower()
        patterns = cls.INTENT_PATTERNS[intent]
        matches = sum(1 for p in patterns if re.search(p, text_lower))
        
        return min(1.0, matches / max(1, len(patterns) // 2))


class ActionSuggester:
    """Suggest actions based on user input and context"""
    
    def __init__(self):
        self._handlers: Dict[MessageIntent, Callable] = {
            MessageIntent.CREATE_WORKFLOW: self._suggest_workflow,
            MessageIntent.GENERATE_IMAGE: self._suggest_image_generation,
            MessageIntent.QUERY_STATUS: self._suggest_status_query,
            MessageIntent.RUN_TASK: self._suggest_task_run
        }
        
    def suggest(self, text: str, intent: MessageIntent, 
                context: Optional[ConversationMemory] = None) -> List[ActionSuggestion]:
        """Generate action suggestions"""
        handler = self._handlers.get(intent)
        if not handler:
            return []
            
        return handler(text, context)
    
    def _suggest_workflow(self, text: str, context: Optional[ConversationMemory]) -> List[ActionSuggestion]:
        """Suggest workflow creation actions"""
        suggestions = []
        text_lower = text.lower()
        
        # Check for n8n keywords
        if any(kw in text_lower for kw in ['n8n', 'automation', 'webhook', 'email', 'slack']):
            suggestions.append(ActionSuggestion(
                action_type=ActionType.CREATE_N8N_WORKFLOW,
                confidence=0.8,
                description="Create an n8n automation workflow",
                parameters={"source_text": text}
            ))
            
        # Check for ComfyUI keywords
        if any(kw in text_lower for kw in ['comfyui', 'image', 'stable', 'diffusion']):
            suggestions.append(ActionSuggestion(
                action_type=ActionType.CREATE_COMFYUI_WORKFLOW,
                confidence=0.8,
                description="Create a ComfyUI image generation workflow",
                parameters={"source_text": text}
            ))
            
        return suggestions
    
    def _suggest_image_generation(self, text: str, context: Optional[ConversationMemory]) -> List[ActionSuggestion]:
        """Suggest image generation actions"""
        # Extract potential prompt
        prompt = text
        
        # Try to extract quoted text as prompt
        quotes = re.findall(r'"([^"]+)"', text)
        if quotes:
            prompt = quotes[0]
            
        suggestions = [
            ActionSuggestion(
                action_type=ActionType.QUEUE_COMFYUI,
                confidence=0.9,
                description="Generate image with ComfyUI",
                parameters={
                    "prompt": prompt,
                    "negative_prompt": "",
                    "width": 1024,
                    "height": 1024
                }
            )
        ]
        
        # Check if might need Colab (heavy task)
        if any(kw in text.lower() for kw in ['batch', 'multiple', 'video', 'training']):
            suggestions.append(ActionSuggestion(
                action_type=ActionType.START_COLAB_TASK,
                confidence=0.7,
                description="Run heavy processing on Google Colab",
                parameters={"task_type": "image_generation", "prompt": prompt}
            ))
            
        return suggestions
    
    def _suggest_status_query(self, text: str, context: Optional[ConversationMemory]) -> List[ActionSuggestion]:
        """Suggest status query actions"""
        return [ActionSuggestion(
            action_type=ActionType.RESPOND_TEXT,
            confidence=0.9,
            description="Query and display system status",
            parameters={"query_type": "status"},
            requires_confirmation=False
        )]
    
    def _suggest_task_run(self, text: str, context: Optional[ConversationMemory]) -> List[ActionSuggestion]:
        """Suggest task execution"""
        suggestions = []
        
        # Check for specific task types
        if 'workflow' in text.lower():
            suggestions.append(ActionSuggestion(
                action_type=ActionType.EXECUTE_COMMAND,
                confidence=0.7,
                description="Execute workflow",
                parameters={"command_type": "workflow"}
            ))
            
        return suggestions


class EnhancedAssistant:
    """
    Enhanced conversational assistant with context awareness,
    intent detection, and intelligent action suggestion.
    """
    
    def __init__(self):
        self.memory = ConversationMemory()
        self.classifier = IntentClassifier()
        self.suggester = ActionSuggester()
        self.system_prompt = self._build_system_prompt()
        
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the assistant"""
        return """You are an AI workflow assistant that helps users create and manage automation workflows.

Capabilities:
- Create n8n automation workflows for tasks like email processing, data sync, API integration
- Create ComfyUI workflows for AI image generation, upscaling, and processing
- Monitor workflow execution and status
- Suggest optimizations and best practices
- Help configure services and settings

When responding:
- Be concise and helpful
- Suggest specific actions when possible
- Ask clarifying questions if the request is ambiguous
- Provide examples when explaining concepts

Available services:
- n8n: Workflow automation platform
- ComfyUI: AI image generation
- Ollama: Local LLM inference
- Google Colab: Heavy computation offloading"""

    def process_message(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message and generate response
        
        Returns structured response with:
        - intent: Detected intent
        - response: Text response
        - actions: Suggested actions
        - context: Current context summary
        """
        # Classify intent
        intent = self.classifier.classify(user_message)
        confidence = self.classifier.get_confidence(user_message, intent)
        
        # Add to memory
        self.memory.add_message(
            role=ConversationRole.USER,
            content=user_message,
            intent=intent,
            metadata={"confidence": confidence}
        )
        
        # Get action suggestions
        suggestions = self.suggester.suggest(user_message, intent, self.memory)
        
        # Generate response based on intent
        response = self._generate_response(user_message, intent, suggestions)
        
        # Add response to memory
        self.memory.add_message(
            role=ConversationRole.ASSISTANT,
            content=response
        )
        
        return {
            "intent": intent.value,
            "confidence": confidence,
            "response": response,
            "actions": [s.to_dict() for s in suggestions],
            "context": self.memory.get_summary(),
            "requires_action": len(suggestions) > 0
        }
    
    def _generate_response(self, message: str, intent: MessageIntent, 
                          suggestions: List[ActionSuggestion]) -> str:
        """Generate appropriate response text"""
        
        if intent == MessageIntent.CREATE_WORKFLOW:
            if suggestions:
                action_desc = suggestions[0].description
                return f"I can help you with that! Based on your request, I suggest: {action_desc}. Would you like me to proceed?"
            return "I can help create a workflow. Please describe what you want to automate in more detail."
            
        elif intent == MessageIntent.GENERATE_IMAGE:
            if suggestions:
                params = suggestions[0].parameters
                prompt = params.get("prompt", message)
                return f"I'll generate an image with the prompt: \"{prompt}\". The image will be 1024x1024. Ready to proceed?"
            return "I can generate images. Please describe what you want to create."
            
        elif intent == MessageIntent.QUERY_STATUS:
            return "I'll check the current status of your workflows and tasks."
            
        elif intent == MessageIntent.GET_HELP:
            return self._generate_help_response(message)
            
        elif intent == MessageIntent.CONFIGURE_SETTINGS:
            return "I can help configure settings. What would you like to change?"
            
        elif intent == MessageIntent.RUN_TASK:
            return "I can run that task. Please confirm or provide additional details."
            
        elif intent == MessageIntent.CHAT:
            return "How can I assist you with your workflows today?"
            
        return "I'm not sure how to help with that. Could you please clarify?"
    
    def _generate_help_response(self, message: str) -> str:
        """Generate help response"""
        message_lower = message.lower()
        
        if 'workflow' in message_lower:
            return """To create a workflow:
1. Describe what you want to automate (e.g., "Create a workflow to send Slack messages when I receive important emails")
2. I'll suggest the best platform (n8n or ComfyUI)
3. Review and confirm the generated workflow
4. Deploy and monitor execution"""
            
        elif 'image' in message_lower:
            return """To generate images:
1. Describe your image (e.g., "Generate an image of a futuristic city at sunset")
2. Optionally specify: size, style, negative prompts
3. I'll queue it in ComfyUI
4. View results when ready"""
            
        elif 'colab' in message_lower:
            return """Google Colab integration allows offloading heavy tasks:
- Large batch image generation
- Model training/fine-tuning
- Video generation
- LLM inference for large models

Just describe the task and I'll handle the Colab setup."""
            
        return """I can help you with:
- **Workflows**: Create automation in n8n or ComfyUI
- **Images**: Generate AI images with Stable Diffusion
- **Colab**: Offload heavy computation
- **Monitoring**: Check task status and results

What would you like to do?"""
    
    def get_context_for_llm(self) -> str:
        """Get full context formatted for LLM"""
        return f"{self.system_prompt}\n\n{self.memory.get_context()}"
    
    def clear_context(self):
        """Start fresh conversation"""
        self.memory.clear()
        
    def get_state(self) -> Dict[str, Any]:
        """Get current assistant state"""
        return {
            "session_start": self.memory.session_start.isoformat(),
            "message_count": len(self.memory.messages),
            "history": [m.to_dict() for m in self.memory.messages],
            "summary": self.memory.get_summary()
        }


class QuickAction:
    """Pre-defined quick actions for common tasks"""
    
    ACTIONS = {
        "generate_portrait": {
            "name": "Generate Portrait",
            "description": "Generate a portrait image",
            "action_type": ActionType.QUEUE_COMFYUI,
            "default_params": {
                "prompt": "portrait of a person, professional photo, high quality",
                "negative_prompt": "blurry, distorted",
                "width": 768,
                "height": 1024
            }
        },
        "generate_landscape": {
            "name": "Generate Landscape",
            "description": "Generate a landscape image",
            "action_type": ActionType.QUEUE_COMFYUI,
            "default_params": {
                "prompt": "beautiful landscape, nature, scenic, high quality",
                "negative_prompt": "urban, buildings",
                "width": 1024,
                "height": 768
            }
        },
        "email_workflow": {
            "name": "Email Automation",
            "description": "Create email processing workflow",
            "action_type": ActionType.CREATE_N8N_WORKFLOW,
            "default_params": {
                "template": "email-automation"
            }
        },
        "slack_bot": {
            "name": "Slack Bot",
            "description": "Create Slack bot workflow",
            "action_type": ActionType.CREATE_N8N_WORKFLOW,
            "default_params": {
                "template": "slack-bot-automation"
            }
        },
        "check_status": {
            "name": "Check Status",
            "description": "Check system and workflow status",
            "action_type": ActionType.RESPOND_TEXT,
            "default_params": {
                "query_type": "status"
            }
        }
    }
    
    @classmethod
    def get_all(cls) -> List[Dict[str, Any]]:
        """Get all quick actions"""
        return [
            {
                "id": action_id,
                "name": action["name"],
                "description": action["description"],
                "action_type": action["action_type"].value
            }
            for action_id, action in cls.ACTIONS.items()
        ]
    
    @classmethod
    def get_action(cls, action_id: str) -> Optional[Dict[str, Any]]:
        """Get specific quick action"""
        return cls.ACTIONS.get(action_id)


# Singleton instance
_assistant: Optional[EnhancedAssistant] = None


def get_assistant() -> EnhancedAssistant:
    """Get the global enhanced assistant instance"""
    global _assistant
    if _assistant is None:
        _assistant = EnhancedAssistant()
    return _assistant
