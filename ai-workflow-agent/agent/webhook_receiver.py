"""
Webhook Receiver - M2 Feature
Handle incoming webhooks from n8n, GitHub, and external services
"""

import uuid
import hmac
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class WebhookSource(str, Enum):
    """Known webhook sources"""
    N8N = "n8n"
    COMFYUI = "comfyui"
    GITHUB = "github"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


@dataclass
class WebhookConfig:
    """Configuration for a webhook endpoint"""
    webhook_id: str
    name: str
    source: WebhookSource
    secret: Optional[str] = None  # For signature verification
    is_active: bool = True
    allowed_events: List[str] = field(default_factory=list)  # Empty = all events
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class WebhookEvent:
    """Incoming webhook event"""
    event_id: str
    webhook_id: str
    source: WebhookSource
    event_type: str
    payload: Dict[str, Any]
    headers: Dict[str, str]
    received_at: datetime
    processed: bool = False
    processed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Type for webhook handlers
WebhookHandler = Callable[[WebhookEvent], Awaitable[Dict[str, Any]]]


class WebhookReceiver:
    """
    Receive and process webhooks from various sources.
    Supports signature verification, event filtering, and custom handlers.
    """
    
    def __init__(self):
        self._webhooks: Dict[str, WebhookConfig] = {}
        self._handlers: Dict[str, List[WebhookHandler]] = {}
        self._events: Dict[str, WebhookEvent] = {}
        self._max_events = 1000  # Keep last 1000 events in memory
        
        # Register default handlers
        self._register_default_handlers()
        
    def _register_default_handlers(self):
        """Register default webhook handlers"""
        # n8n workflow completion handler
        self.register_handler("n8n:workflow.completed", self._handle_n8n_completion)
        self.register_handler("n8n:workflow.failed", self._handle_n8n_failure)
        
        # GitHub handlers
        self.register_handler("github:push", self._handle_github_push)
        self.register_handler("github:workflow_run", self._handle_github_workflow)
        
    # ==================== Webhook Management ====================
    
    def create_webhook(
        self,
        name: str,
        source: WebhookSource,
        secret: Optional[str] = None,
        allowed_events: Optional[List[str]] = None
    ) -> WebhookConfig:
        """Create a new webhook endpoint"""
        webhook_id = str(uuid.uuid4())
        
        config = WebhookConfig(
            webhook_id=webhook_id,
            name=name,
            source=source,
            secret=secret,
            allowed_events=allowed_events or []
        )
        
        self._webhooks[webhook_id] = config
        logger.info(f"Created webhook: {name} ({webhook_id})")
        
        return config
    
    def get_webhook(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Get webhook configuration"""
        return self._webhooks.get(webhook_id)
    
    def list_webhooks(self, source: Optional[WebhookSource] = None) -> List[WebhookConfig]:
        """List all webhooks, optionally filtered by source"""
        webhooks = list(self._webhooks.values())
        if source:
            webhooks = [w for w in webhooks if w.source == source]
        return webhooks
    
    def delete_webhook(self, webhook_id: str) -> bool:
        """Delete a webhook"""
        if webhook_id in self._webhooks:
            del self._webhooks[webhook_id]
            logger.info(f"Deleted webhook: {webhook_id}")
            return True
        return False
    
    def update_webhook(
        self,
        webhook_id: str,
        name: Optional[str] = None,
        secret: Optional[str] = None,
        is_active: Optional[bool] = None,
        allowed_events: Optional[List[str]] = None
    ) -> Optional[WebhookConfig]:
        """Update webhook configuration"""
        config = self._webhooks.get(webhook_id)
        if not config:
            return None
            
        if name is not None:
            config.name = name
        if secret is not None:
            config.secret = secret
        if is_active is not None:
            config.is_active = is_active
        if allowed_events is not None:
            config.allowed_events = allowed_events
            
        return config
    
    # ==================== Handler Registration ====================
    
    def register_handler(self, event_pattern: str, handler: WebhookHandler):
        """
        Register a handler for webhook events
        
        Args:
            event_pattern: Pattern like "source:event_type" or "*:event_type"
            handler: Async function to handle the event
        """
        if event_pattern not in self._handlers:
            self._handlers[event_pattern] = []
        self._handlers[event_pattern].append(handler)
        logger.debug(f"Registered handler for: {event_pattern}")
    
    def unregister_handler(self, event_pattern: str, handler: WebhookHandler) -> bool:
        """Unregister a handler"""
        if event_pattern in self._handlers:
            try:
                self._handlers[event_pattern].remove(handler)
                return True
            except ValueError:
                pass
        return False
    
    # ==================== Webhook Processing ====================
    
    async def receive_webhook(
        self,
        webhook_id: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        event_type: Optional[str] = None
    ) -> WebhookEvent:
        """
        Receive and process a webhook
        
        Args:
            webhook_id: The webhook endpoint ID
            payload: The webhook payload
            headers: HTTP headers (for signature verification)
            event_type: Optional event type override
        """
        headers = headers or {}
        
        # Get webhook config
        config = self._webhooks.get(webhook_id)
        if not config:
            raise ValueError(f"Unknown webhook: {webhook_id}")
            
        if not config.is_active:
            raise ValueError(f"Webhook is inactive: {webhook_id}")
        
        # Detect event type
        event_type = event_type or self._detect_event_type(config.source, payload, headers)
        
        # Check if event is allowed
        if config.allowed_events and event_type not in config.allowed_events:
            raise ValueError(f"Event type not allowed: {event_type}")
        
        # Verify signature if secret is set
        if config.secret:
            if not self._verify_signature(config, payload, headers):
                raise ValueError("Invalid webhook signature")
        
        # Create event
        event = WebhookEvent(
            event_id=str(uuid.uuid4()),
            webhook_id=webhook_id,
            source=config.source,
            event_type=event_type,
            payload=payload,
            headers=headers,
            received_at=datetime.now()
        )
        
        # Store event
        self._events[event.event_id] = event
        self._trim_events()
        
        # Process event
        await self._process_event(event)
        
        return event
    
    async def receive_raw_webhook(
        self,
        source: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        event_type: Optional[str] = None
    ) -> WebhookEvent:
        """
        Receive a webhook without a pre-configured endpoint.
        Creates a temporary webhook for processing.
        """
        headers = headers or {}
        
        # Detect source
        webhook_source = self._detect_source(source, payload, headers)
        event_type = event_type or self._detect_event_type(webhook_source, payload, headers)
        
        # Create event
        event = WebhookEvent(
            event_id=str(uuid.uuid4()),
            webhook_id="raw",
            source=webhook_source,
            event_type=event_type,
            payload=payload,
            headers=headers,
            received_at=datetime.now()
        )
        
        # Store and process
        self._events[event.event_id] = event
        self._trim_events()
        await self._process_event(event)
        
        return event
    
    async def _process_event(self, event: WebhookEvent):
        """Process a webhook event through registered handlers"""
        try:
            # Find matching handlers
            handlers = self._find_handlers(event)
            
            if not handlers:
                logger.debug(f"No handlers for event: {event.source}:{event.event_type}")
                event.result = {"message": "No handlers registered"}
            else:
                results = []
                for handler in handlers:
                    try:
                        result = await handler(event)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Handler error: {e}")
                        results.append({"error": str(e)})
                        
                event.result = {"handlers_executed": len(handlers), "results": results}
                
            event.processed = True
            event.processed_at = datetime.now()
            
        except Exception as e:
            event.error = str(e)
            event.processed = True
            event.processed_at = datetime.now()
            logger.error(f"Event processing failed: {e}")
    
    def _find_handlers(self, event: WebhookEvent) -> List[WebhookHandler]:
        """Find all handlers matching an event"""
        handlers = []
        
        # Exact match: source:event_type
        pattern = f"{event.source.value}:{event.event_type}"
        handlers.extend(self._handlers.get(pattern, []))
        
        # Wildcard source: *:event_type
        pattern = f"*:{event.event_type}"
        handlers.extend(self._handlers.get(pattern, []))
        
        # Wildcard event: source:*
        pattern = f"{event.source.value}:*"
        handlers.extend(self._handlers.get(pattern, []))
        
        # Global wildcard: *:*
        handlers.extend(self._handlers.get("*:*", []))
        
        return handlers
    
    def _detect_source(self, source_hint: str, payload: Dict, headers: Dict) -> WebhookSource:
        """Detect webhook source"""
        source_hint = source_hint.lower()
        
        if source_hint in ["n8n", "n8n-webhook"]:
            return WebhookSource.N8N
        elif source_hint in ["comfyui", "comfy"]:
            return WebhookSource.COMFYUI
        elif source_hint == "github" or "x-github-event" in headers:
            return WebhookSource.GITHUB
        elif source_hint == "custom":
            return WebhookSource.CUSTOM
        else:
            return WebhookSource.UNKNOWN
    
    def _detect_event_type(self, source: WebhookSource, payload: Dict, headers: Dict) -> str:
        """Detect event type from payload/headers"""
        # GitHub
        if source == WebhookSource.GITHUB:
            return headers.get("x-github-event", "unknown")
            
        # n8n
        if source == WebhookSource.N8N:
            return payload.get("event", payload.get("type", "webhook"))
            
        # ComfyUI
        if source == WebhookSource.COMFYUI:
            return payload.get("type", "generation")
            
        # Generic
        return payload.get("event", payload.get("type", "unknown"))
    
    def _verify_signature(self, config: WebhookConfig, payload: Dict, headers: Dict) -> bool:
        """Verify webhook signature"""
        import json
        
        if not config.secret:
            return True
            
        # GitHub signature verification
        if config.source == WebhookSource.GITHUB:
            signature = headers.get("x-hub-signature-256", "")
            if signature.startswith("sha256="):
                expected = "sha256=" + hmac.new(
                    config.secret.encode(),
                    json.dumps(payload, separators=(',', ':')).encode(),
                    hashlib.sha256
                ).hexdigest()
                return hmac.compare_digest(signature, expected)
                
        # Generic HMAC verification
        signature = headers.get("x-webhook-signature", headers.get("x-signature", ""))
        if signature:
            expected = hmac.new(
                config.secret.encode(),
                json.dumps(payload, separators=(',', ':')).encode(),
                hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(signature, expected)
            
        return True  # No signature to verify
    
    def _trim_events(self):
        """Trim old events if over max"""
        if len(self._events) > self._max_events:
            # Remove oldest events
            sorted_events = sorted(self._events.values(), key=lambda e: e.received_at)
            for event in sorted_events[:len(self._events) - self._max_events]:
                del self._events[event.event_id]
    
    # ==================== Event History ====================
    
    def get_event(self, event_id: str) -> Optional[WebhookEvent]:
        """Get a specific event"""
        return self._events.get(event_id)
    
    def get_events(
        self,
        webhook_id: Optional[str] = None,
        source: Optional[WebhookSource] = None,
        limit: int = 50
    ) -> List[WebhookEvent]:
        """Get recent events"""
        events = list(self._events.values())
        
        if webhook_id:
            events = [e for e in events if e.webhook_id == webhook_id]
        if source:
            events = [e for e in events if e.source == source]
            
        events.sort(key=lambda e: e.received_at, reverse=True)
        return events[:limit]
    
    # ==================== Default Handlers ====================
    
    async def _handle_n8n_completion(self, event: WebhookEvent) -> Dict[str, Any]:
        """Handle n8n workflow completion"""
        workflow_id = event.payload.get("workflowId")
        execution_id = event.payload.get("executionId")
        
        logger.info(f"n8n workflow completed: {workflow_id} (execution: {execution_id})")
        
        return {
            "handled": True,
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "status": "completed"
        }
    
    async def _handle_n8n_failure(self, event: WebhookEvent) -> Dict[str, Any]:
        """Handle n8n workflow failure"""
        workflow_id = event.payload.get("workflowId")
        error = event.payload.get("error", "Unknown error")
        
        logger.warning(f"n8n workflow failed: {workflow_id} - {error}")
        
        return {
            "handled": True,
            "workflow_id": workflow_id,
            "error": error,
            "status": "failed"
        }
    
    async def _handle_github_push(self, event: WebhookEvent) -> Dict[str, Any]:
        """Handle GitHub push event"""
        repo = event.payload.get("repository", {}).get("full_name", "unknown")
        ref = event.payload.get("ref", "")
        commits = len(event.payload.get("commits", []))
        
        logger.info(f"GitHub push to {repo}:{ref} ({commits} commits)")
        
        return {
            "handled": True,
            "repo": repo,
            "ref": ref,
            "commits": commits
        }
    
    async def _handle_github_workflow(self, event: WebhookEvent) -> Dict[str, Any]:
        """Handle GitHub workflow run event"""
        workflow = event.payload.get("workflow", {}).get("name", "unknown")
        status = event.payload.get("workflow_run", {}).get("status", "unknown")
        conclusion = event.payload.get("workflow_run", {}).get("conclusion")
        
        logger.info(f"GitHub workflow '{workflow}': {status} ({conclusion})")
        
        return {
            "handled": True,
            "workflow": workflow,
            "status": status,
            "conclusion": conclusion
        }


# Singleton instance
_receiver: Optional[WebhookReceiver] = None


def get_webhook_receiver() -> WebhookReceiver:
    """Get the global webhook receiver instance"""
    global _receiver
    if _receiver is None:
        _receiver = WebhookReceiver()
    return _receiver
