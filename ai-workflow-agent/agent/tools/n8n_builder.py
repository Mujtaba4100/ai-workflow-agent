# n8n Workflow Builder Tool
"""
Generate and deploy n8n workflow JSON templates.
Supports common automation patterns.
"""

import httpx
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from config import settings

logger = logging.getLogger(__name__)


class N8NWorkflowBuilder:
    """
    n8n workflow generator and deployer.
    Creates JSON workflow templates and deploys via n8n API.
    """
    
    def __init__(self):
        self.n8n_host = settings.N8N_HOST
        self.api_key = settings.N8N_API_KEY
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Headers for n8n API
        self.headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            self.headers["X-N8N-API-KEY"] = self.api_key
    
    async def check_health(self) -> str:
        """Check if n8n is running and responsive."""
        try:
            response = await self.client.get(
                f"{self.n8n_host}/healthz"
            )
            if response.status_code == 200:
                return "healthy"
            return "unhealthy"
        except Exception as e:
            logger.debug(f"n8n health check failed: {e}")
            return "unreachable"
    
    async def generate_workflow(self, query: str) -> Dict[str, Any]:
        """
        Generate n8n workflow JSON based on user query.
        
        Args:
            query: User's natural language request
            
        Returns:
            n8n workflow JSON structure
        """
        # Detect workflow type from query
        workflow_type = self._detect_workflow_type(query)
        
        # Generate appropriate template
        templates = {
            "webhook": self._generate_webhook_workflow,
            "schedule": self._generate_schedule_workflow,
            "api_integration": self._generate_api_workflow,
            "email": self._generate_email_workflow,
            "data_transform": self._generate_transform_workflow,
            "notification": self._generate_notification_workflow,
            "generic": self._generate_generic_workflow
        }
        
        generator = templates.get(workflow_type, self._generate_generic_workflow)
        workflow = generator(query)
        
        return workflow
    
    def _detect_workflow_type(self, query: str) -> str:
        """Detect the type of workflow needed from query."""
        query_lower = query.lower()
        
        if any(w in query_lower for w in ["webhook", "http", "api call", "endpoint"]):
            return "webhook"
        elif any(w in query_lower for w in ["schedule", "cron", "every day", "every hour", "periodic"]):
            return "schedule"
        elif any(w in query_lower for w in ["api", "rest", "fetch", "get data"]):
            return "api_integration"
        elif any(w in query_lower for w in ["email", "mail", "send message", "gmail"]):
            return "email"
        elif any(w in query_lower for w in ["transform", "convert", "process", "parse"]):
            return "data_transform"
        elif any(w in query_lower for w in ["notify", "slack", "telegram", "alert"]):
            return "notification"
        else:
            return "generic"
    
    def _generate_webhook_workflow(self, query: str) -> Dict[str, Any]:
        """Generate webhook-triggered workflow."""
        return {
            "name": f"Webhook Workflow - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "nodes": [
                {
                    "parameters": {
                        "httpMethod": "POST",
                        "path": "webhook-trigger",
                        "responseMode": "onReceived",
                        "responseData": "allEntries"
                    },
                    "id": "webhook_1",
                    "name": "Webhook",
                    "type": "n8n-nodes-base.webhook",
                    "typeVersion": 1,
                    "position": [250, 300]
                },
                {
                    "parameters": {},
                    "id": "set_1",
                    "name": "Process Data",
                    "type": "n8n-nodes-base.set",
                    "typeVersion": 1,
                    "position": [450, 300]
                },
                {
                    "parameters": {
                        "functionCode": "// Process incoming data\nconst items = $input.all();\nreturn items;"
                    },
                    "id": "code_1",
                    "name": "Custom Logic",
                    "type": "n8n-nodes-base.code",
                    "typeVersion": 1,
                    "position": [650, 300]
                }
            ],
            "connections": {
                "Webhook": {
                    "main": [[{"node": "Process Data", "type": "main", "index": 0}]]
                },
                "Process Data": {
                    "main": [[{"node": "Custom Logic", "type": "main", "index": 0}]]
                }
            },
            "settings": {
                "executionOrder": "v1"
            },
            "meta": {
                "generated_by": "AI Workflow Agent",
                "query": query,
                "type": "webhook"
            }
        }
    
    def _generate_schedule_workflow(self, query: str) -> Dict[str, Any]:
        """Generate schedule-triggered workflow."""
        return {
            "name": f"Scheduled Workflow - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "nodes": [
                {
                    "parameters": {
                        "rule": {
                            "interval": [{"field": "hours", "hoursInterval": 1}]
                        }
                    },
                    "id": "schedule_1",
                    "name": "Schedule Trigger",
                    "type": "n8n-nodes-base.scheduleTrigger",
                    "typeVersion": 1,
                    "position": [250, 300]
                },
                {
                    "parameters": {
                        "functionCode": "// Scheduled task logic\nconst now = new Date();\nreturn [{ json: { timestamp: now.toISOString(), status: 'executed' } }];"
                    },
                    "id": "code_1",
                    "name": "Execute Task",
                    "type": "n8n-nodes-base.code",
                    "typeVersion": 1,
                    "position": [450, 300]
                }
            ],
            "connections": {
                "Schedule Trigger": {
                    "main": [[{"node": "Execute Task", "type": "main", "index": 0}]]
                }
            },
            "settings": {
                "executionOrder": "v1"
            },
            "meta": {
                "generated_by": "AI Workflow Agent",
                "query": query,
                "type": "schedule"
            }
        }
    
    def _generate_api_workflow(self, query: str) -> Dict[str, Any]:
        """Generate API integration workflow."""
        return {
            "name": f"API Integration - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "nodes": [
                {
                    "parameters": {
                        "httpMethod": "GET",
                        "path": "api-trigger",
                        "responseMode": "responseNode"
                    },
                    "id": "webhook_1",
                    "name": "API Trigger",
                    "type": "n8n-nodes-base.webhook",
                    "typeVersion": 1,
                    "position": [250, 300]
                },
                {
                    "parameters": {
                        "url": "https://api.example.com/data",
                        "method": "GET",
                        "options": {}
                    },
                    "id": "http_1",
                    "name": "HTTP Request",
                    "type": "n8n-nodes-base.httpRequest",
                    "typeVersion": 4,
                    "position": [450, 300]
                },
                {
                    "parameters": {
                        "respondWith": "json",
                        "responseBody": "={{ $json }}"
                    },
                    "id": "respond_1",
                    "name": "Respond",
                    "type": "n8n-nodes-base.respondToWebhook",
                    "typeVersion": 1,
                    "position": [650, 300]
                }
            ],
            "connections": {
                "API Trigger": {
                    "main": [[{"node": "HTTP Request", "type": "main", "index": 0}]]
                },
                "HTTP Request": {
                    "main": [[{"node": "Respond", "type": "main", "index": 0}]]
                }
            },
            "settings": {
                "executionOrder": "v1"
            },
            "meta": {
                "generated_by": "AI Workflow Agent",
                "query": query,
                "type": "api_integration"
            }
        }
    
    def _generate_email_workflow(self, query: str) -> Dict[str, Any]:
        """Generate email workflow."""
        return {
            "name": f"Email Workflow - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "nodes": [
                {
                    "parameters": {
                        "httpMethod": "POST",
                        "path": "send-email",
                        "responseMode": "onReceived"
                    },
                    "id": "webhook_1",
                    "name": "Trigger",
                    "type": "n8n-nodes-base.webhook",
                    "typeVersion": 1,
                    "position": [250, 300]
                },
                {
                    "parameters": {
                        "fromEmail": "={{ $json.from || 'noreply@example.com' }}",
                        "toEmail": "={{ $json.to }}",
                        "subject": "={{ $json.subject }}",
                        "emailType": "text",
                        "message": "={{ $json.body }}"
                    },
                    "id": "email_1",
                    "name": "Send Email",
                    "type": "n8n-nodes-base.emailSend",
                    "typeVersion": 2,
                    "position": [450, 300],
                    "credentials": {
                        "smtp": {
                            "id": "SMTP_CREDENTIAL_ID",
                            "name": "SMTP Account"
                        }
                    }
                }
            ],
            "connections": {
                "Trigger": {
                    "main": [[{"node": "Send Email", "type": "main", "index": 0}]]
                }
            },
            "settings": {
                "executionOrder": "v1"
            },
            "meta": {
                "generated_by": "AI Workflow Agent",
                "query": query,
                "type": "email",
                "note": "Requires SMTP credentials configuration"
            }
        }
    
    def _generate_transform_workflow(self, query: str) -> Dict[str, Any]:
        """Generate data transformation workflow."""
        return {
            "name": f"Data Transform - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "nodes": [
                {
                    "parameters": {
                        "httpMethod": "POST",
                        "path": "transform",
                        "responseMode": "responseNode"
                    },
                    "id": "webhook_1",
                    "name": "Input",
                    "type": "n8n-nodes-base.webhook",
                    "typeVersion": 1,
                    "position": [250, 300]
                },
                {
                    "parameters": {
                        "functionCode": "// Transform data\nconst items = $input.all();\nconst transformed = items.map(item => {\n  return {\n    json: {\n      ...item.json,\n      processed: true,\n      timestamp: new Date().toISOString()\n    }\n  };\n});\nreturn transformed;"
                    },
                    "id": "code_1",
                    "name": "Transform",
                    "type": "n8n-nodes-base.code",
                    "typeVersion": 1,
                    "position": [450, 300]
                },
                {
                    "parameters": {
                        "respondWith": "json",
                        "responseBody": "={{ $json }}"
                    },
                    "id": "respond_1",
                    "name": "Output",
                    "type": "n8n-nodes-base.respondToWebhook",
                    "typeVersion": 1,
                    "position": [650, 300]
                }
            ],
            "connections": {
                "Input": {
                    "main": [[{"node": "Transform", "type": "main", "index": 0}]]
                },
                "Transform": {
                    "main": [[{"node": "Output", "type": "main", "index": 0}]]
                }
            },
            "settings": {
                "executionOrder": "v1"
            },
            "meta": {
                "generated_by": "AI Workflow Agent",
                "query": query,
                "type": "data_transform"
            }
        }
    
    def _generate_notification_workflow(self, query: str) -> Dict[str, Any]:
        """Generate notification workflow (Slack/Telegram)."""
        return {
            "name": f"Notification Workflow - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "nodes": [
                {
                    "parameters": {
                        "httpMethod": "POST",
                        "path": "notify",
                        "responseMode": "onReceived"
                    },
                    "id": "webhook_1",
                    "name": "Trigger",
                    "type": "n8n-nodes-base.webhook",
                    "typeVersion": 1,
                    "position": [250, 300]
                },
                {
                    "parameters": {
                        "channel": "={{ $json.channel || '#general' }}",
                        "text": "={{ $json.message }}",
                        "otherOptions": {}
                    },
                    "id": "slack_1",
                    "name": "Slack",
                    "type": "n8n-nodes-base.slack",
                    "typeVersion": 2,
                    "position": [450, 250],
                    "credentials": {
                        "slackApi": {
                            "id": "SLACK_CREDENTIAL_ID",
                            "name": "Slack Account"
                        }
                    }
                },
                {
                    "parameters": {
                        "chatId": "={{ $json.chat_id }}",
                        "text": "={{ $json.message }}"
                    },
                    "id": "telegram_1",
                    "name": "Telegram",
                    "type": "n8n-nodes-base.telegram",
                    "typeVersion": 1,
                    "position": [450, 350],
                    "credentials": {
                        "telegramApi": {
                            "id": "TELEGRAM_CREDENTIAL_ID",
                            "name": "Telegram Bot"
                        }
                    }
                }
            ],
            "connections": {
                "Trigger": {
                    "main": [
                        [
                            {"node": "Slack", "type": "main", "index": 0},
                            {"node": "Telegram", "type": "main", "index": 0}
                        ]
                    ]
                }
            },
            "settings": {
                "executionOrder": "v1"
            },
            "meta": {
                "generated_by": "AI Workflow Agent",
                "query": query,
                "type": "notification",
                "note": "Requires Slack/Telegram credentials configuration"
            }
        }
    
    def _generate_generic_workflow(self, query: str) -> Dict[str, Any]:
        """Generate generic workflow template."""
        return {
            "name": f"Workflow - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "nodes": [
                {
                    "parameters": {
                        "httpMethod": "POST",
                        "path": "workflow-trigger",
                        "responseMode": "responseNode"
                    },
                    "id": "webhook_1",
                    "name": "Start",
                    "type": "n8n-nodes-base.webhook",
                    "typeVersion": 1,
                    "position": [250, 300]
                },
                {
                    "parameters": {
                        "functionCode": f"// Generated for: {query}\n// Add your custom logic here\nconst input = $input.all();\nreturn input;"
                    },
                    "id": "code_1",
                    "name": "Process",
                    "type": "n8n-nodes-base.code",
                    "typeVersion": 1,
                    "position": [450, 300]
                },
                {
                    "parameters": {
                        "respondWith": "json",
                        "responseBody": "={{ $json }}"
                    },
                    "id": "respond_1",
                    "name": "End",
                    "type": "n8n-nodes-base.respondToWebhook",
                    "typeVersion": 1,
                    "position": [650, 300]
                }
            ],
            "connections": {
                "Start": {
                    "main": [[{"node": "Process", "type": "main", "index": 0}]]
                },
                "Process": {
                    "main": [[{"node": "End", "type": "main", "index": 0}]]
                }
            },
            "settings": {
                "executionOrder": "v1"
            },
            "meta": {
                "generated_by": "AI Workflow Agent",
                "query": query,
                "type": "generic"
            }
        }
    
    async def deploy_workflow(self, workflow: Dict[str, Any]) -> str:
        """
        Deploy workflow to n8n instance via API.
        
        Args:
            workflow: n8n workflow JSON
            
        Returns:
            Workflow ID if successful
        """
        try:
            response = await self.client.post(
                f"{self.n8n_host}/api/v1/workflows",
                headers=self.headers,
                json=workflow
            )
            
            if response.status_code in [200, 201]:
                data = response.json()
                workflow_id = data.get("id", "unknown")
                logger.info(f"Workflow deployed: {workflow_id}")
                return workflow_id
            else:
                logger.error(f"Deploy failed: {response.status_code} - {response.text}")
                raise Exception(f"Deploy failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Deploy error: {e}")
            raise
    
    async def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows in n8n."""
        try:
            response = await self.client.get(
                f"{self.n8n_host}/api/v1/workflows",
                headers=self.headers
            )
            
            if response.status_code == 200:
                return response.json().get("data", [])
            return []
            
        except Exception as e:
            logger.error(f"List workflows error: {e}")
            return []
    
    async def activate_workflow(self, workflow_id: str, active: bool = True) -> bool:
        """Activate or deactivate a workflow."""
        try:
            response = await self.client.patch(
                f"{self.n8n_host}/api/v1/workflows/{workflow_id}",
                headers=self.headers,
                json={"active": active}
            )
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Activate workflow error: {e}")
            return False
