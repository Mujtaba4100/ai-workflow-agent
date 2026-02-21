# Additional n8n Workflow Templates
"""
Extended workflow templates for common automation patterns.
Milestone 1: More comprehensive template library.
"""

from datetime import datetime
from typing import Dict, Any


def get_workflow_templates() -> Dict[str, callable]:
    """Return all available workflow templates."""
    return {
        "database_sync": database_sync_workflow,
        "file_processor": file_processor_workflow,
        "social_media": social_media_workflow,
        "crm_integration": crm_integration_workflow,
        "monitoring": monitoring_workflow,
        "data_pipeline": data_pipeline_workflow,
        "chatbot": chatbot_workflow,
        "report_generator": report_generator_workflow,
    }


def database_sync_workflow(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate database synchronization workflow."""
    return {
        "name": f"Database Sync - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "nodes": [
            {
                "parameters": {
                    "rule": {"interval": [{"field": "hours", "hoursInterval": 1}]}
                },
                "id": "schedule_1",
                "name": "Schedule",
                "type": "n8n-nodes-base.scheduleTrigger",
                "typeVersion": 1,
                "position": [250, 300]
            },
            {
                "parameters": {
                    "operation": "executeQuery",
                    "query": "SELECT * FROM source_table WHERE updated_at > NOW() - INTERVAL '1 hour'"
                },
                "id": "postgres_source",
                "name": "Source DB",
                "type": "n8n-nodes-base.postgres",
                "typeVersion": 2,
                "position": [450, 300],
                "credentials": {"postgres": {"id": "SOURCE_DB_ID", "name": "Source Database"}}
            },
            {
                "parameters": {
                    "operation": "insert",
                    "table": "destination_table",
                    "columns": "={{ Object.keys($json).join(',') }}"
                },
                "id": "postgres_dest",
                "name": "Destination DB",
                "type": "n8n-nodes-base.postgres",
                "typeVersion": 2,
                "position": [650, 300],
                "credentials": {"postgres": {"id": "DEST_DB_ID", "name": "Destination Database"}}
            }
        ],
        "connections": {
            "Schedule": {"main": [[{"node": "Source DB", "type": "main", "index": 0}]]},
            "Source DB": {"main": [[{"node": "Destination DB", "type": "main", "index": 0}]]}
        },
        "settings": {"executionOrder": "v1"},
        "meta": {"type": "database_sync", "generated_by": "AI Workflow Agent"}
    }


def file_processor_workflow(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate file processing workflow."""
    return {
        "name": f"File Processor - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "nodes": [
            {
                "parameters": {"path": "/data/input", "events": ["change"]},
                "id": "file_trigger",
                "name": "File Trigger",
                "type": "n8n-nodes-base.localFileTrigger",
                "typeVersion": 1,
                "position": [250, 300]
            },
            {
                "parameters": {"filePath": "={{ $json.path }}"},
                "id": "read_file",
                "name": "Read File",
                "type": "n8n-nodes-base.readBinaryFiles",
                "typeVersion": 1,
                "position": [450, 300]
            },
            {
                "parameters": {
                    "functionCode": """// Process file content
const items = $input.all();
const processed = items.map(item => {
  // Add processing logic here
  return {
    json: {
      ...item.json,
      processed: true,
      processedAt: new Date().toISOString()
    }
  };
});
return processed;"""
                },
                "id": "process",
                "name": "Process",
                "type": "n8n-nodes-base.code",
                "typeVersion": 1,
                "position": [650, 300]
            },
            {
                "parameters": {
                    "fileName": "={{ 'processed_' + $json.filename }}",
                    "filePath": "/data/output/"
                },
                "id": "write_file",
                "name": "Save Output",
                "type": "n8n-nodes-base.writeBinaryFile",
                "typeVersion": 1,
                "position": [850, 300]
            }
        ],
        "connections": {
            "File Trigger": {"main": [[{"node": "Read File", "type": "main", "index": 0}]]},
            "Read File": {"main": [[{"node": "Process", "type": "main", "index": 0}]]},
            "Process": {"main": [[{"node": "Save Output", "type": "main", "index": 0}]]}
        },
        "settings": {"executionOrder": "v1"},
        "meta": {"type": "file_processor", "generated_by": "AI Workflow Agent"}
    }


def social_media_workflow(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate social media automation workflow."""
    return {
        "name": f"Social Media - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "nodes": [
            {
                "parameters": {
                    "httpMethod": "POST",
                    "path": "post-social",
                    "responseMode": "onReceived"
                },
                "id": "webhook",
                "name": "Trigger",
                "type": "n8n-nodes-base.webhook",
                "typeVersion": 1,
                "position": [250, 300]
            },
            {
                "parameters": {
                    "functionCode": """// Prepare content for different platforms
const content = $input.first().json;
return [
  {
    json: {
      platform: 'twitter',
      text: content.message.substring(0, 280),
      media: content.image_url
    }
  },
  {
    json: {
      platform: 'linkedin',
      text: content.message,
      media: content.image_url
    }
  }
];"""
                },
                "id": "prepare",
                "name": "Prepare Content",
                "type": "n8n-nodes-base.code",
                "typeVersion": 1,
                "position": [450, 300]
            },
            {
                "parameters": {
                    "text": "={{ $json.text }}",
                    "additionalFields": {}
                },
                "id": "twitter",
                "name": "Post to Twitter",
                "type": "n8n-nodes-base.twitter",
                "typeVersion": 2,
                "position": [650, 200],
                "credentials": {"twitterOAuth2Api": {"id": "TWITTER_ID", "name": "Twitter"}}
            },
            {
                "parameters": {
                    "text": "={{ $json.text }}",
                    "shareMediaCategory": "NONE"
                },
                "id": "linkedin",
                "name": "Post to LinkedIn",
                "type": "n8n-nodes-base.linkedIn",
                "typeVersion": 1,
                "position": [650, 400],
                "credentials": {"linkedInOAuth2Api": {"id": "LINKEDIN_ID", "name": "LinkedIn"}}
            }
        ],
        "connections": {
            "Trigger": {"main": [[{"node": "Prepare Content", "type": "main", "index": 0}]]},
            "Prepare Content": {"main": [
                [{"node": "Post to Twitter", "type": "main", "index": 0}],
                [{"node": "Post to LinkedIn", "type": "main", "index": 0}]
            ]}
        },
        "settings": {"executionOrder": "v1"},
        "meta": {"type": "social_media", "generated_by": "AI Workflow Agent"}
    }


def crm_integration_workflow(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate CRM integration workflow."""
    return {
        "name": f"CRM Integration - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "nodes": [
            {
                "parameters": {
                    "httpMethod": "POST",
                    "path": "new-lead",
                    "responseMode": "responseNode"
                },
                "id": "webhook",
                "name": "New Lead",
                "type": "n8n-nodes-base.webhook",
                "typeVersion": 1,
                "position": [250, 300]
            },
            {
                "parameters": {
                    "resource": "contact",
                    "operation": "create",
                    "email": "={{ $json.email }}",
                    "additionalFields": {
                        "firstName": "={{ $json.first_name }}",
                        "lastName": "={{ $json.last_name }}",
                        "phone": "={{ $json.phone }}"
                    }
                },
                "id": "hubspot",
                "name": "Create in HubSpot",
                "type": "n8n-nodes-base.hubspot",
                "typeVersion": 2,
                "position": [450, 300],
                "credentials": {"hubspotApi": {"id": "HUBSPOT_ID", "name": "HubSpot"}}
            },
            {
                "parameters": {
                    "fromEmail": "sales@company.com",
                    "toEmail": "={{ $json.email }}",
                    "subject": "Welcome to our platform!",
                    "emailType": "html",
                    "message": "<h1>Welcome!</h1><p>Thank you for your interest.</p>"
                },
                "id": "email",
                "name": "Send Welcome Email",
                "type": "n8n-nodes-base.emailSend",
                "typeVersion": 2,
                "position": [650, 300],
                "credentials": {"smtp": {"id": "SMTP_ID", "name": "SMTP"}}
            },
            {
                "parameters": {
                    "respondWith": "json",
                    "responseBody": "={{ JSON.stringify({success: true, contact_id: $json.id}) }}"
                },
                "id": "respond",
                "name": "Response",
                "type": "n8n-nodes-base.respondToWebhook",
                "typeVersion": 1,
                "position": [850, 300]
            }
        ],
        "connections": {
            "New Lead": {"main": [[{"node": "Create in HubSpot", "type": "main", "index": 0}]]},
            "Create in HubSpot": {"main": [[{"node": "Send Welcome Email", "type": "main", "index": 0}]]},
            "Send Welcome Email": {"main": [[{"node": "Response", "type": "main", "index": 0}]]}
        },
        "settings": {"executionOrder": "v1"},
        "meta": {"type": "crm_integration", "generated_by": "AI Workflow Agent"}
    }


def monitoring_workflow(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate monitoring and alerting workflow."""
    return {
        "name": f"Monitoring - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "nodes": [
            {
                "parameters": {
                    "rule": {"interval": [{"field": "minutes", "minutesInterval": 5}]}
                },
                "id": "schedule",
                "name": "Every 5 Minutes",
                "type": "n8n-nodes-base.scheduleTrigger",
                "typeVersion": 1,
                "position": [250, 300]
            },
            {
                "parameters": {
                    "url": "={{ $json.endpoint || 'https://api.example.com/health' }}",
                    "method": "GET",
                    "options": {"timeout": 10000}
                },
                "id": "check",
                "name": "Health Check",
                "type": "n8n-nodes-base.httpRequest",
                "typeVersion": 4,
                "position": [450, 300]
            },
            {
                "parameters": {
                    "conditions": {
                        "boolean": [
                            {"value1": "={{ $json.status }}", "value2": 200, "operation": "notEqual"}
                        ]
                    }
                },
                "id": "if_error",
                "name": "If Error",
                "type": "n8n-nodes-base.if",
                "typeVersion": 1,
                "position": [650, 300]
            },
            {
                "parameters": {
                    "channel": "#alerts",
                    "text": "🚨 *Alert*: Service health check failed!\nStatus: {{ $json.status }}\nTime: {{ new Date().toISOString() }}"
                },
                "id": "slack",
                "name": "Alert Slack",
                "type": "n8n-nodes-base.slack",
                "typeVersion": 2,
                "position": [850, 200],
                "credentials": {"slackApi": {"id": "SLACK_ID", "name": "Slack"}}
            },
            {
                "parameters": {},
                "id": "noop",
                "name": "All Good",
                "type": "n8n-nodes-base.noOp",
                "typeVersion": 1,
                "position": [850, 400]
            }
        ],
        "connections": {
            "Every 5 Minutes": {"main": [[{"node": "Health Check", "type": "main", "index": 0}]]},
            "Health Check": {"main": [[{"node": "If Error", "type": "main", "index": 0}]]},
            "If Error": {
                "main": [
                    [{"node": "Alert Slack", "type": "main", "index": 0}],
                    [{"node": "All Good", "type": "main", "index": 0}]
                ]
            }
        },
        "settings": {"executionOrder": "v1"},
        "meta": {"type": "monitoring", "generated_by": "AI Workflow Agent"}
    }


def data_pipeline_workflow(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate data processing pipeline workflow."""
    return {
        "name": f"Data Pipeline - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "nodes": [
            {
                "parameters": {
                    "httpMethod": "POST",
                    "path": "ingest-data",
                    "responseMode": "onReceived"
                },
                "id": "webhook",
                "name": "Data Input",
                "type": "n8n-nodes-base.webhook",
                "typeVersion": 1,
                "position": [250, 300]
            },
            {
                "parameters": {
                    "functionCode": """// Validate incoming data
const data = $input.all();
const validated = data.filter(item => {
  const json = item.json;
  return json.id && json.timestamp && json.value !== undefined;
});
if (validated.length === 0) {
  throw new Error('No valid data received');
}
return validated;"""
                },
                "id": "validate",
                "name": "Validate",
                "type": "n8n-nodes-base.code",
                "typeVersion": 1,
                "position": [450, 300]
            },
            {
                "parameters": {
                    "functionCode": """// Transform data
const items = $input.all();
return items.map(item => ({
  json: {
    ...item.json,
    processed_value: item.json.value * 1.1,
    source: 'pipeline',
    ingested_at: new Date().toISOString()
  }
}));"""
                },
                "id": "transform",
                "name": "Transform",
                "type": "n8n-nodes-base.code",
                "typeVersion": 1,
                "position": [650, 300]
            },
            {
                "parameters": {
                    "operation": "insert",
                    "table": "processed_data"
                },
                "id": "store",
                "name": "Store",
                "type": "n8n-nodes-base.postgres",
                "typeVersion": 2,
                "position": [850, 300],
                "credentials": {"postgres": {"id": "DB_ID", "name": "Database"}}
            }
        ],
        "connections": {
            "Data Input": {"main": [[{"node": "Validate", "type": "main", "index": 0}]]},
            "Validate": {"main": [[{"node": "Transform", "type": "main", "index": 0}]]},
            "Transform": {"main": [[{"node": "Store", "type": "main", "index": 0}]]}
        },
        "settings": {"executionOrder": "v1"},
        "meta": {"type": "data_pipeline", "generated_by": "AI Workflow Agent"}
    }


def chatbot_workflow(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate AI chatbot workflow."""
    return {
        "name": f"AI Chatbot - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "nodes": [
            {
                "parameters": {
                    "httpMethod": "POST",
                    "path": "chat",
                    "responseMode": "responseNode"
                },
                "id": "webhook",
                "name": "Chat Input",
                "type": "n8n-nodes-base.webhook",
                "typeVersion": 1,
                "position": [250, 300]
            },
            {
                "parameters": {
                    "url": "http://ollama:11434/api/generate",
                    "method": "POST",
                    "sendBody": True,
                    "bodyParameters": {
                        "parameters": [
                            {"name": "model", "value": "qwen2.5:7b"},
                            {"name": "prompt", "value": "={{ $json.message }}"},
                            {"name": "stream", "value": "false"}
                        ]
                    }
                },
                "id": "ollama",
                "name": "Ask LLM",
                "type": "n8n-nodes-base.httpRequest",
                "typeVersion": 4,
                "position": [450, 300]
            },
            {
                "parameters": {
                    "respondWith": "json",
                    "responseBody": "={{ JSON.stringify({response: $json.response, model: 'qwen2.5'}) }}"
                },
                "id": "respond",
                "name": "Response",
                "type": "n8n-nodes-base.respondToWebhook",
                "typeVersion": 1,
                "position": [650, 300]
            }
        ],
        "connections": {
            "Chat Input": {"main": [[{"node": "Ask LLM", "type": "main", "index": 0}]]},
            "Ask LLM": {"main": [[{"node": "Response", "type": "main", "index": 0}]]}
        },
        "settings": {"executionOrder": "v1"},
        "meta": {"type": "chatbot", "generated_by": "AI Workflow Agent"}
    }


def report_generator_workflow(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate automated report workflow."""
    return {
        "name": f"Report Generator - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "nodes": [
            {
                "parameters": {
                    "rule": {"interval": [{"field": "days", "daysInterval": 1}]},
                    "triggerAtTime": "09:00"
                },
                "id": "schedule",
                "name": "Daily at 9 AM",
                "type": "n8n-nodes-base.scheduleTrigger",
                "typeVersion": 1,
                "position": [250, 300]
            },
            {
                "parameters": {
                    "operation": "executeQuery",
                    "query": """SELECT 
                        DATE(created_at) as date,
                        COUNT(*) as total,
                        SUM(amount) as revenue
                    FROM orders 
                    WHERE created_at >= NOW() - INTERVAL '7 days'
                    GROUP BY DATE(created_at)
                    ORDER BY date"""
                },
                "id": "query",
                "name": "Get Data",
                "type": "n8n-nodes-base.postgres",
                "typeVersion": 2,
                "position": [450, 300],
                "credentials": {"postgres": {"id": "DB_ID", "name": "Database"}}
            },
            {
                "parameters": {
                    "functionCode": """// Generate HTML report
const data = $input.all();
let html = '<h1>Weekly Report</h1><table border="1">';
html += '<tr><th>Date</th><th>Orders</th><th>Revenue</th></tr>';
data.forEach(row => {
  html += '<tr>';
  html += '<td>' + row.json.date + '</td>';
  html += '<td>' + row.json.total + '</td>';
  html += '<td>$' + row.json.revenue.toFixed(2) + '</td>';
  html += '</tr>';
});
html += '</table>';
return [{json: {html, generated: new Date().toISOString()}}];"""
                },
                "id": "generate",
                "name": "Generate Report",
                "type": "n8n-nodes-base.code",
                "typeVersion": 1,
                "position": [650, 300]
            },
            {
                "parameters": {
                    "fromEmail": "reports@company.com",
                    "toEmail": "team@company.com",
                    "subject": "Weekly Report - {{ new Date().toLocaleDateString() }}",
                    "emailType": "html",
                    "message": "={{ $json.html }}"
                },
                "id": "email",
                "name": "Send Report",
                "type": "n8n-nodes-base.emailSend",
                "typeVersion": 2,
                "position": [850, 300],
                "credentials": {"smtp": {"id": "SMTP_ID", "name": "SMTP"}}
            }
        ],
        "connections": {
            "Daily at 9 AM": {"main": [[{"node": "Get Data", "type": "main", "index": 0}]]},
            "Get Data": {"main": [[{"node": "Generate Report", "type": "main", "index": 0}]]},
            "Generate Report": {"main": [[{"node": "Send Report", "type": "main", "index": 0}]]}
        },
        "settings": {"executionOrder": "v1"},
        "meta": {"type": "report_generator", "generated_by": "AI Workflow Agent"}
    }
