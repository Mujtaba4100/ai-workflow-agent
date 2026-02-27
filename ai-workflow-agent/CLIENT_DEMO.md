# AI Workflow Agent - Client Demo Guide

## 🎯 What This System Does

This is an **intelligent workflow automation platform** that:

1. **Analyzes user requests** using AI (CrewAI) to decide the best tool
2. **Builds workflows** automatically in n8n or ComfyUI
3. **Manages dashboards** with Appsmith
4. **Stores data** in Directus (headless CMS)
5. **Offloads heavy tasks** to Google Colab (saves local GPU)

---

## 🚀 Quick Demo Steps

### **Step 1: Start the System**
```powershell
cd e:\Python\new\ai-workflow-agent
docker-compose up -d
```

Wait 2-3 minutes for all services to initialize.

### **Step 2: Access the UIs**

| Service | URL | Purpose |
|---------|-----|---------|
| **Appsmith** | http://localhost | Visual dashboard builder |
| **Directus** | http://localhost:8055 | Data management & API |
| **AI Agent API** | http://localhost:8000/docs | Backend API documentation |
| **n8n** | http://localhost:5678 | Workflow automation |
| **Portainer** | http://localhost:9000 | Docker management |

### **Step 3: Login Credentials**

**Directus:**
- Email: `admin@example.com`
- Password: `directus2026`

**Appsmith:**
- Create your account on first visit

---

## 📊 Demo Scenarios

### **Demo 1: AI Decision Making**
Show the AI deciding which tool to use.

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Create a workflow to process customer emails and send automated responses"}'
```

**Expected:** AI decides → n8n (automation workflow)

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Generate product images using AI for my e-commerce store"}'
```

**Expected:** AI decides → ComfyUI (generative AI)

---

### **Demo 2: Colab Offloading**
Show automatic GPU task offloading.

```bash
curl -X POST http://localhost:8000/colab/offload \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Train a neural network on image classification",
    "gpu_required": true,
    "max_execution_time": 3600
  }'
```

**Expected Response:**
```json
{
  "decided_offload": true,
  "decision_reason": "Task requires GPU - offloading to Colab",
  "tunnel_url": "https://xxx.ngrok.io"
}
```

---

### **Demo 3: Dashboard Creation**
Create an Appsmith dashboard from template.

```bash
curl -X POST http://localhost:8000/dashboard/create \
  -H "Content-Type: application/json" \
  -d '{"template": "workflow_status"}'
```

Available templates:
- `workflow_status` - Monitor running workflows
- `container_logs` - View Docker container logs
- `agent_decisions` - Track AI decision history

---

### **Demo 4: Web Page Analysis (DOM → JSON → Widgets)**
Analyze any webpage and convert to Appsmith widgets.

```bash
curl -X POST http://localhost:8000/dashboard/analyze-page \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "convert_to_appsmith": true
  }'
```

**Expected Response:**
```json
{
  "analysis": {
    "element_count": 45,
    "layout_type": "single_column"
  },
  "appsmith_widgets": [
    {"type": "text", "config": {...}},
    {"type": "table", "config": {...}}
  ]
}
```

---

### **Demo 5: Data Storage in Directus**
Store and retrieve data.

```bash
# Store workflow metadata
curl -X POST "http://localhost:8000/directus/store?collection=workflow_metadata" \
  -H "Content-Type: application/json" \
  -d '{"workflow_name": "Email Automation", "status": "running"}'

# Search data
curl "http://localhost:8000/directus/search/workflow_metadata/Email"
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        USER                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 APPSMITH (Frontend UI)                       │
│              http://localhost                                │
│    - Visual dashboards                                       │
│    - 3 built-in templates                                    │
│    - Custom widget creation                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               AI AGENT API (FastAPI)                         │
│              http://localhost:8000                           │
│    - CrewAI decision engine                                  │
│    - GitHub repository search                                │
│    - Colab offloading logic                                  │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│    DIRECTUS     │ │      n8n        │ │    ComfyUI      │
│  (Data/Auth)    │ │  (Automation)   │ │ (Generative AI) │
│  Port: 8055     │ │  Port: 5678     │ │  Port: 8188     │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                   │                   │
          └───────────────────┴───────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    POSTGRESQL                                │
│                    Port: 5432                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              GOOGLE COLAB (Remote GPU)                       │
│    - ColabCode for notebook execution                        │
│    - pyngrok for secure tunneling                            │
│    - Automatic fallback to local                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 Feature Checklist (What's Implemented)

### ✅ Milestone 1: Core AI Agent
- [x] CrewAI decision engine
- [x] Project type classification (n8n / ComfyUI / Hybrid / External)
- [x] GitHub repository search
- [x] Workflow template generation
- [x] Docker helper for container management

### ✅ Milestone 2: Colab Offloading Layer
- [x] ColabCode notebook generation
- [x] pyngrok tunnel management
- [x] Automatic offload decision (based on GPU need, task duration)
- [x] Fallback to local execution
- [x] Task complexity analysis

### ✅ Milestone 3: Dashboard Layer
- [x] Appsmith client with templates
- [x] Playwright DOM analyzer
- [x] Layout converter (DOM → Appsmith widgets)
- [x] Directus integration (auth, users, storage)
- [x] 5 pre-configured collections

---

## 🔧 API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Detailed service health |
| GET | `/milestones` | Feature overview |
| POST | `/analyze` | AI analyzes user query |
| POST | `/build` | Generate complete workflow |
| POST | `/github/search` | Search GitHub repos |

### Colab Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/colab/offload` | Offload task to Colab |
| GET | `/colab/status/{id}` | Check execution status |
| POST | `/colab/tunnel/create` | Create ngrok tunnel |

### Dashboard Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/dashboard/create` | Create from template |
| GET | `/dashboard/list` | List all dashboards |
| POST | `/dashboard/analyze-page` | Analyze web page |

### Directus Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/directus/auth` | Login to Directus |
| GET | `/directus/collections` | List collections |
| POST | `/directus/store` | Store data |
| GET | `/directus/search/{collection}/{query}` | Search data |

---

## 🧪 Running Tests

```powershell
cd e:\Python\new\ai-workflow-agent
pytest -v
```

**Expected:** 92 tests passing
- M1 Core Agent: 8 tests
- M2 Colab Layer: 32 tests
- M3 Dashboard Layer: 52 tests

---

## 📞 Support

For questions about:
- **Appsmith dashboards**: See Appsmith docs
- **Directus API**: See Directus docs
- **n8n workflows**: See n8n docs
- **This project**: Check SETUP_GUIDE.md

---

## 🚀 Next Steps (Future Enhancements)

1. **SSO Integration** - Connect Appsmith auth to Directus
2. **Embedded Browser** - Add iframe-based web navigation
3. **More Templates** - Additional dashboard templates
4. **Webhook Triggers** - Real-time workflow triggers
5. **Multi-tenant Support** - Organization/team management
