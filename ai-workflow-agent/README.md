# AI Workflow Agent

**Intelligent Workflow Assistant Platform** for automated n8n, ComfyUI, and hybrid workflow generation.

## 🎯 Overview

This system provides:
- **Natural Language Interface** - Describe what you want, the agent decides how to build it
- **Multi-Platform Support** - Generates workflows for n8n (automation) and ComfyUI (AI generation)
- **GitHub Integration** - Search, clone, and deploy external projects automatically
- **Docker Automation** - Build, run, and troubleshoot containers with LLM-assisted error fixing
- **Colab Offloading** - Heavy tasks can be sent to Google Colab to save local resources

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER REQUEST                            │
│              "Create image generation workflow"             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   DECISION AGENT                            │
│            (CrewAI + Ollama/Qwen2.5)                        │
│                                                             │
│   Analyzes request → Decides: n8n / ComfyUI / Hybrid / Repo │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┼───────────┬────────────┐
          ▼           ▼           ▼            ▼
     ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
     │   n8n   │ │ ComfyUI │ │ Hybrid  │ │  Repo   │
     │ Builder │ │ Builder │ │  Mode   │ │ Helper  │
     └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
          │           │           │            │
          ▼           ▼           ▼            ▼
     ┌─────────────────────────────────────────────┐
     │              OUTPUT / DEPLOYMENT            │
     │  - JSON workflows                           │
     │  - API deployment                           │
     │  - Docker containers                        │
     └─────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU (optional, for local AI)
- Git

### Installation

1. **Clone the repository**
```bash
git clone <this-repo>
cd ai-workflow-agent
```

2. **Configure environment**
```bash
cp agent/.env.example agent/.env
# Edit .env with your tokens (GitHub, Ngrok - optional)
```

3. **Start all services**
```bash
docker compose up -d
```

4. **Wait for services to initialize** (~2-5 minutes first time)

5. **Access the services**
- Agent API: http://localhost:8000
- n8n: http://localhost:5678
- ComfyUI: http://localhost:8188
- Portainer: http://localhost:9000

### First Request

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Create a workflow that generates images from text prompts"}'
```

## 📚 API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed service status |
| `/analyze` | POST | Analyze query, decide project type |
| `/build` | POST | Full pipeline: analyze → generate → deploy |
| `/github/search` | POST | Search GitHub repositories |
| `/docker/build` | POST | Clone repo and build Docker |
| `/n8n/deploy` | POST | Deploy workflow to n8n |
| `/comfyui/execute` | POST | Execute ComfyUI workflow |

### Example: Build Workflow

```bash
curl -X POST http://localhost:8000/build \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Create an automation that sends Slack notifications when a webhook is received"
  }'
```

Response:
```json
{
  "success": true,
  "project_type": "n8n",
  "workflow": { ... },
  "message": "n8n workflow generated successfully"
}
```

### Example: Search GitHub

```bash
curl -X POST http://localhost:8000/github/search \
  -H "Content-Type: application/json" \
  -d '{
    "keywords": "stable diffusion webui",
    "max_results": 3
  }'
```

## 🛠️ Services

| Service | Port | Purpose |
|---------|------|---------|
| Agent API | 8000 | Core decision agent |
| n8n | 5678 | Workflow automation |
| ComfyUI | 8188 | AI image generation |
| Ollama | 11434 | Local LLM inference |
| PostgreSQL | 5432 | Database |
| Portainer | 9000 | Docker management |

## 📁 Project Structure

```
ai-workflow-agent/
├── docker-compose.yml       # All services
├── agent/
│   ├── Dockerfile           # Agent container
│   ├── requirements.txt     # Python dependencies
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── decision_agent.py    # Core decision logic
│   ├── tools/
│   │   ├── github_search.py    # GitHub search tool
│   │   ├── n8n_builder.py      # n8n workflow generator
│   │   ├── comfyui_builder.py  # ComfyUI workflow generator
│   │   └── docker_helper.py    # Docker automation
│   └── colab/
│       └── __init__.py      # Colab integration
├── workflows/               # Generated workflows
├── projects/                # Cloned repositories
└── README.md
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OLLAMA_HOST` | Yes | Ollama server URL |
| `OLLAMA_MODEL` | Yes | Model to use (default: qwen2.5:7b) |
| `N8N_HOST` | Yes | n8n server URL |
| `COMFYUI_HOST` | Yes | ComfyUI server URL |
| `GITHUB_TOKEN` | No | GitHub API token (increases rate limit) |
| `NGROK_AUTH_TOKEN` | No | Ngrok token (for Colab tunneling) |

### Ollama Model

The default model is `qwen2.5:7b`. To pull it manually:

```bash
docker exec -it ollama ollama pull qwen2.5:7b
```

## 🎯 Development Phases

### ✅ Phase 0 - Setup & PoC (Current)
- [x] Docker Compose with all services
- [x] Decision agent core
- [x] GitHub search tool
- [x] n8n workflow generator
- [x] ComfyUI workflow generator
- [x] Docker helper
- [x] Basic Colab connector

### 🔄 Milestone 1 - Core AI Agent
- [ ] Advanced CrewAI integration
- [ ] Improved decision logic
- [ ] More workflow templates
- [ ] Error handling refinement

### ⏳ Milestone 2 - Full Colab Integration
- [ ] ColabCode integration
- [ ] Auto-offload logic
- [ ] Fallback system
- [ ] Result synchronization

### ⏳ Milestone 3 - Dashboard Layer
- [ ] Appsmith integration
- [ ] Directus backend
- [ ] Playwright DOM analyzer
- [ ] Visual workflow builder

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with:** CrewAI, Ollama (Qwen2.5), n8n, ComfyUI, FastAPI, Docker
