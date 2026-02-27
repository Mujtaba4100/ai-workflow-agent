# AI Workflow Agent - Complete Setup Guide

## 🎯 Overview

This guide will help you set up and access all components of the AI Workflow Agent system, including the **visual dashboards** (Appsmith) and **data management** (Directus) UIs.

---

## 📋 System Architecture

### **Milestone 1: Core AI Agent**
- **CrewAI** decision-making engine
- **GitHub Search** for repository discovery
- **Workflow builders** for n8n and ComfyUI

### **Milestone 2: Colab Offloading Layer**
- **ColabCode** for notebook generation
- **pyngrok** for tunnel management
- **Automatic offload decision** based on task complexity
- **Fallback handler** for graceful degradation

### **Milestone 3: Dashboard Layer**
- **Appsmith** visual dashboard builder
- **Playwright** web page analyzer
- **Directus** headless CMS for data storage

---

## 🚀 Quick Start

### **Step 1: Start All Services**

```powershell
cd e:\Python\new\ai-workflow-agent
docker-compose up -d
```

This will start:
- ✅ Ollama (AI models) - Port 11434
- ✅ n8n (Workflow automation) - Port 5678
- ✅ ComfyUI (Generative workflows) - Port 8188
- ✅ PostgreSQL (Database) - Port 5432
- ✅ MongoDB (Appsmith DB) - Port 27017
- ✅ **Directus** (Data management UI) - **Port 8055** ⭐
- ✅ **Appsmith** (Dashboard UI) - **Port 80** ⭐
- ✅ Portainer (Docker management) - Port 9000
- ✅ AI Agent (FastAPI) - Port 8000

### **Step 2: Wait for Services to Initialize**

Monitor logs to ensure all services are ready:

```powershell
docker-compose logs -f appsmith directus
```

- **Appsmith** takes ~2-3 minutes for first startup
- **Directus** takes ~30 seconds

Press `Ctrl+C` to stop following logs.

### **Step 3: Verify Services Are Running**

```powershell
docker-compose ps
```

All services should show status: `Up`

---

## 🌐 Access the UIs

### **1️⃣ Appsmith - Visual Dashboard Builder**

**URL:** http://localhost

**First-Time Setup:**
1. Open http://localhost in your browser
2. Click **"Sign Up"** to create your admin account
3. Enter:
   - **Name:** Your name
   - **Email:** admin@localhost (or your email)
   - **Password:** Choose a secure password
4. Click **"Create Account"**
5. You'll be taken to the dashboard home page

**What You Can Do:**
- Create custom dashboards from templates
- Monitor workflow executions in real-time
- View Docker container logs
- Track AI agent decisions
- Build interactive data visualizations

**Pre-Built Templates Available:**
- `workflow_status` - Real-time workflow monitoring
- `container_logs` - Docker container log viewer
- `agent_decisions` - AI decision tracking dashboard

### **2️⃣ Directus - Data Management UI**

**URL:** http://localhost:8055

**Login Credentials:**
- **Email:** `admin@example.com`
- **Password:** `directus2026`

**First-Time Setup:**
1. Open http://localhost:8055
2. Log in with the credentials above
3. You'll see the Directus admin panel

**Pre-Configured Collections:**
- `dashboard_layouts` - Saved Appsmith dashboard configurations
- `workflow_metadata` - n8n/ComfyUI workflow data
- `project_data` - Project and repository information
- `user_settings` - User preferences and configurations
- `analysis_results` - Playwright page analysis results

**What You Can Do:**
- Browse and search all stored data
- Create custom data collections
- Manage users and permissions
- Export data to CSV/JSON
- Build custom APIs with GraphQL/REST

### **3️⃣ AI Agent API - Backend Service**

**URL:** http://localhost:8000

**API Documentation:** http://localhost:8000/docs

**Key Endpoints:**

**M1: Core Decision Making**
- `POST /analyze` - Analyze user queries
- `POST /build` - Generate workflows
- `POST /github/search` - Search repositories

**M2: Colab Offloading**
- `POST /colab/offload` - Offload tasks to Colab
- `GET /colab/status/{id}` - Check execution status
- `POST /colab/tunnel/create` - Create pyngrok tunnel

**M3: Dashboard Layer**
- `POST /dashboard/create` - Create Appsmith dashboard
- `POST /dashboard/analyze-page` - Analyze web page
- `POST /directus/auth` - Authenticate with Directus
- `POST /directus/store` - Store data in collections

### **4️⃣ Other Services**

- **n8n:** http://localhost:5678 (Workflow automation)
- **ComfyUI:** http://localhost:8188 (Generative workflows)
- **Portainer:** http://localhost:9000 (Docker management)

---

## 📊 Testing the Integration

### **Test 1: Create Your First Dashboard**

Use the API to create a dashboard from a template:

```powershell
curl -X POST http://localhost:8000/dashboard/create `
  -H "Content-Type: application/json" `
  -d '{"template": "workflow_status"}'
```

**Expected Response:**
```json
{
  "success": true,
  "message": "Dashboard created from template: workflow_status",
  "dashboard_id": "abc123",
  "dashboard_url": "http://localhost/app/abc123"
}
```

Now open the `dashboard_url` in your browser to see your dashboard!

### **Test 2: Authenticate with Directus**

```powershell
curl -X POST http://localhost:8000/directus/auth `
  -H "Content-Type: application/json" `
  -d '{"email": "admin@example.com", "password": "directus2026"}'
```

**Expected Response:**
```json
{
  "success": true,
  "message": "Authentication successful",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_at": "2026-01-15T12:00:00"
}
```

### **Test 3: Store Data in Directus**

```powershell
curl -X POST "http://localhost:8000/directus/store?collection=workflow_metadata" `
  -H "Content-Type: application/json" `
  -d '{"workflow_name": "test_workflow", "status": "running", "created_at": "2026-01-14T10:00:00"}'
```

Then verify in the Directus UI:
1. Go to http://localhost:8055
2. Click **"workflow_metadata"** collection
3. You should see your stored item!

### **Test 4: Offload a Task to Colab**

```powershell
curl -X POST http://localhost:8000/colab/offload `
  -H "Content-Type: application/json" `
  -d '{
    "task_description": "Train a small ML model on MNIST dataset",
    "code": "import tensorflow as tf\nprint(tf.__version__)",
    "requirements": ["tensorflow", "numpy"],
    "gpu_required": true,
    "max_execution_time": 1800
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "message": "Task offloaded to Colab successfully",
  "execution_id": "exec_12345",
  "tunnel_url": "https://abc123.ngrok.io",
  "notebook_url": "https://colab.research.google.com/drive/...",
  "decided_offload": true,
  "decision_reason": "Task requires GPU and will run for 1800s (benefits from Colab)"
}
```

### **Test 5: Analyze a Web Page**

```powershell
curl -X POST http://localhost:8000/dashboard/analyze-page `
  -H "Content-Type: application/json" `
  -d '{
    "url": "https://example.com",
    "convert_to_appsmith": true
  }'
```

This will:
1. Use Playwright to analyze the page structure
2. Extract all DOM elements
3. Convert them to Appsmith widget configurations
4. Return the analysis and widget data

---

## 🔧 Common Tasks

### **View Logs for a Specific Service**

```powershell
# View AI Agent logs
docker-compose logs -f agent

# View Appsmith logs
docker-compose logs -f appsmith

# View Directus logs
docker-compose logs -f directus
```

### **Restart a Service**

```powershell
# Restart Appsmith
docker-compose restart appsmith

# Restart Directus
docker-compose restart directus
```

### **Stop All Services**

```powershell
docker-compose down
```

### **Stop and Remove All Data (Fresh Start)**

```powershell
docker-compose down -v
```

⚠️ **Warning:** This will delete all data in Appsmith, Directus, and PostgreSQL!

---

## 📚 Using the Dashboards

### **Workflow Status Dashboard (Template)**

**URL:** Create via API or manually in Appsmith

**Features:**
- Live workflow execution status
- Success/failure rates
- Execution timeline
- Recent activity log

**How to Create:**
```powershell
curl -X POST http://localhost:8000/dashboard/create `
  -H "Content-Type: application/json" `
  -d '{"template": "workflow_status"}'
```

### **Container Logs Dashboard (Template)**

**Features:**
- Real-time Docker container logs
- Log filtering by container
- Error highlighting
- Log export functionality

**How to Create:**
```powershell
curl -X POST http://localhost:8000/dashboard/create `
  -H "Content-Type: application/json" `
  -d '{"template": "container_logs"}'
```

### **Agent Decisions Dashboard (Template)**

**Features:**
- AI decision tracking
- Confidence scores over time
- Project type distribution
- Suggested tools analysis

**How to Create:**
```powershell
curl -X POST http://localhost:8000/dashboard/create `
  -H "Content-Type: application/json" `
  -d '{"template": "agent_decisions"}'
```

---

## 🧪 Running All Tests

Verify that all components are working correctly:

```powershell
cd e:\Python\new\ai-workflow-agent
pytest -v
```

**Expected Output:**
```
test_m1_core_agent.py ................ (8 passed)
test_m2_colab_layer.py ................................ (32 passed)
test_m3_dashboard.py ...................................................... (52 passed)

===================== 92 passed in 15.23s =====================
```

---

## 🎨 Customizing Dashboards

### **Option 1: Use Pre-Built Templates (Easiest)**

The system includes 3 ready-to-use templates:
1. `workflow_status` - Monitor n8n/ComfyUI workflows
2. `container_logs` - View Docker logs
3. `agent_decisions` - Track AI decisions

### **Option 2: Create Custom Dashboards in Appsmith UI**

1. Go to http://localhost
2. Click **"Create New"** → **"Application"**
3. Choose **"Start From Scratch"**
4. Drag and drop widgets:
   - **Table** - Display data from Directus
   - **Chart** - Visualize metrics
   - **Button** - Trigger actions
   - **Text** - Add labels
5. Connect to data sources:
   - Use Directus API: http://directus:8055
   - Use AI Agent API: http://agent:8000

### **Option 3: Use Playwright to Clone Existing Pages**

Analyze any web page and convert it to Appsmith:

```powershell
curl -X POST http://localhost:8000/dashboard/analyze-page `
  -H "Content-Type: application/json" `
  -d '{
    "url": "https://your-favorite-dashboard.com",
    "convert_to_appsmith": true
  }'
```

The response will include `appsmith_widgets` that you can import!

---

## 🔐 Security Notes

### **Default Passwords (Change These!)**

- **Directus Admin:** `directus2026`
- **PostgreSQL:** `agent_secret_2026`
- **Appsmith:** Set during first login

### **Production Deployment**

Before deploying to production:

1. **Change all default passwords** in `docker-compose.yml`
2. **Enable HTTPS** for Appsmith (ports 443 configured)
3. **Restrict CORS** in Directus (currently set to `*`)
4. **Set strong secrets** for encryption keys
5. **Configure firewall rules** to limit access

---

## 🆘 Troubleshooting

### **Problem: Appsmith won't load**

**Solution:**
```powershell
# Check if MongoDB is running
docker-compose ps mongo

# Restart Appsmith
docker-compose restart appsmith mongo

# Wait 2-3 minutes and try again
```

### **Problem: Directus shows database error**

**Solution:**
```powershell
# Check PostgreSQL connection
docker-compose logs postgres

# Restart Directus and PostgreSQL
docker-compose restart directus postgres
```

### **Problem: "Connection refused" on API calls**

**Solution:**
```powershell
# Check if agent service is running
docker-compose ps agent

# View agent logs for errors
docker-compose logs agent

# Restart agent
docker-compose restart agent
```

### **Problem: Can't access http://localhost**

**Solution:**
```powershell
# Check if port 80 is already in use
netstat -an | Select-String ":80"

# If port 80 is taken, edit docker-compose.yml:
# Change "80:80" to "8080:80" for Appsmith
# Then access via http://localhost:8080
```

---

## 📖 API Examples

### **Example 1: Full Workflow - Create Dashboard with Live Data**

```powershell
# Step 1: Authenticate with Directus
$auth = curl -X POST http://localhost:8000/directus/auth `
  -H "Content-Type: application/json" `
  -d '{"email": "admin@localhost", "password": "directus2026"}' | ConvertFrom-Json

Write-Host "Token: $($auth.token)"

# Step 2: Store some sample workflow data
curl -X POST "http://localhost:8000/directus/store?collection=workflow_metadata" `
  -H "Content-Type: application/json" `
  -d '{"workflow_name": "Image Generation", "status": "running", "progress": 45}'

curl -X POST "http://localhost:8000/directus/store?collection=workflow_metadata" `
  -H "Content-Type: application/json" `
  -d '{"workflow_name": "Data Processing", "status": "completed", "progress": 100}'

# Step 3: Create dashboard to visualize this data
$dashboard = curl -X POST http://localhost:8000/dashboard/create `
  -H "Content-Type: application/json" `
  -d '{"template": "workflow_status"}' | ConvertFrom-Json

Write-Host "Dashboard URL: $($dashboard.dashboard_url)"

# Step 4: Open in browser (Windows)
Start-Process $dashboard.dashboard_url
```

### **Example 2: Analyze a GitHub Repo and Store Results**

```powershell
# Step 1: Search for repositories
$repos = curl -X POST http://localhost:8000/github/search `
  -H "Content-Type: application/json" `
  -d '{"keywords": "fastapi machine learning", "max_results": 3}' | ConvertFrom-Json

# Step 2: Store the recommendation in Directus
curl -X POST "http://localhost:8000/directus/store?collection=analysis_results" `
  -H "Content-Type: application/json" `
  -d "{\"type\": \"github_search\", \"query\": \"fastapi machine learning\", \"recommendation\": \"$($repos.recommendation)\"}"

Write-Host "Found $($repos.repositories.Count) repositories"
Write-Host "Recommendation: $($repos.recommendation)"
```

---

## 🎯 Next Steps

1. **Explore the Appsmith UI** - Create custom dashboards
2. **Browse Directus Collections** - See how data is organized
3. **Test API Endpoints** - Use http://localhost:8000/docs
4. **Run the Test Suite** - Verify all 92 tests pass
5. **Check /milestones Endpoint** - See feature overview

```powershell
curl http://localhost:8000/milestones
```

---

## 📝 Summary

You now have:
- ✅ **M1:** AI decision-making with CrewAI and GitHub search
- ✅ **M2:** Colab offloading with ColabCode, pyngrok, and fallback
- ✅ **M3:** Visual dashboards (Appsmith) and data storage (Directus)

**All services running on:**
- Appsmith UI: http://localhost
- Directus UI: http://localhost:8055
- AI Agent API: http://localhost:8000
- API Docs: http://localhost:8000/docs

**Tests:** 92/92 passing ✅

**Ready to build workflows!** 🚀

---

## 📞 Quick Reference

| Service | URL | Credentials |
|---------|-----|-------------|
| **Appsmith** | http://localhost | Create on first login |
| **Directus** | http://localhost:8055 | admin@example.com / directus2026 |
| **AI Agent API** | http://localhost:8000 | No auth required |
| **API Docs** | http://localhost:8000/docs | Interactive documentation |
| **n8n** | http://localhost:5678 | No auth required |
| **ComfyUI** | http://localhost:8188 | No auth required |
| **Portainer** | http://localhost:9000 | Create on first login |

---

**Happy Building! 🎉**
