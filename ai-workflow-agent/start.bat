@echo off
REM AI Workflow Agent - Quick Start Script (Windows)
REM This script sets up and starts all services

echo ============================================
echo    AI WORKFLOW AGENT - QUICK START
echo ============================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Docker is not installed or not in PATH
    echo Please install Docker Desktop from: https://docker.com
    pause
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Docker is not running
    echo Please start Docker Desktop and try again
    pause
    exit /b 1
)

echo [1/4] Docker is ready
echo.

REM Create .env if it doesn't exist
if not exist "agent\.env" (
    echo [2/4] Creating .env file...
    copy "agent\.env.example" "agent\.env"
    echo.
    echo IMPORTANT: Edit agent\.env with your tokens if needed
    echo.
) else (
    echo [2/4] .env file already exists
)

REM Pull the Ollama model first (optional but recommended)
echo [3/4] Starting services... (this may take a few minutes on first run)
echo.

docker compose up -d

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to start services
    echo Check docker-compose.yml for errors
    pause
    exit /b 1
)

echo.
echo [4/4] Waiting for services to initialize...
timeout /t 30 /nobreak >nul

echo.
echo ============================================
echo    SERVICES STARTED SUCCESSFULLY!
echo ============================================
echo.
echo Access points:
echo   - Agent API:  http://localhost:8000
echo   - n8n:        http://localhost:5678
echo   - ComfyUI:    http://localhost:8188
echo   - Portainer:  http://localhost:9000
echo.
echo To stop services: docker compose down
echo To view logs:     docker compose logs -f
echo.
echo Press any key to open Agent API in browser...
pause >nul
start http://localhost:8000
