@echo off
chcp 65001 >nul
setlocal

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

if not exist "logs" mkdir logs

echo ==========================================
echo   Intelligent Matching Engine
echo ==========================================
echo.

if "%1"=="api" goto :start_api
if "%1"=="webui" goto :start_webui
if "%1"=="all" goto :start_all
if "%1"=="docker-build" goto :docker_build
if "%1"=="docker-up" goto :docker_up
if "%1"=="docker-down" goto :docker_down
if "%1"=="docker-logs" goto :docker_logs

goto :help

:start_api
echo Starting API server on port 5310...
uvicorn api.main:app --host 0.0.0.0 --port 5310
rem  --reload
goto :end

:start_webui
echo Starting WebUI on port 5320...
streamlit run webui/main.py --server.port 5320 --server.address 0.0.0.0
goto :end

:start_all
echo Starting all services...
echo API: http://localhost:5310
echo WebUI: http://localhost:5320
echo.
start "IME API" cmd /k "uvicorn api.main:app --host 0.0.0.0 --port 5310 --reload"
timeout /t 3 /nobreak >nul
streamlit run webui/main.py --server.port 5320 --server.address 0.0.0.0
goto :end

:docker_build
echo Building Docker images...
docker-compose build
goto :end

:docker_up
echo Starting Docker containers...
docker-compose up -d
goto :end

:docker_down
echo Stopping Docker containers...
docker-compose down
goto :end

:docker_logs
echo Showing Docker logs...
docker-compose logs -f
goto :end

:help
echo Usage: start.cmd {api^|webui^|all^|docker-build^|docker-up^|docker-down^|docker-logs}
echo.
echo Commands:
echo   api           Start API server only
echo   webui         Start WebUI only
echo   all           Start both API and WebUI
echo   docker-build  Build Docker images
echo   docker-up     Start containers in background
echo   docker-down   Stop containers
echo   docker-logs   View container logs
goto :end

:end
endlocal
