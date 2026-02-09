#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

echo "=========================================="
echo "  Intelligent Matching Engine"
echo "=========================================="
echo ""

case "$1" in
    api)
        echo "Starting API server on port 5310..."
        uvicorn api.main:app --host 0.0.0.0 --port 5310 --reload
        ;;
    webui)
        echo "Starting WebUI on port 5320..."
        streamlit run webui/main.py --server.port 5320 --server.address 0.0.0.0
        ;;
    all)
        echo "Starting all services..."
        echo "API: http://localhost:5310"
        echo "WebUI: http://localhost:5320"
        echo ""
        uvicorn api.main:app --host 0.0.0.0 --port 5310 &
        API_PID=$!
        streamlit run webui/main.py --server.port 5320 --server.address 0.0.0.0 &
        WEBUI_PID=$!
        wait $API_PID $WEBUI_PID
        ;;
    docker-build)
        echo "Building Docker images..."
        docker-compose build
        ;;
    docker-up)
        echo "Starting Docker containers..."
        docker-compose up -d
        ;;
    docker-down)
        echo "Stopping Docker containers..."
        docker-compose down
        ;;
    docker-logs)
        echo "Showing Docker logs..."
        docker-compose logs -f
        ;;
    *)
        echo "Usage: $0 {api|webui|all|docker-build|docker-up|docker-down|docker-logs}"
        echo ""
        echo "Commands:"
        echo "  api           Start API server only"
        echo "  webui         Start WebUI only"
        echo "  all           Start both API and WebUI"
        echo "  docker-build  Build Docker images"
        echo "  docker-up     Start containers in background"
        echo "  docker-down   Stop containers"
        echo "  docker-logs   View container logs"
        exit 1
        ;;
esac
