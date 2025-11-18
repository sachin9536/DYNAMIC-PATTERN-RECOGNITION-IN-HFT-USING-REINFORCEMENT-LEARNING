#!/bin/bash

# Script to run the FastAPI server

set -e

echo "🚀 Starting Market Anomaly Detection API..."

# Check if we're in the right directory
if [ ! -f "src/api/app.py" ]; then
    echo "❌ Error: src/api/app.py not found. Please run from project root."
    exit 1
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Default configuration
HOST=${API_HOST:-"0.0.0.0"}
PORT=${API_PORT:-8000}
WORKERS=${API_WORKERS:-1}
RELOAD=${API_RELOAD:-false}
LOG_LEVEL=${LOG_LEVEL:-"info"}

echo "📋 Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Workers: $WORKERS"
echo "  Reload: $RELOAD"
echo "  Log Level: $LOG_LEVEL"

# Check if uvicorn is available
if ! command -v uvicorn &> /dev/null; then
    echo "❌ Error: uvicorn not found. Please install with: pip install uvicorn[standard]"
    exit 1
fi

# Start the server
echo "🌐 Starting server at http://$HOST:$PORT"
echo "📚 API documentation will be available at http://$HOST:$PORT/docs"
echo "🔍 Health check: http://$HOST:$PORT/health"
echo ""

if [ "$RELOAD" = "true" ]; then
    echo "🔄 Running in development mode with auto-reload..."
    uvicorn src.api.app:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --log-level "$LOG_LEVEL"
else
    echo "🏭 Running in production mode..."
    uvicorn src.api.app:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL"
fi