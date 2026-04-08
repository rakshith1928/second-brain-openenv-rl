#!/bin/bash
# Starts 3 uvicorn workers — one per task — on ports 8001, 8002, 8003
# Also starts the main server on port 8000 (default task)

set -e

echo "[start.sh] Launching Second Brain environment servers..."

# Start task-specific servers in background
TASK_NAME=note_categorization  uvicorn server.app:app --host 0.0.0.0 --port 8001 &
TASK_NAME=memory_retrieval     uvicorn server.app:app --host 0.0.0.0 --port 8002 &
TASK_NAME=knowledge_synthesis  uvicorn server.app:app --host 0.0.0.0 --port 8003 &

# Main server on port 8000 (uses default TASK_NAME env var or note_categorization)
uvicorn server.app:app --host 0.0.0.0 --port 8000 &

echo "[start.sh] All servers started. Waiting..."
wait
