FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy everything from root into /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    openenv-core \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    websockets \
    openai

# Set PYTHONPATH so all imports resolve correctly
# /app        → models.py, client.py, __init__.py
# /app/server → app.py, data.py, second_brain_env_environment.py
ENV PYTHONPATH="/app:/app/server"
ENV TASK_NAME=note_categorization
ENV PORT=8000

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["sh", "-c", "cd /app && uvicorn server.app:app --host 0.0.0.0 --port 8000"]