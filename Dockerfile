FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy everything from root into /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH so all imports resolve correctly
ENV PYTHONPATH="/app:/app/server"
ENV TASK_NAME=note_categorization
ENV PORT=8000

EXPOSE 8000 8001 8002 8003

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Make start script executable & run
RUN chmod +x /app/start.sh
CMD ["python", "inference.py"]