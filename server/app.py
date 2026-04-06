"""
FastAPI server for Second Brain OpenEnv environment.
Exposes reset(), step(), state() over HTTP + WebSocket.
"""
import os
from typing import Optional
from fastapi import FastAPI, Query
from openenv.core.env_server import create_app

from second_brain_env_environment import SecondBrainEnvironment
from models import SecondBrainAction

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VALID_TASKS = ["note_categorization", "memory_retrieval", "knowledge_synthesis"]


def make_app(task_name: str = "note_categorization") -> FastAPI:
    env = SecondBrainEnvironment(task_name=task_name)
    app = create_app(env, action_type=SecondBrainAction)
    return app


# Default app — task controlled via env var
_task = os.getenv("TASK_NAME", "note_categorization")
app = make_app(_task)


@app.get("/tasks")
def list_tasks():
    """List all available tasks."""
    return {
        "tasks": [
            {
                "name": "note_categorization",
                "difficulty": "easy",
                "description": "Categorize 10 notes into correct categories.",
                "max_steps": 12,
                "max_score": 1.0,
            },
            {
                "name": "memory_retrieval",
                "difficulty": "medium",
                "description": "Retrieve relevant notes to answer 5 questions.",
                "max_steps": 15,
                "max_score": 1.0,
            },
            {
                "name": "knowledge_synthesis",
                "difficulty": "hard",
                "description": "Synthesize answers from 50-note knowledge base.",
                "max_steps": 20,
                "max_score": 1.0,
            },
        ]
    }


@app.get("/health")
def health():
    return {"status": "ok", "environment": "second_brain_env"}
