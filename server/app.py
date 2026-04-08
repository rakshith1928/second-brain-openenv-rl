import os
from functools import partial
from fastapi import FastAPI
from openenv.core.env_server import create_web_interface_app
from server.second_brain_env_environment import SecondBrainEnvironment
from models import SecondBrainAction, SecondBrainObservation

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VALID_TASKS = ["note_categorization", "memory_retrieval", "knowledge_synthesis"]

_task = os.getenv("TASK_NAME", "note_categorization")

def create_task_env():
    return SecondBrainEnvironment(task_name=_task)

app = create_web_interface_app(
    env=create_task_env,
    action_cls=SecondBrainAction,
    observation_cls=SecondBrainObservation,
    env_name=_task,
)

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"name": "note_categorization", "difficulty": "easy",   "max_steps": 12},
            {"name": "memory_retrieval",    "difficulty": "medium",  "max_steps": 15},
            {"name": "knowledge_synthesis", "difficulty": "hard",    "max_steps": 20},
        ]
    }

@app.get("/health")
def health():
    return {"status": "ok", "environment": "second_brain_env", "task": _task}

def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()