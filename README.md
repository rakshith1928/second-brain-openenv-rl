---
title: Second Brain Environment
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
---


# 🧠 Second Brain — OpenEnv Environment

> A Personal Knowledge Management environment where AI agents learn to capture, organize, and retrieve information — simulating the real-world challenge of managing a second brain.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow)](https://huggingface.co/spaces/RAc1928/second-brain-env)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)

---

## 🌍 Motivation

Millions of people struggle with **information overload** daily. We save articles, capture meeting notes, jot down ideas — but rarely retrieve them when we actually need them. 

This environment trains AI agents to act as a **Personal Knowledge Manager**: capturing raw notes, organizing them correctly, and retrieving the right information at the right time. This is a genuine real-world skill that benefits students, professionals, researchers, and anyone who manages information.

---

## 🎯 Tasks

### Task 1 — `note_categorization` 🟢 Easy
| Property | Value |
|---|---|
| **Goal** | Assign each of 10 raw notes to the correct category |
| **Categories** | `work`, `personal`, `reference`, `action_item` |
| **Max Steps** | 12 |
| **Reward** | +0.10 per correct, -0.02 per wrong, 0.00 per skip |
| **Score Formula** | `correct_count / 10` |

### Task 2 — `memory_retrieval` 🟡 Medium
| Property | Value |
|---|---|
| **Goal** | Find the most relevant note in a 30-note KB for each of 5 questions |
| **Max Steps** | 15 |
| **Reward** | +0.20 exact match, +0.05–0.16 near miss, -0.05 miss, -0.10 repeated failure |
| **Score Formula** | `avg(retrieval_scores) / 5` |

### Task 3 — `knowledge_synthesis` 🔴 Hard
| Property | Value |
|---|---|
| **Goal** | Synthesize multi-theme insights from a 50-note knowledge base |
| **Max Steps** | 20 |
| **Reward** | Up to +0.35 per question (theme coverage + note coverage + length) |
| **Score Formula** | `avg(synthesis_scores) / 3` |

---

## 📐 Action Space

```python
class SecondBrainAction(BaseModel):
    action_type: str   # "categorize" | "retrieve" | "synthesize" | "tag" | "skip"
    content: str       # category name / search query / synthesized answer
    note_id: Optional[str]       # target note ID (optional)
    tags: Optional[List[str]]    # tags to apply (optional)
```

**Action types:**
- `categorize` — assign a category label to the current note (Task 1)
- `retrieve`   — search the knowledge base with a query string (Task 2 & 3)
- `synthesize` — produce a final synthesized answer (Task 3)
- `skip`       — skip the current item (no reward)

---

## 👁️ Observation Space

```python
class SecondBrainObservation(BaseModel):
    current_note: Optional[Dict]        # note being processed (Task 1)
    query: str                          # question to answer (Task 2 & 3)
    retrieved_notes: Optional[List]     # top notes from KB search
    knowledge_base_size: int            # total notes in KB
    step_count: int                     # current step
    task_name: str                      # active task
    reward: float                       # reward from last action
    done: bool                          # episode finished?
    score: float                        # running score in [0.0, 1.0]
    feedback: str                       # human-readable feedback
    valid_categories: List[str]         # valid labels for Task 1
    remaining_items: int                # items left to process
```

---

## 🏗️ Setup & Usage

### Install
```bash
pip install openenv-core python-dotenv
pip install git+https://huggingface.co/spaces/RAc1928/second-brain-env
```

### Run locally with Docker
```bash
git clone https://huggingface.co/spaces/RAc1928/second-brain-env
cd second-brain-env

docker build -t second-brain-env .
docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8003:8003 second-brain-env
```

### Connect as a client
```python
import asyncio
from second_brain_env import SecondBrainEnv, SecondBrainAction

async def main():
    async with SecondBrainEnv(base_url="http://localhost:8000") as env:
        # Reset
        result = await env.reset()
        print(result.observation.feedback)

        # Task 1: categorize a note
        result = await env.step(SecondBrainAction(
            action_type="categorize",
            content="work"
        ))
        print(f"Reward: {result.observation.reward}")
        print(f"Score:  {result.observation.score}")

asyncio.run(main())
```

### Run baseline inference
```bash
export HF_TOKEN=hf_your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export SECOND_BRAIN_URL=http://localhost:8000

python inference.py
```

### Validate submission
```bash
openenv validate
```

---

## 📊 Baseline Scores

Scores produced by the `Qwen/Qwen2.5-72B-Instruct` model via HuggingFace router:

| Task | Difficulty | Baseline Score |
|---|---|---|
| `note_categorization` | 🟢 Easy | 0.70 |
| `memory_retrieval` | 🟡 Medium | 0.52 |
| `knowledge_synthesis` | 🔴 Hard | 0.38 |
| **Average** | | **0.53** |

---

## 🔁 Reward Design

Rewards are **dense** — every step produces a signal:

```
Task 1 (categorization):
  Correct category  → +0.10  (agent gets immediate confirmation)
  Wrong category    → -0.02  (small penalty, not catastrophic)
  Skip              →  0.00  (neutral — no progress)

Task 2 (retrieval):
  Exact match       → +0.20  (top-1 is the correct note)
  Near miss rank 2  → +0.16  (partial credit)
  Near miss rank 3  → +0.12
  Not in top 5      → -0.05  (try a different query)
  3 consecutive bad → -0.10  (penalty for looping)

Task 3 (synthesis):
  Per relevant note found   → +0.05
  Synthesis theme coverage  → up to +0.12
  Synthesis note coverage   → up to +0.11
  Answer length bonus       → up to +0.07
  Multi-theme connection    → +0.05 bonus
  Hallucinated facts        → -0.10 penalty
```

---

## 🐳 Docker

```bash
# Build
docker build -t second-brain-env .

# Run (starts all 3 task servers automatically)
docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8003:8003 second-brain-env

# Test
curl -X POST http://localhost:8000/reset
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "categorize", "content": "work"}'
```

---

## 📁 Project Structure

```
second_brain_env/
├── inference.py          ← baseline inference script (root level)
├── openenv.yaml          ← environment manifest
├── pyproject.toml        ← dependencies
├── README.md
├── __init__.py           ← exports Action, Observation, Env
├── models.py             ← Pydantic typed models
├── client.py             ← WebSocket client
└── server/
    ├── app.py            ← FastAPI server
    ├── second_brain_env_environment.py  ← step/reset/state logic
    ├── data.py           ← seed notes and knowledge base
    ├── requirements.txt
    └── Dockerfile
```

---

## 👥 Team

**Team TwinCoders**
- Rachana N 
- Rakshith N

Built for the Meta × PyTorch × HuggingFace OpenEnv Hackathon 2026.
