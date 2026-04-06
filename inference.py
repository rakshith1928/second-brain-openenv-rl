"""
inference.py — Second Brain OpenEnv Baseline Inference Script
=============================================================
Runs all 3 tasks against the Second Brain environment using an LLM agent.
Emits structured stdout logs in [START] / [STEP] / [END] format.

Environment variables:
  API_BASE_URL      LLM endpoint (default: HuggingFace router)
  MODEL_NAME        Model identifier
  HF_TOKEN          HuggingFace API key
  SECOND_BRAIN_URL  Running environment URL (default: localhost:8000)
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from client import SecondBrainEnv
from models import SecondBrainAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("SECOND_BRAIN_URL", "http://localhost:8000")
BENCHMARK    = "second_brain_env"
MAX_STEPS    = 20
TEMPERATURE  = 0.3
MAX_TOKENS   = 256
SUCCESS_THRESHOLD = 0.5

TASKS = [
    "note_categorization",
    "memory_retrieval",
    "knowledge_synthesis",
]

# ---------------------------------------------------------------------------
# Logging helpers — EXACT format required by judges
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # Sanitize action string — no newlines
    action_clean = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# System prompts per task
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "note_categorization": textwrap.dedent("""
        You are a personal knowledge manager. You will receive notes one at a time.
        For each note, classify it into exactly one category:
          - work        : professional tasks, meetings, projects
          - personal    : personal life, family, hobbies
          - reference   : articles, tips, resources to save
          - action_item : something that needs to be done

        Respond with a JSON object ONLY. Example:
        {"action_type": "categorize", "content": "work"}
    """).strip(),

    "memory_retrieval": textwrap.dedent("""
        You are a personal knowledge retrieval agent. You have access to a knowledge base.
        Given a question, write a concise search query to find the most relevant note.

        Respond with a JSON object ONLY. Example:
        {"action_type": "retrieve", "content": "project deadline John April"}
    """).strip(),

    "knowledge_synthesis": textwrap.dedent("""
        You are a personal knowledge synthesis agent. Given a complex question:
        1. First use "retrieve" actions to collect relevant notes (2-3 retrieves).
        2. Then use a "synthesize" action with a comprehensive answer covering key themes.

        For retrieve: {"action_type": "retrieve", "content": "search query here"}
        For synthesize: {"action_type": "synthesize", "content": "your full answer here"}

        Always respond with a JSON object ONLY. No explanation outside JSON.
    """).strip(),
}

# ---------------------------------------------------------------------------
# Agent decision function
# ---------------------------------------------------------------------------

def get_agent_action(
    client: OpenAI,
    task_name: str,
    obs: dict,
    step: int,
    history: List[str],
) -> SecondBrainAction:
    """Call the LLM to get the next action."""

    # Build context from observation
    context_parts = [f"Step: {step}"]

    if obs.get("current_note"):
        note = obs["current_note"]
        context_parts.append(f"Note to categorize: {note.get('text', '')}")

    if obs.get("query"):
        context_parts.append(f"Question: {obs['query']}")

    if obs.get("retrieved_notes"):
        notes_text = "\n".join(
            f"  [{n['id']}] {n['text']}" for n in obs["retrieved_notes"][:3]
        )
        context_parts.append(f"Retrieved notes:\n{notes_text}")

    if obs.get("feedback"):
        context_parts.append(f"Last feedback: {obs['feedback']}")

    context_parts.append(f"Remaining items: {obs.get('remaining_items', 0)}")
    context_parts.append(f"Current score: {obs.get('score', 0.0):.3f}")

    if history:
        context_parts.append("Recent actions:\n" + "\n".join(history[-3:]))

    user_prompt = "\n".join(context_parts)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task_name]},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Parse JSON response
        # Strip markdown fences if present
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        data = json.loads(text)
        return SecondBrainAction(
            action_type=data.get("action_type", "skip"),
            content=data.get("content", ""),
            note_id=data.get("note_id"),
            tags=data.get("tags"),
        )

    except Exception as exc:
        print(f"[DEBUG] LLM parse error: {exc}", flush=True)
        # Fallback actions per task
        fallbacks = {
            "note_categorization": SecondBrainAction(action_type="categorize", content="work"),
            "memory_retrieval":    SecondBrainAction(action_type="retrieve",   content="important note"),
            "knowledge_synthesis": SecondBrainAction(action_type="retrieve",   content="key information"),
        }
        return fallbacks[task_name]

# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, task_name: str) -> float:
    """Run a single task episode. Returns final score in [0, 1]."""

    env = SecondBrainEnv(base_url=ENV_URL, task_name=task_name)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation.model_dump() if hasattr(result.observation, "model_dump") else dict(result.observation)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_agent_action(client, task_name, obs, step, history)

            # Format action string for logging
            action_str = f"{action.action_type}({action.content[:60]})"

            try:
                result  = await env.step(action)
                obs     = result.observation.model_dump() if hasattr(result.observation, "model_dump") else dict(result.observation)
                reward  = float(obs.get("reward", 0.0))
                done    = bool(result.done)
                error   = None
            except Exception as e:
                reward = 0.0
                done   = True
                error  = str(e)[:80]

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action_str} → reward {reward:+.2f}")

            if done:
                break

        # Final score from observation
        score = float(obs.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

# ---------------------------------------------------------------------------
# Main — run all 3 tasks
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores = {}
    for task_name in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_name}", flush=True)
        print(f"{'='*60}", flush=True)
        score = await run_task(client, task_name)
        all_scores[task_name] = score

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("BASELINE SCORES SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for task, score in all_scores.items():
        status = "✓ PASS" if score >= SUCCESS_THRESHOLD else "✗ FAIL"
        print(f"  {status}  {task}: {score:.3f}", flush=True)
    overall = sum(all_scores.values()) / len(all_scores)
    print(f"\n  Overall average: {overall:.3f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())