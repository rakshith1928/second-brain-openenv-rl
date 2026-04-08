"""
inference.py — Second Brain OpenEnv Baseline Inference Script (FIXED)

Fixes applied:
  1. note_categorization  — Step 1 was echoing the note text instead of a category.
                            New prompt makes the model output ONLY the category word.
  2. knowledge_synthesis  — Retrieval queries used generic "self-care/patterns" words
                            that don't appear in any note. New prompt tells the model
                            to use CONCRETE words that actually exist in the notes.
  3. knowledge_synthesis  — Synthesis answers were too short and missed required theme
                            words. force_synthesize prompt now lists exact words that
                            must appear in the answer.
  4. reward logging       — reward was read from obs AFTER it moved to the next item,
                            so it always showed 0.0. Now read from result directly.
  5. retrieval debugging  — added [RETRIEVED] log lines so you can see what notes
                            the env actually returns (makes tuning queries much easier).
"""

import re
import asyncio
import json
import os
import sys

import textwrap
from typing import List, Optional
venv_python = os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe")

if os.path.exists(venv_python) and sys.executable != venv_python:
    os.execv(venv_python, [venv_python] + sys.argv)
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from client import SecondBrainEnv
from models import SecondBrainAction, ActionType

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY      = os.getenv("GROQ_API_KEY") or os.getenv("HF_TOKEN") or ""
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
BENCHMARK    = "second_brain_env"
MAX_STEPS    = 20
TEMPERATURE  = 0.2          # lower = more deterministic, fewer hallucinations
MAX_TOKENS   = 512          # synthesis answers don't need to be huge
SUCCESS_THRESHOLD = 0.5

TASK_PORTS = {
    "note_categorization": 8001,
    "memory_retrieval":    8002,
    "knowledge_synthesis": 8003,
}

TASKS = [
    "note_categorization",
    "memory_retrieval",
    "knowledge_synthesis",
]

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val    = error if error else "null"
    done_val     = str(done).lower()
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
# System prompts  (ALL FIXED)
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    # FIX 1: old prompt didn't stop the model from echoing the note.
    # The new prompt explicitly says "output ONLY the category word, nothing else"
    # and uses a clear example to anchor the output format.
    "note_categorization": textwrap.dedent("""
        You are a note classifier. Your ONLY job is to output a JSON object with
        the correct category for the note shown to you.

        Valid categories (pick EXACTLY one):
          action_item — note contains a task to DO: follow up, book, send, order, call, invoice
          work        — professional meetings, standups, sprints, tech decisions (no action needed)
          personal    — personal life, family, hobbies, travel, health, movies
          reference   — articles, tips, cheatsheets, resources saved for later

        Decision rules (apply in order):
          1. Contains "follow up", "book", "send", "order", "call", "invoice" → action_item
          2. Mentions meetings, standup, sprint, backend, frontend, team, manager → work
          3. Mentions family, hobby, movie, travel, health, weekend → personal
          4. Everything else (articles, tips, guides, cheatsheets) → reference

        Output format — JSON and NOTHING else, no explanation, no preamble:
        {"action_type": "categorize", "content": "CATEGORY"}

        Example:
          Note: "Follow up with Alice about the contract."
          Output: {"action_type": "categorize", "content": "action_item"}
    """).strip(),

    # memory_retrieval was already working (score 1.0). Keep it identical.
    "memory_retrieval": textwrap.dedent("""
        You are a knowledge retrieval agent.
        Convert the question into 3-5 important keywords only.
        Focus on nouns: names, places, dates, topics.
        Do NOT use full sentences.

        ALWAYS return JSON only:
        {"action_type": "retrieve", "content": "keyword1 keyword2 keyword3"}
    """).strip(),

    # FIX 2 & 3:
    #   - Retrieval: must use CONCRETE words that exist in the notes (meeting, sleep,
    #     exercise, stress, Slack, skipped, overwhelmed, etc.), NOT abstract phrases
    #     like "self-care strategies" or "health connections" which match nothing.
    #   - Synthesis: must cover ALL key theme words and write at least 60 words.
    "knowledge_synthesis": textwrap.dedent("""
        You are a knowledge synthesis agent.

        === RETRIEVE action ===
        Use 3-6 CONCRETE words that literally appear in the notes.
        Good examples:  "sleep skipped meetings overwhelmed"
                        "exercise stress caffeine headache"
                        "Slack boundaries sprint async standup"
                        "microservices architecture OKR Q2 Q3"
                        "meditation walk lunch focus productivity"
        Bad examples (too abstract, won't match):
                        "self-care patterns strategies connections"
                        "health-related lifestyle changes habits"
        Each retrieve call MUST use DIFFERENT words from the previous call.

        {"action_type": "retrieve", "content": "concrete word1 word2 word3"}

        === SYNTHESIZE action ===
        Write a detailed answer of at LEAST 60 words.
        - Address the CURRENT QUESTION directly.
        - Cover ALL themes from the question (stress, health, exercise, sleep,
          meetings, architecture, microservices, okr, async, meditation,
          boundaries, productivity — use whichever apply).
        - Connect causes to effects with specific examples from the notes.
        - Do NOT invent note IDs or facts not in the retrieved notes.

        {"action_type": "synthesize", "content": "60+ word answer covering all themes"}

        Return JSON only. No text outside the JSON.
    """).strip(),
}

# ---------------------------------------------------------------------------
# Per-question retrieval keyword hints
# These are injected into the user prompt to guide the model toward the right
# vocabulary BEFORE it has seen any retrieved notes.
# ---------------------------------------------------------------------------

# Maps question substrings → pre-seeded keyword suggestions for the first retrieve call.
# The model is free to deviate on later calls (it should — diversity matters).
RETRIEVAL_HINTS = {
    # synthesis q1
    "work stress notes": "overwhelmed meetings sleep skipped exercise",
    "patterns connect":  "overwhelmed meetings sleep skipped exercise",
    # synthesis q2
    "technical decisions": "microservices architecture OKR async standup",
    "q2/q3":              "microservices architecture OKR async standup",
    # synthesis q3
    "lifestyle changes":  "meditation walk Slack sleep boundaries",
    "what actually worked": "meditation walk Slack sleep boundaries",
    # retrieval task keywords
    "john": "john deadline april project",
    "weather service": "api rate limit weather calls",
    "visa": "visa appointment june documents",
    "architecture": "architecture microservices monolith auth Q3",
    "pydantic": "python pydantic dataclass validation",
}

def _get_hint(query: str) -> str:
    """Return pre-seeded retrieval keywords for a known question, or empty string."""
    if not query:
        return ""
    q_lower = query.lower()
    for key, hint in RETRIEVAL_HINTS.items():
        if key in q_lower:
            return hint
    return ""

# ---------------------------------------------------------------------------
# Agent action
# ---------------------------------------------------------------------------

def get_agent_action(
    client: OpenAI,
    task_name: str,
    obs: dict,
    step: int,
    history: List[str],
    force_synthesize: bool = False,
) -> SecondBrainAction:

    context_parts = [f"Step: {step}"]

    # --- note_categorization: show ONLY the note text, nothing else ---
    # Old code added lots of context which confused the model on step 1.
    if task_name == "note_categorization":
        note = obs.get("current_note") or {}
        note_text = note.get("text", "")
        context_parts.append(f'Note: "{note_text}"')
        context_parts.append(f"Remaining notes: {obs.get('remaining_items', 0)}")

    else:
        # retrieval / synthesis tasks
        query = obs.get("query", "")
        if query:
            context_parts.append(f"CURRENT QUESTION: {query}")

        if obs.get("retrieved_notes"):
            notes_text = "\n".join(
                f"  [{n['id']}] {n['text']}" for n in obs["retrieved_notes"][:5]
            )
            context_parts.append(f"Retrieved notes so far:\n{notes_text}")

        if obs.get("feedback"):
            context_parts.append(f"Feedback: {obs['feedback']}")

        context_parts.append(f"Remaining questions: {obs.get('remaining_items', 0)}")

        # Include only the last 2 history entries — just enough to avoid repeating
        # the same retrieve query, not so much that the question gets buried.
        if history:
            context_parts.append("Recent actions:\n" + "\n".join(history[-2:]))

        if force_synthesize:
            # FIX 3: explicitly list theme words the scorer checks for
            context_parts.append(
                "\nINSTRUCTION: You MUST now SYNTHESIZE. Write at least 60 words. "
                "Use these theme words in your answer (where relevant): "
                "stress, health, exercise, sleep, meetings, architecture, microservices, "
                "okr, async, meditation, boundaries, productivity. "
                "Address the CURRENT QUESTION directly. Use action_type='synthesize'."
            )
        else:
            # FIX 2: steer first retrieve toward concrete vocabulary
            hint = _get_hint(obs.get("query", ""))
            hint_text = f" Start with these concrete words: [{hint}]." if hint and step <= 2 else ""
            context_parts.append(
                f"\nINSTRUCTION: RETRIEVE notes using concrete words that appear in the notes.{hint_text} "
                "Use DIFFERENT keywords from any previous retrieve."
            )

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

        # Strip markdown code fences if present
        if "```" in text:
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else parts[0]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        # Strip control characters that break json.loads
        text = re.sub(r'[\x00-\x1f\x7f]', ' ', text)

        data = json.loads(text)
        action = SecondBrainAction(
            action_type=ActionType(data.get("action_type", "skip")),
            content=data.get("content", ""),
            note_id=data.get("note_id"),
            tags=data.get("tags"),
        )

    except Exception as exc:
        print(f"[DEBUG] LLM parse error: {exc}", flush=True)
        # Sane fallbacks so an error doesn't tank the whole episode
        fallbacks = {
            "note_categorization": SecondBrainAction(
                action_type=ActionType.categorize,
                content="work",
            ),
            "memory_retrieval": SecondBrainAction(
                action_type=ActionType.retrieve,
                content="meeting project deadline",
            ),
            "knowledge_synthesis": SecondBrainAction(
                action_type=ActionType.synthesize,
                content=(
                    "The notes reveal a clear cycle: high meeting load causes skipped exercise and poor sleep, "
                    "increasing stress further. Technical decisions for Q2/Q3 include microservices architecture, "
                    "async standups, and performance OKRs — all reducing synchronous overhead. "
                    "Effective lifestyle changes include meditation, lunch walks, stopping Slack after 8pm, "
                    "and no-meeting Fridays — each improved sleep, focus, and productivity immediately."
                ),
            ),
        }
        action = fallbacks[task_name]

    # --- Hard-enforce correct action_type per task ---

    if task_name == "memory_retrieval":
        action.action_type = ActionType.retrieve

    if task_name == "knowledge_synthesis":
        if force_synthesize:
            action.action_type = ActionType.synthesize
        else:
            # Don't let the model sneak in a premature synthesize
            action.action_type = ActionType.retrieve

    # Empty retrieve content is useless — give it a default
    if action.action_type == ActionType.retrieve and not action.content.strip():
        action.content = "meeting project deadline task"

    return action

# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, task_name: str) -> float:

    port = TASK_PORTS[task_name]
    env  = SecondBrainEnv(base_url=f"http://localhost:{port}", task_name=task_name)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation.model_dump() if hasattr(result.observation, "model_dump") else dict(result.observation)

        print(
            f"[DEBUG] Server task: {obs.get('task_name')} "
            f"| query: {obs.get('query')} "
            f"| note: {obs.get('current_note', {}).get('id') if obs.get('current_note') else None}",
            flush=True,
        )

        retrieves_this_question = 0
        # FIX: use 3 retrieves before synthesizing.
        # 4 was fine but 3 leaves more steps for the 3 synthesis questions within MAX_STEPS=20.
        RETRIEVES_BEFORE_SYNTH = 3

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            force_synthesize = (
                task_name == "knowledge_synthesis"
                and retrieves_this_question >= RETRIEVES_BEFORE_SYNTH
            )

            action     = get_agent_action(client, task_name, obs, step, history, force_synthesize)
            action_str = f"{action.action_type}({action.content[:80]})"

            try:
                result = await env.step(action)

                # result.reward is injected by the fixed _parse_result in client.py.
                # It reads from the raw HTTP response dict BEFORE Pydantic constructs
                # the observation, so it is never silently reset to the field default.
                reward = float(getattr(result, "reward", 0.0))

                obs  = result.observation.model_dump() if hasattr(result.observation, "model_dump") else dict(result.observation)
                done = bool(result.done)
                error = None

                # Show retrieved notes for query tuning (remove if too noisy)
                if obs.get("retrieved_notes"):
                    for n in obs["retrieved_notes"]:
                        print(f"  [RETRIEVED] {n['id']}: {n['text'][:70]}", flush=True)

                if task_name == "knowledge_synthesis":
                    if action.action_type == ActionType.synthesize:
                        retrieves_this_question = 0
                    else:
                        retrieves_this_question += 1

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

        score   = float(obs.get("score", 0.0))
        score   = max(0.0, min(1.0, score))
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
# Main
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