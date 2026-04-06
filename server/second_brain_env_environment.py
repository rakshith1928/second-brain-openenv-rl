"""
Second Brain Environment — Core Logic
Implements reset(), step(), state() for all 3 tasks.
"""
import copy
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

# We import models relative to the server's working directory
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import SecondBrainAction, SecondBrainObservation, SecondBrainState, ActionType
from server.data import (
    TASK1_NOTES,
    TASK2_KNOWLEDGE_BASE,
    TASK2_QUESTIONS,
    TASK3_KNOWLEDGE_BASE,
    TASK3_QUESTIONS,
    VALID_CATEGORIES,
    keyword_overlap_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_STEPS = {
    "note_categorization": 12,
    "memory_retrieval": 15,
    "knowledge_synthesis": 20,
}

VALID_TASKS = list(MAX_STEPS.keys())


class SecondBrainEnvironment(Environment):
    """
    Second Brain — Personal Knowledge Management OpenEnv Environment.

    Three tasks:
      1. note_categorization  (Easy)   — categorize 10 notes correctly
      2. memory_retrieval     (Medium) — retrieve relevant notes for 5 questions
      3. knowledge_synthesis  (Hard)   — synthesize answers from 50-note KB
    """

    def __init__(self, task_name: str = "note_categorization"):
        if task_name not in VALID_TASKS:
            raise ValueError(f"task_name must be one of {VALID_TASKS}")
        self._task_name = task_name
        self._state = SecondBrainState(
            episode_id=str(uuid4()),
            task_name=task_name,
            step_count=0,
            score=0.0,
            done=False,
        )
        self._reset_internal()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_internal(self):
        """Reset all episode-level variables."""
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._rewards: List[float] = []

        if self._task_name == "note_categorization":
            self._notes_queue = copy.deepcopy(TASK1_NOTES)
            self._current_note_idx = 0
            self._correct_count = 0

        elif self._task_name == "memory_retrieval":
            self._kb = copy.deepcopy(TASK2_KNOWLEDGE_BASE)
            self._questions_queue = copy.deepcopy(TASK2_QUESTIONS)
            self._current_q_idx = 0
            self._retrieval_scores: List[float] = []
            self._consecutive_wrong = 0

        elif self._task_name == "knowledge_synthesis":
            self._kb = copy.deepcopy(TASK3_KNOWLEDGE_BASE)
            self._questions_queue = copy.deepcopy(TASK3_QUESTIONS)
            self._current_q_idx = 0
            self._synthesis_scores: List[float] = []
            self._collected_note_ids: List[str] = []

        # Update typed state
        self._state = SecondBrainState(
            episode_id=self._state.episode_id,
            step_count=0,
            task_name=self._task_name,
            score=0.0,
            correct_count=0,
            retrieval_scores=[],
            synthesis_scores=[],
            collected_note_ids=[],
            done=False,
        )

    def _current_score(self) -> float:
        """Compute normalized score in [0, 1]."""
        if self._task_name == "note_categorization":
            total = len(TASK1_NOTES)
            return round(self._correct_count / total, 4)

        elif self._task_name == "memory_retrieval":
            if not self._retrieval_scores:
                return 0.0
            return round(sum(self._retrieval_scores) / len(TASK2_QUESTIONS), 4)

        elif self._task_name == "knowledge_synthesis":
            if not self._synthesis_scores:
                return 0.0
            return round(sum(self._synthesis_scores) / len(TASK3_QUESTIONS), 4)

        return 0.0

    # ------------------------------------------------------------------
    # Task 1 — Note Categorization
    # ------------------------------------------------------------------

    def _step_categorization(self, action: SecondBrainAction) -> SecondBrainObservation:
        if self._current_note_idx >= len(self._notes_queue):
            self._done = True
            return self._build_obs(reward=0.0, feedback="All notes processed.", done=True)

        current_note = self._notes_queue[self._current_note_idx]
        correct = current_note["correct_category"]

        if action.action_type == ActionType.skip:
            reward = 0.0
            feedback = f"Skipped. Correct category was '{correct}'."
        elif action.action_type == ActionType.categorize:
            if action.content.strip().lower() == correct:
                reward = 0.10
                self._correct_count += 1
                feedback = f"✓ Correct! '{correct}' is right."
            else:
                reward = -0.02
                feedback = f"✗ Wrong. You said '{action.content}', correct is '{correct}'."
        else:
            reward = -0.01
            feedback = f"Invalid action '{action.action_type}' for this task. Use 'categorize' or 'skip'."

        self._current_note_idx += 1
        done = self._current_note_idx >= len(self._notes_queue)
        self._done = done

        remaining = len(self._notes_queue) - self._current_note_idx
        next_note = (
            self._notes_queue[self._current_note_idx]
            if not done else None
        )

        return self._build_obs(
            reward=reward,
            feedback=feedback,
            done=done,
            current_note=next_note,
            remaining=remaining,
        )

    # ------------------------------------------------------------------
    # Task 2 — Memory Retrieval
    # ------------------------------------------------------------------

    def _step_retrieval(self, action: SecondBrainAction) -> SecondBrainObservation:
        if self._current_q_idx >= len(self._questions_queue):
            self._done = True
            return self._build_obs(reward=0.0, feedback="All questions answered.", done=True)

        current_q = self._questions_queue[self._current_q_idx]

        if action.action_type == ActionType.retrieve:
            query = action.content
            # Score all KB notes against the query
            scored = []
            for note in self._kb:
                score = keyword_overlap_score(query, note["text"])
                # Bonus if query keywords appear in correct note
                if note["id"] == current_q["correct_note_id"]:
                    score = max(score, keyword_overlap_score(
                        " ".join(current_q["keywords"]), note["text"]
                    ))
                scored.append((score, note))

            scored.sort(key=lambda x: x[0], reverse=True)
            top_notes = [n for _, n in scored[:5]]

            # Check if correct note is in top results
            top_ids = [n["id"] for n in top_notes]
            correct_id = current_q["correct_note_id"]

            if top_ids and top_ids[0] == correct_id:
                reward = 0.20
                retrieval_score = 1.0
                self._consecutive_wrong = 0
                feedback = f"✓ Perfect retrieval! Found the exact right note first."
            elif correct_id in top_ids:
                rank = top_ids.index(correct_id) + 1
                reward = max(0.05, 0.20 - (rank - 1) * 0.04)
                retrieval_score = reward / 0.20
                self._consecutive_wrong = 0
                feedback = f"✓ Correct note found at rank {rank}."
            else:
                reward = -0.05
                retrieval_score = 0.0
                self._consecutive_wrong += 1
                if self._consecutive_wrong >= 3:
                    reward -= 0.05  # extra penalty for repeated failures
                feedback = f"✗ Correct note not in top 5. Try a different query."

            self._retrieval_scores.append(retrieval_score)
            self._current_q_idx += 1
            done = self._current_q_idx >= len(self._questions_queue)
            self._done = done

            next_q = (
                self._questions_queue[self._current_q_idx]["question"]
                if not done else None
            )
            remaining = len(self._questions_queue) - self._current_q_idx

            return self._build_obs(
                reward=reward,
                feedback=feedback,
                done=done,
                query=next_q,
                retrieved_notes=top_notes[:3],
                remaining=remaining,
            )

        elif action.action_type == ActionType.skip:
            self._retrieval_scores.append(0.0)
            self._current_q_idx += 1
            done = self._current_q_idx >= len(self._questions_queue)
            self._done = done
            feedback = "Skipped question. Score 0 recorded."
            next_q = (
                self._questions_queue[self._current_q_idx]["question"]
                if not done else None
            )
            return self._build_obs(
                reward=0.0, feedback=feedback, done=done,
                query=next_q, remaining=len(self._questions_queue) - self._current_q_idx
            )
        else:
            return self._build_obs(
                reward=-0.01,
                feedback=f"Use 'retrieve' action with a search query string.",
                done=False,
                query=current_q["question"],
                remaining=len(self._questions_queue) - self._current_q_idx,
            )

    # ------------------------------------------------------------------
    # Task 3 — Knowledge Synthesis
    # ------------------------------------------------------------------

    def _step_synthesis(self, action: SecondBrainAction) -> SecondBrainObservation:
        if self._current_q_idx >= len(self._questions_queue):
            self._done = True
            return self._build_obs(reward=0.0, feedback="All synthesis questions answered.", done=True)

        current_q = self._questions_queue[self._current_q_idx]

        if action.action_type == ActionType.retrieve:
            # Agent is collecting relevant notes before synthesizing
            query = action.content
            scored = []
            for note in self._kb:
                score = keyword_overlap_score(query, note["text"])
                scored.append((score, note))
            scored.sort(key=lambda x: x[0], reverse=True)
            top_notes = [n for _, n in scored[:5]]

            # Count how many relevant notes were found
            relevant_ids = set(current_q["relevant_note_ids"])
            found_relevant = [n for n in top_notes if n["id"] in relevant_ids]

            # Accumulate found note IDs (agent may do multiple retrieves)
            for n in found_relevant:
                if n["id"] not in self._collected_note_ids:
                    self._collected_note_ids.append(n["id"])

            reward = len(found_relevant) * 0.05
            feedback = f"Found {len(found_relevant)} relevant notes. Keep retrieving or synthesize."

            return self._build_obs(
                reward=reward,
                feedback=feedback,
                done=False,
                query=current_q["question"],
                retrieved_notes=top_notes[:3],
                remaining=len(self._questions_queue) - self._current_q_idx,
            )

        elif action.action_type == ActionType.synthesize:
            answer = action.content.lower()
            expected = current_q["expected_insight"].lower()
            themes = current_q["key_themes"]

            # Score: theme coverage
            themes_found = sum(1 for t in themes if t in answer)
            theme_score = themes_found / len(themes)

            # Score: relevant notes collected
            relevant_ids = set(current_q["relevant_note_ids"])
            coverage = len(set(self._collected_note_ids) & relevant_ids) / len(relevant_ids)

            # Score: answer length (penalize very short answers)
            length_score = min(1.0, len(answer.split()) / 30)

            # Hallucination penalty: if answer contains made-up note IDs
            hallucination_penalty = 0.0
            all_kb_ids = {n["id"] for n in self._kb}
            for word in answer.split():
                if word.startswith("k") and word not in all_kb_ids and len(word) == 4:
                    hallucination_penalty += 0.1

            synthesis_score = (
                0.40 * theme_score +
                0.35 * coverage +
                0.25 * length_score -
                hallucination_penalty
            )
            synthesis_score = round(max(0.0, min(1.0, synthesis_score)), 4)

            reward = synthesis_score * 0.30  # max 0.30 per question

            # Bonus for connecting multiple themes
            if themes_found >= 3:
                reward += 0.05
                feedback = f"✓ Great synthesis! Covered {themes_found}/{len(themes)} themes. Score: {synthesis_score:.2f}"
            else:
                feedback = f"Synthesis scored {synthesis_score:.2f}. Covered {themes_found}/{len(themes)} themes."

            self._synthesis_scores.append(synthesis_score)
            self._collected_note_ids = []  # reset for next question
            self._current_q_idx += 1
            done = self._current_q_idx >= len(self._questions_queue)
            self._done = done

            next_q = (
                self._questions_queue[self._current_q_idx]["question"]
                if not done else None
            )

            return self._build_obs(
                reward=reward,
                feedback=feedback,
                done=done,
                query=next_q,
                remaining=len(self._questions_queue) - self._current_q_idx,
            )

        elif action.action_type == ActionType.skip:
            self._synthesis_scores.append(0.0)
            self._collected_note_ids = []
            self._current_q_idx += 1
            done = self._current_q_idx >= len(self._questions_queue)
            self._done = done
            return self._build_obs(
                reward=0.0, feedback="Skipped. Score 0 recorded.", done=done,
                remaining=len(self._questions_queue) - self._current_q_idx,
            )
        else:
            return self._build_obs(
                reward=-0.01,
                feedback="Use 'retrieve' to collect notes, then 'synthesize' to answer.",
                done=False,
                query=current_q["question"],
                remaining=len(self._questions_queue) - self._current_q_idx,
            )

    # ------------------------------------------------------------------
    # Build observation helper
    # ------------------------------------------------------------------

    def _build_obs(
        self,
        reward: float,
        feedback: str,
        done: bool,
        current_note: Optional[Dict] = None,
        query: Optional[str] = None,
        retrieved_notes: Optional[List[Dict]] = None,
        remaining: int = 0,
    ) -> SecondBrainObservation:
        self._cumulative_reward += reward
        self._rewards.append(reward)
        score = self._current_score()
        kb_size = (
            len(TASK2_KNOWLEDGE_BASE) if self._task_name == "memory_retrieval"
            else len(TASK3_KNOWLEDGE_BASE) if self._task_name == "knowledge_synthesis"
            else 0
        )
        return SecondBrainObservation(
            current_note=current_note,
            query=query,
            retrieved_notes=retrieved_notes,
            knowledge_base_size=kb_size,
            step_count=self._step_count,
            task_name=self._task_name,
            reward=reward,
            done=done,
            score=score,
            feedback=feedback,
            valid_categories=VALID_CATEGORIES,
            remaining_items=remaining,
        )

    # ------------------------------------------------------------------
    # OpenEnv Interface
    # ------------------------------------------------------------------

    def reset(self) -> SecondBrainObservation:
        self._state = SecondBrainState(
            episode_id=str(uuid4()),
            task_name=self._task_name,
            step_count=0,
            score=0.0,
            done=False,
        )
        self._reset_internal()

        if self._task_name == "note_categorization":
            first_note = self._notes_queue[0]
            return self._build_obs(
                reward=0.0,
                feedback="Categorize each note. Valid categories: work, personal, reference, action_item.",
                done=False,
                current_note=first_note,
                remaining=len(self._notes_queue),
            )
        elif self._task_name == "memory_retrieval":
            first_q = self._questions_queue[0]["question"]
            return self._build_obs(
                reward=0.0,
                feedback="Use 'retrieve' action with a search query to find relevant notes.",
                done=False,
                query=first_q,
                remaining=len(self._questions_queue),
            )
        elif self._task_name == "knowledge_synthesis":
            first_q = self._questions_queue[0]["question"]
            return self._build_obs(
                reward=0.0,
                feedback="Use 'retrieve' to collect notes, then 'synthesize' with your answer.",
                done=False,
                query=first_q,
                remaining=len(self._questions_queue),
            )

    def step(self, action: SecondBrainAction) -> SecondBrainObservation:
        self._step_count += 1
        self._state.step_count = self._step_count
        self._state.task_name = self._task_name
        self._state.score = self._current_score()
        self._state.done = self._done
        # Hard step limit
        max_steps = MAX_STEPS[self._task_name]
        if self._step_count > max_steps:
            self._done = True
            return self._build_obs(
                reward=-0.10,
                feedback=f"Episode ended: exceeded {max_steps} steps.",
                done=True,
            )

        if self._done:
            return self._build_obs(
                reward=0.0,
                feedback="Episode already finished. Call reset() to start again.",
                done=True,
            )

        if self._task_name == "note_categorization":
            return self._step_categorization(action)
        elif self._task_name == "memory_retrieval":
            return self._step_retrieval(action)
        elif self._task_name == "knowledge_synthesis":
            return self._step_synthesis(action)

    @property
    def state(self) -> SecondBrainState:
        """Return full typed state — synced with current episode variables."""
        self._state.step_count = self._step_count
        self._state.score = self._current_score()
        self._state.done = self._done
        self._state.task_name = self._task_name
        if self._task_name == "note_categorization":
            self._state.correct_count = self._correct_count
        elif self._task_name == "memory_retrieval":
            self._state.retrieval_scores = list(getattr(self, "_retrieval_scores", []))
        elif self._task_name == "knowledge_synthesis":
            self._state.synthesis_scores = list(getattr(self, "_synthesis_scores", []))
            self._state.collected_note_ids = list(getattr(self, "_collected_note_ids", []))
        return self._state