"""
Typed models for the Second Brain OpenEnv environment.
Defines Action, Observation, and State using proper OpenEnv inheritance.
"""
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import Field
 
from openenv.core.env_server import Action, Observation
from openenv.core.env_server.types import State
 
 
# ---------------------------------------------------------------------------
# ActionType Enum — prevents typos like "categorise" instead of "categorize"
# ---------------------------------------------------------------------------
 
class ActionType(str, Enum):
    categorize = "categorize"
    retrieve   = "retrieve"
    synthesize = "synthesize"
    tag        = "tag"
    skip       = "skip"
 
 
# ---------------------------------------------------------------------------
# Action — inherits from OpenEnv Action
# ---------------------------------------------------------------------------
 
class SecondBrainAction(Action):
    """
    A single action the agent can take in the Second Brain environment.
 
    action_type options:
      - categorize : assign a category to the current note (Task 1)
      - retrieve   : query the knowledge base with a search string (Task 2 & 3)
      - synthesize : produce a final synthesized answer (Task 3)
      - tag        : add tags to a note
      - skip       : skip the current item (no reward)
 
    Fields:
      action_type  : validated enum — prevents typos
      content      : category name / search query / synthesized answer
      note_id      : optional — which note to act on
      tags         : optional list of tags
    """
    action_type: ActionType = Field(
        ...,
        description="One of: categorize | retrieve | synthesize | tag | skip"
    )
    content: str = Field(
        ...,
        description="Main payload: category name, search query, or synthesized answer"
    )
    note_id: Optional[str] = Field(
        default=None,
        description="Target note ID for retrieve or tag actions"
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="List of tags to apply (used with action_type='tag')"
    )
 
 
# ---------------------------------------------------------------------------
# Observation — inherits from OpenEnv Observation
# ---------------------------------------------------------------------------
 
class SecondBrainObservation(Observation):
    """
    What the agent observes after each step.
 
    Fields:
      current_note      : the note currently being processed (Task 1)
      query             : the retrieval question (Task 2 & 3)
      retrieved_notes   : notes returned by a retrieve action
      knowledge_base_size : how many notes are in the KB
      step_count        : current step number
      task_name         : which task is active
      reward            : reward from the last action
      done              : whether the episode is finished
      score             : running cumulative score (0.0–1.0)
      feedback          : human-readable feedback on last action
      valid_categories  : categories the agent may use (Task 1)
      remaining_items   : how many items still need processing
    """
    current_note: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The note currently presented to the agent"
    )
    query: Optional[str] = Field(
        default=None,
        description="The retrieval question the agent must answer"
    )
    retrieved_notes: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Notes returned after a retrieve action"
    )
    knowledge_base_size: int = Field(
        default=0,
        description="Total notes in the knowledge base"
    )
    step_count: int = Field(default=0)
    task_name: str = Field(default="")
    reward: float = Field(default=0.0)
    done: bool = Field(default=False)
    score: float = Field(default=0.0, description="Running score in [0, 1]")
    feedback: str = Field(default="", description="Feedback on the last action")
    valid_categories: List[str] = Field(
        default_factory=lambda: ["work", "personal", "reference", "action_item"],
        description="Valid category labels for Task 1"
    )
    remaining_items: int = Field(
        default=0,
        description="Number of items still to be processed this episode"
    )
 
 
# ---------------------------------------------------------------------------
# State — extends OpenEnv State with environment-specific fields
# ---------------------------------------------------------------------------
 
class SecondBrainState(State):
    """
    Full internal state of the Second Brain environment.
    Used by state() endpoint and for debugging/graders.
 
    Fields:
      task_name    : which task is currently running
      step_count   : how many steps have been taken
      score        : current normalized score in [0.0, 1.0]
      correct_count       : Task 1 — correct categorizations so far
      retrieval_scores    : Task 2 — list of per-question retrieval scores
      synthesis_scores    : Task 3 — list of per-question synthesis scores
      collected_note_ids  : Task 3 — notes collected before synthesizing
      done         : whether the episode has ended
    """
    task_name: str = Field(default="")
    step_count: int = Field(default=0)
    score: float = Field(default=0.0)
    correct_count: int = Field(default=0)
    retrieval_scores: List[float] = Field(default_factory=list)
    synthesis_scores: List[float] = Field(default_factory=list)
    collected_note_ids: List[str] = Field(default_factory=list)
    done: bool = Field(default=False)