"""
Second Brain Environment — WebSocket Client
Connects to the running server and exposes reset()/step()/state().

FIX: The openenv HTTP layer deserializes the observation using the Pydantic
schema, but `reward` has a default of 0.0 — so if the server response omits
the field (or it gets stripped by the base class serializer) it silently
resets to 0. The fix: read `reward` from the raw response dict BEFORE
constructing the typed observation, then attach it as `result.reward` so
inference.py can read it independently of the observation object.
"""
from openenv.core.env_client import EnvClient
from models import SecondBrainAction, SecondBrainObservation, SecondBrainState
from types import SimpleNamespace


class SecondBrainEnv(EnvClient):
    action_type      = SecondBrainAction
    observation_type = SecondBrainObservation
    state_type       = SecondBrainState

    def __init__(self, base_url: str = "http://localhost:8000", task_name: str = "note_categorization"):
        super().__init__(base_url=base_url)
        self._task_name = task_name

    def _step_payload(self, action: SecondBrainAction) -> dict:
        return action.model_dump()

    def _parse_result(self, data: dict):
        obs_data = data.get("observation", {})

        # Pull reward out of the raw dict BEFORE Pydantic touches it.
        # The base EnvClient may not forward all fields — this is the only
        # reliable place to read it.
        reward = float(obs_data.get("reward", 0.0))

        obs  = self.observation_type(**obs_data)
        done = data.get("done", False)

        # Attach reward at the top level of the result so inference.py can
        # read result.reward without depending on obs.reward surviving intact.
        return SimpleNamespace(
            observation=obs,
            done=done,
            reward=reward,          # ← top-level, always correct
        )

    def _parse_state(self, data: dict):
        return self.state_type(**data)