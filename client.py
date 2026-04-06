"""
Second Brain Environment — WebSocket Client
Connects to the running server and exposes reset()/step()/state().
"""
from openenv.core.env_client import EnvClient
from models import SecondBrainAction, SecondBrainObservation , SecondBrainState


class SecondBrainEnv(EnvClient):
    """
    Client for the Second Brain OpenEnv environment.

    Usage:
        async with SecondBrainEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
            result = await env.step(SecondBrainAction(
                action_type="categorize",
                content="work"
            ))

    Or from a deployed HF Space:
        async with SecondBrainEnv(base_url="https://RAc1928-second-brain-env.hf.space") as env:
            ...
    """

    action_type      = SecondBrainAction
    observation_type = SecondBrainObservation
    state_type       = SecondBrainState  

    def __init__(self, base_url: str = "http://localhost:8000", task_name: str = "note_categorization"):
        super().__init__(base_url=base_url)
        self._task_name = task_name