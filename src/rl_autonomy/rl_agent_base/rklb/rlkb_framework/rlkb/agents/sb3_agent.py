from typing import Any

from rlkb.core.agent_base import BaseAgent


class SB3Agent(BaseAgent):
    """Thin wrapper around a Stable-Baselines3 policy."""

    def __init__(self, model: Any, training: bool = True):
        super().__init__(action_space=model.action_space, observation_space=getattr(model, "observation_space", None), training=training)
        self.model = model

    def act(self, obs, deterministic: bool = False):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        # Lazy import to keep SB3 optional.
        alg_type = type(self.model)
        self.model = alg_type.load(path)
        self.action_space = self.model.action_space
        self.observation_space = getattr(self.model, "observation_space", None)
