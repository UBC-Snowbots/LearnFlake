from abc import ABC, abstractmethod
from typing import Optional

import gymnasium as gym


class BaseAgent(ABC):
    def __init__(self, action_space: gym.Space, observation_space: Optional[gym.Space] = None, training: bool = True):
        self.action_space = action_space
        self.observation_space = observation_space
        self.training = training

    @abstractmethod
    def act(self, obs, deterministic: bool = False):
        """Produce an action compatible with the environment action space."""
        raise NotImplementedError

    def observe(self, transition) -> None:
        """Optional hook for storing transitions."""
        return None

    def train_step(self) -> dict:
        """Optional per-step training hook."""
        return {}

    def set_training_mode(self, training: bool) -> None:
        self.training = training

    def save(self, path: str) -> None:
        """Optional save hook."""
        return None

    def load(self, path: str) -> None:
        """Optional load hook."""
        return None
