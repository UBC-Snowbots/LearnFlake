import numpy as np

from rlkb.core.agent_base import BaseAgent


class RandomAgent(BaseAgent):
    def act(self, obs, deterministic: bool = False):
        return self.action_space.sample()
