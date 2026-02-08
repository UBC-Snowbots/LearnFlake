import numpy as np

from rlkb.core.agent_base import BaseAgent


class HeuristicPressAgent(BaseAgent):
    def __init__(self, action_space, observation_space=None, training: bool = False, xy_gain: float = 5.0, press_tol: float = 0.02):
        super().__init__(action_space, observation_space, training)
        self.xy_gain = xy_gain
        self.press_tol = press_tol

    def act(self, obs, deterministic: bool = True):
        ee_pos = np.asarray(obs["ee_pos"], dtype=np.float32)
        target = np.asarray(obs["target_center_xy"], dtype=np.float32)

        xy_error = target - ee_pos[:2]
        xy_cmd = np.clip(xy_error * self.xy_gain, -1.0, 1.0)

        close_enough = np.linalg.norm(xy_error) < self.press_tol
        dz = -1.0 if close_enough else 0.0
        press_intent = 1.0 if close_enough else 0.0

        action = np.array([xy_cmd[0], xy_cmd[1], dz, press_intent], dtype=np.float32)

        if hasattr(self.action_space, "low") and hasattr(self.action_space, "high"):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        return action
