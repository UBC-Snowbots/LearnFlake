from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MockKeyboardEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 100, dwell_steps: int = 5, press_depth_z: float = -0.02, step_scale: float = 0.02):
        super().__init__()
        self.max_steps = max_steps
        self.dwell_steps = dwell_steps
        self.press_depth_z = press_depth_z
        self.step_scale = step_scale

        self.target_center_xy = np.array([0.1, 0.1], dtype=np.float32)
        self.target_half_extents_xy = np.array([0.05, 0.05], dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "ee_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "target_center_xy": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
                "target_half_extents_xy": spaces.Box(low=0.0, high=np.inf, shape=(2,), dtype=np.float32),
                "t_frac": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "contact": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

        self.ee_pos = np.zeros(3, dtype=np.float32)
        self.steps = 0
        self.dwell_count = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[dict, dict]:
        super().reset(seed=seed)
        self.steps = 0
        self.dwell_count = 0
        self.ee_pos = np.array([0.0, 0.0, 0.05], dtype=np.float32)
        obs = self._get_obs()
        info = {"success": False, "dwell": self.dwell_count, "step": self.steps}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        delta = action[:3] * self.step_scale
        self.ee_pos = self.ee_pos + delta
        press_intent = action[3]

        contact = self._compute_contact(press_intent)
        self.dwell_count = self.dwell_count + 1 if contact else 0
        success = self.dwell_count >= self.dwell_steps

        reward = self._compute_reward(contact, success)
        self.steps += 1

        obs = self._get_obs(contact)
        terminated = bool(success)
        truncated = self.steps >= self.max_steps

        info = {
            "success": success,
            "dwell": self.dwell_count,
            "step": self.steps,
            "contact": contact,
        }
        return obs, reward, terminated, truncated, info

    def _compute_contact(self, press_intent: float) -> bool:
        in_xy = np.all(np.abs(self.ee_pos[:2] - self.target_center_xy) <= self.target_half_extents_xy)
        pressing = self.ee_pos[2] <= self.press_depth_z and press_intent > 0.0
        return bool(in_xy and pressing)

    def _compute_reward(self, contact: bool, success: bool) -> float:
        xy_err = np.linalg.norm(self.ee_pos[:2] - self.target_center_xy)
        reward = -float(xy_err)
        if contact:
            reward += 0.1
        if success:
            reward += 5.0
        return reward

    def _get_obs(self, contact: Optional[bool] = None) -> dict:
        contact_val = contact if contact is not None else False
        t_frac = np.array([min(1.0, self.steps / max(1, self.max_steps))], dtype=np.float32)
        return {
            "ee_pos": self.ee_pos.astype(np.float32),
            "target_center_xy": self.target_center_xy,
            "target_half_extents_xy": self.target_half_extents_xy,
            "t_frac": t_frac,
            "contact": np.array([1.0 if contact_val else 0.0], dtype=np.float32),
        }
