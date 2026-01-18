from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RoboSuiteKeyboardTask(gym.Wrapper):
    """Task wrapper mapping RoboSuite observations into canonical keyboard-control dicts."""

    def __init__(self, env: gym.Env, dwell_steps: int = 5):
        super().__init__(env)
        self.dwell_steps = dwell_steps
        self.dwell_count = 0
        self.steps = 0

        self.observation_space = spaces.Dict(
            {
                "ee_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "target_center_xy": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
                "target_half_extents_xy": spaces.Box(low=0.0, high=np.inf, shape=(2,), dtype=np.float32),
                "t_frac": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "contact": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[dict, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.dwell_count = 0
        self.steps = 0
        canonical = self._to_canonical_obs(obs, contact=False)
        info = info or {}
        info.update({"success": False, "dwell": self.dwell_count, "step": self.steps})
        return canonical, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # TODO: map RoboSuite contacts/keyboard presses into contact detection.
        contact = False
        self.dwell_count = self.dwell_count + 1 if contact else 0
        self.steps += 1
        success = self.dwell_count >= self.dwell_steps

        canonical = self._to_canonical_obs(obs, contact=contact)
        info = info or {}
        info.update({"success": success, "dwell": self.dwell_count, "step": self.steps})
        return canonical, reward, terminated, truncated, info

    def _to_canonical_obs(self, raw_obs, contact: bool) -> dict:
        """Placeholder mapping; replace with real RoboSuite observations."""
        # TODO: extract EE pose and task targets from RoboSuite observations.
        ee_pos = np.zeros(3, dtype=np.float32)
        target_center_xy = np.zeros(2, dtype=np.float32)
        target_half_extents_xy = np.ones(2, dtype=np.float32) * 0.05
        horizon = getattr(self.env.unwrapped, "horizon", 1)
        t_frac = np.array([min(1.0, self.steps / max(1, horizon))], dtype=np.float32)

        return {
            "ee_pos": ee_pos,
            "target_center_xy": target_center_xy,
            "target_half_extents_xy": target_half_extents_xy,
            "t_frac": t_frac,
            "contact": np.array([1.0 if contact else 0.0], dtype=np.float32),
        }
