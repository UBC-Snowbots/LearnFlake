"""Mode-aware action wrapper with first-order smoothing.

Approach mode: the policy emits 7-D actions but we hard-mask the solenoid
(action[6] = -1, fully retracted) so the arm can't accidentally extend the
plunger while moving.

Strike mode: the policy emits 7-D actions but we hard-mask the joints
(action[:6] = 0, no joint motion) so the arm holds pose during the press.

In both modes the action is fed through a first-order low-pass filter:
``a_filt = α * a_filt_prev + (1-α) * a_new`` with ``α = 0.4`` (TRACKER §6.3).
This caps the policy's effective Lipschitz constant — important for
sim-to-real (real PD servos break under bang-bang RL output).

The smoothing is applied to the *joint* dimensions only. The solenoid is
binary; smoothing it would just delay the threshold crossing.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

import gymnasium as gym

Mode = Literal["approach", "strike"]


class ActionAdapter(gym.Wrapper):
    """Apply mode-specific masking and per-step action smoothing."""

    def __init__(
        self,
        env: gym.Env,
        *,
        mode: Mode = "approach",
        smooth_alpha: float = 0.4,
    ):
        super().__init__(env)
        if mode not in ("approach", "strike"):
            raise ValueError(f"mode must be 'approach' or 'strike', got {mode!r}")
        self.mode: Mode = mode
        self.alpha = float(smooth_alpha)
        self._a_filt_joints: np.ndarray | None = None
        # action_space stays 7-D — the policy gets the same shape regardless
        # of mode; the wrapper just zeros out irrelevant dims.

    def set_mode(self, mode: Mode) -> None:
        """Switch masking mode mid-episode (TRACKER §39 true Approach→Strike
        chaining). Resets the smoothing filter so the new mode starts clean."""
        if mode not in ("approach", "strike"):
            raise ValueError(f"mode must be 'approach' or 'strike', got {mode!r}")
        self.mode = mode
        self._a_filt_joints = None

    def reset(self, **kwargs):
        self._a_filt_joints = None
        return self.env.reset(**kwargs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(7)
        applied = self._mask_and_smooth(action)
        return self.env.step(applied)

    def _mask_and_smooth(self, action: np.ndarray) -> np.ndarray:
        if self.mode == "approach":
            joints = action[:6]
            solenoid = -1.0  # locked retracted
        else:  # strike
            joints = np.zeros(6, dtype=np.float32)
            solenoid = float(action[-1])

        # First-order smoothing on joint dims (in-place)
        if self._a_filt_joints is None:
            self._a_filt_joints = joints.astype(np.float32).copy()
        else:
            self._a_filt_joints = (
                self.alpha * self._a_filt_joints + (1.0 - self.alpha) * joints
            ).astype(np.float32)

        out = np.empty(7, dtype=np.float32)
        out[:6] = self._a_filt_joints
        out[6] = solenoid
        return out
