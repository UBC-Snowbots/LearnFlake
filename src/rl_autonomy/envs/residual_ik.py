"""Residual-on-IK env wrapper + factory (TRACKER §38).

The agent's action is a **tube-clipped residual** added to the live M1 DLS-IK
expert action:

    a_final[:6] = clip( a_ik[:6] + tube * residual[:6], -1, 1 )   # joints
    a_final[6]  = -1                                              # solenoid retracted (Approach)

Why this is the right ceiling-raiser (TRACKER §35.2 / §37.1): DAgger plateaued at
the IK expert's ~62% per-attempt quality (it can only *match* the expert). A
residual trained with RL can *exceed* it — but naive online RL failed for v1
(v3–v7) because its cm-scale exploration is wider than the 4 mm success basin.
The **tube clip** (small, e.g. 0.15) confines the residual — and therefore the
exploration — to a neighbourhood of the already-good IK trajectory, so RL refines
inside the basin instead of wandering out. The IK does the gross motion; the
residual only learns the local correction (precision + workspace-edge) the IK
gets wrong. Deployable: the IK runs on the real arm too, so this ships as
"IK + learned correction".

Wrapper stack (outer → inner) built by ``make_residual_env``:

    ObsAdapter → FrameStackWrapper → ActionAdapter(smooth+mask the RESIDUAL)
        → ResidualIKWrapper(add the crisp IK action) → KeyboardGymEnv → KeyboardEnv

ActionAdapter sits *outside* the residual wrapper so the learned residual is
band-limited (good for sim-to-real) while the IK action stays crisp/closed-loop.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from .keyboard_env import KeyboardEnv, Mode
from ._wrapper_utils import require_inner
from rl_autonomy.algos.expert_ik import IKExpert


class ResidualIKWrapper(gym.Wrapper):
    """Add a tube-clipped residual (the agent's action) to the live IK action.

    Wraps ``KeyboardGymEnv``. The wrapped env's action is the 7-D residual in
    [-1, 1]; only the 6 joint dims are used (the solenoid is held retracted in
    Approach). Observations pass through unchanged.
    """

    def __init__(self, env: gym.Env, *, tube: float = 0.15, mode: Mode = "approach"):
        super().__init__(env)
        if tube <= 0.0:
            raise ValueError(f"tube must be > 0; got {tube}")
        self.tube = float(tube)
        self.mode: Mode = mode
        self._kb: KeyboardEnv = require_inner(env, KeyboardEnv)
        self._expert = IKExpert()
        # Telemetry: how hard the residual is pushing against the tube wall.
        self._last_residual_frac = 0.0
        # When True, pass the action straight through (no IK add, no solenoid
        # mask) — used during the Strike phase of true Approach→Strike chaining
        # (TRACKER §39), where the arm holds and only the solenoid acts.
        self.bypass = False

    def reset(self, **kwargs):
        self.bypass = False
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.bypass:
            return self.env.step(np.asarray(action, dtype=np.float32).reshape(7))
        residual = np.asarray(action, dtype=np.float32).reshape(7)
        a_ik = self._expert.action(self._kb)               # crisp, closed-loop
        a_final = np.empty(7, dtype=np.float32)
        a_final[:6] = np.clip(a_ik[:6] + self.tube * residual[:6], -1.0, 1.0)
        a_final[6] = -1.0                                  # solenoid retracted (Approach)
        # fraction of the tube the (clipped) joint residual is using, for logging
        self._last_residual_frac = float(np.mean(np.abs(residual[:6])))
        obs, reward, terminated, truncated, info = self.env.step(a_final)
        info = dict(info)
        info["residual_frac"] = self._last_residual_frac
        return obs, reward, terminated, truncated, info


def make_residual_env(
    *,
    tube: float = 0.15,
    reward_mode: str = "xy_focus",
    keyboard_offset: tuple[float, float] = (-0.10, -0.10),
    random_key: bool = True,
    domain_rand: bool = False,
    frame_stack: int = 3,
    horizon: int | None = None,
    seed: int | None = None,
    smooth_alpha: float = 0.4,
):
    """Build the residual-on-IK Approach env (mode is always 'approach').

    Mirrors ``make_env`` but inserts ``ResidualIKWrapper`` between
    ``KeyboardGymEnv`` and ``ActionAdapter`` so the agent acts in residual space.
    """
    from .action_adapter import ActionAdapter
    from .obs_adapter import KeyboardGymEnv, ObsAdapter, FrameStackWrapper
    from .domain_rand import DomainRandWrapper

    base = KeyboardEnv(
        mode="approach", render=False, random_key=random_key,
        horizon=horizon, reward_mode=reward_mode, keyboard_offset=keyboard_offset,
    )
    gym_env = KeyboardGymEnv(base, mode="approach", seed=seed)
    res = ResidualIKWrapper(gym_env, tube=tube, mode="approach")
    env = ActionAdapter(res, mode="approach", smooth_alpha=smooth_alpha)
    env = FrameStackWrapper(env, k=frame_stack)
    env = ObsAdapter(env)
    if domain_rand:
        env = DomainRandWrapper(env)
    return env
