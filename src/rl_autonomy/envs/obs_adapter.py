"""Gymnasium wrapper + observation adapters for KeyboardEnv.

Three layered wrappers, inner → outer:

  1. ``KeyboardGymEnv`` — turns the robosuite-native env into a
     ``gymnasium.Env``. Returns a Dict observation: every named field from
     ``KeyboardEnv._build_obs_dict()`` plus a separate ``critic`` view with
     privileged extras.

  2. ``FrameStackWrapper`` — concatenates the last k actor frames. The
     critic view is *not* stacked (TRACKER §9.3 — privileged single frame).

  3. ``ObsAdapter`` — flattens the Dict into two ``Box`` arrays:
     ``observation['actor']`` (post-stack), ``observation['critic']``
     (single frame, privileged), with ``RunningMeanStd`` normalization on
     the actor input.

This split lets the algorithm (RLPD-SAC, Phase 2) feed the actor and critic
different inputs — the asymmetric AC pattern (TRACKER §4.2).
"""
from __future__ import annotations

from collections import deque
from typing import Any, Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from .keyboard_env import KeyboardEnv, Mode, quat_to_rot, rot_to_6d
from .normalizer import RunningMeanStd


# ---------------------------------------------------------------------------
# Schema: which fields go into the actor / critic flat arrays. Listing them
# explicitly (instead of "everything in the dict") forces a deliberate
# decision per field and makes asymmetric AC obvious.
# ---------------------------------------------------------------------------

# Actor sees only deployable signals. Keys reference KeyboardEnv._build_obs_dict.
ACTOR_FIELDS: list[tuple[str, int]] = [
    # name in dict → flat dim contributed
    ("joint_pos_sin_cos", 12),     # synthesized below from "joint_pos"
    ("joint_vel", 6),
    ("eef_pos", 3),
    ("eef_rot_6d", 6),             # synthesized below from "eef_quat"
    ("actuator_extended", 1),
    ("target_offset_eef", 3),
    ("aruco_obs", 3),              # noisy, can drop out
    ("rangefinder", 1),
    ("contact_force_norm", 1),
]
# Sum: 12 + 6 + 3 + 6 + 1 + 3 + 3 + 1 + 1 = 36 — matches TRACKER §9.1.
# In v1 we use the same `contact_force_norm` scalar for both the "real"
# contact force and the synthetic Moteus torque-delta TRACKER §9.1 #9.
# Once the synthetic_moteus_node is added (Phase 5) we may split them; that
# would bump the dim from 36 to 37. Re-test then.
ACTOR_OBS_DIM: int = sum(d for _, d in ACTOR_FIELDS)


# Critic sees actor fields *minus* aruco (which is noisy — replace with GT
# offset) plus privileged extras.
CRITIC_FIELDS: list[tuple[str, int]] = [
    # Drop "aruco_obs" from the actor list:
    ("joint_pos_sin_cos", 12),
    ("joint_vel", 6),
    ("eef_pos", 3),
    ("eef_rot_6d", 6),
    ("actuator_extended", 1),
    ("target_offset_eef", 3),     # this IS the ground-truth replacement for aruco
    ("rangefinder", 1),
    ("contact_force_norm", 1),
    # Privileged extras:
    ("contact_force_vec", 3),
    ("actuator_pos", 1),          # continuous, not just binary
    ("tilt_rad", 1),
]
CRITIC_OBS_DIM: int = sum(d for _, d in CRITIC_FIELDS)
# Sum: 12+6+3+6+1+3+1+1+3+1+1 = 38.


def _joint_pos_sin_cos(joint_pos: np.ndarray) -> np.ndarray:
    """Wrap-safe encoding for continuous joints."""
    return np.concatenate([np.sin(joint_pos), np.cos(joint_pos)]).astype(np.float32)


def _eef_rot_6d(eef_quat: np.ndarray) -> np.ndarray:
    """Continuous 6-D rotation rep (Zhou+ CVPR 2019)."""
    return rot_to_6d(quat_to_rot(eef_quat)).astype(np.float32)


def _flatten_dict_to_actor(obs_dict: dict[str, np.ndarray]) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for name, _ in ACTOR_FIELDS:
        if name == "joint_pos_sin_cos":
            pieces.append(_joint_pos_sin_cos(obs_dict["joint_pos"]))
        elif name == "eef_rot_6d":
            pieces.append(_eef_rot_6d(obs_dict["eef_quat"]))
        else:
            pieces.append(np.asarray(obs_dict[name], dtype=np.float32).reshape(-1))
    return np.concatenate(pieces).astype(np.float32)


def _flatten_dict_to_critic(obs_dict: dict[str, np.ndarray]) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for name, _ in CRITIC_FIELDS:
        if name == "joint_pos_sin_cos":
            pieces.append(_joint_pos_sin_cos(obs_dict["joint_pos"]))
        elif name == "eef_rot_6d":
            pieces.append(_eef_rot_6d(obs_dict["eef_quat"]))
        else:
            pieces.append(np.asarray(obs_dict[name], dtype=np.float32).reshape(-1))
    return np.concatenate(pieces).astype(np.float32)


# ---------------------------------------------------------------------------
# Layer 1 — Gymnasium wrapper around KeyboardEnv
# ---------------------------------------------------------------------------

class KeyboardGymEnv(gym.Env):
    """gymnasium.Env over a robosuite KeyboardEnv.

    Actions: ``Box(7,)`` in ``[-1, 1]`` (6 joint deltas + 1 solenoid).
    Observations: ``Dict({'actor': Box, 'critic': Box})`` produced by feeding
    the labelled obs dict through ``_flatten_dict_to_actor/critic``.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: KeyboardEnv, *, mode: Mode = "approach", seed: int | None = None):
        super().__init__()
        self._env = env
        self.mode: Mode = mode

        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "actor": spaces.Box(-np.inf, np.inf, shape=(ACTOR_OBS_DIM,), dtype=np.float32),
            "critic": spaces.Box(-np.inf, np.inf, shape=(CRITIC_OBS_DIM,), dtype=np.float32),
        })

        if seed is not None:
            self.reset(seed=seed)

    # ---- gymnasium API ----

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            np.random.seed(seed)
        self._env.reset()
        obs = self._build_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        _, reward, terminated, info = self._env.step(action)
        success = bool(self._env._check_success())
        if success:
            terminated = True
        truncated = bool(terminated and not success and info.get("TimeLimit.truncated", False))
        # robosuite's `done` rolls success+timeout together. Re-derive cleanly:
        # if not success and step count == horizon then truncated, not terminated.
        if not success and self._env.horizon is not None:
            truncated = bool(self._env.timestep >= self._env.horizon)
            terminated = False if truncated else terminated
        info["is_success"] = success
        info["target_key"] = self._env.target_key
        obs = self._build_obs()
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self._env.has_renderer:
            self._env.render()

    def close(self):
        self._env.close()

    # ---- helpers ----

    def _build_obs(self) -> dict[str, np.ndarray]:
        obs_dict = self._env.get_obs_dict()
        return {
            "actor": _flatten_dict_to_actor(obs_dict),
            "critic": _flatten_dict_to_critic(obs_dict),
        }

    @property
    def underlying(self) -> KeyboardEnv:
        """Direct access to the robosuite env for tools/tests."""
        return self._env


# ---------------------------------------------------------------------------
# Layer 2 — FrameStack on the actor view only
# ---------------------------------------------------------------------------

class FrameStackWrapper(gym.Wrapper):
    """Concatenate the last k actor observations.

    Critic view is passed through unchanged (single-frame privileged input
    per TRACKER §9.3).
    """

    def __init__(self, env: gym.Env, k: int = 3):
        super().__init__(env)
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self._actor_buf: deque[np.ndarray] = deque(maxlen=k)

        actor_dim = env.observation_space["actor"].shape[0]
        critic_dim = env.observation_space["critic"].shape[0]
        self.observation_space = spaces.Dict({
            "actor": spaces.Box(-np.inf, np.inf, shape=(actor_dim * k,), dtype=np.float32),
            "critic": spaces.Box(-np.inf, np.inf, shape=(critic_dim,), dtype=np.float32),
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._actor_buf.clear()
        # Pad initial buffer by repeating the first observation.
        for _ in range(self.k):
            self._actor_buf.append(obs["actor"])
        return self._stacked(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._actor_buf.append(obs["actor"])
        return self._stacked(obs), reward, terminated, truncated, info

    def _stacked(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        actor = np.concatenate(list(self._actor_buf)).astype(np.float32)
        return {"actor": actor, "critic": obs["critic"]}


# ---------------------------------------------------------------------------
# Layer 3 — RunningMeanStd normalization on actor input
# ---------------------------------------------------------------------------

class ObsAdapter(gym.Wrapper):
    """Normalize the (already-stacked) actor view with running statistics.

    The critic view is not normalized here — its scale is meaningful (raw
    metres for forces, etc.) and the critic is wider/regularized enough
    to handle it. This matches RLPD's recipe.
    """

    def __init__(self, env: gym.Env, *, training: bool = True, clip: float = 10.0):
        super().__init__(env)
        actor_shape = env.observation_space["actor"].shape
        self._rms = RunningMeanStd.zeros(actor_shape)
        self.training = training
        self.clip = clip

    @property
    def rms(self) -> RunningMeanStd:
        """Access the RunningMeanStd accumulator (so train/eval envs can share)."""
        return self._rms

    @rms.setter
    def rms(self, new_rms: RunningMeanStd) -> None:
        """Replace the accumulator. Used to share train env's stats with eval env."""
        self._rms = new_rms

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._normalize(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize(obs), reward, terminated, truncated, info

    def _normalize(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        actor = obs["actor"]
        if self.training:
            self._rms.update(actor[None, :])
        actor_norm = self._rms.normalize(actor, clip=self.clip)
        return {"actor": actor_norm, "critic": obs["critic"]}

    # ---- persistence ----
    def save_stats(self, path: str) -> None:
        self._rms.save(path)

    def load_stats(self, path: str) -> None:
        self._rms = RunningMeanStd.load(path)
