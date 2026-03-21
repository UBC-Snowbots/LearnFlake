"""
Gymnasium-compatible goal-conditioned navigation environment for Rover2026.

This is Policy Sub-Meta-1 in the HRL stack:
- input: robot state + target hover position (x, y, z)
- output action: delta xyz in operational/task space (clipped to [-1, 1])
- reward: dense distance + smoothness penalty + sparse success bonus
"""

from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Must be set before MuJoCo / robosuite imports.
os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)


class UpgradedEnvHRL(gym.Env):
    """
    Goal-conditioned SAC environment for hover navigation.

    Notes:
    - We intentionally avoid image observations for sample efficiency.
    - Action space is task-space delta xyz; gripper/contact is not learned here.
    - The env is reset each episode to approximate a "cold state".
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    # Keyboard layout in meters in world frame (example calibration values).
    # Meta-1 should select a key id; Sub-Meta-1 samples or receives one target.
    KEY_TO_POS = {
        1: np.array([0.05, -0.08, 0.84], dtype=np.float32),
        2: np.array([0.07, -0.08, 0.84], dtype=np.float32),
        3: np.array([0.09, -0.08, 0.84], dtype=np.float32),
        4: np.array([0.05, -0.06, 0.84], dtype=np.float32),
        5: np.array([0.07, -0.06, 0.84], dtype=np.float32),
        6: np.array([0.09, -0.06, 0.84], dtype=np.float32),
    }

    def __init__(
        self,
        render: bool = False,
        domain_randomization: bool = True,
        control_freq: int = 20,
        horizon: int = 150,
        success_threshold_m: float = 0.01,
        hover_height_m: float = 0.015,
        velocity_penalty_coef: float = 0.03,
        controller: str = "OSC_POSE",
        robot_name: str = "Rover2026",
    ) -> None:
        super().__init__()
        self.render_enabled = render
        self.domain_randomization = domain_randomization
        self.max_episode_steps = int(horizon)
        self.success_threshold_m = float(success_threshold_m)
        self.hover_height_m = float(hover_height_m)
        self.velocity_penalty_coef = float(velocity_penalty_coef)

        # Prefer operational-space controller for delta xyz commands.
        arm_cfg = suite.load_part_controller_config(default_controller=controller)
        controller_cfg = refactor_composite_controller_config(arm_cfg, robot_name, ["right"])

        self.env = suite.make(
            env_name="Lift",
            robots=[robot_name],
            controller_configs=controller_cfg,
            has_renderer=render,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            ignore_done=False,
            control_freq=control_freq,
            horizon=horizon,
            reward_shaping=False,
        )

        self._steps = 0
        self._goal_xyz = np.zeros(3, dtype=np.float32)
        self._goal_key = 1
        self._rng = np.random.default_rng()
        self._action_dim = int(self.env.action_dim)

        # Build spaces from one reset.
        obs = self.env.reset()
        robot_state = self._extract_robot_state(obs)
        self._robot_state_dim = int(robot_state.size)
        self._obs_dim = self._robot_state_dim + 3

        # Policy outputs normalized delta xyz in task-space.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

    def _extract_robot_state(self, obs: dict[str, Any]) -> np.ndarray:
        eef = np.array(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32).ravel()
        qpos = np.array(obs.get("robot0_joint_pos", np.zeros(6)), dtype=np.float32).ravel()
        qvel = np.array(obs.get("robot0_joint_vel", np.zeros(6)), dtype=np.float32).ravel()
        qpos_sin = np.sin(qpos).astype(np.float32)
        qpos_cos = np.cos(qpos).astype(np.float32)
        return np.concatenate([eef, qpos_sin, qpos_cos, qvel]).astype(np.float32)

    def _sample_goal(self) -> tuple[int, np.ndarray]:
        key = int(self._rng.choice(list(self.KEY_TO_POS.keys())))
        goal = self.KEY_TO_POS[key].copy()
        goal[2] += self.hover_height_m

        if self.domain_randomization:
            goal[0] += self._rng.uniform(-0.003, 0.003)
            goal[1] += self._rng.uniform(-0.003, 0.003)
        return key, goal.astype(np.float32)

    def _get_obs(self, obs: dict[str, Any]) -> np.ndarray:
        robot_state = self._extract_robot_state(obs)
        return np.concatenate([robot_state, self._goal_xyz]).astype(np.float32)

    def _compose_env_action(self, delta_xyz: np.ndarray) -> np.ndarray:
        """
        Map delta xyz action into the full robosuite action vector.
        Remaining dimensions are zeroed (no contact policy in this stage).
        """
        full_action = np.zeros(self._action_dim, dtype=np.float32)
        full_action[:3] = np.clip(delta_xyz, -1.0, 1.0)
        return full_action

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._steps = 0
        obs = self.env.reset()

        # Cold-state episode with a fresh target key hover point.
        self._goal_key, self._goal_xyz = self._sample_goal()

        out_obs = self._get_obs(obs)
        info = {
            "goal_key": self._goal_key,
            "goal_xyz": self._goal_xyz.copy(),
        }
        return out_obs, info

    def step(self, action: np.ndarray):
        self._steps += 1
        action = np.asarray(action, dtype=np.float32).reshape(3)

        env_action = self._compose_env_action(action)
        obs, _, env_done, info = self.env.step(env_action)

        eef = np.array(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32).ravel()
        qvel = np.array(obs.get("robot0_joint_vel", np.zeros(6)), dtype=np.float32).ravel()

        distance = float(np.linalg.norm(eef - self._goal_xyz))
        velocity_norm = float(np.linalg.norm(qvel))

        dense = -distance
        smooth = -self.velocity_penalty_coef * velocity_norm
        success = distance < self.success_threshold_m
        sparse = 100.0 if success else 0.0
        reward = dense + smooth + sparse

        terminated = bool(success)
        truncated = bool((self._steps >= self.max_episode_steps) or env_done)

        out_obs = self._get_obs(obs)
        info = dict(info or {})
        info.update(
            {
                "goal_key": self._goal_key,
                "goal_xyz": self._goal_xyz.copy(),
                "distance_to_goal": distance,
                "velocity_norm": velocity_norm,
                "reward_dense": dense,
                "reward_smoothness": smooth,
                "reward_sparse": sparse,
                "is_success": success,
            }
        )
        return out_obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_enabled:
            return self.env.render()
        return None

    def close(self):
        self.env.close()
