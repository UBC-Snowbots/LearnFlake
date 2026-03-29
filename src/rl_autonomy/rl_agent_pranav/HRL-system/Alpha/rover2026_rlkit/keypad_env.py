"""
KeypadReachEnv -- Gym wrapper around KeypadLift for the Alpha reach policy.

Same observation/action space as Rover2026ReachEnv so trained policy
checkpoints are directly compatible. Adds:
  - set_target(xyz) to override random goal with a specific key position
  - current_obs() to read observation without stepping
  - Physical keys visible in the simulation
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import gym
import numpy as np
from gym import spaces

from alpha_env_utils import ensure_alpha_import_paths

os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

ensure_alpha_import_paths()

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)

from keypad_lift_env import KeypadLift
from reach_env import RoverReachEnvConfig
from layout_compat import DEFAULT_KEY_LAYOUT


@dataclass
class KeypadReachEnvConfig(RoverReachEnvConfig):
    """Extends RoverReachEnvConfig with a much larger horizon for multi-key sequences."""
    horizon: int = 2000
    max_delta_m: float = 0.007
    residual_action_scale_m: float = 0.005
    sample_key_targets: bool = False
    key_target_noise_xy_m: float = 0.0
    fixed_target_key: Optional[str] = None
    objective_mode: str = "hover"
    proportional_gain: float = 0.3
    hover_height_m: float = 0.03
    hover_xy_weight: float = 2.0
    hover_z_weight: float = 1.0
    hover_bonus: float = 5.0
    hover_success_xy_tolerance_m: float = 0.02
    hover_success_z_tolerance_m: float = 0.01
    hover_shape_radius_m: float = 0.05
    hover_far_penalty_weight: float = 0.1
    press_bonus: float = 10.0
    press_success_xy_tolerance_m: float = 0.012
    press_success_z_tolerance_m: float = 0.02
    terminate_on_success: bool = True


class KeypadReachEnv(gym.Env):
    """
    Gym env wrapping KeypadLift (Lift + 10 physical keys).

    Observation and action spaces are identical to Rover2026ReachEnv so
    existing trained policies work without modification.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}
    _EEF_SITE_CANDIDATES = (
        "robot0_grip_site",
        "gripper0_right_grip_site",
        "gripper0_grip_site",
    )

    def __init__(self, config: KeypadReachEnvConfig):
        super().__init__()
        self.config = config
        self._rng = np.random.default_rng(config.seed)

        arm_controller_config = suite.load_part_controller_config(
            default_controller=config.controller
        )
        controller_config = refactor_composite_controller_config(
            arm_controller_config,
            config.robot_name,
            ["right"],
        )

        # Use KeypadLift instead of suite.make("Lift")
        self.env = KeypadLift(
            robots=[config.robot_name],
            controller_configs=controller_config,
            has_renderer=config.render,
            has_offscreen_renderer=config.offscreen_render,
            use_camera_obs=False,
            ignore_done=True,
            reward_shaping=False,
            control_freq=config.control_freq,
            horizon=config.horizon,
        )

        self._step_count = 0
        self._target_xyz = np.zeros(3, dtype=np.float32)
        self._target_key_id: Optional[str] = None
        self._press_target_xyz = np.zeros(3, dtype=np.float32)
        self._hover_target_xyz = np.zeros(3, dtype=np.float32)
        self._last_base_action = np.zeros(3, dtype=np.float32)
        self._eef_site_id = self._resolve_eef_site_id()

        obs_dict = self.env.reset()
        obs = self._flatten_obs(obs_dict)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    # ── observation helpers (same as Rover2026ReachEnv) ──────────────

    def _extract_robot_state(self, obs: Dict[str, Any]) -> np.ndarray:
        eef = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32).ravel()
        qpos = np.asarray(obs.get("robot0_joint_pos", np.zeros(6)), dtype=np.float32).ravel()
        qvel = np.asarray(obs.get("robot0_joint_vel", np.zeros(6)), dtype=np.float32).ravel()
        return np.concatenate([eef, qpos, qvel], dtype=np.float32)

    def _flatten_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        robot_state = self._extract_robot_state(obs)
        hover_delta = self._hover_target_xyz - robot_state[:3]
        return np.concatenate([robot_state, hover_delta, self._last_base_action], dtype=np.float32)

    def _compose_env_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(3)
        action = np.clip(action, -1.0, 1.0)
        delta_m = action * self.config.max_delta_m
        full_action = np.zeros(self.env.action_dim, dtype=np.float32)
        full_action[:3] = delta_m / self.config.max_delta_m
        return full_action

    def _proportional_action(self, eef_xyz: np.ndarray) -> np.ndarray:
        hover_delta_world = self._hover_target_xyz - np.asarray(eef_xyz, dtype=np.float32).reshape(3)
        rotation = self._eef_rotation_world()
        hover_delta_local = rotation.T @ hover_delta_world
        raw_delta_m = float(self.config.proportional_gain) * hover_delta_local
        base_action = raw_delta_m / float(self.config.max_delta_m)
        return np.clip(base_action, -1.0, 1.0).astype(np.float32)

    def _resolve_eef_site_id(self) -> int | None:
        sim = getattr(self.env, "sim", None)
        if sim is None:
            return None
        for site_name in self._EEF_SITE_CANDIDATES:
            try:
                return int(sim.model.site_name2id(site_name))
            except Exception:
                continue
        return None

    def _eef_rotation_world(self) -> np.ndarray:
        sim = getattr(self.env, "sim", None)
        if sim is None or self._eef_site_id is None:
            return np.eye(3, dtype=np.float32)
        xmat = np.asarray(sim.data.site_xmat[self._eef_site_id], dtype=np.float32).reshape(3, 3)
        return xmat

    # ── target control ───────────────────────────────────────────────

    def set_target(self, target_xyz: np.ndarray) -> None:
        """Override the random target with a specific 3D position (e.g. a key)."""
        self._set_press_and_hover_targets(np.asarray(target_xyz, dtype=np.float32))
        obs_dict = self.env._get_observations()
        eef = np.asarray(obs_dict.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32).ravel()
        self._last_base_action = self._proportional_action(eef)

    def current_obs(self) -> np.ndarray:
        """Return the current flattened observation without stepping."""
        obs_dict = self.env._get_observations()
        return self._flatten_obs(obs_dict)

    # ── Gym interface ────────────────────────────────────────────────

    def seed(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return [seed]

    def reset(self):
        self._step_count = 0
        obs_dict = self.env.reset()
        eef = np.asarray(obs_dict.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32).ravel()
        self._set_press_and_hover_targets(self._sample_goal(eef))
        self._last_base_action = self._proportional_action(eef)
        return self._flatten_obs(obs_dict)

    def _sample_goal(self, eef_xyz: np.ndarray) -> np.ndarray:
        if self.config.sample_key_targets:
            return self._sample_key_goal()
        x = float(eef_xyz[0] + self._rng.uniform(-self.config.target_noise_xy_m, self.config.target_noise_xy_m))
        y = float(eef_xyz[1] + self._rng.uniform(-self.config.target_noise_xy_m, self.config.target_noise_xy_m))
        z = float(eef_xyz[2] + self._rng.uniform(-self.config.target_noise_z_m, self.config.target_noise_z_m))
        z = float(np.clip(z, self.config.target_min_z, self.config.target_max_z))
        self._target_key_id = None
        return np.asarray([x, y, z], dtype=np.float32)

    def _sample_key_goal(self) -> np.ndarray:
        if self.config.fixed_target_key is not None:
            key_id = str(self.config.fixed_target_key)
        else:
            key_id = str(self._rng.choice(sorted(DEFAULT_KEY_LAYOUT)))
        key_xyz = self.env.get_key_position(key_id).astype(np.float32)
        key_xyz[0] += self._rng.uniform(-self.config.key_target_noise_xy_m, self.config.key_target_noise_xy_m)
        key_xyz[1] += self._rng.uniform(-self.config.key_target_noise_xy_m, self.config.key_target_noise_xy_m)
        self._target_key_id = key_id
        return key_xyz

    def _set_press_and_hover_targets(self, press_target_xyz: np.ndarray) -> None:
        self._press_target_xyz = np.asarray(press_target_xyz, dtype=np.float32).copy()
        self._hover_target_xyz = self._press_target_xyz.copy()
        self._hover_target_xyz[2] += float(self.config.hover_height_m)
        self._target_xyz = self._hover_target_xyz.copy()

    def step(self, action: np.ndarray):
        self._step_count += 1

        residual_action = np.asarray(action, dtype=np.float32).reshape(3)
        residual_action = np.clip(residual_action, -1.0, 1.0)
        obs_before = self.env._get_observations()
        eef_before = np.asarray(obs_before.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32).ravel()
        base_action = self._proportional_action(eef_before)
        residual_scale = float(self.config.residual_action_scale_m / self.config.max_delta_m)
        full_action = np.clip(base_action + residual_action * residual_scale, -1.0, 1.0)
        self._last_base_action = base_action.copy()

        env_action = self._compose_env_action(full_action)
        obs_dict, _, _, info = self.env.step(env_action)

        eef = np.asarray(obs_dict.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32).ravel()
        self._last_base_action = self._proportional_action(eef)
        hover_delta = eef - self._hover_target_xyz
        hover_xy_error = float(np.linalg.norm(hover_delta[:2]))
        hover_z_error = float(abs(hover_delta[2]))
        hover_distance = float(np.linalg.norm(hover_delta))

        press_delta = eef - self._press_target_xyz
        press_xy_error = float(np.linalg.norm(press_delta[:2]))
        press_z_error = float(abs(press_delta[2]))
        press_distance = float(np.linalg.norm(press_delta))

        hover_locked = bool(
            hover_xy_error <= self.config.hover_success_xy_tolerance_m
            and hover_z_error <= self.config.hover_success_z_tolerance_m
        )
        press_success = bool(
            press_xy_error <= self.config.press_success_xy_tolerance_m
            and press_z_error <= self.config.press_success_z_tolerance_m
        )

        hover_shape_radius = max(float(self.config.hover_shape_radius_m), 1e-6)
        hover_shaping = (
            1.0 - (hover_distance / hover_shape_radius)
            if hover_distance < hover_shape_radius
            else -self.config.hover_far_penalty_weight * hover_distance
        )

        if self.config.objective_mode == "press":
            reward = (
                hover_shaping
                - self.config.hover_xy_weight * hover_xy_error
                - self.config.hover_z_weight * hover_z_error
                - self.config.action_l2_weight * float(np.linalg.norm(residual_action))
                + (self.config.hover_bonus if hover_locked else 0.0)
                + (self.config.press_bonus if press_success else 0.0)
            )
            success = press_success
        elif self.config.objective_mode == "hover":
            reward = (
                hover_shaping
                - self.config.hover_xy_weight * hover_xy_error
                - self.config.hover_z_weight * hover_z_error
                - self.config.action_l2_weight * float(np.linalg.norm(residual_action))
                + (self.config.hover_bonus if hover_locked else 0.0)
            )
            success = hover_locked
        else:
            raise ValueError(f"Unsupported keypad objective_mode: {self.config.objective_mode}")

        done = bool(self._step_count >= self.config.horizon or (self.config.terminate_on_success and success))
        obs = self._flatten_obs(obs_dict)

        out_info = dict(info or {})
        out_info.update({
            "distance_to_goal": hover_distance,
            "standoff_error": hover_distance,
            "is_success": float(success),
            "hover_locked": float(hover_locked),
            "press_success": float(press_success),
            "hover_xy_error": hover_xy_error,
            "hover_z_error": hover_z_error,
            "press_distance": press_distance,
            "press_xy_error": press_xy_error,
            "press_z_error": press_z_error,
            "target_xyz": self._hover_target_xyz.copy(),
            "hover_target_xyz": self._hover_target_xyz.copy(),
            "press_target_xyz": self._press_target_xyz.copy(),
            "target_key_id": -1.0 if self._target_key_id is None else float(int(self._target_key_id)),
            "eef_xyz": eef.copy(),
            "base_action": base_action.copy(),
            "residual_action": residual_action.copy(),
            "full_action": full_action.copy(),
        })

        return obs, float(reward), done, out_info

    def render(self, mode: str = "human"):
        if mode == "human":
            if self.config.render:
                return self.env.render()
            return None
        if mode == "rgb_array":
            return self.env.sim.render(
                camera_name=self.config.camera_name,
                width=self.config.camera_width,
                height=self.config.camera_height,
            )[::-1]
        raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        self.env.close()
