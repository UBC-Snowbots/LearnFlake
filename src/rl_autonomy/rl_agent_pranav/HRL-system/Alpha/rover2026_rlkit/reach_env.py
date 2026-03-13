from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import gym
import numpy as np
from gym import spaces

# Headless-safe default for MuJoCo rendering.
os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBO_PATH = os.path.join(ROOT, "..", "external_pkgs", "RoboSuite")
if os.path.exists(ROBO_PATH) and ROBO_PATH not in sys.path:
    sys.path.insert(0, ROBO_PATH)

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)


@dataclass
class RoverReachEnvConfig:
    env_name: str = "Lift"
    robot_name: str = "Rover2026"
    controller: str = "OSC_POSE"
    control_freq: int = 20
    horizon: int = 200
    success_threshold_m: float = 0.05
    success_bonus: float = 5.0
    distance_weight: float = 1.0
    action_l2_weight: float = 0.01
    max_delta_m: float = 0.04
    target_noise_xy_m: float = 0.08
    target_noise_z_m: float = 0.06
    target_min_z: float = 0.82
    target_max_z: float = 1.25
    render: bool = False
    offscreen_render: bool = False
    camera_name: str = "birdview"
    camera_width: int = 640
    camera_height: int = 480
    seed: int = 0


class Rover2026ReachEnv(gym.Env):
    """
    RLKit-compatible Robosuite reaching env for Rover2026.

    Observations: [eef_xyz(3), joint_pos(6), joint_vel(6), target_xyz(3)]
    Actions: normalized Cartesian deltas [dx, dy, dz] in [-1, 1], mapped to meters via max_delta_m
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config: RoverReachEnvConfig):
        super().__init__()
        self.config = config
        self._rng = np.random.default_rng(config.seed)

        arm_controller_config = suite.load_part_controller_config(default_controller=config.controller)
        controller_config = refactor_composite_controller_config(
            arm_controller_config,
            config.robot_name,
            ["right"],
        )

        self.env = suite.make(
            env_name=config.env_name,
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

        obs_dict = self.env.reset()
        obs = self._flatten_obs(obs_dict)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def _extract_robot_state(self, obs: Dict[str, Any]) -> np.ndarray:
        eef = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32).ravel()
        qpos = np.asarray(obs.get("robot0_joint_pos", np.zeros(6)), dtype=np.float32).ravel()
        qvel = np.asarray(obs.get("robot0_joint_vel", np.zeros(6)), dtype=np.float32).ravel()
        return np.concatenate([eef, qpos, qvel], dtype=np.float32)

    def _sample_goal(self, eef_xyz: np.ndarray) -> np.ndarray:
        x = float(eef_xyz[0] + self._rng.uniform(-self.config.target_noise_xy_m, self.config.target_noise_xy_m))
        y = float(eef_xyz[1] + self._rng.uniform(-self.config.target_noise_xy_m, self.config.target_noise_xy_m))
        z = float(eef_xyz[2] + self._rng.uniform(-self.config.target_noise_z_m, self.config.target_noise_z_m))
        z = float(np.clip(z, self.config.target_min_z, self.config.target_max_z))
        return np.asarray([x, y, z], dtype=np.float32)

    def _flatten_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        robot_state = self._extract_robot_state(obs)
        return np.concatenate([robot_state, self._target_xyz], dtype=np.float32)

    def _compose_env_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(3)
        action = np.clip(action, -1.0, 1.0)
        delta_m = action * self.config.max_delta_m

        full_action = np.zeros(self.env.action_dim, dtype=np.float32)
        # OSC_POSE command ordering is [dx, dy, dz, dRx, dRy, dRz, (gripper)]
        full_action[:3] = delta_m / self.config.max_delta_m
        return full_action

    def seed(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return [seed]

    def reset(self):
        self._step_count = 0
        obs_dict = self.env.reset()
        eef = np.asarray(obs_dict.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32).ravel()
        self._target_xyz = self._sample_goal(eef)
        return self._flatten_obs(obs_dict)

    def step(self, action: np.ndarray):
        self._step_count += 1

        env_action = self._compose_env_action(action)
        obs_dict, _, _, info = self.env.step(env_action)

        eef = np.asarray(obs_dict.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32).ravel()
        action = np.asarray(action, dtype=np.float32).reshape(3)
        distance = float(np.linalg.norm(eef - self._target_xyz))
        success = bool(distance <= self.config.success_threshold_m)

        reward = (
            -self.config.distance_weight * distance
            -self.config.action_l2_weight * float(np.linalg.norm(action))
            + (self.config.success_bonus if success else 0.0)
        )

        done = bool(success or self._step_count >= self.config.horizon)
        obs = self._flatten_obs(obs_dict)

        out_info = dict(info or {})
        out_info.update(
            {
                "distance_to_goal": distance,
                "is_success": float(success),
                "target_xyz": self._target_xyz.copy(),
                "eef_xyz": eef.copy(),
            }
        )

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

    def get_diagnostics(self, paths: List[Dict[str, Any]]):
        if not paths:
            return {}

        successes = []
        final_distances = []
        for path in paths:
            env_infos = path.get("env_infos", [])
            if not env_infos:
                continue
            path_success = max(float(info.get("is_success", 0.0)) for info in env_infos)
            final_distance = float(env_infos[-1].get("distance_to_goal", np.nan))
            successes.append(path_success)
            final_distances.append(final_distance)

        if not successes:
            return {}

        return {
            "success_rate": float(np.mean(successes)),
            "final_distance_mean": float(np.nanmean(final_distances)),
        }

    def close(self):
        self.env.close()
