#!/usr/bin/env python3
"""
Traverse — Coarse stage.

Move laterally from one key to another at hover height. Success within 3cm XY.
6-dim arm action, solenoid locked retracted.

Usage:
    python -m skills.traverse.train_coarse
    python -m skills.traverse.train_coarse --eval checkpoints/traverse_coarse/best_model.zip
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from skills.train_utils import GYM_TYPE, run_cli

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from keyboard_env import TraverseEnv


class CoarseTraverseGymEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render: bool = False, random_key: bool = True):
        super().__init__()
        self._env = TraverseEnv(render=render, random_key=random_key)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._env.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self._env.reset()
        obs = self._env._flat_obs().astype(np.float32)
        return (obs, {}) if GYM_TYPE == "gymnasium" else obs

    def step(self, action):
        full_action = np.append(action, 0.0)
        _, reward, done, info = self._env.step(full_action)
        obs = self._env._flat_obs().astype(np.float32)

        env = self._env
        eef_pos = env.sim.data.site_xpos[env._eef_site_id]
        key_pos = env.sim.data.body_xpos[env._key_body_ids[env._target_key]]
        pf = env.robots[0].robot_model.naming_prefix
        eef_quat = env._get_observations(force_update=False).get(
            f"{pf}eef_quat", np.array([1, 0, 0, 0]))

        info['xy_dist']    = float(np.linalg.norm(eef_pos[:2] - key_pos[:2]))
        info['z_error']    = float(abs((eef_pos[2] - key_pos[2]) - TraverseEnv.HOVER_HEIGHT))
        info['tilt_rad']   = float(TraverseEnv._eef_tilt_from_vertical(eef_quat))
        info['source_key'] = env._source_key
        info['target_key'] = env._target_key

        if GYM_TYPE == "gymnasium":
            return obs, float(reward), done, False, info
        return obs, float(reward), done, info

    def render(self, mode='human'): self._env.render()
    def close(self): self._env.close()


def _print_eval(ep, n, reward, info, success):
    src = info.get('source_key', '?')
    tgt = info.get('target_key', '?')
    xy  = info.get('xy_dist', float('nan'))
    ze  = info.get('z_error', float('nan'))
    s = 'SUCCESS' if success else 'timeout'
    print(f"  ep {ep+1:3d}/{n}  {src}->{tgt}  reward={reward:7.1f}  "
          f"xy={xy*100:5.2f}cm  z_err={ze*100:5.2f}cm  {s}")


if __name__ == '__main__':
    run_cli(CoarseTraverseGymEnv, "traverse_coarse", default_timesteps=200_000,
            print_fn=_print_eval)
