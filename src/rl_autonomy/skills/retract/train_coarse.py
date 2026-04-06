#!/usr/bin/env python3
"""
Retract — Coarse stage.

Retract solenoid and lift EEF back to hover height after pressing.
7-dim action (6 arm joints + 1 solenoid).

Usage:
    python -m skills.retract.train_coarse
    python -m skills.retract.train_coarse --eval checkpoints/retract_coarse/best_model.zip
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

from keyboard_env import RetractEnv


class CoarseRetractGymEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render: bool = False, random_key: bool = True):
        super().__init__()
        self._env = RetractEnv(render=render, random_key=random_key)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._env.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self._env.reset()
        obs = self._env._flat_obs().astype(np.float32)
        return (obs, {}) if GYM_TYPE == "gymnasium" else obs

    def step(self, action):
        _, reward, done, info = self._env.step(action)
        obs = self._env._flat_obs().astype(np.float32)

        env = self._env
        act_pos = float(env.sim.data.qpos[env._actuator_qpos_addr])
        eef_pos = env.sim.data.site_xpos[env._eef_site_id]
        key_pos = env.sim.data.body_xpos[env._key_body_ids[env._target_key]]

        info['actuator_pos'] = act_pos
        info['z_error'] = float(abs((eef_pos[2] - key_pos[2]) - RetractEnv.HOVER_HEIGHT))
        info['target_key'] = env._target_key

        if GYM_TYPE == "gymnasium":
            return obs, float(reward), done, False, info
        return obs, float(reward), done, info

    def render(self, mode='human'): self._env.render()
    def close(self): self._env.close()


def _print_eval(ep, n, reward, info, success):
    key = info.get('target_key', '?')
    act = info.get('actuator_pos', float('nan'))
    ze  = info.get('z_error', float('nan'))
    s = 'SUCCESS' if success else 'timeout'
    print(f"  ep {ep+1:3d}/{n}  key={key}  reward={reward:7.1f}  "
          f"act={act*100:5.2f}cm  z_err={ze*100:5.2f}cm  {s}")


if __name__ == '__main__':
    run_cli(CoarseRetractGymEnv, "retract_coarse", default_timesteps=100_000,
            print_fn=_print_eval)
