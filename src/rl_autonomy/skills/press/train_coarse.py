#!/usr/bin/env python3
"""
Press — Coarse stage.

Extend solenoid to make contact with the target key and hold for 3 steps.
Arm joints locked to 0. 1-dim solenoid action space.

Usage:
    python -m skills.press.train_coarse
    python -m skills.press.train_coarse --eval checkpoints/press_coarse/best_model.zip
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

from keyboard_env import PressKeyEnv


class CoarsePressGymEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render: bool = False, random_key: bool = True):
        super().__init__()
        self._env = PressKeyEnv(render=render, random_key=random_key)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._env.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self._env.reset()
        obs = self._env._flat_obs().astype(np.float32)
        return (obs, {}) if GYM_TYPE == "gymnasium" else obs

    def step(self, action):
        full_action = np.zeros(self._env.action_dim)
        full_action[-1] = float(action[0])
        _, reward, done, info = self._env.step(full_action)
        obs = self._env._flat_obs().astype(np.float32)

        info['actuator_pos'] = float(self._env.sim.data.qpos[self._env._actuator_qpos_addr])
        info['contact_steps'] = self._env._contact_steps
        info['target_key'] = self._env._target_key

        if GYM_TYPE == "gymnasium":
            return obs, float(reward), done, False, info
        return obs, float(reward), done, info

    def render(self, mode='human'): self._env.render()
    def close(self): self._env.close()


def _print_eval(ep, n, reward, info, success):
    key = info.get('target_key', '?')
    s = 'SUCCESS' if success else 'timeout'
    print(f"  ep {ep+1:3d}/{n}  key={key}  reward={reward:8.1f}  {s}")


if __name__ == '__main__':
    run_cli(CoarsePressGymEnv, "press_coarse", default_timesteps=90_000,
            sac_kwargs=dict(buffer_size=50_000, learning_starts=500),
            print_fn=_print_eval)
