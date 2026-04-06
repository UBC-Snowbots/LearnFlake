#!/usr/bin/env python3
"""
Press — Domain randomization stage.

Fine press with DomainRandWrapper for sim-to-real transfer.

Usage:
    python -m skills.press.train_domain_rand
    python -m skills.press.train_domain_rand --eval checkpoints/press_dr/best_model.zip
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

from keyboard_env import FinePressEnv, DomainRandWrapper


class DRPressGymEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render: bool = False, random_key: bool = True):
        super().__init__()
        base = FinePressEnv(render=render, random_key=random_key)
        self._dr = DomainRandWrapper(base)
        self._env = base

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(base.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self._dr.reset()
        obs = self._dr._flat_obs().astype(np.float32)
        return (obs, {}) if GYM_TYPE == "gymnasium" else obs

    def step(self, action):
        full_action = np.zeros(self._env.action_dim)
        full_action[-1] = float(action[0])
        _, reward, done, info = self._dr.step(full_action)
        obs = self._dr._flat_obs().astype(np.float32)

        info['contact_steps'] = self._env._contact_steps
        info['target_key'] = self._env._target_key

        if GYM_TYPE == "gymnasium":
            return obs, float(reward), done, False, info
        return obs, float(reward), done, info

    def render(self, mode='human'): self._dr.render()
    def close(self): self._dr.close()


def _print_eval(ep, n, reward, info, success):
    key  = info.get('target_key', '?')
    dr_s = info.get('dr_action_scale', 1.0)
    dr_l = info.get('dr_latency', 0)
    s = 'SUCCESS' if success else 'timeout'
    print(f"  ep {ep+1:3d}/{n}  key={key}  reward={reward:8.1f}  "
          f"dr_scale={dr_s:.2f} lat={dr_l}  {s}")


if __name__ == '__main__':
    run_cli(DRPressGymEnv, "press_dr", default_timesteps=150_000,
            sac_kwargs=dict(buffer_size=50_000, learning_starts=500),
            print_fn=_print_eval)
