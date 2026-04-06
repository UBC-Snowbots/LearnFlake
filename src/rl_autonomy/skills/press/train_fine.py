#!/usr/bin/env python3
"""
Press — Fine stage.

Precise center-of-key contact with force control. Requires 5 consecutive
hold steps and penalizes XY drift during press and force overshoot.
1-dim solenoid action, arm locked.

Usage:
    python -m skills.press.train_fine
    python -m skills.press.train_fine --eval checkpoints/press_fine/best_model.zip
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

from keyboard_env import FinePressEnv


class FinePressGymEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render: bool = False, random_key: bool = True):
        super().__init__()
        self._env = FinePressEnv(render=render, random_key=random_key)
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

        env = self._env
        eef_pos = env.sim.data.site_xpos[env._eef_site_id]
        key_pos = env.sim.data.body_xpos[env._key_body_ids[env._target_key]]
        force = float(np.linalg.norm(env.sim.data.cfrc_ext[env._actuator_body_id][3:]))

        info['xy_drift']      = float(np.linalg.norm(eef_pos[:2] - key_pos[:2]))
        info['contact_force']  = force
        info['contact_steps']  = env._contact_steps
        info['target_key']     = env._target_key

        if GYM_TYPE == "gymnasium":
            return obs, float(reward), done, False, info
        return obs, float(reward), done, info

    def render(self, mode='human'): self._env.render()
    def close(self): self._env.close()


def _print_eval(ep, n, reward, info, success):
    key   = info.get('target_key', '?')
    drift = info.get('xy_drift', float('nan'))
    force = info.get('contact_force', float('nan'))
    s = 'SUCCESS' if success else 'timeout'
    print(f"  ep {ep+1:3d}/{n}  key={key}  reward={reward:8.1f}  "
          f"drift={drift*1000:5.1f}mm  force={force:5.1f}N  {s}")


if __name__ == '__main__':
    run_cli(FinePressGymEnv, "press_fine", default_timesteps=120_000,
            sac_kwargs=dict(buffer_size=50_000, learning_starts=500),
            print_fn=_print_eval)
