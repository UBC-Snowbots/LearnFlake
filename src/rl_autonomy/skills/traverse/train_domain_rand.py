#!/usr/bin/env python3
"""
Traverse — Domain randomization stage.

Fine traverse with DomainRandWrapper for sim-to-real transfer.

Usage:
    python -m skills.traverse.train_domain_rand
    python -m skills.traverse.train_domain_rand --eval checkpoints/traverse_dr/best_model.zip
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

from keyboard_env import FineTraverseEnv, DomainRandWrapper


class DRTraverseGymEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render: bool = False, random_key: bool = True):
        super().__init__()
        base = FineTraverseEnv(render=render, random_key=random_key)
        self._dr = DomainRandWrapper(base)
        self._env = base

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(base.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self._dr.reset()
        obs = self._dr._flat_obs().astype(np.float32)
        return (obs, {}) if GYM_TYPE == "gymnasium" else obs

    def step(self, action):
        full_action = np.append(action, 0.0)
        _, reward, done, info = self._dr.step(full_action)
        obs = self._dr._flat_obs().astype(np.float32)

        env = self._env
        eef_pos = env.sim.data.site_xpos[env._eef_site_id]
        key_pos = env.sim.data.body_xpos[env._key_body_ids[env._target_key]]

        info['xy_dist']    = float(np.linalg.norm(eef_pos[:2] - key_pos[:2]))
        info['z_error']    = float(abs((eef_pos[2] - key_pos[2]) - FineTraverseEnv.HOVER_HEIGHT))
        info['source_key'] = env._source_key
        info['target_key'] = env._target_key

        if GYM_TYPE == "gymnasium":
            return obs, float(reward), done, False, info
        return obs, float(reward), done, info

    def render(self, mode='human'): self._dr.render()
    def close(self): self._dr.close()


def _print_eval(ep, n, reward, info, success):
    src  = info.get('source_key', '?')
    tgt  = info.get('target_key', '?')
    xy   = info.get('xy_dist', float('nan'))
    ze   = info.get('z_error', float('nan'))
    dr_s = info.get('dr_action_scale', 1.0)
    dr_l = info.get('dr_latency', 0)
    s = 'SUCCESS' if success else 'timeout'
    print(f"  ep {ep+1:3d}/{n}  {src}->{tgt}  reward={reward:7.1f}  "
          f"xy={xy*100:5.2f}cm  z_err={ze*100:5.2f}cm  "
          f"dr_scale={dr_s:.2f} lat={dr_l}  {s}")


if __name__ == '__main__':
    run_cli(DRTraverseGymEnv, "traverse_dr", default_timesteps=300_000,
            print_fn=_print_eval)
