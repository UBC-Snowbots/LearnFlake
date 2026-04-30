"""M2 acceptance (TRACKER §15) — RLPD-SAC on Pendulum-v1.

Algorithm-correctness sanity check. The asymmetric critic is "fed" the same
obs as the actor on Pendulum (no privileged state available), which is just
RLPD-SAC. Goal: ≥ -150 mean episode return in 50,000 env steps within
~5 minutes wallclock on this machine.

If this fails the algo has a bug; do NOT proceed to the keyboard env.

Run inside rover_gpu:
    python3 -m rl_autonomy.tools.m2_pendulum [--steps 50000]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import gymnasium as gym

from rl_autonomy.algos import RLPDSAC, RLPDConfig


class _ActorCriticDictWrapper(gym.Wrapper):
    """Project Box → Dict({'actor', 'critic'}) so RLPDSAC's API is satisfied.

    For Pendulum the actor and critic see identical observations; this is
    the symmetric AC degenerate case of our framework.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("only Box obs supported by this wrapper")
        self.observation_space = gym.spaces.Dict({
            "actor": env.observation_space,
            "critic": env.observation_space,
        })

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return {"actor": obs.astype(np.float32), "critic": obs.astype(np.float32)}, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        d = {"actor": obs.astype(np.float32), "critic": obs.astype(np.float32)}
        return d, r, term, trunc, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--device", default=None, help="cuda or cpu (auto if unset)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[M2] RLPD-SAC on Pendulum-v1, target ≥-150 in {args.steps} steps")
    env = gym.make("Pendulum-v1")
    env.action_space.seed(args.seed)
    env = _ActorCriticDictWrapper(env)

    eval_env = gym.make("Pendulum-v1")
    eval_env = _ActorCriticDictWrapper(eval_env)

    cfg = RLPDConfig(
        # Pendulum baseline: match SB3's defaults as closely as we can while
        # keeping the RLPD architectural differences (LayerNorm critic, GELU,
        # wider critic). target_entropy_scale=1.0 → target_entropy=-1 matches
        # standard SAC on Pendulum (our default 0.5 is the RoboPianist setting,
        # tuned for high-action-dim tasks).
        update_to_data=1,
        target_entropy_scale=1.0,
        warmstart_steps=1_000,
        batch_size=256,
        buffer_size=200_000,
        demo_buffer_size=1,                   # no demos
        demo_fraction_init=0.0,
        demo_fraction_final=0.0,
        seed=args.seed,
    )
    agent = RLPDSAC(env=env, config=cfg, device=args.device, eval_env=eval_env)
    print(f"[M2] device={agent.device}  obs_dim={agent.actor_dim}  action_dim={agent.action_dim}")

    t0 = time.time()
    history = agent.learn(total_timesteps=args.steps, log_every=2_000, eval_every=10_000)
    wall = time.time() - t0

    rets = history["episode_return"]
    if len(rets) < 5:
        print(f"[M2] FAIL — only {len(rets)} episodes completed in {args.steps} steps")
        return 1

    last_n = min(20, len(rets))
    mean_ret = float(np.mean(rets[-last_n:]))
    print(f"[M2] wallclock={wall:.1f}s, episodes={len(rets)}, "
          f"last-{last_n} mean return={mean_ret:.2f}")

    if mean_ret >= -150.0:
        print("[M2] PASSED — RLPD-SAC trains correctly on Pendulum.")
        return 0
    else:
        print(f"[M2] FAILED — mean return {mean_ret:.2f} < -150.0 target.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
