"""Profile a short Approach training run to find the actual bottleneck.

Splits time across: env.step (MuJoCo physics), obs construction,
replay buffer, and the GPU train_step. Writes a cumulative profile to
/tmp/profile.pstats; prints the top 30 cumulative + top 30 tottime entries.
"""
from __future__ import annotations

import cProfile
import pstats
import sys
import warnings

import numpy as np
import torch

from rl_autonomy.algos import RLPDSAC, RLPDConfig
from rl_autonomy.curricula import KeyPhaseCurriculum
from rl_autonomy.envs import make_env
from rl_autonomy.scripts.train_approach import CurriculumEnv, _share_normalizer

warnings.filterwarnings("ignore")


def run():
    np.random.seed(0)
    torch.manual_seed(0)
    train_env = make_env(mode="approach", frame_stack=3, domain_rand=False, seed=0)
    eval_env = make_env(mode="approach", frame_stack=3, domain_rand=False, seed=1)
    train_env = CurriculumEnv(train_env, KeyPhaseCurriculum(seed=0))
    eval_env = CurriculumEnv(eval_env, KeyPhaseCurriculum(seed=2))
    _share_normalizer(train_env, eval_env)

    cfg = RLPDConfig(update_to_data=2, warmstart_steps=200, batch_size=256,
                     buffer_size=20_000, demo_buffer_size=1,
                     demo_fraction_init=0.0, demo_fraction_final=0.0)
    agent = RLPDSAC(env=train_env, config=cfg, eval_env=eval_env)
    # Tight loop: 2k steps. Long enough for stable timing, short enough to
    # finish profiling quickly.
    agent.learn(total_timesteps=2_000, log_every=10_000, eval_every=0)


if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()
    pr.dump_stats("/tmp/profile.pstats")

    st = pstats.Stats("/tmp/profile.pstats").strip_dirs()
    print("\n=== TOP 30 by CUMULATIVE time ===")
    st.sort_stats("cumulative").print_stats(30)
    print("\n=== TOP 30 by TOTAL time (excluding callees) ===")
    st.sort_stats("tottime").print_stats(30)
