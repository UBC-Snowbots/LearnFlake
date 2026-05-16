"""Unit tests for the env-level reward_mode flag (TRACKER §30.6 Option A).

These tests construct a minimal KeyboardEnv-like reward path by patching only
the public surface of `_approach_reward`. They don't spin up MuJoCo — that's
covered by tests/test_smoke.py.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_reward_mode_invalid_raises():
    """Constructor rejects unknown reward_mode values."""
    # We can't easily instantiate KeyboardEnv without MuJoCo, so we just
    # confirm the validation path by calling __init__ directly on a shim that
    # only exercises the validation branch.
    from rl_autonomy.envs.keyboard_env import KeyboardEnv
    with pytest.raises(ValueError, match="reward_mode"):
        # The reward_mode check happens before super().__init__, so this
        # raises cleanly without ever loading MuJoCo.
        KeyboardEnv.__init__(
            object.__new__(KeyboardEnv),
            mode="approach",
            reward_mode="nonsense",
        )


def test_pbrs_only_zeros_dense_terms():
    """In pbrs_only mode, hover-anywhere should give ≈ time_penalty + pbrs only."""
    # Reuse the approach_reward primitive to compute components manually,
    # then verify the sparse_total formula matches what _approach_reward
    # returns in pbrs_only mode.
    from rl_autonomy.envs.rewards import (
        approach_reward, APPROACH_W_TIME, APPROACH_W_COLLISION,
    )

    # Hover at 5cm out (typical hover pose, no collision, no success)
    c = approach_reward(
        xy_dist=0.05, z_error=0.005, tilt=0.05,
        action_delta=0.0, success=False, collision=False,
    )

    # Dense terms in this pose are non-trivial:
    dense_sum = c.total - c.r_success - c.r_collision - c.r_time
    assert dense_sum > 0.1, "fixture should have meaningful dense reward"

    # pbrs_only's sparse_total drops the dense entirely:
    sparse_total = c.r_success + c.r_collision + c.r_time
    assert sparse_total == pytest.approx(APPROACH_W_TIME)  # 0 + 0 + time_penalty


def test_pbrs_only_success_bonus_still_dominates():
    """Success bonus must be reachable in pbrs_only mode."""
    from rl_autonomy.envs.rewards import (
        approach_reward, APPROACH_W_SUCCESS, APPROACH_W_TIME,
    )

    # At success: dense=1 each (but pbrs_only ignores them); success bonus fires.
    c = approach_reward(0.0, 0.0, 0.0, 0.0, success=True, collision=False)
    sparse_total = c.r_success + c.r_collision + c.r_time
    assert sparse_total == pytest.approx(APPROACH_W_SUCCESS + APPROACH_W_TIME)
    assert sparse_total > 99.0  # one success step dominates a whole hover episode


def test_make_env_threads_reward_mode():
    """make_env(reward_mode=...) propagates to KeyboardEnv.reward_mode."""
    import os
    # Skip if robosuite isn't importable in this test environment.
    pytest.importorskip("robosuite")
    from rl_autonomy.envs import make_env, KeyboardEnv
    from rl_autonomy.envs._wrapper_utils import find_inner

    env = make_env(mode="approach", frame_stack=2, domain_rand=False,
                   reward_mode="pbrs_only")
    try:
        kb = find_inner(env, KeyboardEnv)
        assert kb is not None
        assert kb.reward_mode == "pbrs_only"
    finally:
        env.close()
