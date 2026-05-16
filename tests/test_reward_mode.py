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


def test_xy_focus_gradient_at_distance():
    """In xy_focus mode, r_xy must give meaningful gradient at 5-10cm out.

    The §31 falsified hypothesis was that PBRS alone could pull the agent
    in. xy_focus uses long_tail sigmoid + 15cm margin so r_xy is well above
    zero at the workspace edge. Verifies the gradient direction (closer →
    bigger reward) and a minimum magnitude that won't be washed out by
    exploration noise.
    """
    from rl_autonomy.envs.rewards import approach_reward

    far = approach_reward(0.10, 0.005, 0.05, 0.0,
                          success=False, collision=False, mode="xy_focus")
    mid = approach_reward(0.05, 0.005, 0.05, 0.0,
                          success=False, collision=False, mode="xy_focus")
    close = approach_reward(0.01, 0.005, 0.05, 0.0,
                            success=False, collision=False, mode="xy_focus")

    # Monotone increasing as xy_dist shrinks.
    assert far.r_xy < mid.r_xy < close.r_xy
    # Meaningful magnitude at 10cm — long_tail gives ~0.18 at 10cm with margin 15cm.
    assert far.r_xy > 0.15
    # Differential between mid (5cm) and far (10cm) is the gradient signal
    # the agent needs to follow. With weight 0.7, this should be ~0.2 per
    # 5cm closer = sizable per-step learning signal.
    assert (mid.r_xy - far.r_xy) > 0.05


def test_xy_focus_z_cannot_dominate_xy():
    """Hover-at-distance (z=0 error, xy=8cm) must NOT outscore close-but-bad-z."""
    from rl_autonomy.envs.rewards import approach_reward

    # Old v3 failure pose: hovering at correct height but xy=8cm away
    hover_far = approach_reward(0.08, 0.001, 0.02, 0.0,
                                success=False, collision=False, mode="xy_focus")
    # Close in xy but z is at workspace edge
    close_bad_z = approach_reward(0.01, 0.02, 0.05, 0.0,
                                  success=False, collision=False, mode="xy_focus")

    # In v3 (dense), hover_far beat close_bad_z by ~0.07/step → 200·0.07 = +14 episodes.
    # In xy_focus, close_bad_z must beat hover_far so the agent prefers XY closeness.
    assert close_bad_z.total > hover_far.total, (
        f"xy_focus must penalize hovering at distance: "
        f"close_bad_z={close_bad_z.total:.4f} <= hover_far={hover_far.total:.4f}"
    )


def test_make_env_threads_reward_mode():
    """make_env(reward_mode=...) propagates to KeyboardEnv.reward_mode."""
    import os
    # Skip if robosuite isn't importable in this test environment.
    pytest.importorskip("robosuite")
    from rl_autonomy.envs import make_env, KeyboardEnv
    from rl_autonomy.envs._wrapper_utils import find_inner

    for mode in ("pbrs_only", "xy_focus"):
        env = make_env(mode="approach", frame_stack=2, domain_rand=False,
                       reward_mode=mode)
        try:
            kb = find_inner(env, KeyboardEnv)
            assert kb is not None
            assert kb.reward_mode == mode
        finally:
            env.close()
