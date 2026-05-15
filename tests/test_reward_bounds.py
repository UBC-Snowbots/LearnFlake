"""Rewards stay in their declared range — no Bellman-target blowups."""
from __future__ import annotations

import warnings
import numpy as np
import pytest

warnings.filterwarnings("ignore")


def test_approach_reward_bounds_static_claim():
    """Per-step bounds after §25 collapse fix: total ∈ ~[-1.025, 100.475].

    History:
      §5.2 initial: [-0.2, 2.0]  (hover beat success, fixed §23)
      §23 fix:     [-2.05, 200.95]  (success now dominates, but critic overestimated)
      §25 fix:     [-1.025, 100.475]  (halved everything; narrower critic target)
    """
    from rl_autonomy.envs.rewards import approach_reward_bounds
    lo, hi = approach_reward_bounds()
    assert lo == pytest.approx(-1.025)
    assert hi == pytest.approx(100.475)


def test_approach_reward_within_bounds_on_random_inputs():
    from rl_autonomy.envs.rewards import approach_reward, approach_reward_bounds
    lo, hi = approach_reward_bounds()
    rng = np.random.default_rng(0)
    for _ in range(2000):
        xy = float(rng.uniform(0, 1.0))           # 1 m max XY error
        z = float(rng.uniform(0, 0.5))            # 50 cm max Z error
        tilt = float(rng.uniform(0, np.pi))       # full hemisphere
        delta = float(rng.uniform(0, 2.0))
        success = bool(rng.integers(2))
        collision = bool(rng.integers(2))
        c = approach_reward(xy, z, tilt, delta, success, collision)
        assert lo - 1e-6 <= c.total <= hi + 1e-6, (
            f"reward {c.total} outside [{lo}, {hi}] for "
            f"xy={xy} z={z} tilt={tilt} delta={delta} success={success} collision={collision}"
        )


def test_approach_reward_perfect_state():
    from rl_autonomy.envs.rewards import (
        approach_reward, APPROACH_W_TIME, APPROACH_W_SUCCESS,
        APPROACH_W_XY, APPROACH_W_Z, APPROACH_W_TILT, APPROACH_W_SMOOTH,
    )
    c = approach_reward(0.0, 0.0, 0.0, 0.0, success=True, collision=False)
    # all dense terms = 1.0 (each tolerance() returns 1 inside its bound),
    # weighted by their W_* coefficients, plus success bonus, plus per-step
    # time penalty. Computed symbolically so the test survives future
    # reward re-tuning without lying.
    dense_sum = (
        APPROACH_W_XY + APPROACH_W_Z + APPROACH_W_TILT + APPROACH_W_SMOOTH
    )
    assert c.total == pytest.approx(APPROACH_W_SUCCESS + dense_sum + APPROACH_W_TIME)
    assert c.r_success == APPROACH_W_SUCCESS
    assert c.r_collision == 0.0


def test_approach_reward_success_dominates_dense_episode():
    """Per TRACKER §23: a 200-step hovering episode at near-max dense must
    yield strictly less reward than a single-step success episode."""
    from rl_autonomy.envs.rewards import approach_reward, APPROACH_W_TIME
    horizon = 200
    # Hover at the bound: dense ≈ 1.0 per step, no success, no collision
    hover_per_step = approach_reward(0.005, 0.006, 0.1, 0.0,
                                     success=False, collision=False).total
    hover_episode = horizon * hover_per_step
    # Success on the very first step (hypothetical lower bound on success ep)
    success_step = approach_reward(0.0, 0.0, 0.0, 0.0,
                                   success=True, collision=False).total
    assert success_step > hover_episode, (
        f"hovering for full episode ({hover_episode:.1f}) outweighs single-step success "
        f"({success_step:.1f}); reward shape still buggy"
    )


def test_approach_reward_collision_penalizes():
    from rl_autonomy.envs.rewards import approach_reward
    no_col = approach_reward(0.5, 0.1, 0.5, 0.0, success=False, collision=False)
    col = approach_reward(0.5, 0.1, 0.5, 0.0, success=False, collision=True)
    assert col.total < no_col.total


def test_strike_reward_terminal_only_at_hold():
    from rl_autonomy.envs.rewards import strike_reward, STRIKE_HOLD_STEPS
    # Two contact ticks: per-tick reward but no terminal yet
    c, done, hold = strike_reward(in_contact=True, hold_counter=0,
                                  actuator_extension=0.04, extending=True)
    assert not done and hold == 1
    assert c.r_terminal == 0.0
    c, done, hold = strike_reward(in_contact=True, hold_counter=hold,
                                  actuator_extension=0.04, extending=True)
    assert not done and hold == 2
    # Third contact tick triggers terminal
    c, done, hold = strike_reward(in_contact=True, hold_counter=hold,
                                  actuator_extension=0.04, extending=True)
    assert done
    assert hold == STRIKE_HOLD_STEPS
    assert c.r_terminal > 0


def test_strike_reward_resets_counter_on_loss_of_contact():
    from rl_autonomy.envs.rewards import strike_reward
    _, _, hold = strike_reward(in_contact=True, hold_counter=0,
                               actuator_extension=0.04, extending=True)
    _, _, hold = strike_reward(in_contact=True, hold_counter=hold,
                               actuator_extension=0.04, extending=True)
    # Lose contact mid-press
    c, done, hold = strike_reward(in_contact=False, hold_counter=hold,
                                  actuator_extension=0.04, extending=True)
    assert hold == 0
    assert c.r_contact == 0.0
    assert c.r_extension > 0  # still rewarded for partial extension
    assert not done


def test_pbrs_term_zero_when_potential_unchanged():
    from rl_autonomy.envs.rewards import pbrs_term
    assert pbrs_term(phi_s=-0.5, phi_s_prime=-0.5 / 0.99, gamma=0.99) == pytest.approx(0.0, abs=1e-12)
