"""Rewards stay in their declared range — no Bellman-target blowups."""
from __future__ import annotations

import warnings
import numpy as np
import pytest

warnings.filterwarnings("ignore")


def test_approach_reward_bounds_static_claim():
    """The TRACKER §5.2 claim: total ∈ [-0.2, 2.0]."""
    from rl_autonomy.envs.rewards import approach_reward_bounds
    lo, hi = approach_reward_bounds()
    assert lo == pytest.approx(-0.2)
    assert hi == pytest.approx(2.0)


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
    from rl_autonomy.envs.rewards import approach_reward
    c = approach_reward(0.0, 0.0, 0.0, 0.0, success=True, collision=False)
    # 0.5 + 0.3 + 0.15 + 0.05 (all dense terms = 1.0) + 1.0 success bonus
    assert c.total == pytest.approx(1.0 + 0.5 + 0.3 + 0.15 + 0.05)
    assert c.r_success == 1.0
    assert c.r_collision == 0.0


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
