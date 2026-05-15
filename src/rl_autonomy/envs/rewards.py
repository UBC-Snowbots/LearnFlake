"""Tolerance-based reward functions for Approach and Strike skills.

Built on `dm_control.utils.rewards.tolerance` (RoboPianist's reward primitive).
Each shaping term is bounded in [0, 1]. The success bonus is large (100) so
finishing the task and ending the episode strictly dominates accumulating
dense reward over a full 200-step horizon.

See TRACKER §5 for the design rationale and §23 for the 2026-04-29
mid-training fix that introduced the large success bonus + time penalty
after observing a hovering optimum with the original 1.0 / -0.2 weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# dm_control's tolerance is the only shaping primitive we use. Importing
# eagerly so missing-dep failures surface at import time, not at first
# step() — a far less debuggable place to fail.
from dm_control.utils import rewards as dm_rewards


# ---------------------------------------------------------------------------
# Approach
# ---------------------------------------------------------------------------

# Tolerance bounds — match the success criteria in TRACKER §14.
APPROACH_XY_BOUNDS = (0.0, 0.004)   # 4 mm
APPROACH_Z_BOUNDS = (0.0, 0.005)    # 5 mm
APPROACH_TILT_BOUNDS = (0.0, 0.087)  # ~5° in rad
APPROACH_SMOOTH_BOUNDS = (0.0, 0.05)  # action delta L2

# Margins — the distance from the bound at which the tolerance value drops
# to 0.1 (the dm_control default). Wider margin = gentler gradient.
APPROACH_XY_MARGIN = 0.05            # 5 cm — roughly half the keyboard span
APPROACH_Z_MARGIN = 0.04             # 4 cm
APPROACH_TILT_MARGIN = 0.30          # ~17°
APPROACH_SMOOTH_MARGIN = 0.5

# Component weights — sum approximately to 0.5 so the dense reward per step
# stays in [0, 0.5]. Halved from the original 1.0 sum (commit 9550935 → c1fb555)
# after the §25 collapse: with max dense = 1.0 the critic had to fit a 220-unit
# jump at success boundaries, which destabilized Q estimates under UTD=10.
# Halving the dense reward gives the critic a narrower target distribution
# (success-vs-hover gap reduced from ~220 to ~120) while preserving the
# gradient direction.
APPROACH_W_XY = 0.25
APPROACH_W_Z = 0.15
APPROACH_W_TILT = 0.075
APPROACH_W_SMOOTH = 0.025

# Sparse / penalty weights — applied on top of dense shaping.
#
# IMPORTANT: success bonus must dominate the per-episode dense-reward sum.
# Episode horizon = 200; max dense per step ≈ 0.5; so hover episode caps at
# ~100. Success bonus = 100 plus a per-step time penalty (-5/episode) makes
# the success policy strictly dominate hover: ~123 vs ~95 = 28-point margin.
#
# History: original v1 had 1.0 / -0.2 / no time penalty (hover beat success
# by 7×, §23). Bumped to 200 / -2 / -0.05 in §23. That fixed the hover
# optimum but the 220-unit critic target jump caused Q-overestimation
# cascades under UTD=10 (§25). Halved everything 2026-05-15.
APPROACH_W_SUCCESS = 100.0
APPROACH_W_COLLISION = -1.0
APPROACH_W_TIME = -0.025          # per-step penalty; -5 over a 200-step horizon


@dataclass
class ApproachRewardComponents:
    """Per-step breakdown for logging."""
    r_xy: float
    r_z: float
    r_tilt: float
    r_smooth: float
    r_success: float
    r_collision: float
    r_time: float
    total: float


def approach_reward(
    xy_dist: float,
    z_error: float,
    tilt: float,
    action_delta: float,
    success: bool,
    collision: bool,
) -> ApproachRewardComponents:
    """Compute the Approach reward per TRACKER §5.2.

    Args:
        xy_dist: ‖dxy‖ in metres, target offset in EEF frame XY.
        z_error: |z - hover_height| in metres.
        tilt: angle from vertical of the actuator push direction (rad).
        action_delta: ‖a_t - a_{t-1}‖ for smoothness term.
        success: True iff all three tolerances are simultaneously satisfied.
        collision: True iff the EEF or actuator collided with the keyboard surface.

    Returns:
        Component breakdown plus total. Total ∈ approximately [-2.01, 101].
        (Wide range because success bonus = 100 dominates; per-episode
        return is bounded ~ [-2.01·H, dense·H + 100].)
    """
    r_xy = float(dm_rewards.tolerance(
        xy_dist, bounds=APPROACH_XY_BOUNDS,
        margin=APPROACH_XY_MARGIN, sigmoid="gaussian",
    ))
    r_z = float(dm_rewards.tolerance(
        z_error, bounds=APPROACH_Z_BOUNDS,
        margin=APPROACH_Z_MARGIN, sigmoid="gaussian",
    ))
    r_tilt = float(dm_rewards.tolerance(
        tilt, bounds=APPROACH_TILT_BOUNDS,
        margin=APPROACH_TILT_MARGIN, sigmoid="gaussian",
    ))
    r_smooth = float(dm_rewards.tolerance(
        action_delta, bounds=APPROACH_SMOOTH_BOUNDS,
        margin=APPROACH_SMOOTH_MARGIN, sigmoid="linear",
    ))

    dense = (
        APPROACH_W_XY * r_xy
        + APPROACH_W_Z * r_z
        + APPROACH_W_TILT * r_tilt
        + APPROACH_W_SMOOTH * r_smooth
    )
    r_success = APPROACH_W_SUCCESS if success else 0.0
    r_collision = APPROACH_W_COLLISION if collision else 0.0
    r_time = APPROACH_W_TIME

    return ApproachRewardComponents(
        r_xy=r_xy,
        r_z=r_z,
        r_tilt=r_tilt,
        r_smooth=r_smooth,
        r_success=r_success,
        r_collision=r_collision,
        r_time=r_time,
        total=dense + r_success + r_collision + r_time,
    )


def approach_success(
    xy_dist: float,
    z_error: float,
    tilt: float,
) -> bool:
    """Approach success: simultaneous threshold on all three quantities."""
    return (
        xy_dist < APPROACH_XY_BOUNDS[1]
        and z_error < APPROACH_Z_BOUNDS[1]
        and tilt < APPROACH_TILT_BOUNDS[1]
    )


# ---------------------------------------------------------------------------
# Strike
# ---------------------------------------------------------------------------

@dataclass
class StrikeRewardComponents:
    r_contact: float
    r_extension: float
    r_terminal: float
    total: float


STRIKE_HOLD_STEPS = 3            # consecutive contact ticks for success
STRIKE_W_PER_CONTACT = 1.0       # per-tick contact reward
STRIKE_W_TERMINAL = 1.0          # added once on the final hold tick
STRIKE_W_EXTENSION = 0.1         # progress reward while extending without contact
STRIKE_MAX_EXTENSION = 0.04      # m, full solenoid stroke


def strike_reward(
    in_contact: bool,
    hold_counter: int,
    actuator_extension: float,
    extending: bool,
) -> tuple[StrikeRewardComponents, bool, int]:
    """Compute Strike reward per TRACKER §5.4.

    Args:
        in_contact: True iff (force > F_thresh) ∧ (|q̇_actuator| < V_thresh).
        hold_counter: number of consecutive contact ticks observed *before* this step.
        actuator_extension: current solenoid extension in metres ∈ [0, STRIKE_MAX_EXTENSION].
        extending: True iff the policy is commanding the solenoid to extend.

    Returns:
        (components, done, new_hold_counter). `done=True` iff hold_counter
        reaches STRIKE_HOLD_STEPS, which triggers the terminal bonus.
    """
    r_contact = 0.0
    r_extension = 0.0
    r_terminal = 0.0
    new_hold = hold_counter
    done = False

    if in_contact:
        new_hold = hold_counter + 1
        r_contact = STRIKE_W_PER_CONTACT
        if new_hold >= STRIKE_HOLD_STEPS:
            r_terminal = STRIKE_W_TERMINAL
            done = True
    else:
        new_hold = 0
        if extending:
            progress = min(1.0, max(0.0, actuator_extension / STRIKE_MAX_EXTENSION))
            r_extension = STRIKE_W_EXTENSION * progress

    components = StrikeRewardComponents(
        r_contact=r_contact,
        r_extension=r_extension,
        r_terminal=r_terminal,
        total=r_contact + r_extension + r_terminal,
    )
    return components, done, new_hold


# ---------------------------------------------------------------------------
# PBRS — potential-based reward shaping (TRACKER §5.3)
# ---------------------------------------------------------------------------

def approach_potential(xy_dist: float, z_error: float, tilt: float) -> float:
    """Φ(s) for Approach. Negative monotone in 'distance to goal'."""
    return -(xy_dist + 0.5 * z_error + 0.05 * tilt)


def pbrs_term(
    phi_s: float,
    phi_s_prime: float,
    gamma: float = 0.99,
) -> float:
    """Policy-invariant shaping term: γΦ(s') − Φ(s)."""
    return gamma * phi_s_prime - phi_s


# Convenience used by tests so the bounds claim in TRACKER §5.2 can be
# checked numerically.

def approach_reward_bounds() -> tuple[float, float]:
    """Return (min, max) of Approach total reward across the reachable input domain.

    Used by tests/test_reward_bounds.py.
    """
    # Worst case: collision + dense terms all zero (very far from goal) + time penalty
    lo = APPROACH_W_COLLISION + APPROACH_W_TIME
    # Best case: success bonus + every dense term = 1 + time penalty (always present)
    hi = (
        APPROACH_W_SUCCESS
        + APPROACH_W_XY
        + APPROACH_W_Z
        + APPROACH_W_TILT
        + APPROACH_W_SMOOTH
        + APPROACH_W_TIME
    )
    return lo, hi
