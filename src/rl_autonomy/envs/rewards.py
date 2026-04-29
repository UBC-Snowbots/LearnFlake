"""Tolerance-based reward functions for Approach and Strike skills.

Built on `dm_control.utils.rewards.tolerance` (RoboPianist's reward primitive).
Each shaping term is bounded in [0, 1]; total dense reward is in [-1, 2].

See TRACKER §5 for the full design and rationale (PBRS layered separately
inside the env to preserve the optimal policy of the sparse-success MDP).
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

# Component weights — sum approximately to 1.0 so the dense reward stays in [0,1].
APPROACH_W_XY = 0.5
APPROACH_W_Z = 0.3
APPROACH_W_TILT = 0.15
APPROACH_W_SMOOTH = 0.05

# Sparse / penalty weights — applied on top of dense shaping.
APPROACH_W_SUCCESS = 1.0
APPROACH_W_COLLISION = -0.2


@dataclass
class ApproachRewardComponents:
    """Per-step breakdown for logging."""
    r_xy: float
    r_z: float
    r_tilt: float
    r_smooth: float
    r_success: float
    r_collision: float
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
        Component breakdown plus total. Total ∈ approximately [-0.2, 2.0].
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

    return ApproachRewardComponents(
        r_xy=r_xy,
        r_z=r_z,
        r_tilt=r_tilt,
        r_smooth=r_smooth,
        r_success=r_success,
        r_collision=r_collision,
        total=dense + r_success + r_collision,
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

    Used by tests/test_reward_bounds.py to assert TRACKER's "[-0.2, 2.0]" claim.
    """
    # Worst case: collision flag set, every dense term is 0 (very far from goal).
    lo = APPROACH_W_COLLISION
    # Best case: success bonus + every dense term = 1.
    hi = (
        APPROACH_W_SUCCESS
        + APPROACH_W_XY
        + APPROACH_W_Z
        + APPROACH_W_TILT
        + APPROACH_W_SMOOTH
    )
    return lo, hi
