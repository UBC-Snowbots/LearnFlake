"""rl_autonomy.envs — sim environment for the keyboard typing task.

Public API:
    KeyboardEnv:    single mode-switched env class (mode='approach'|'strike').
    make_env:       factory that returns an env wrapped with action smoothing,
                    obs normalization, frame stacking, and (optionally) DR.
    DomainRandWrapper:  per-episode physics & sensor randomization.
    KEYBOARD_LAYOUT: list of (name, x_local_m, y_local_m, width_u) tuples,
                    Redragon K552 TKL, 87 keys.
"""

from .keyboard_layout import (
    AVAILABLE_KEYS,
    KEYBOARD_LAYOUT,
    KEY_PITCH,
    KEY_HALF,
    KEY_H,
    ARUCO_NOISE_STD,
    ARUCO_VISIBLE_DIST,
    ARUCO_FALLOFF_DIST,
    ARUCO_MAX_TILT,
    CONTACT_FORCE_THRESHOLD,
    STALL_VEL_THRESHOLD,
)
from .keyboard_env import KeyboardEnv, make_env
from .domain_rand import DomainRandWrapper
from .normalizer import RunningMeanStd

__all__ = [
    "KeyboardEnv",
    "make_env",
    "DomainRandWrapper",
    "RunningMeanStd",
    "KEYBOARD_LAYOUT",
    "AVAILABLE_KEYS",
    "KEY_PITCH",
    "KEY_HALF",
    "KEY_H",
    "ARUCO_NOISE_STD",
    "ARUCO_VISIBLE_DIST",
    "ARUCO_FALLOFF_DIST",
    "ARUCO_MAX_TILT",
    "CONTACT_FORCE_THRESHOLD",
    "STALL_VEL_THRESHOLD",
]
