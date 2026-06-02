"""Tests for the key-aware base-init pose (TRACKER §36).

The single ABOVE_KEYBOARD_QPOS init leaves the left-side keys at the arm's
workspace edge, where the DLS-IK expert misses by 5-22mm. key_aware_init
pre-rotates the base (shoulder) toward the target key's column so the expert
(and the DAgger policy it teaches) can reach them. These tests pin the offset
logic and the behavioral win.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")


def test_init_qpos_off_is_plain_pose():
    from rl_autonomy.envs.keyboard_env import KeyboardEnv, ABOVE_KEYBOARD_QPOS
    env = KeyboardEnv(mode="approach", horizon=20, key_aware_init=False)
    try:
        for key in ("a", "l", "g"):
            assert np.array_equal(env.init_qpos_for_key(key), ABOVE_KEYBOARD_QPOS)
    finally:
        env.close()


def test_init_qpos_left_key_positive_offset_right_key_zero():
    from rl_autonomy.envs.keyboard_env import (
        KeyboardEnv, ABOVE_KEYBOARD_QPOS, KEY_AWARE_SHOULDER_MAX,
    )
    env = KeyboardEnv(mode="approach", horizon=20, key_aware_init=True)
    try:
        base_shoulder = ABOVE_KEYBOARD_QPOS[0]
        # 'a' is far left (y_local ~ +0.13) -> positive shoulder offset.
        a = env.init_qpos_for_key("a")
        assert a[0] > base_shoulder + 0.2
        assert a[0] - base_shoulder <= KEY_AWARE_SHOULDER_MAX + 1e-9
        # 'l' is right-of-centre (y_local < 0) -> clamped to zero offset.
        l = env.init_qpos_for_key("l")
        assert np.isclose(l[0], base_shoulder)
        # only the shoulder (joint 0) changes.
        assert np.array_equal(a[1:], ABOVE_KEYBOARD_QPOS[1:])
    finally:
        env.close()


def test_make_env_threads_key_aware_init():
    from rl_autonomy.envs import make_env, KeyboardEnv
    from rl_autonomy.envs._wrapper_utils import find_inner
    env = make_env(mode="approach", frame_stack=2, domain_rand=False,
                   random_key=False, key_aware_init=True)
    try:
        kb = find_inner(env, KeyboardEnv)
        assert kb is not None and kb.key_aware_init is True
    finally:
        env.close()


def test_key_aware_init_closes_left_key_xy_gap():
    """key-aware init must close the IK expert's XY reach on a left key.

    The flag fixes the workspace-edge XY miss (16.6mm -> <1mm for 'a'); that XY
    reduction is the robust, meaningful effect. (Full 3-tolerance success
    additionally needs z/tilt alignment, which is the controller's job and a
    noisier knife-edge, so we test the XY gap the init actually addresses.)
    """
    from rl_autonomy.envs import make_env, KeyboardEnv
    from rl_autonomy.envs._wrapper_utils import find_inner
    from rl_autonomy.envs.action_adapter import ActionAdapter
    from rl_autonomy.algos import IKExpert

    def best_xy(key_aware: bool, key: str = "a") -> float:
        env = make_env(mode="approach", frame_stack=3, domain_rand=False,
                       random_key=False, key_aware_init=key_aware)
        kb = find_inner(env, KeyboardEnv)
        aa = find_inner(env, ActionAdapter)
        if aa is not None:
            aa.alpha = 0.0
        expert = IKExpert()
        best = 9.9
        try:
            for seed in range(2):
                np.random.seed(seed)
                kb.set_target_key(key); env.reset(); kb.set_target_key(key)
                for _ in range(200):
                    _, _, term, trunc, info = env.step(expert.action(kb))
                    xy, _, _ = kb._compute_approach_errors()
                    best = min(best, xy)
                    if term or trunc:
                        break
        finally:
            env.close()
        return best

    on = best_xy(True)
    off = best_xy(False)
    assert on < 0.005, f"key-aware init should reach 'a' within 5mm, got {on*1000:.1f}mm"
    assert off > 0.010, f"without it 'a' should miss by >10mm, got {off*1000:.1f}mm"
    assert on < 0.5 * off                              # at least halves the gap
