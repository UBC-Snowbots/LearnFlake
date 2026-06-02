"""Tests for the residual-on-IK wrapper + factory (TRACKER §38)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")


def test_residual_env_builds_with_right_spaces():
    from rl_autonomy.envs import make_residual_env
    env = make_residual_env(tube=0.15, random_key=False)
    try:
        obs, _ = env.reset()
        assert "actor" in obs and "critic" in obs
        assert obs["actor"].shape[0] % 36 == 0      # stacked 36-D actor frames
        assert obs["critic"].shape[0] == 38
        assert env.action_space.shape == (7,)
        # one step with a real residual works
        obs, r, term, trunc, info = env.step(np.zeros(7, dtype=np.float32))
        assert "residual_frac" in info
    finally:
        env.close()


def test_zero_residual_drives_like_ik():
    """With residual=0, a_final == a_ik, so the arm should approach a reachable
    key just like the bare IK expert (sanity that the IK base is applied)."""
    from rl_autonomy.envs import make_residual_env, KeyboardEnv
    from rl_autonomy.envs._wrapper_utils import find_inner
    env = make_residual_env(tube=0.15, random_key=False, keyboard_offset=(-0.10, -0.10))
    kb = find_inner(env, KeyboardEnv)
    try:
        np.random.seed(0)
        kb.set_target_key("j"); env.reset(); kb.set_target_key("j")
        xy0, _, _ = kb._compute_approach_errors()
        for _ in range(120):
            _, _, term, trunc, _ = env.step(np.zeros(7, dtype=np.float32))
            if term or trunc:
                break
        xy1, _, _ = kb._compute_approach_errors()
        assert xy1 < 0.5 * xy0, f"zero-residual env should drive like IK: {xy0:.3f}->{xy1:.3f}"
    finally:
        env.close()


def test_tube_caps_joint_deviation_and_masks_solenoid():
    """a_final[:6] must stay within `tube` of the IK action; solenoid == -1."""
    from rl_autonomy.envs import make_residual_env, KeyboardEnv, ResidualIKWrapper
    from rl_autonomy.envs._wrapper_utils import find_inner
    from rl_autonomy.algos import IKExpert
    tube = 0.15
    env = make_residual_env(tube=tube, random_key=False, keyboard_offset=(-0.10, -0.10))
    kb = find_inner(env, KeyboardEnv)
    res_wrap = find_inner(env, ResidualIKWrapper)
    try:
        np.random.seed(1)
        kb.set_target_key("k"); env.reset(); kb.set_target_key("k")
        a_ik = IKExpert().action(kb)                 # IK action at the current (post-reset) state

        captured = {}
        inner = res_wrap.env                          # KeyboardGymEnv
        orig_step = inner.step
        def rec(a):
            captured["a"] = np.asarray(a, dtype=np.float32).copy()
            return orig_step(a)
        inner.step = rec
        # First step after reset: ActionAdapter's filter initialises to the input
        # (no smoothing yet), so the residual reaches the wrapper unsmoothed.
        env.step(np.ones(7, dtype=np.float32))
        inner.step = orig_step

        a_final = captured["a"]
        # joint dims: within tube of the IK action (clipping only shrinks the gap)
        assert np.all(np.abs(a_final[:6] - a_ik[:6]) <= tube + 1e-5)
        # and the residual actually moved the joints (not a no-op), up to the cap
        assert np.any(np.abs(a_final[:6] - a_ik[:6]) > 1e-4)
        # solenoid locked retracted
        assert a_final[6] == -1.0
    finally:
        env.close()


def test_tube_must_be_positive():
    from rl_autonomy.envs import make_residual_env
    with pytest.raises(ValueError):
        make_residual_env(tube=0.0)


def test_chain_mode_switch_produces_strike_action():
    """True-chain mechanics (TRACKER §39): ActionAdapter.set_mode('strike') +
    ResidualIKWrapper.bypass=True must deliver [0,0,0,0,0,0,solenoid] to the
    inner env (arm frozen, solenoid passes) — no IK added."""
    from rl_autonomy.envs import make_residual_env, ResidualIKWrapper
    from rl_autonomy.envs._wrapper_utils import find_inner
    from rl_autonomy.envs.action_adapter import ActionAdapter
    env = make_residual_env(tube=0.15, random_key=False, keyboard_offset=(-0.10, -0.10))
    res = find_inner(env, ResidualIKWrapper)
    aa = find_inner(env, ActionAdapter)
    try:
        env.reset()
        aa.set_mode("strike")
        res.bypass = True
        captured = {}
        inner = res.env
        orig = inner.step
        def rec(a):
            captured["a"] = np.asarray(a, dtype=np.float32).copy()
            return orig(a)
        inner.step = rec
        env.step(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0], dtype=np.float32))
        inner.step = orig
        a = captured["a"]
        assert np.allclose(a[:6], 0.0), f"joints must be frozen in strike, got {a[:6]}"
        assert a[6] == 1.0, "solenoid action must pass through in strike+bypass"
    finally:
        env.close()


def test_action_adapter_set_mode_validates():
    from rl_autonomy.envs import make_env
    from rl_autonomy.envs._wrapper_utils import find_inner
    from rl_autonomy.envs.action_adapter import ActionAdapter
    env = make_env(mode="approach", frame_stack=2, domain_rand=False, random_key=False)
    aa = find_inner(env, ActionAdapter)
    try:
        aa.set_mode("strike"); assert aa.mode == "strike"
        aa.set_mode("approach"); assert aa.mode == "approach"
        with pytest.raises(ValueError):
            aa.set_mode("nonsense")
    finally:
        env.close()
