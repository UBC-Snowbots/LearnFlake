"""Tests for the shared DLS-IK expert (rl_autonomy.algos.expert_ik).

The expert is the M1 hand-coded controller, factored out of tools/gen_demos so
DAgger can query it at policy-visited states (TRACKER §35). These tests pin:
  - action shape / bounds / solenoid masking,
  - IKExpert.action == ik_step(kb, derived target) equivalence,
  - the controller is *competent* (closed-loop, it pulls XY toward the key) —
    this is the regression guard that the refactor didn't change the dynamics,
  - gen_demos delegates to the shared implementation (no silent divergence).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")


def _fresh_env(key: str = "j"):
    from rl_autonomy.envs import KeyboardEnv
    env = KeyboardEnv(mode="approach", horizon=200)
    env.reset()
    env.set_target_key(key)
    return env


def test_ik_expert_action_shape_bounds_and_solenoid():
    from rl_autonomy.algos.expert_ik import IKExpert
    kb = _fresh_env("j")
    a = IKExpert().action(kb)
    try:
        assert a.shape == (7,)
        assert a.dtype == np.float32
        assert np.all(a >= -1.0) and np.all(a <= 1.0)
        assert a[6] == -1.0, "Approach must hold the solenoid retracted"
    finally:
        kb.close()


def test_expert_action_matches_ik_step_with_derived_target():
    """IKExpert.action(kb) must equal ik_step(kb, expert.target_for(kb))."""
    from rl_autonomy.algos.expert_ik import IKExpert, ik_step
    kb = _fresh_env("h")
    try:
        expert = IKExpert()
        target = expert.target_for(kb)
        a_expert = expert.action(kb)
        a_direct = ik_step(kb, target)
        # Deterministic function of identical sim state → bit-identical.
        assert np.array_equal(a_expert, a_direct)
    finally:
        kb.close()


def test_target_for_is_above_key_by_hover_height():
    from rl_autonomy.algos.expert_ik import IKExpert
    kb = _fresh_env("g")
    try:
        target = IKExpert().target_for(kb)
        key_pos = kb.sim.data.body_xpos[kb._key_body_ids["g"]].copy()
        assert np.allclose(target[:2], key_pos[:2])
        assert np.isclose(target[2] - key_pos[2], kb.hover_height)
    finally:
        kb.close()


def test_expert_is_competent_through_full_pipeline():
    """The expert must succeed at a rate consistent with M1's documented ~44%.

    This is the behavioral regression guard for the gen_demos -> expert_ik
    refactor. It runs the expert through the **full wrapped env** the way it is
    actually used (gen_demos.run_one_episode, with ActionAdapter smoothing
    disabled) — NOT the bare KeyboardEnv, whose raw .step bypasses the
    ActionAdapter scaling the controller was tuned for. Key 'j' (home-row,
    deepest workspace) is M1's strongest. Over several seeds we require a
    nonzero success rate well clear of flakiness.
    """
    from rl_autonomy.envs import make_env, KeyboardEnv
    from rl_autonomy.envs._wrapper_utils import find_inner
    from rl_autonomy.envs.action_adapter import ActionAdapter
    from rl_autonomy.tools.gen_demos import run_one_episode

    env = make_env(mode="approach", frame_stack=3, domain_rand=False,
                   random_key=False, seed=0)
    kb = find_inner(env, KeyboardEnv)
    aa = find_inner(env, ActionAdapter)
    if aa is not None:
        aa.alpha = 0.0          # matches gen_demos demo-recording setup
    try:
        n_success = 0
        n_trials = 6
        for seed in range(n_trials):
            np.random.seed(seed)
            _, success, _ = run_one_episode(env, kb, "j", max_steps=200)
            n_success += int(success)
        # Documented M1 rate ~44%; require >=2/6 so the guard catches a broken
        # controller without being flaky on the expert's inherent ~44% rate.
        assert n_success >= 2, f"expert success {n_success}/{n_trials} on 'j' — controller likely broken"
    finally:
        env.close()


def test_gen_demos_delegates_to_shared_expert():
    """gen_demos._jacobian_step must BE the shared ik_step (no duplicate copy)."""
    from rl_autonomy.tools import gen_demos
    from rl_autonomy.algos.expert_ik import ik_step
    assert gen_demos._jacobian_step is ik_step
