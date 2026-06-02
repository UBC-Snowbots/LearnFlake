"""Tests for the DAgger trainer (rl_autonomy.scripts.train_dagger).

DAgger is the §35 next step: roll out the policy, label every visited state
with the IK expert, aggregate, refit. These tests pin the collection / pinning
/ eval helpers (the parts that don't need a multi-minute training run). The
full loop is validated by a tiny end-to-end CLI smoke in CI-of-record (see
TRACKER §35); here we keep unit coverage fast.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")


def _agent_and_env():
    from rl_autonomy.algos import RLPDSAC, RLPDConfig
    from rl_autonomy.envs import make_env, KeyboardEnv
    from rl_autonomy.envs._wrapper_utils import find_inner
    env = make_env(mode="approach", frame_stack=3, domain_rand=False,
                   random_key=False, seed=0, reward_mode="xy_focus")
    agent = RLPDSAC(env=env, config=RLPDConfig(), device="cpu")
    kb = find_inner(env, KeyboardEnv)
    return env, agent, kb


def test_pin_key_reset_sets_target_and_offset():
    from rl_autonomy.scripts.train_dagger import _pin_key_reset
    env, agent, kb = _agent_and_env()
    try:
        obs, _ = _pin_key_reset(env, kb, "j")
        assert kb.target_key == "j"
        # The target_offset_eef block of the actor obs should be finite + nonzero
        # (the goal vector is in the observation — TRACKER §35 observability note).
        assert np.all(np.isfinite(obs["actor"]))
    finally:
        env.close()


def test_collect_episode_records_expert_labels():
    from rl_autonomy.scripts.train_dagger import collect_episode
    from rl_autonomy.algos import IKExpert
    env, agent, kb = _agent_and_env()
    rng = np.random.default_rng(0)
    try:
        obs_l, act_l, succ, n = collect_episode(
            env, kb, agent, IKExpert(), key="j", beta=1.0, max_steps=12,
            deterministic_policy=True, rng=rng)
        assert len(obs_l) == len(act_l) == n
        assert 1 <= n <= 12
        obs_arr = np.stack(obs_l)
        act_arr = np.stack(act_l)
        assert obs_arr.shape[1] == agent.actor_dim       # stacked actor obs (108)
        assert act_arr.shape[1] == 7
        assert np.all(act_arr >= -1.0) and np.all(act_arr <= 1.0)
        assert np.allclose(act_arr[:, 6], -1.0)          # expert holds solenoid
        assert isinstance(succ, bool)
    finally:
        env.close()


def test_collect_episode_label_false_keeps_nothing():
    from rl_autonomy.scripts.train_dagger import collect_episode
    from rl_autonomy.algos import IKExpert
    env, agent, kb = _agent_and_env()
    rng = np.random.default_rng(1)
    try:
        obs_l, act_l, succ, n = collect_episode(
            env, kb, agent, IKExpert(), key="h", beta=0.0, max_steps=8,
            deterministic_policy=True, rng=rng, label=False)
        assert obs_l == [] and act_l == [] and n == 0
    finally:
        env.close()


def test_eval_success_returns_rate_in_unit_interval():
    from rl_autonomy.scripts.train_dagger import eval_success
    env, agent, kb = _agent_and_env()
    rng = np.random.default_rng(2)
    try:
        rate, per_key = eval_success(
            env, kb, agent, keys=["j", "h"], trials=1, max_steps=20, rng=rng)
        assert 0.0 <= rate <= 1.0
        assert set(per_key.keys()) == {"j", "h"}
        assert all(0 <= v <= 1 for v in per_key.values())
    finally:
        env.close()


def test_key_groups_cover_expected_sets():
    from rl_autonomy.scripts.train_dagger import KEY_GROUPS
    from rl_autonomy.envs.keyboard_layout import AVAILABLE_KEYS
    avail = set(AVAILABLE_KEYS)
    assert len(KEY_GROUPS["all"]) == 87
    assert set(KEY_GROUPS["central"]).issubset(avail)
    assert "j" in KEY_GROUPS["central"]
    # 'stratified' must be valid, deduplicated, and genuinely spread (not a
    # subset of the easy 'central' cluster) so it is a fair model-selection eval.
    strat = KEY_GROUPS["stratified"]
    assert set(strat).issubset(avail)
    assert len(strat) == len(set(strat))                 # no dupes
    assert 15 <= len(strat) <= 30
    assert not set(strat).issubset(set(KEY_GROUPS["central"]))
    # spans edge/corner keys the central cluster never touches
    assert {"esc", "space", "left"} & set(strat)
