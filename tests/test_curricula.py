"""Curriculum tests."""
from __future__ import annotations

import pytest


def test_key_phase_starts_in_phase_A():
    from rl_autonomy.curricula import KeyPhaseCurriculum
    from rl_autonomy.envs.keyboard_layout import PHASE_A_KEYS
    cur = KeyPhaseCurriculum(seed=0)
    assert cur.current_phase == 0
    for _ in range(50):
        k = cur.sample_key()
        assert k in PHASE_A_KEYS


def test_key_phase_advances_on_high_success():
    from rl_autonomy.curricula import KeyPhaseCurriculum
    cur = KeyPhaseCurriculum(advance_threshold=0.85, window=20, seed=0)
    # Below window — never advances
    for _ in range(10):
        cur.report_outcome("g", True)
    assert cur.current_phase == 0
    # Fill window with failures — still doesn't advance
    for _ in range(20):
        cur.report_outcome("g", False)
    assert cur.current_phase == 0
    # Fill window with 90% success → should advance once
    for i in range(20):
        cur.report_outcome("g", success=(i < 18))   # 18/20 = 90%
    assert cur.current_phase == 1


def test_key_phase_caps_at_phase_C():
    from rl_autonomy.curricula import KeyPhaseCurriculum
    cur = KeyPhaseCurriculum(advance_threshold=0.5, window=2, seed=0)
    # Force advance through all phases
    for _ in range(20):
        cur.report_outcome("g", True)
    assert cur.current_phase == cur.n_phases - 1


def test_state_replay_v1_demo_free_is_noop():
    from rl_autonomy.curricula import StateReplayCurriculum
    cur = StateReplayCurriculum()
    assert not cur.is_active()
    assert cur.sample_reset_state() is None


def test_state_replay_with_demos_unsupported_in_v1():
    from rl_autonomy.curricula import StateReplayCurriculum
    with pytest.raises(NotImplementedError):
        StateReplayCurriculum(demo_buffer=object())
