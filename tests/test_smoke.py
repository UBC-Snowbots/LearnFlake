"""Smoke test — package imports, env instantiates, env steps."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")


def test_package_imports():
    import rl_autonomy
    assert rl_autonomy.__version__


def test_envs_package_imports():
    from rl_autonomy.envs import (
        KeyboardEnv, make_env, DomainRandWrapper, RunningMeanStd,
        KEYBOARD_LAYOUT, AVAILABLE_KEYS,
    )
    assert len(KEYBOARD_LAYOUT) == 87
    assert len(AVAILABLE_KEYS) == 87
    assert len(set(AVAILABLE_KEYS)) == 87


def test_configs_path():
    from rl_autonomy.configs import CONTROLLER_JP_PATH
    import json
    cfg = json.load(open(CONTROLLER_JP_PATH))
    assert cfg["body_parts"]["arms"]["right"]["type"] == "JOINT_POSITION"


def test_keyboard_env_instantiates_and_steps():
    from rl_autonomy.envs import KeyboardEnv
    env = KeyboardEnv(mode="approach", horizon=20)
    env.reset()
    assert env.action_dim == 7
    for _ in range(5):
        env.step(np.zeros(7, dtype=np.float32))
    obs = env.get_obs_dict()
    # All required obs keys present
    expected = {
        "joint_pos", "joint_vel", "eef_pos", "eef_quat",
        "actuator_extended", "actuator_pos", "actuator_vel",
        "target_key_pos_world", "target_offset_eef",
        "aruco_obs", "rangefinder",
        "contact_force_norm", "contact_force_vec",
        "tilt_rad",
    }
    assert set(obs.keys()) == expected
    env.close()


@pytest.mark.parametrize("mode", ["approach", "strike"])
def test_make_env_yields_gym_compatible_env(mode):
    from rl_autonomy.envs import make_env
    env = make_env(mode=mode, frame_stack=3, domain_rand=False, horizon=20)
    obs, info = env.reset(seed=0)
    assert "actor" in obs and "critic" in obs
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, r, term, trunc, info = env.step(a)
    assert isinstance(r, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    env.close()
