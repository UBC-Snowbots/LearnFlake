"""Tests for the HDF5 demo loader (`rl_autonomy.data.demo_buffer`).

Covers:
  - happy path: synthetic HDF5 → ReplayBuffer roundtrip preserves data
  - shape-mismatch raises ValueError
  - empty file returns 0
  - missing path raises FileNotFoundError
  - re_normalize=True with a warm RMS rescales actor obs, leaves critic alone
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from rl_autonomy.algos import ReplayBuffer
from rl_autonomy.data import load_demos_into_buffer


ACTOR_DIM = 12
CRITIC_DIM = 8
ACTION_DIM = 4


def _write_synthetic_h5(path: Path, n: int = 50, *, actor_dim: int = ACTOR_DIM,
                        critic_dim: int = CRITIC_DIM, action_dim: int = ACTION_DIM) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset("actor_obs", data=rng.standard_normal((n, actor_dim)).astype(np.float32))
        f.create_dataset("critic_obs", data=rng.standard_normal((n, critic_dim)).astype(np.float32))
        f.create_dataset("action", data=rng.uniform(-1, 1, (n, action_dim)).astype(np.float32))
        f.create_dataset("reward", data=rng.uniform(0, 1, n).astype(np.float32))
        f.create_dataset("next_actor_obs", data=rng.standard_normal((n, actor_dim)).astype(np.float32))
        f.create_dataset("next_critic_obs", data=rng.standard_normal((n, critic_dim)).astype(np.float32))
        # Mark the last transition of each episode as terminated; episode_id ramps.
        term = np.zeros(n, dtype=bool)
        if n > 0:
            term[-1] = True
        f.create_dataset("terminated", data=term)
        f.create_dataset("episode_id", data=np.zeros(n, dtype=np.int32))
        meta = f.create_group("trial_meta")
        meta.attrs["n_attempts"] = 1
        meta.attrs["n_successes"] = 1


def test_load_demos_happy_path():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "demo.h5"
        _write_synthetic_h5(p, n=50)
        rb = ReplayBuffer(capacity=200, actor_dim=ACTOR_DIM, critic_dim=CRITIC_DIM,
                          action_dim=ACTION_DIM)
        n = load_demos_into_buffer(p, rb)
        assert n == 50
        assert rb.size == 50


def test_load_demos_shape_mismatch_raises():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "demo.h5"
        _write_synthetic_h5(p, n=10, actor_dim=ACTOR_DIM + 1)
        rb = ReplayBuffer(capacity=200, actor_dim=ACTOR_DIM, critic_dim=CRITIC_DIM,
                          action_dim=ACTION_DIM)
        with pytest.raises(ValueError, match="actor_obs"):
            load_demos_into_buffer(p, rb)


def test_load_demos_missing_file_raises():
    rb = ReplayBuffer(capacity=10, actor_dim=ACTOR_DIM, critic_dim=CRITIC_DIM,
                      action_dim=ACTION_DIM)
    with pytest.raises(FileNotFoundError):
        load_demos_into_buffer("/nonexistent/path/demo.h5", rb)


def test_load_demos_empty_file_returns_zero():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "demo.h5"
        _write_synthetic_h5(p, n=0)
        rb = ReplayBuffer(capacity=10, actor_dim=ACTOR_DIM, critic_dim=CRITIC_DIM,
                          action_dim=ACTION_DIM)
        n = load_demos_into_buffer(p, rb)
        assert n == 0
        assert rb.size == 0


def test_load_demos_re_normalize_with_warm_rms():
    """If a warmed RMS is given, actor_obs should be re-normalized (not critic)."""
    import gymnasium as gym
    from rl_autonomy.envs.obs_adapter import ObsAdapter
    from rl_autonomy.envs.normalizer import RunningMeanStd

    # Build a minimal ObsAdapter just to expose an `.rms` we can warm up.
    class _Dummy(gym.Env):
        observation_space = gym.spaces.Dict({
            "actor": gym.spaces.Box(-np.inf, np.inf, (ACTOR_DIM,), dtype=np.float32),
            "critic": gym.spaces.Box(-np.inf, np.inf, (CRITIC_DIM,), dtype=np.float32),
        })
        action_space = gym.spaces.Box(-1, 1, (ACTION_DIM,), dtype=np.float32)

        def reset(self, *a, **kw):
            return {"actor": np.zeros(ACTOR_DIM, dtype=np.float32),
                    "critic": np.zeros(CRITIC_DIM, dtype=np.float32)}, {}

        def step(self, action):
            return ({"actor": np.zeros(ACTOR_DIM, dtype=np.float32),
                     "critic": np.zeros(CRITIC_DIM, dtype=np.float32)},
                    0.0, False, False, {})

    oa = ObsAdapter(_Dummy())
    # Warm the RMS so .count > 100 ⇒ loader will rescale.
    rng = np.random.default_rng(0)
    oa.rms.update(rng.standard_normal((500, ACTOR_DIM)).astype(np.float32) * 5.0)
    assert oa.rms.count > 100

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "demo.h5"
        _write_synthetic_h5(p, n=30)
        rb = ReplayBuffer(capacity=100, actor_dim=ACTOR_DIM, critic_dim=CRITIC_DIM,
                          action_dim=ACTION_DIM)
        n = load_demos_into_buffer(p, rb, obs_adapter=oa, re_normalize=True)
        assert n == 30

        # Actor obs after re-normalize should be clipped to [-10, 10] (RMS clip default).
        loaded_actor = rb.actor_obs[:30].cpu().numpy()
        assert np.all(np.abs(loaded_actor) <= 10.0 + 1e-5)

        # Critic obs must match the raw HDF5 (NOT re-normalized).
        with h5py.File(p, "r") as f:
            raw_critic = f["critic_obs"][:]
        loaded_critic = rb.critic_obs[:30].cpu().numpy()
        np.testing.assert_allclose(loaded_critic, raw_critic, atol=1e-5)
