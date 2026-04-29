"""Lock down the observation schema so future edits surface here."""
from __future__ import annotations

import warnings
import numpy as np

warnings.filterwarnings("ignore")


def test_actor_obs_dim():
    from rl_autonomy.envs.obs_adapter import ACTOR_OBS_DIM, ACTOR_FIELDS
    # Sum of declared dims must match exported constant
    declared = sum(d for _, d in ACTOR_FIELDS)
    assert declared == ACTOR_OBS_DIM
    # Per TRACKER §9.1
    assert ACTOR_OBS_DIM == 36


def test_critic_obs_dim():
    from rl_autonomy.envs.obs_adapter import CRITIC_OBS_DIM, CRITIC_FIELDS
    declared = sum(d for _, d in CRITIC_FIELDS)
    assert declared == CRITIC_OBS_DIM
    # Privileged extras: contact_force_vec(3) + actuator_pos(1) + tilt(1) added
    # to actor base minus aruco. Keep this exact so a future bump is intentional.
    assert CRITIC_OBS_DIM == 38


def test_actor_critic_disjoint_aruco():
    """Actor sees aruco_obs; critic sees ground-truth target_offset_eef instead."""
    from rl_autonomy.envs.obs_adapter import ACTOR_FIELDS, CRITIC_FIELDS
    actor_names = {n for n, _ in ACTOR_FIELDS}
    critic_names = {n for n, _ in CRITIC_FIELDS}
    assert "aruco_obs" in actor_names
    assert "aruco_obs" not in critic_names


def test_frame_stack_multiplies_actor_dim():
    from rl_autonomy.envs import make_env
    env_k1 = make_env(mode="approach", frame_stack=1, domain_rand=False, horizon=10)
    env_k3 = make_env(mode="approach", frame_stack=3, domain_rand=False, horizon=10)
    obs_k1, _ = env_k1.reset(seed=0)
    obs_k3, _ = env_k3.reset(seed=0)
    assert obs_k1["actor"].shape[0] * 3 == obs_k3["actor"].shape[0]
    # Critic is NOT stacked
    assert obs_k1["critic"].shape == obs_k3["critic"].shape
    env_k1.close()
    env_k3.close()


def test_obs_after_step_finite_and_correct_shape():
    from rl_autonomy.envs import make_env
    env = make_env(mode="approach", frame_stack=3, domain_rand=False, horizon=10)
    obs, _ = env.reset(seed=0)
    assert obs["actor"].shape == env.observation_space["actor"].shape
    assert obs["critic"].shape == env.observation_space["critic"].shape
    for _ in range(5):
        a = np.zeros(7, dtype=np.float32)
        obs, *_ = env.step(a)
        assert np.all(np.isfinite(obs["actor"]))
        assert np.all(np.isfinite(obs["critic"]))
    env.close()


def test_dr_sample_in_info():
    """When DR is on, info['dr'] must contain every sampled axis."""
    from rl_autonomy.envs import make_env
    from rl_autonomy.envs.domain_rand import DR_RANGES
    env = make_env(mode="approach", frame_stack=1, domain_rand=True, horizon=5)
    env.reset(seed=0)
    a = np.zeros(7, dtype=np.float32)
    _, _, _, _, info = env.step(a)
    assert "dr" in info
    for axis in DR_RANGES:
        assert axis in info["dr"], f"missing DR axis {axis} in info['dr']"
    env.close()
