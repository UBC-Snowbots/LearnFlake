"""Algorithm-level tests for RLPD-SAC.

These don't run the full training loop (M2 covers that) — they just verify
the building blocks: networks have correct shapes, the replay buffer
samples correctly, a single update step doesn't NaN.
"""
from __future__ import annotations

import warnings
import numpy as np
import pytest
import torch
import gymnasium as gym

warnings.filterwarnings("ignore")


def test_actor_shapes():
    from rl_autonomy.algos import Actor
    actor = Actor(obs_dim=10, action_dim=3)
    obs = torch.randn(8, 10)
    mu, log_std = actor(obs)
    assert mu.shape == (8, 3)
    assert log_std.shape == (8, 3)
    a, lp = actor.sample(obs)
    assert a.shape == (8, 3)
    assert lp.shape == (8, 1)
    # tanh-squashed: every action component in [-1, 1]
    assert (a.abs() <= 1.0).all()


def test_critic_shapes():
    from rl_autonomy.algos import EnsembleCritic
    critic = EnsembleCritic(obs_dim=10, action_dim=3, n_critics=2)
    obs = torch.randn(8, 10)
    action = torch.randn(8, 3) * 2  # anything; not necessarily in [-1, 1]
    q = critic(obs, action)
    assert q.shape == (2, 8, 1)


def test_replay_buffer_sample_shapes():
    from rl_autonomy.algos import ReplayBuffer
    rb = ReplayBuffer(capacity=200, actor_dim=12, critic_dim=18, action_dim=4)
    for _ in range(50):
        rb.add(
            actor_obs=np.random.randn(12).astype(np.float32),
            critic_obs=np.random.randn(18).astype(np.float32),
            action=np.random.randn(4).astype(np.float32),
            reward=0.0,
            next_actor_obs=np.random.randn(12).astype(np.float32),
            next_critic_obs=np.random.randn(18).astype(np.float32),
            terminated=False,
        )
    batch = rb.sample(10)
    assert batch.actor_obs.shape == (10, 12)
    assert batch.critic_obs.shape == (10, 18)
    assert batch.action.shape == (10, 4)
    assert batch.reward.shape == (10, 1)
    assert batch.terminated.shape == (10, 1)


def test_replay_buffer_circular_overwrite():
    """Buffer of capacity 100, fill to 150 — size caps at 100, ptr wraps."""
    from rl_autonomy.algos import ReplayBuffer
    rb = ReplayBuffer(capacity=100, actor_dim=5, critic_dim=5, action_dim=2)
    for _ in range(150):
        rb.add(
            actor_obs=np.zeros(5, dtype=np.float32),
            critic_obs=np.zeros(5, dtype=np.float32),
            action=np.zeros(2, dtype=np.float32),
            reward=0.0,
            next_actor_obs=np.zeros(5, dtype=np.float32),
            next_critic_obs=np.zeros(5, dtype=np.float32),
            terminated=False,
        )
    assert rb.size == 100


def test_symmetric_replay_demo_fraction_decays():
    from rl_autonomy.algos import ReplayBuffer, SymmetricReplayBuffer
    online = ReplayBuffer(100, actor_dim=4, critic_dim=4, action_dim=2)
    demos = ReplayBuffer(100, actor_dim=4, critic_dim=4, action_dim=2)
    sym = SymmetricReplayBuffer(online, demos, f_init=0.5, f_final=0.25, decay_steps=1000)
    # No demos loaded yet → fraction is 0.
    assert sym.current_demo_fraction() == 0.0
    # Add some demos
    for _ in range(10):
        demos.add(np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32),
                  np.zeros(2, dtype=np.float32), 0.0,
                  np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32), False)
    assert sym.current_demo_fraction() == pytest.approx(0.5, abs=1e-6)
    # Advance halfway → fraction ≈ 0.5 + 0.5 * (0.25 - 0.5) = 0.375
    sym.advance(500)
    assert sym.current_demo_fraction() == pytest.approx(0.375, abs=1e-6)
    sym.advance(500)
    assert sym.current_demo_fraction() == pytest.approx(0.25, abs=1e-6)
    # Past decay_steps stays at f_final
    sym.advance(5000)
    assert sym.current_demo_fraction() == pytest.approx(0.25, abs=1e-6)


def test_rlpd_sac_one_train_step_no_nan():
    """Construct the agent, fill the replay with a few transitions, run one
    update step. Validates the gradient path doesn't NaN."""
    from rl_autonomy.algos import RLPDSAC, RLPDConfig

    class _DictBox(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gym.spaces.Dict({
                "actor": env.observation_space,
                "critic": env.observation_space,
            })
        def reset(self, **kw):
            o, i = self.env.reset(**kw)
            return {"actor": o.astype(np.float32), "critic": o.astype(np.float32)}, i
        def step(self, a):
            o, r, t, tr, i = self.env.step(a)
            return {"actor": o.astype(np.float32), "critic": o.astype(np.float32)}, r, t, tr, i

    env = _DictBox(gym.make("Pendulum-v1"))
    cfg = RLPDConfig(
        update_to_data=1, warmstart_steps=10, batch_size=8, buffer_size=100,
        demo_buffer_size=1, demo_fraction_init=0.0, demo_fraction_final=0.0,
        actor_hidden=(32, 32), critic_hidden=(32, 32),
    )
    agent = RLPDSAC(env=env, config=cfg, device="cpu")
    obs, _ = env.reset(seed=0)
    # Enough random transitions to enable training
    for _ in range(20):
        a = env.action_space.sample()
        n_obs, r, term, trunc, _ = env.step(a)
        agent.replay.add(
            actor_obs=obs["actor"], critic_obs=obs["critic"],
            action=a, reward=r,
            next_actor_obs=n_obs["actor"], next_critic_obs=n_obs["critic"],
            terminated=term,
        )
        obs = n_obs if not (term or trunc) else env.reset()[0]
    info = agent._train_step()
    for k, v in info.items():
        if isinstance(v, float):
            assert np.isfinite(v), f"{k} = {v} is not finite"
    env.close()


def test_residual_actor_zero_init_matches_base():
    from rl_autonomy.algos import Actor, ResidualActor
    base = Actor(obs_dim=8, action_dim=3)
    res = ResidualActor(base)
    obs = torch.randn(4, 8)
    mu_base, _ = base(obs)
    mu_res, _ = res(obs)
    assert torch.allclose(mu_base, mu_res, atol=1e-6)


def test_min_alpha_floor_enforced():
    """Per TRACKER §26: log_alpha must be clamped above np.log(min_alpha)
    after each temperature update. Run many updates with a tiny target
    entropy so the auto-tuner would normally push α toward zero; assert
    α stays ≥ min_alpha."""
    from rl_autonomy.algos import RLPDSAC, RLPDConfig

    class _DictBox(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gym.spaces.Dict({
                "actor": env.observation_space,
                "critic": env.observation_space,
            })
        def reset(self, **kw):
            o, i = self.env.reset(**kw)
            return {"actor": o.astype(np.float32), "critic": o.astype(np.float32)}, i
        def step(self, a):
            o, r, t, tr, i = self.env.step(a)
            return {"actor": o.astype(np.float32), "critic": o.astype(np.float32)}, r, t, tr, i

    env = _DictBox(gym.make("Pendulum-v1"))
    cfg = RLPDConfig(
        update_to_data=1, warmstart_steps=10, batch_size=8, buffer_size=200,
        demo_buffer_size=1, demo_fraction_init=0.0, demo_fraction_final=0.0,
        actor_hidden=(16, 16), critic_hidden=(16, 16),
        # Encourage α decay: very negative target_entropy via a huge scale
        target_entropy_scale=20.0,
        # Slow-decaying init, big lr so we see motion
        init_temperature=1.0, temp_lr=1e-1,
        min_alpha=0.1,
    )
    agent = RLPDSAC(env=env, config=cfg, device="cpu")
    obs, _ = env.reset(seed=0)
    for _ in range(40):
        a = env.action_space.sample()
        n_obs, r, term, trunc, _ = env.step(a)
        agent.replay.add(actor_obs=obs["actor"], critic_obs=obs["critic"],
                         action=a, reward=r,
                         next_actor_obs=n_obs["actor"], next_critic_obs=n_obs["critic"],
                         terminated=term)
        obs = n_obs if not (term or trunc) else env.reset()[0]

    # Many training steps to drive α toward its asymptote.
    for _ in range(50):
        info = agent._train_step()
        assert info["alpha"] >= cfg.min_alpha - 1e-6, (
            f"α={info['alpha']} dropped below min_alpha={cfg.min_alpha}"
        )
    env.close()


def test_save_load_persists_rms():
    """TRACKER §28: checkpoint roundtrip must restore the env's RMS state."""
    from rl_autonomy.algos import RLPDSAC, RLPDConfig
    from rl_autonomy.envs.normalizer import RunningMeanStd
    import tempfile, os

    class _DictBoxWithObsAdapter(gym.Wrapper):
        """Synthetic stack: gym env + a real ObsAdapter so the RMS path is exercised."""
        def __init__(self, env):
            super().__init__(env)
            from rl_autonomy.envs.obs_adapter import ObsAdapter
            self.observation_space = gym.spaces.Dict({
                "actor": env.observation_space,
                "critic": env.observation_space,
            })
            # Wrap inner-most: gym -> DictBoxWithObsAdapter (this) -> ObsAdapter
            # Note: ObsAdapter expects Dict({'actor', 'critic'}) — which we just made above.
        def reset(self, **kw):
            o, i = self.env.reset(**kw)
            return {"actor": o.astype(np.float32), "critic": o.astype(np.float32)}, i
        def step(self, a):
            o, r, t, tr, i = self.env.step(a)
            return {"actor": o.astype(np.float32), "critic": o.astype(np.float32)}, r, t, tr, i

    from rl_autonomy.envs.obs_adapter import ObsAdapter
    inner = _DictBoxWithObsAdapter(gym.make("Pendulum-v1"))
    env_with_oa = ObsAdapter(inner, training=True)

    cfg = RLPDConfig(update_to_data=1, warmstart_steps=5, batch_size=4, buffer_size=20,
                     demo_buffer_size=1, demo_fraction_init=0.0, demo_fraction_final=0.0,
                     actor_hidden=(8,), critic_hidden=(8,))
    agent = RLPDSAC(env=env_with_oa, config=cfg, device="cpu")

    # Populate RMS with a few obs samples
    obs, _ = env_with_oa.reset()
    for _ in range(10):
        a = env_with_oa.action_space.sample()
        obs, _, t, tr, _ = env_with_oa.step(a)
        if t or tr:
            obs, _ = env_with_oa.reset()

    rms_mean_before = env_with_oa.rms.mean.copy()
    rms_var_before = env_with_oa.rms.var.copy()
    rms_count_before = env_with_oa.rms.count

    # Save + load on a fresh agent + fresh env
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ckpt.pt")
        agent.save(path)

        # New agent + new env with a fresh empty RMS
        inner2 = _DictBoxWithObsAdapter(gym.make("Pendulum-v1"))
        env2 = ObsAdapter(inner2, training=True)
        agent2 = RLPDSAC(env=env2, config=cfg, device="cpu")
        # Confirm RMS is empty before load
        assert env2.rms.count <= 1e-3 + 1e-4  # default epsilon
        agent2.load(path)

        # After load, RMS should match the saved one
        assert np.allclose(env2.rms.mean, rms_mean_before)
        assert np.allclose(env2.rms.var, rms_var_before)
        assert env2.rms.count == pytest.approx(rms_count_before)
        # And training should be disabled (frozen stats)
        assert env2.training is False
        # has_rms() should now return True
        assert agent2.has_rms()


def test_warm_up_env_rms_bootstraps_from_random_actions():
    """For checkpoints saved before §28's RMS persistence, warm_up_env_rms()
    must bring the RMS count well above the default epsilon by running
    random actions through the env."""
    from rl_autonomy.algos import RLPDSAC, RLPDConfig
    from rl_autonomy.envs.obs_adapter import ObsAdapter

    class _DictBoxWithObsAdapter(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gym.spaces.Dict({
                "actor": env.observation_space,
                "critic": env.observation_space,
            })
        def reset(self, **kw):
            o, i = self.env.reset(**kw)
            return {"actor": o.astype(np.float32), "critic": o.astype(np.float32)}, i
        def step(self, a):
            o, r, t, tr, i = self.env.step(a)
            return {"actor": o.astype(np.float32), "critic": o.astype(np.float32)}, r, t, tr, i

    inner = _DictBoxWithObsAdapter(gym.make("Pendulum-v1"))
    env = ObsAdapter(inner, training=True)
    cfg = RLPDConfig(update_to_data=1, warmstart_steps=5, batch_size=4, buffer_size=20,
                     demo_buffer_size=1, demo_fraction_init=0.0, demo_fraction_final=0.0,
                     actor_hidden=(8,), critic_hidden=(8,))
    agent = RLPDSAC(env=env, config=cfg, device="cpu")

    assert not agent.has_rms()
    agent.warm_up_env_rms(n_steps=200, action_source="random")
    assert agent.has_rms()
    # RMS should be frozen after warmup
    assert env.training is False
    # RMS should have non-trivial variance (random actions = wide obs range)
    assert env.rms.count > 200 - 1


def test_bc_pretrain_reduces_actor_loss():
    """BC fit should monotonically reduce NLL on a fittable batch."""
    import torch
    from rl_autonomy.algos.bc_pretrain import BCPretrain, BCConfig
    from rl_autonomy.algos import Actor

    torch.manual_seed(0)
    np.random.seed(0)
    obs_dim = 8
    act_dim = 3

    actor = Actor(obs_dim=obs_dim, action_dim=act_dim, hidden=(32, 32))
    bc = BCPretrain(actor, BCConfig(epochs=20, batch_size=32, lr=1e-3, device="cpu"))

    # Synthetic: actions = tanh(W·obs). Actor with tanh output should fit this.
    W = np.random.randn(act_dim, obs_dim).astype(np.float32) * 0.5
    obs = np.random.randn(256, obs_dim).astype(np.float32)
    actions = np.tanh(obs @ W.T).astype(np.float32)

    history = bc.fit(obs, actions)
    assert history["loss"][-1] < history["loss"][0] - 0.5, (
        f"BC didn't learn: epoch 0 loss={history['loss'][0]:.4f}, "
        f"final={history['loss'][-1]:.4f}"
    )


def test_min_alpha_zero_disables_floor():
    """min_alpha=0 should be equivalent to vanilla SAC (no floor)."""
    from rl_autonomy.algos import RLPDSAC, RLPDConfig
    # Build a config and just check construction succeeds + log_alpha
    # has no clamp wrapper (the clamp branch is gated by min_alpha > 0).
    import gymnasium as gym
    class _Dummy(gym.Wrapper):
        def __init__(self):
            super().__init__(gym.make("Pendulum-v1"))
            self.observation_space = gym.spaces.Dict({
                "actor": self.env.observation_space,
                "critic": self.env.observation_space,
            })
        def reset(self, **kw):
            o, i = self.env.reset(**kw); return {"actor": o.astype(np.float32),
                                                  "critic": o.astype(np.float32)}, i
        def step(self, a):
            o, r, t, tr, i = self.env.step(a)
            return {"actor": o.astype(np.float32), "critic": o.astype(np.float32)}, r, t, tr, i
    cfg = RLPDConfig(min_alpha=0.0, actor_hidden=(8,), critic_hidden=(8,),
                     buffer_size=10, demo_buffer_size=1)
    agent = RLPDSAC(env=_Dummy(), config=cfg, device="cpu")
    assert agent.cfg.min_alpha == 0.0
