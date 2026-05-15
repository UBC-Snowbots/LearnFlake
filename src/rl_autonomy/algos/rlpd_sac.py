"""RLPD-SAC — Reinforcement Learning with Prior Data, asymmetric variant.

Implements the algorithm from Ball et al. NeurIPS 2023 with two minor
adaptations:

  1. Asymmetric actor-critic — actor sees ``obs['actor']`` (deployable
     observation), critic sees ``obs['critic']`` (privileged sim state).
     Standard RLPD assumes symmetric obs; ours doesn't.

  2. Auto-tuned entropy temperature with target ``H = -0.5·|action_dim|``
     (RoboPianist setting; halfway between -1·dim and 0).

Other features per TRACKER §3.1:
  - Twin critics + LayerNorm
  - High update-to-data (UTD) ratio (default 10)
  - Symmetric demo+online sampling via SymmetricReplayBuffer
  - Polyak-updated target critic, no target actor (SAC standard)
  - GELU activations, AdamW optimizer, weight_decay 1e-4
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import gymnasium as gym

from .networks import Actor, EnsembleCritic
from .replay_buffer import Batch, ReplayBuffer, SymmetricReplayBuffer


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class RLPDConfig:
    # Discounting
    gamma: float = 0.99
    tau: float = 0.005

    # Entropy
    target_entropy_scale: float = 0.5      # target_entropy = -scale * dim(action)
    init_temperature: float = 1.0
    min_alpha: float = 0.1                 # floor on the auto-tuned α; prevents
                                           # exploration collapse on sparse-success
                                           # tasks without demos (TRACKER §26).
                                           # Set to 0 to disable (vanilla SAC behavior).

    # Optimizer
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    weight_decay: float = 1e-4

    # Sampling
    batch_size: int = 512
    update_to_data: int = 10
    warmstart_steps: int = 5_000

    # Replay
    buffer_size: int = 1_000_000
    demo_buffer_size: int = 50_000
    demo_fraction_init: float = 0.5
    demo_fraction_final: float = 0.25
    demo_fraction_decay_steps: int = 500_000

    # Architecture
    actor_hidden: tuple[int, ...] = (256, 256, 256)
    critic_hidden: tuple[int, ...] = (512, 512, 512)
    n_critics: int = 2
    use_layer_norm: bool = True

    # Practical
    seed: int = 0


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RLPDSAC:
    """Asymmetric RLPD-SAC with separate actor / critic observation views.

    The training env's observation space must be ``Dict({'actor': Box, 'critic': Box})``.
    Build one with ``rl_autonomy.envs.make_env(...)``.
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[RLPDConfig] = None,
        device: torch.device | str | None = None,
        eval_env: gym.Env | None = None,
    ):
        self.env = env
        self.eval_env = eval_env or env
        self.cfg = config or RLPDConfig()

        # Validate the env shape — raise early on mistakes.
        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.Dict) or set(obs_space.spaces) != {"actor", "critic"}:
            raise ValueError(
                "env.observation_space must be Dict({'actor': Box, 'critic': Box}); "
                f"got {obs_space}. Use rl_autonomy.envs.make_env(...)."
            )
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError(f"action_space must be Box, got {env.action_space}")

        self.actor_dim = obs_space["actor"].shape[0]
        self.critic_dim = obs_space["critic"].shape[0]
        self.action_dim = env.action_space.shape[0]

        # Action scaling: actor outputs tanh-squashed values in [-1, 1]; the
        # env may use a different range (e.g. Pendulum: [-2, 2]). Without
        # rescaling the policy can't reach the env's action limits — silently
        # underperforms on every benchmark whose action_space != [-1, 1].
        action_low = env.action_space.low
        action_high = env.action_space.high
        if not np.all(np.isfinite(action_low)) or not np.all(np.isfinite(action_high)):
            raise ValueError("action_space must have finite bounds")
        self._action_scale = torch.as_tensor(
            (action_high - action_low) / 2.0, dtype=torch.float32
        )
        self._action_bias = torch.as_tensor(
            (action_high + action_low) / 2.0, dtype=torch.float32
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._action_scale = self._action_scale.to(self.device)
        self._action_bias = self._action_bias.to(self.device)

        # RNG
        self._rng = np.random.default_rng(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        # Networks
        self.actor = Actor(
            self.actor_dim, self.action_dim,
            hidden=self.cfg.actor_hidden, use_layer_norm=self.cfg.use_layer_norm,
        ).to(self.device)
        self.critic = EnsembleCritic(
            self.critic_dim, self.action_dim,
            n_critics=self.cfg.n_critics,
            hidden=self.cfg.critic_hidden, use_layer_norm=self.cfg.use_layer_norm,
        ).to(self.device)
        self.critic_target = EnsembleCritic(
            self.critic_dim, self.action_dim,
            n_critics=self.cfg.n_critics,
            hidden=self.cfg.critic_hidden, use_layer_norm=self.cfg.use_layer_norm,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Cache the parameter lists for the Polyak target update. Walking
        # nn.Module.parameters() each call is a generator over the whole
        # module tree (~25 LayerNorms + Linears + biases per critic × 2 critics).
        # With UTD=10 that's 200k+ Module._named_members traversals per 1k env
        # steps — profile shows it was 10% of wallclock. Caching the lists once
        # is sound because critic + critic_target are not mutated after init.
        self._critic_params = list(self.critic.parameters())
        self._critic_target_params = list(self.critic_target.parameters())

        # Auto-tuned entropy temperature, parametrized as log_alpha for stability
        self.log_alpha = nn.Parameter(
            torch.tensor(np.log(self.cfg.init_temperature), dtype=torch.float32, device=self.device)
        )
        self.target_entropy = -self.cfg.target_entropy_scale * float(self.action_dim)

        # Optimizers
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=self.cfg.actor_lr, weight_decay=self.cfg.weight_decay
        )
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(), lr=self.cfg.critic_lr, weight_decay=self.cfg.weight_decay
        )
        self.temp_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.temp_lr)

        # Replay
        online = ReplayBuffer(
            capacity=self.cfg.buffer_size,
            actor_dim=self.actor_dim, critic_dim=self.critic_dim,
            action_dim=self.action_dim, device=self.device,
        )
        demos = ReplayBuffer(
            capacity=self.cfg.demo_buffer_size,
            actor_dim=self.actor_dim, critic_dim=self.critic_dim,
            action_dim=self.action_dim, device=self.device,
        )
        self.replay = SymmetricReplayBuffer(
            online, demos,
            f_init=self.cfg.demo_fraction_init,
            f_final=self.cfg.demo_fraction_final,
            decay_steps=self.cfg.demo_fraction_decay_steps,
        )

        # Counters
        self._env_steps = 0
        self._gradient_steps = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _scale_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Map tanh output [-1, 1] → env action range."""
        return raw_action * self._action_scale + self._action_bias

    def _unscale_action(self, env_action: torch.Tensor) -> torch.Tensor:
        """Inverse of _scale_action."""
        return (env_action - self._action_bias) / self._action_scale

    @torch.no_grad()
    def predict(
        self, obs: dict[str, np.ndarray], deterministic: bool = False
    ) -> tuple[np.ndarray, None]:
        """SB3-compatible signature: returns (action, state). state is unused.

        The returned action is already in the env's native scale.
        """
        actor_obs = torch.as_tensor(obs["actor"], dtype=torch.float32, device=self.device)
        if actor_obs.ndim == 1:
            actor_obs = actor_obs.unsqueeze(0)
        raw, _ = self.actor.sample(actor_obs, deterministic=deterministic)
        scaled = self._scale_action(raw)
        return scaled.squeeze(0).cpu().numpy().astype(np.float32), None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _update_critic(self, batch: Batch) -> dict[str, float]:
        with torch.no_grad():
            next_action_raw, next_log_prob = self.actor.sample(batch.next_actor_obs)
            next_action = self._scale_action(next_action_raw)              # to env scale
            target_q = self.critic_target(batch.next_critic_obs, next_action)   # (n, B, 1)
            target_q = target_q.min(dim=0).values                                # (B, 1)
            target_q = target_q - self.alpha.detach() * next_log_prob
            y = batch.reward + self.cfg.gamma * (1.0 - batch.terminated) * target_q

        # batch.action is already env-scale (it's what got executed in the env).
        q = self.critic(batch.critic_obs, batch.action)                          # (n, B, 1)
        critic_loss = sum(F.mse_loss(q[i], y) for i in range(q.shape[0])) / q.shape[0]

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        return {"critic_loss": float(critic_loss.detach()), "target_q_mean": float(y.mean().detach())}

    def _update_actor_and_temp(self, batch: Batch) -> dict[str, float]:
        action_raw, log_prob = self.actor.sample(batch.actor_obs)
        action = self._scale_action(action_raw)                            # to env scale
        # Q value of *the actor's* action — fed into the critic with the
        # PRIVILEGED critic obs from the same transition (asymmetric AC).
        q = self.critic(batch.critic_obs, action).min(dim=0).values
        actor_loss = (self.alpha.detach() * log_prob - q).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # Temperature: minimize alpha * (-log_prob - target_entropy)
        # Equivalently: alpha * (target_entropy + log_prob).
        # Note: log_prob is summed over action dims and is NEGATIVE for
        # high-entropy policies, so -log_prob is POSITIVE.
        temp_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.temp_opt.zero_grad(set_to_none=True)
        temp_loss.backward()
        self.temp_opt.step()

        # Floor log_alpha. Without this, the auto-tuner can drive α to ~0.03
        # on sparse-success tasks (target_entropy met by a near-deterministic
        # policy), killing exploration. Once exploration dies the policy can't
        # escape local minima — observed in §26 collapse. Setting min_alpha=0.1
        # is a soft floor that preserves the auto-tune dynamics while
        # preventing total exploration collapse. min_alpha=0 = vanilla SAC.
        if self.cfg.min_alpha > 0:
            with torch.no_grad():
                self.log_alpha.clamp_(min=float(np.log(self.cfg.min_alpha)))

        return {
            "actor_loss": float(actor_loss.detach()),
            "actor_entropy": float(-log_prob.mean().detach()),
            "alpha": float(self.alpha.detach()),
        }

    def _polyak_update(self) -> None:
        # Use torch._foreach_lerp_ to update all target tensors in a single
        # fused kernel launch (PyTorch ≥1.13). Equivalent math:
        #     p_t ← (1 − τ) p_t + τ p
        # but runs ~5× faster than Python-loop .mul_().add_() on cuda because
        # it avoids re-entering CPU between every tensor.
        with torch.no_grad():
            torch._foreach_lerp_(
                self._critic_target_params, self._critic_params, self.cfg.tau,
            )

    def _train_step(self) -> dict[str, float]:
        """One env-step worth of gradient updates: UTD critic updates + 1 actor update."""
        utd = self.cfg.update_to_data
        info = {}
        for _ in range(utd):
            batch = self.replay.sample(self.cfg.batch_size)
            info_critic = self._update_critic(batch)
            self._polyak_update()
            self._gradient_steps += 1
        # One actor + temperature update per env step (DroQ/RLPD recipe — actor
        # update lags critic).
        batch = self.replay.sample(self.cfg.batch_size)
        info_actor = self._update_actor_and_temp(batch)
        info.update(info_critic)
        info.update(info_actor)
        info["demo_fraction"] = self.replay.current_demo_fraction()
        return info

    # ------------------------------------------------------------------
    # Public training loop
    # ------------------------------------------------------------------

    def learn(
        self,
        total_timesteps: int,
        log_every: int = 1_000,
        eval_every: int = 0,
        eval_episodes: int = 5,
        progress: Optional[Callable[[int, dict], None]] = None,
    ) -> dict[str, list]:
        """Run the agent for ``total_timesteps`` env steps.

        Returns a history dict: {'env_step': [...], 'episode_return': [...],
        'critic_loss': [...], 'actor_loss': [...], 'alpha': [...],
        'eval_return': [...], 'eval_step': [...]}.

        ``progress`` callback (if given) is invoked every step with
        ``(env_step, info_dict)`` for live plotting.
        """
        history: dict[str, list] = {
            "env_step": [], "episode_return": [], "critic_loss": [],
            "actor_loss": [], "alpha": [], "eval_return": [], "eval_step": [],
        }

        obs, _ = self.env.reset(seed=self.cfg.seed)
        episode_return = 0.0
        episode_len = 0
        episode_count = 0

        t0 = time.time()

        while self._env_steps < total_timesteps:
            # Action selection: uniform random until warmstart, policy after
            if self._env_steps < self.cfg.warmstart_steps:
                action = self.env.action_space.sample().astype(np.float32)
            else:
                action, _ = self.predict(obs, deterministic=False)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)

            # Store with `terminated` (not `done`) so truncations don't
            # incorrectly zero the bootstrap target.
            self.replay.add(
                actor_obs=obs["actor"], critic_obs=obs["critic"],
                action=action, reward=reward,
                next_actor_obs=next_obs["actor"], next_critic_obs=next_obs["critic"],
                terminated=terminated,
            )

            episode_return += reward
            episode_len += 1
            self._env_steps += 1
            self.replay.advance(1)
            obs = next_obs

            # Train if we have enough warm-up data
            train_info = {}
            if self._env_steps >= self.cfg.warmstart_steps and self.replay.size > self.cfg.batch_size:
                train_info = self._train_step()

            if done:
                history["env_step"].append(self._env_steps)
                history["episode_return"].append(episode_return)
                obs, _ = self.env.reset()
                episode_count += 1
                episode_return = 0.0
                episode_len = 0

            if train_info:
                history["critic_loss"].append(train_info.get("critic_loss"))
                history["actor_loss"].append(train_info.get("actor_loss"))
                history["alpha"].append(train_info.get("alpha"))

            if progress is not None and self._env_steps % 100 == 0:
                progress(self._env_steps, train_info or {})

            if log_every and self._env_steps % log_every == 0:
                last_returns = history["episode_return"][-20:]
                mean_ret = np.mean(last_returns) if last_returns else float("nan")
                fps = self._env_steps / max(time.time() - t0, 1e-3)
                print(
                    f"[step {self._env_steps:>7}] "
                    f"ep_return(20)={mean_ret:>8.2f}  "
                    f"critic_loss={train_info.get('critic_loss', float('nan')):>8.3f}  "
                    f"actor_loss={train_info.get('actor_loss', float('nan')):>8.3f}  "
                    f"α={train_info.get('alpha', float('nan')):>6.3f}  "
                    f"fps={fps:>5.0f}",
                    flush=True,
                )

            if eval_every and self._env_steps % eval_every == 0:
                eval_ret = self._evaluate(eval_episodes)
                history["eval_return"].append(eval_ret)
                history["eval_step"].append(self._env_steps)
                print(f"[step {self._env_steps:>7}] eval_return={eval_ret:.2f}", flush=True)

        return history

    @torch.no_grad()
    def _evaluate(self, n_episodes: int) -> float:
        returns = []
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            ep_ret = 0.0
            done = False
            while not done:
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                ep_ret += reward
                done = bool(terminated or truncated)
            returns.append(ep_ret)
        return float(np.mean(returns))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu().numpy(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "temp_opt": self.temp_opt.state_dict(),
                "config": self.cfg.__dict__,
                "env_steps": self._env_steps,
                "gradient_steps": self._gradient_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(torch.as_tensor(ckpt["log_alpha"], device=self.device))
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.temp_opt.load_state_dict(ckpt["temp_opt"])
        self._env_steps = int(ckpt.get("env_steps", 0))
        self._gradient_steps = int(ckpt.get("gradient_steps", 0))
