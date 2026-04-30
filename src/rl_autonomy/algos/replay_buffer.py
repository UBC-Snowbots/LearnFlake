"""Replay buffers.

Two flavours:

  ReplayBuffer
      Standard ring buffer. Stores actor and critic observation views
      separately so an asymmetric AC can fetch each. All tensors held on
      `device` (CPU or GPU) — GPU-resident buffers eliminate the train-step
      memcopy bottleneck for UTD=10.

  SymmetricReplayBuffer
      Wraps a *demo* buffer and an *online* buffer. Each call to .sample(N)
      draws ⌊N·f⌋ from demos and the rest from online. The fraction f
      decays linearly from f_init → f_final over `decay_steps` env steps.
      This is the RLPD trick for "RL with prior data".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class Batch:
    actor_obs: torch.Tensor          # (N, actor_dim)
    critic_obs: torch.Tensor         # (N, critic_dim)
    action: torch.Tensor             # (N, action_dim)
    reward: torch.Tensor             # (N, 1)
    next_actor_obs: torch.Tensor
    next_critic_obs: torch.Tensor
    terminated: torch.Tensor         # (N, 1) — 1.0 if episode ended (true terminal, not truncation)


class ReplayBuffer:
    """Single-buffer ring storage on `device`."""

    def __init__(
        self,
        capacity: int,
        actor_dim: int,
        critic_dim: int,
        action_dim: int,
        device: torch.device | str = "cpu",
    ):
        self.capacity = int(capacity)
        self.device = torch.device(device)

        # Allocate as float32 tensors directly on `device`. For 1M transitions
        # × (108 + 38 + 7 + 1 + 108 + 38 + 1) ≈ 300 floats × 4 bytes ≈ 1.2 GB.
        # Fits comfortably on the 8 GB RTX 5060.
        self.actor_obs = torch.zeros((capacity, actor_dim), device=self.device)
        self.critic_obs = torch.zeros((capacity, critic_dim), device=self.device)
        self.action = torch.zeros((capacity, action_dim), device=self.device)
        self.reward = torch.zeros((capacity, 1), device=self.device)
        self.next_actor_obs = torch.zeros((capacity, actor_dim), device=self.device)
        self.next_critic_obs = torch.zeros((capacity, critic_dim), device=self.device)
        self.terminated = torch.zeros((capacity, 1), device=self.device)

        self._size = 0
        self._ptr = 0

    @property
    def size(self) -> int:
        return self._size

    def add(
        self,
        actor_obs: np.ndarray,
        critic_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_actor_obs: np.ndarray,
        next_critic_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        i = self._ptr
        self.actor_obs[i] = torch.as_tensor(actor_obs, device=self.device)
        self.critic_obs[i] = torch.as_tensor(critic_obs, device=self.device)
        self.action[i] = torch.as_tensor(action, device=self.device)
        self.reward[i, 0] = float(reward)
        self.next_actor_obs[i] = torch.as_tensor(next_actor_obs, device=self.device)
        self.next_critic_obs[i] = torch.as_tensor(next_critic_obs, device=self.device)
        self.terminated[i, 0] = float(bool(terminated))
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, generator: Optional[torch.Generator] = None) -> Batch:
        if self._size == 0:
            raise RuntimeError("cannot sample from an empty buffer")
        idx = torch.randint(
            0, self._size, (batch_size,), device=self.device, generator=generator
        )
        return Batch(
            actor_obs=self.actor_obs[idx],
            critic_obs=self.critic_obs[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            next_actor_obs=self.next_actor_obs[idx],
            next_critic_obs=self.next_critic_obs[idx],
            terminated=self.terminated[idx],
        )


class SymmetricReplayBuffer:
    """Demos + online with a controlled split per batch.

    ``f(t) = f_init + (f_final − f_init) · min(1, t / decay_steps)`` is the
    fraction of each batch drawn from the demo buffer. ``t`` is incremented
    by the caller via ``advance(env_steps)``. With f_init=0.5 → f_final=0.25,
    early training is heavily demo-biased; late training is demo-light.

    If ``demo_buffer.size == 0`` (no demos available — v1 case), behaves
    exactly like the underlying online buffer.
    """

    def __init__(
        self,
        online: ReplayBuffer,
        demos: ReplayBuffer,
        f_init: float = 0.5,
        f_final: float = 0.25,
        decay_steps: int = 500_000,
    ):
        if online.actor_obs.shape[1] != demos.actor_obs.shape[1]:
            raise ValueError("online and demo buffers must have same actor_dim")
        self.online = online
        self.demos = demos
        self.f_init = float(f_init)
        self.f_final = float(f_final)
        self.decay_steps = int(decay_steps)
        self._t = 0

    @property
    def size(self) -> int:
        return self.online.size + self.demos.size

    def add(self, *args, **kwargs) -> None:
        """Add to the **online** buffer. Demo buffer is loaded once, externally."""
        self.online.add(*args, **kwargs)

    def advance(self, n_env_steps: int) -> None:
        self._t += int(n_env_steps)

    def current_demo_fraction(self) -> float:
        if self.demos.size == 0:
            return 0.0
        ratio = min(1.0, self._t / max(1, self.decay_steps))
        return self.f_init + (self.f_final - self.f_init) * ratio

    def sample(self, batch_size: int, generator: Optional[torch.Generator] = None) -> Batch:
        f = self.current_demo_fraction()
        n_demo = int(round(batch_size * f))
        n_online = batch_size - n_demo
        if n_online > 0 and self.online.size > 0:
            online = self.online.sample(n_online, generator=generator)
        else:
            online = None
        if n_demo > 0 and self.demos.size > 0:
            demo = self.demos.sample(n_demo, generator=generator)
        else:
            demo = None

        if online is None and demo is None:
            raise RuntimeError("nothing to sample — both buffers empty")
        if online is None:
            return demo
        if demo is None:
            return online

        return Batch(
            actor_obs=torch.cat([online.actor_obs, demo.actor_obs]),
            critic_obs=torch.cat([online.critic_obs, demo.critic_obs]),
            action=torch.cat([online.action, demo.action]),
            reward=torch.cat([online.reward, demo.reward]),
            next_actor_obs=torch.cat([online.next_actor_obs, demo.next_actor_obs]),
            next_critic_obs=torch.cat([online.next_critic_obs, demo.next_critic_obs]),
            terminated=torch.cat([online.terminated, demo.terminated]),
        )
