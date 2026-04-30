"""Actor and Critic networks for RLPD-SAC.

Designed per TRACKER §4:
  - Actor: 3 × 256 hidden, GELU, LayerNorm. Tanh-squashed diagonal Gaussian.
  - Critic: 3 × 512 hidden, GELU, LayerNorm (RLPD's stability lifesaver).
  - EnsembleCritic: holds N critics; loss is computed for each, target uses min.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# Numerical-stability bounds on the actor's log-std.
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _make_mlp(input_dim: int, hidden: Sequence[int], use_layer_norm: bool) -> nn.Sequential:
    """Linear → LayerNorm → GELU stacked.

    Order matches RLPD's JAX reference impl (Dense → LayerNorm → ReLU). LN
    after the Linear ensures pre-activation features have stable statistics
    even as upstream weights drift during training. The original RLPD paper
    finds this is the difference between converging and not.
    """
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        if use_layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.GELU())
        prev = h
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Tanh-squashed diagonal-Gaussian actor (the SAC standard)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: Sequence[int] = (256, 256, 256),
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.trunk = _make_mlp(obs_dim, hidden, use_layer_norm=use_layer_norm)
        self.head = nn.Linear(hidden[-1], 2 * action_dim)
        # Default-init the head to small values so the policy starts near μ=0, σ≈1.
        nn.init.uniform_(self.head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.trunk(obs)
        mu, log_std = self.head(feats).chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action, return (action_in_[-1,1], log_prob)."""
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        if deterministic:
            x = mu
        else:
            eps = torch.randn_like(mu)
            x = mu + std * eps
        # Tanh squash. log_prob correction: log|det dy/dx| = sum log(1 - tanh(x)^2)
        a = torch.tanh(x)
        # Use the numerically-stable form from spinningup:
        #   log_prob = -0.5*((x-mu)/std)^2 - log(std) - 0.5*log(2π) - sum log(1 - tanh(x)^2)
        # The Gaussian log-prob:
        log_prob = -0.5 * ((x - mu) / (std + 1e-8)).pow(2) - log_std - 0.5 * torch.log(
            torch.tensor(2.0 * torch.pi, device=obs.device)
        )
        # Tanh correction. Using the stable identity:
        #   log(1 - tanh(x)^2) = 2 * (log(2) - x - softplus(-2x))
        log_prob -= 2.0 * (
            torch.log(torch.tensor(2.0, device=obs.device)) - x - F.softplus(-2.0 * x)
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return a, log_prob


class Critic(nn.Module):
    """Single Q(s, a) → scalar. Wider than the actor (RLPD/BRO recipe)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: Sequence[int] = (512, 512, 512),
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.trunk = _make_mlp(obs_dim + action_dim, hidden, use_layer_norm=use_layer_norm)
        self.head = nn.Linear(hidden[-1], 1)
        nn.init.uniform_(self.head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.head(self.trunk(x))


class EnsembleCritic(nn.Module):
    """N independent Critics. Returns a stacked tensor; min is taken externally."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_critics: int = 2,
        hidden: Sequence[int] = (512, 512, 512),
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.critics = nn.ModuleList(
            [Critic(obs_dim, action_dim, hidden=hidden, use_layer_norm=use_layer_norm)
             for _ in range(n_critics)]
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return tensor of shape (n_critics, batch, 1)."""
        return torch.stack([c(obs, action) for c in self.critics], dim=0)
