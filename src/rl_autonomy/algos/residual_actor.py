"""Residual actor: frozen base BC actor + learnable residual head.

Per TRACKER §8.3 / ResiP / Residual Off-Policy RL. Gated for v1.

Behavior:
    a_total = clip(base_actor(obs) + residual_actor(obs), -1, 1)

The base is frozen at construction; only the residual receives gradient.
At zero-init, residual ≈ 0 so a_total ≈ base — i.e. RLPD on a residual
starts from BC behavior, then refines.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .networks import Actor


class ResidualActor(nn.Module):
    """Wraps a frozen base actor + a learnable residual.

    Both must have matching ``obs_dim`` and ``action_dim``. The residual's
    final-layer weights are zeroed so the initial total action equals the
    base action exactly — useful for RL fine-tuning to start from a known
    safe behavior.
    """

    def __init__(self, base: Actor, residual: Optional[Actor] = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.residual = residual or Actor(
            obs_dim=base.obs_dim,
            action_dim=base.action_dim,
            hidden=(256, 256, 256),
            use_layer_norm=True,
        )
        # Zero-init the residual head so the residual starts at exactly 0.
        nn.init.zeros_(self.residual.head.weight)
        nn.init.zeros_(self.residual.head.bias)

    @property
    def obs_dim(self) -> int:
        return self.base.obs_dim

    @property
    def action_dim(self) -> int:
        return self.base.action_dim

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Combined (μ, log_std) — sum the means, sum the log-vars (≈ sum log_stds).
        mu_b, ls_b = self.base(obs)
        mu_r, ls_r = self.residual(obs)
        # Combined std: sqrt(σ_b² + σ_r²). In log space: 0.5 log(exp(2 ls_b) + exp(2 ls_r)).
        ls_combined = 0.5 * torch.logsumexp(
            torch.stack([2 * ls_b, 2 * ls_r], dim=0), dim=0
        )
        return mu_b + mu_r, ls_combined

    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        # Same tanh-squashed sample as Actor.sample, on the combined
        # (mu, log_std). Inline to avoid double-tanh.
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        if deterministic:
            x = mu
        else:
            x = mu + std * torch.randn_like(mu)
        a = torch.tanh(x)
        # Same numerics as Actor.sample
        log_prob = -0.5 * ((x - mu) / (std + 1e-8)).pow(2) - log_std - 0.5 * torch.log(
            torch.tensor(2.0 * torch.pi, device=obs.device)
        )
        log_prob -= 2.0 * (torch.log(torch.tensor(2.0, device=obs.device)) - x
                           - torch.nn.functional.softplus(-2.0 * x))
        return a, log_prob.sum(dim=-1, keepdim=True)
