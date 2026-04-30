"""Behavior-cloning pretrain — gated for v1 (user direction: skip demos).

Trains an :class:`Actor` (same architecture used by RLPD-SAC) to imitate a
demo dataset. The two-headed (μ, log σ) output makes this a maximum-likelihood
fit to a Tanh-Normal distribution, which is exactly the SAC actor's
parameterization — so the trained weights drop into RLPD without surgery.

Loaded only when demos exist. v1 ships this as functional dead code; v1.1
will wire it into the algorithm port.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .networks import Actor


@dataclass
class BCConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"


class BCPretrain:
    """Train an Actor to predict demonstrator actions via NLL on Tanh-Normal."""

    def __init__(self, actor: Actor, config: BCConfig | None = None):
        self.actor = actor
        self.cfg = config or BCConfig()
        self.opt = torch.optim.AdamW(
            self.actor.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

    def fit(self, obs: np.ndarray, actions: np.ndarray) -> dict[str, list[float]]:
        device = torch.device(self.cfg.device)
        self.actor.to(device)
        x = torch.as_tensor(obs, dtype=torch.float32, device=device)
        y = torch.as_tensor(actions, dtype=torch.float32, device=device)
        n = x.shape[0]

        history = {"epoch": [], "loss": []}
        for epoch in range(self.cfg.epochs):
            perm = torch.randperm(n, device=device)
            losses: list[float] = []
            for i in range(0, n, self.cfg.batch_size):
                idx = perm[i:i + self.cfg.batch_size]
                xb, yb = x[idx], y[idx]
                # NLL of yb under tanh-Normal(μ, σ) parameterized by self.actor.
                # Equivalent to maximizing log p(yb | xb).
                pre_tanh = torch.atanh(yb.clamp(-0.999, 0.999))
                mu, log_std = self.actor(xb)
                std = log_std.exp()
                log_prob = -0.5 * ((pre_tanh - mu) / (std + 1e-8)).pow(2) - log_std \
                    - 0.5 * np.log(2.0 * np.pi)
                # Tanh correction
                log_prob -= 2.0 * (np.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
                loss = -log_prob.sum(dim=-1).mean()

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
                losses.append(float(loss.detach()))

            history["epoch"].append(epoch)
            history["loss"].append(float(np.mean(losses)))
        return history
