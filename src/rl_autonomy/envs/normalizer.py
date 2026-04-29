"""Welford-style online running mean + std for observation normalization.

Used by the obs adapter to standardise raw observations before they reach
the policy. State is `(mean, var, count)`; updates are batched per env step.
Save / load with numpy `.npz` so the same statistics ride alongside saved
policy checkpoints.

Matches the algorithm in `stable_baselines3.common.running_mean_std` but
without the SB3 dependency, so it can be used outside the SB3 training loop
(e.g. inside the env wrapper, in tests, in deployment).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RunningMeanStd:
    """Running mean and variance over a fixed-shape vector."""

    mean: np.ndarray
    var: np.ndarray
    count: float

    @classmethod
    def zeros(cls, shape: tuple[int, ...], epsilon: float = 1e-4) -> "RunningMeanStd":
        return cls(
            mean=np.zeros(shape, dtype=np.float64),
            var=np.ones(shape, dtype=np.float64),
            count=epsilon,
        )

    def update(self, batch: np.ndarray) -> None:
        """Update statistics with a batch of samples (shape: [N, *shape])."""
        if batch.shape[1:] != self.mean.shape:
            raise ValueError(
                f"batch shape {batch.shape} does not match accumulator shape {self.mean.shape}"
            )
        batch_mean = batch.mean(axis=0).astype(np.float64)
        batch_var = batch.var(axis=0).astype(np.float64)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """Normalize ``x`` using current statistics, clipped to ±clip after scaling."""
        std = np.sqrt(self.var + 1e-8)
        out = (x.astype(np.float64) - self.mean) / std
        return np.clip(out, -clip, clip).astype(np.float32)

    # ---- serialization ----
    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, var=self.var, count=np.array([self.count]))

    @classmethod
    def load(cls, path: str) -> "RunningMeanStd":
        data = np.load(path)
        return cls(mean=data["mean"], var=data["var"], count=float(data["count"][0]))
