"""Manual key-phase curriculum (TRACKER §7.2).

Three phases of widening key sets:

    Phase A (central alphanumeric, ~20 keys reachable from the home pose)
    Phase B (home + qwerty + bottom + numbers, ~50 keys)
    Phase C (all 87 keys including modifier row, function row, nav cluster)

Advance when the rolling success rate over the current phase exceeds
``advance_threshold``. The user (or wrapper) calls ``sample_key()`` at the
start of each episode and ``report_outcome(key, success)`` after the env
declares the episode done.

Stays in Phase C indefinitely once reached.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..envs.keyboard_layout import PHASE_A_KEYS, PHASE_B_KEYS, PHASE_C_KEYS


@dataclass
class KeyPhaseCurriculum:
    """Phased key sampler with auto-advance.

    Args:
        advance_threshold: rolling success rate over the current phase needed
            before advancing. Default 0.85 (TRACKER §14).
        window: how many recent episodes to average over.
        seed: RNG seed for reproducible key sampling.
    """
    advance_threshold: float = 0.85
    window: int = 200
    seed: int = 0

    _phases: list[list[str]] = field(init=False)
    _phase_idx: int = field(init=False, default=0)
    _outcomes: deque = field(init=False)
    _rng: np.random.Generator = field(init=False)
    _advance_count: int = field(init=False, default=0)

    def __post_init__(self):
        self._phases = [list(PHASE_A_KEYS), list(PHASE_B_KEYS), list(PHASE_C_KEYS)]
        self._outcomes = deque(maxlen=self.window)
        self._rng = np.random.default_rng(self.seed)

    @property
    def current_phase(self) -> int:
        return self._phase_idx

    @property
    def current_phase_keys(self) -> list[str]:
        return self._phases[self._phase_idx]

    @property
    def n_phases(self) -> int:
        return len(self._phases)

    def sample_key(self) -> str:
        keys = self.current_phase_keys
        return str(self._rng.choice(keys))

    def report_outcome(self, key: str, success: bool) -> None:
        self._outcomes.append(bool(success))
        # Try to advance after every report — cheap, no harm if we can't.
        self.maybe_advance()

    def rolling_success_rate(self) -> float:
        if not self._outcomes:
            return 0.0
        return float(np.mean(list(self._outcomes)))

    def maybe_advance(self) -> bool:
        """Advance if conditions are met. Returns True iff advanced."""
        if self._phase_idx >= len(self._phases) - 1:
            return False
        if len(self._outcomes) < self.window:
            return False
        if self.rolling_success_rate() < self.advance_threshold:
            return False
        self._phase_idx += 1
        self._advance_count += 1
        # Reset the rolling window so the next phase starts fresh
        self._outcomes.clear()
        return True

    def state_dict(self) -> dict:
        return {
            "phase_idx": self._phase_idx,
            "outcomes": list(self._outcomes),
            "advance_count": self._advance_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self._phase_idx = int(state["phase_idx"])
        self._outcomes = deque(state["outcomes"], maxlen=self.window)
        self._advance_count = int(state.get("advance_count", 0))
