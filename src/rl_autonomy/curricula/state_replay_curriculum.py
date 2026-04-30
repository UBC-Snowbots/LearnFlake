"""DemoStart-style state-replay curriculum (TRACKER §7.1).

Given a buffer of demonstration states, reset the env to a sampled demo
state, weighted by per-state success rate so resets concentrate where the
policy has ~50% success rate (the active learning frontier).

v1 SKIPS DEMOS per user direction — there are no demonstration buffers
to draw from. This module ships as a no-op pass-through that always
defers to the env's default reset. v1.1 adds the actual implementation.
"""
from __future__ import annotations

from typing import Any, Optional


class StateReplayCurriculum:
    """No-op for v1. Records the API future versions will implement.

    Future fields (per TRACKER §7.1):
        weight_per_state: dict[state_id → float]
        success_per_state: dict[state_id → success rolling avg]
        sample_temperature: float — softens the weight distribution

    Future methods:
        sample_reset_state() -> sim_state
        report_outcome(state_id, success)
    """

    def __init__(self, demo_buffer: Optional[Any] = None):
        self.demo_buffer = demo_buffer
        if demo_buffer is not None:
            # When demos are wired in (v1.1), the buffer's transitions will
            # become candidate reset states. v1: there's no buffer; we'll
            # raise rather than silently misbehave.
            raise NotImplementedError(
                "StateReplayCurriculum with demos is a v1.1 feature. "
                "v1 ships demo-free; pass demo_buffer=None."
            )

    def is_active(self) -> bool:
        return self.demo_buffer is not None

    def sample_reset_state(self) -> None:
        """Returns None → caller falls back to env's default reset."""
        return None

    def report_outcome(self, state_id: Any, success: bool) -> None:
        """No-op until demos are wired in."""
        return None
