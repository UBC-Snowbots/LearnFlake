"""Debouncing per-person decisions into one-shot events, and the event record we publish."""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass


@dataclass
class GestureEvent:
    label: str  # "beckon" (or "wave")
    confidence: float
    track_id: int
    arm: str | None
    stamp: float  # seconds, same clock as the input frames
    bbox: list[float]  # x1, y1, x2, y2 in image pixels
    image_size: list[int]  # width, height
    bearing_deg: float  # horizontal angle to the person; ROS REP-103 sign: positive = LEFT of centre
    person_height_px: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def bearing_deg(cx: float, image_width: int, hfov_deg: float) -> float:
    """Pinhole bearing of pixel column cx. Positive = left of the optical axis (REP-103)."""
    f = (image_width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return -math.degrees(math.atan2(cx - image_width / 2.0, f))


@dataclass
class DebounceConfig:
    votes_needed: int = 3  # this many positive decisions ...
    votes_window: int = 5  # ... out of the last this many decisions
    min_confidence: float = 0.35
    cooldown_s: float = 4.0  # per person and label, after an event fires
    labels: tuple[str, ...] = ("beckon",)  # which labels produce events


class Debouncer:
    def __init__(self, cfg: DebounceConfig):
        self.cfg = cfg
        self._votes: dict[int, deque] = {}
        self._last_fire: dict[tuple[int, str], float] = {}

    def update(self, track_id: int, t: float, label: str, confidence: float) -> bool:
        """Feed one decision; True when an event for `label` should fire now."""
        v = self._votes.setdefault(track_id, deque(maxlen=self.cfg.votes_window))
        v.append(label if confidence >= self.cfg.min_confidence else "none")
        if label not in self.cfg.labels:
            return False
        if sum(x == label for x in v) < self.cfg.votes_needed:
            return False
        last = self._last_fire.get((track_id, label))
        if last is not None and t - last < self.cfg.cooldown_s:
            return False
        self._last_fire[(track_id, label)] = t
        return True

    def forget(self, live_ids: set[int]) -> None:
        for tid in [k for k in self._votes if k not in live_ids]:
            del self._votes[tid]
        for key in [k for k in self._last_fire if k[0] not in live_ids]:
            del self._last_fire[key]
