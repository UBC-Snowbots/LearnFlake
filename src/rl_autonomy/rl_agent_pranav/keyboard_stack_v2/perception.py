from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PixelBBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("Bounding box x_max must be greater than x_min")
        if self.y_max <= self.y_min:
            raise ValueError("Bounding box y_max must be greater than y_min")

    @property
    def width(self) -> int:
        return int(self.x_max - self.x_min)

    @property
    def height(self) -> int:
        return int(self.y_max - self.y_min)

    @property
    def area(self) -> int:
        return int(self.width * self.height)

    @property
    def center_xy(self) -> tuple[float, float]:
        return (
            float(self.x_min + self.width / 2.0),
            float(self.y_min + self.height / 2.0),
        )


@dataclass(frozen=True)
class KeyDetection:
    key_id: str
    bbox: PixelBBox
    confidence: float
    source: str = "unknown"

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError("Detection confidence must be in [0, 1]")


@dataclass(frozen=True)
class TargetSelection:
    requested_key_id: str
    detection: KeyDetection | None
    matched: bool
    reason: str


class KeyDetector(ABC):
    @abstractmethod
    def detect(self, frame) -> list[KeyDetection]:
        raise NotImplementedError


class MockKeyDetector(KeyDetector):
    def __init__(self, detections: Iterable[KeyDetection] | None = None):
        self._detections = list(detections or [])

    def detect(self, frame) -> list[KeyDetection]:
        return list(self._detections)


class TargetKeySelector:
    """
    Resolve a requested symbolic key id against a flat set of detections.

    Current ranking is:
    1. matching key id
    2. confidence threshold
    3. highest confidence
    4. largest box area
    """

    def __init__(self, min_confidence: float = 0.25):
        self.min_confidence = float(min_confidence)

    def select(self, requested_key_id: str, detections: Iterable[KeyDetection]) -> TargetSelection:
        requested_key_id = str(requested_key_id)
        all_detections = list(detections)
        candidates = [
            detection
            for detection in all_detections
            if detection.key_id == requested_key_id and float(detection.confidence) >= self.min_confidence
        ]

        if not candidates:
            if any(detection.key_id == requested_key_id for detection in all_detections):
                return TargetSelection(
                    requested_key_id=requested_key_id,
                    detection=None,
                    matched=False,
                    reason=f"key {requested_key_id} seen but below confidence threshold",
                )
            return TargetSelection(
                requested_key_id=requested_key_id,
                detection=None,
                matched=False,
                reason=f"key {requested_key_id} not detected",
            )

        best = max(candidates, key=lambda detection: (float(detection.confidence), detection.bbox.area))
        return TargetSelection(
            requested_key_id=requested_key_id,
            detection=best,
            matched=True,
            reason=f"selected key {requested_key_id} from {len(candidates)} candidate(s)",
        )
