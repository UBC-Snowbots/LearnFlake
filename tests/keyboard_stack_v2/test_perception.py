from __future__ import annotations

import sys
from pathlib import Path

import pytest


V2_DIR = Path(__file__).resolve().parents[2] / "src" / "rl_autonomy" / "rl_agent_pranav" / "keyboard_stack_v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from perception import KeyDetection, MockKeyDetector, PixelBBox, TargetKeySelector


def box(x_min: int, y_min: int, x_max: int, y_max: int) -> PixelBBox:
    return PixelBBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def detection(key_id: str, conf: float, bbox: PixelBBox, source: str = "mock") -> KeyDetection:
    return KeyDetection(key_id=str(key_id), confidence=conf, bbox=bbox, source=source)


def test_bbox_computes_area_and_center() -> None:
    bbox = box(10, 20, 30, 50)
    assert bbox.width == 20
    assert bbox.height == 30
    assert bbox.area == 600
    assert bbox.center_xy == (20.0, 35.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(x_min=5, y_min=0, x_max=5, y_max=10), "x_max"),
        (dict(x_min=1, y_min=2, x_max=3, y_max=2), "y_max"),
    ],
)
def test_bbox_rejects_invalid_extents(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PixelBBox(**kwargs)


def test_detection_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        detection("1", 1.2, box(0, 0, 10, 10))


def test_mock_detector_returns_stable_detection_list() -> None:
    dets = [detection("2", 0.8, box(0, 0, 10, 10))]
    detector = MockKeyDetector(detections=dets)
    out = detector.detect(frame=None)
    assert out == dets
    assert out is not dets


def test_selector_picks_requested_key_only() -> None:
    selector = TargetKeySelector(min_confidence=0.25)
    dets = [
        detection("1", 0.95, box(0, 0, 10, 10)),
        detection("2", 0.80, box(10, 10, 20, 20)),
    ]
    selection = selector.select("2", dets)
    assert selection.matched is True
    assert selection.detection is not None
    assert selection.detection.key_id == "2"


def test_selector_ignores_target_below_threshold() -> None:
    selector = TargetKeySelector(min_confidence=0.50)
    dets = [detection("5", 0.40, box(0, 0, 10, 10))]
    selection = selector.select("5", dets)
    assert selection.matched is False
    assert selection.detection is None
    assert "below confidence threshold" in selection.reason


def test_selector_reports_missing_target() -> None:
    selector = TargetKeySelector(min_confidence=0.25)
    dets = [detection("1", 0.90, box(0, 0, 10, 10))]
    selection = selector.select("9", dets)
    assert selection.matched is False
    assert selection.detection is None
    assert selection.reason == "key 9 not detected"


def test_selector_prefers_higher_confidence_then_area() -> None:
    selector = TargetKeySelector(min_confidence=0.25)
    dets = [
        detection("3", 0.80, box(0, 0, 50, 50)),
        detection("3", 0.95, box(0, 0, 10, 10)),
        detection("3", 0.95, box(0, 0, 20, 20)),
    ]
    selection = selector.select("3", dets)
    assert selection.matched is True
    assert selection.detection is not None
    assert selection.detection.confidence == 0.95
    assert selection.detection.bbox.area == 400
