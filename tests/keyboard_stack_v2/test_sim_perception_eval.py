from __future__ import annotations

import math
import sys
from pathlib import Path


V2_DIR = Path(__file__).resolve().parents[2] / "src" / "rl_autonomy" / "rl_agent_pranav" / "keyboard_stack_v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from perception import KeyDetection, PixelBBox
from sim_perception_eval import DetectionMetrics, bbox_iou, format_detection_report, match_detections


def box(x_min: int, y_min: int, x_max: int, y_max: int) -> PixelBBox:
    return PixelBBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def detection(key_id: str, confidence: float, bbox: PixelBBox) -> KeyDetection:
    return KeyDetection(key_id=key_id, confidence=confidence, bbox=bbox, source="test")


def test_bbox_iou_is_zero_for_non_overlapping_boxes() -> None:
    assert bbox_iou(box(0, 0, 10, 10), box(20, 20, 30, 30)) == 0.0


def test_bbox_iou_matches_expected_overlap() -> None:
    iou = bbox_iou(box(0, 0, 10, 10), box(5, 5, 15, 15))
    assert math.isclose(iou, 25.0 / 175.0, rel_tol=1e-9)


def test_match_detections_counts_tp_fp_and_fn() -> None:
    ground_truth = [
        detection("1", 1.0, box(0, 0, 10, 10)),
        detection("2", 1.0, box(20, 20, 30, 30)),
    ]
    predictions = [
        detection("1", 0.9, box(1, 1, 11, 11)),
        detection("2", 0.8, box(100, 100, 110, 110)),
        detection("9", 0.7, box(40, 40, 50, 50)),
    ]

    metrics = match_detections(ground_truth=ground_truth, predictions=predictions, iou_threshold=0.5)
    assert metrics.true_positives == 1
    assert metrics.false_positives == 2
    assert metrics.false_negatives == 1
    assert metrics.exact_match is False
    assert len(metrics.matched_ious) == 1


def test_format_detection_report_includes_core_accuracy_fields() -> None:
    metrics = DetectionMetrics(
        total_frames=3,
        total_ground_truth_boxes=30,
        total_predictions=28,
        true_positives=27,
        false_positives=1,
        false_negatives=3,
        precision=27 / 28,
        recall=27 / 30,
        f1=0.9310344827586207,
        mean_matched_iou=0.82,
        exact_frame_match_rate=2 / 3,
        visible_target_selection_accuracy=0.9,
        evaluated_target_selections=30,
        correct_target_selections=27,
    )

    report = format_detection_report(metrics=metrics, detector_name="test_detector")
    assert "Detector: test_detector" in report
    assert "Precision/recall/F1: 96.4%/90.0%/93.1%" in report
    assert "False positives / false negatives: 1 / 3" in report
    assert "Mean matched IoU: 0.820" in report
    assert "Exact frame match rate: 66.7%" in report
    assert "Visible target selection accuracy: 90.0% (27/30)" in report
