from __future__ import annotations

import math
import sys
from pathlib import Path


V2_DIR = Path(__file__).resolve().parents[2] / "src" / "rl_autonomy" / "rl_agent_pranav" / "keyboard_stack_v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from metrics import (
    EngineScenario,
    SelectionCase,
    evaluate_engine_scenarios,
    evaluate_target_selection,
)
from perception import KeyDetection, PixelBBox, TargetKeySelector


def box(x_min: int, y_min: int, x_max: int, y_max: int) -> PixelBBox:
    return PixelBBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def detection(key_id: str, confidence: float, bbox: PixelBBox) -> KeyDetection:
    return KeyDetection(key_id=key_id, confidence=confidence, bbox=bbox, source="mock")


def test_selection_metrics_report_exact_match_accuracy() -> None:
    selector = TargetKeySelector(min_confidence=0.5)
    cases = [
        SelectionCase(
            requested_key_id="1",
            detections=[detection("1", 0.9, box(0, 0, 10, 10))],
            expected_key_id="1",
        ),
        SelectionCase(
            requested_key_id="2",
            detections=[detection("2", 0.4, box(0, 0, 10, 10))],
            expected_key_id=None,
        ),
        SelectionCase(
            requested_key_id="3",
            detections=[detection("5", 0.9, box(0, 0, 10, 10))],
            expected_key_id=None,
        ),
        SelectionCase(
            requested_key_id="4",
            detections=[
                detection("4", 0.8, box(0, 0, 10, 10)),
                detection("4", 0.95, box(0, 0, 20, 20)),
            ],
            expected_key_id="4",
        ),
    ]

    metrics = evaluate_target_selection(selector, cases)
    assert metrics.total_cases == 4
    assert metrics.exact_matches == 4
    assert metrics.misses == 0
    assert metrics.false_matches == 0
    assert metrics.true_positives == 2
    assert metrics.true_negatives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.exact_match_accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0


def test_engine_accuracy_metrics_report_success_and_completion_rates() -> None:
    scenarios = [
        EngineScenario(code="12", expected_success=True, expected_completed_keys=2),
        EngineScenario(code="5", verify_plan={"5": [False, True]}, expected_success=True, expected_completed_keys=1),
        EngineScenario(code="70", verify_plan={"7": [False, False]}, expected_success=False, expected_completed_keys=0),
    ]

    metrics = evaluate_engine_scenarios(scenarios)
    assert metrics.total_runs == 3
    assert metrics.successful_runs == 2
    assert metrics.failed_runs == 1
    assert metrics.expected_outcome_matches == 3
    assert metrics.total_requested_keys == 5
    assert metrics.attempted_keys == 4
    assert metrics.completed_keys == 3
    assert metrics.failed_keys == 1
    assert metrics.first_pass_verified_keys == 2
    assert metrics.retry_verified_keys == 1
    assert metrics.total_verify_attempts == 6
    assert math.isclose(metrics.run_success_rate, 2 / 3, rel_tol=1e-9)
    assert math.isclose(metrics.expected_outcome_accuracy, 1.0, rel_tol=1e-9)
    assert math.isclose(metrics.key_completion_rate, 3 / 5, rel_tol=1e-9)
    assert math.isclose(metrics.first_pass_completion_rate, 2 / 5, rel_tol=1e-9)
    assert math.isclose(metrics.retry_recovery_rate, 1 / 5, rel_tol=1e-9)
    assert math.isclose(metrics.average_attempts_per_attempted_key, 6 / 4, rel_tol=1e-9)
    assert math.isclose(metrics.average_attempts_per_completed_key, 6 / 3, rel_tol=1e-9)
