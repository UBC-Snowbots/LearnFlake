from __future__ import annotations

try:
    from .metrics import (
        EngineScenario,
        SelectionCase,
        evaluate_engine_scenarios,
        evaluate_target_selection,
    )
    from .perception import KeyDetection, PixelBBox, TargetKeySelector
except ImportError:
    from metrics import EngineScenario, SelectionCase, evaluate_engine_scenarios, evaluate_target_selection
    from perception import KeyDetection, PixelBBox, TargetKeySelector


def box(x_min: int, y_min: int, x_max: int, y_max: int) -> PixelBBox:
    return PixelBBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def detection(key_id: str, confidence: float, bbox: PixelBBox) -> KeyDetection:
    return KeyDetection(key_id=str(key_id), confidence=float(confidence), bbox=bbox, source="synthetic")


def build_controlled_selection_suite() -> list[SelectionCase]:
    return [
        SelectionCase(
            requested_key_id="1",
            detections=[detection("1", 0.95, box(0, 0, 12, 12))],
            expected_key_id="1",
        ),
        SelectionCase(
            requested_key_id="2",
            detections=[detection("2", 0.40, box(0, 0, 12, 12))],
            expected_key_id=None,
        ),
        SelectionCase(
            requested_key_id="3",
            detections=[detection("5", 0.90, box(0, 0, 12, 12))],
            expected_key_id=None,
        ),
        SelectionCase(
            requested_key_id="4",
            detections=[
                detection("4", 0.82, box(0, 0, 12, 12)),
                detection("4", 0.96, box(0, 0, 20, 20)),
            ],
            expected_key_id="4",
        ),
        SelectionCase(
            requested_key_id="7",
            detections=[
                detection("1", 0.92, box(0, 0, 16, 16)),
                detection("7", 0.88, box(20, 0, 36, 16)),
            ],
            expected_key_id="7",
        ),
    ]


def build_noisy_selection_suite() -> list[SelectionCase]:
    return [
        SelectionCase(
            requested_key_id="1",
            detections=[detection("1", 0.95, box(0, 0, 12, 12))],
            expected_key_id="1",
        ),
        SelectionCase(
            requested_key_id="2",
            detections=[detection("2", 0.49, box(0, 0, 12, 12))],
            expected_key_id="2",
        ),
        SelectionCase(
            requested_key_id="3",
            detections=[detection("3", 0.91, box(0, 0, 16, 16))],
            expected_key_id=None,
        ),
        SelectionCase(
            requested_key_id="4",
            detections=[detection("9", 0.93, box(0, 0, 18, 18))],
            expected_key_id=None,
        ),
        SelectionCase(
            requested_key_id="5",
            detections=[
                detection("5", 0.85, box(0, 0, 10, 10)),
                detection("5", 0.60, box(0, 0, 16, 16)),
            ],
            expected_key_id="5",
        ),
    ]


def build_engine_reliability_suite() -> list[EngineScenario]:
    return [
        EngineScenario(code="12", expected_success=True, expected_completed_keys=2),
        EngineScenario(code="5", verify_plan={"5": [False, True]}, expected_success=True, expected_completed_keys=1),
        EngineScenario(code="70", verify_plan={"7": [False, False]}, expected_success=False, expected_completed_keys=0),
        EngineScenario(
            code="890",
            verify_plan={"8": [True], "9": [False, True], "0": [True]},
            expected_success=True,
            expected_completed_keys=3,
        ),
    ]


def generate_accuracy_report(min_confidence: float = 0.5) -> str:
    selector = TargetKeySelector(min_confidence=min_confidence)

    controlled = evaluate_target_selection(selector, build_controlled_selection_suite())
    noisy = evaluate_target_selection(selector, build_noisy_selection_suite())
    engine = evaluate_engine_scenarios(build_engine_reliability_suite())

    return "\n".join(
        [
            f"Selection controlled exact-match accuracy: {controlled.exact_match_accuracy:.1%} "
            f"({controlled.exact_matches}/{controlled.total_cases})",
            f"Selection controlled precision/recall: {controlled.precision:.1%}/{controlled.recall:.1%}",
            f"Selection noisy exact-match accuracy: {noisy.exact_match_accuracy:.1%} "
            f"({noisy.exact_matches}/{noisy.total_cases})",
            f"Selection noisy precision/recall: {noisy.precision:.1%}/{noisy.recall:.1%}",
            f"Engine success rate: {engine.run_success_rate:.1%} "
            f"({engine.successful_runs}/{engine.total_runs})",
            f"Engine key completion rate: {engine.key_completion_rate:.1%} "
            f"({engine.completed_keys}/{engine.total_requested_keys})",
            f"Engine first-pass/retry recovery: {engine.first_pass_completion_rate:.1%}/"
            f"{engine.retry_recovery_rate:.1%}",
            f"Engine expected-outcome accuracy: {engine.expected_outcome_accuracy:.1%} "
            f"({engine.expected_outcome_matches}/{engine.total_runs})",
        ]
    )


def main() -> None:
    print(generate_accuracy_report())


if __name__ == "__main__":
    main()
