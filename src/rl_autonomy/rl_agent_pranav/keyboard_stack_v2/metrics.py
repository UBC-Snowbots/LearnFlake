from __future__ import annotations

from dataclasses import dataclass

try:
    from .backends import MockKeyPressBackend
    from .layout import DEFAULT_KEY_LAYOUT
    from .omega import KeyboardDeciderEngine, SequentialCodeDecider
    from .perception import KeyDetection, TargetKeySelector
except ImportError:
    from backends import MockKeyPressBackend
    from layout import DEFAULT_KEY_LAYOUT
    from omega import KeyboardDeciderEngine, SequentialCodeDecider
    from perception import KeyDetection, TargetKeySelector


@dataclass(frozen=True)
class SelectionCase:
    requested_key_id: str
    detections: list[KeyDetection]
    expected_key_id: str | None


@dataclass(frozen=True)
class EngineScenario:
    code: str
    verify_plan: dict[str, list[bool]] | None = None
    max_retries: int = 1
    expected_success: bool = True
    expected_completed_keys: int | None = None


@dataclass(frozen=True)
class SelectionMetrics:
    total_cases: int
    exact_matches: int
    misses: int
    false_matches: int
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    exact_match_accuracy: float
    precision: float
    recall: float


@dataclass(frozen=True)
class EngineAccuracyMetrics:
    total_runs: int
    successful_runs: int
    failed_runs: int
    expected_outcome_matches: int
    total_requested_keys: int
    attempted_keys: int
    completed_keys: int
    failed_keys: int
    first_pass_verified_keys: int
    retry_verified_keys: int
    total_verify_attempts: int
    run_success_rate: float
    expected_outcome_accuracy: float
    key_completion_rate: float
    first_pass_completion_rate: float
    retry_recovery_rate: float
    average_attempts_per_attempted_key: float
    average_attempts_per_completed_key: float


def evaluate_target_selection(
    selector: TargetKeySelector,
    cases: list[SelectionCase],
) -> SelectionMetrics:
    exact_matches = 0
    misses = 0
    false_matches = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for case in cases:
        selection = selector.select(case.requested_key_id, case.detections)
        predicted_key = selection.detection.key_id if selection.detection is not None else None
        actual_positive = case.expected_key_id is not None
        predicted_positive = predicted_key is not None

        if predicted_key == case.expected_key_id:
            exact_matches += 1
            if actual_positive:
                true_positives += 1
            else:
                true_negatives += 1
        elif case.expected_key_id is None:
            false_matches += 1
            false_positives += 1
        else:
            misses += 1
            false_negatives += 1
            if predicted_positive:
                false_positives += 1

    total_cases = len(cases)
    accuracy = float(exact_matches / total_cases) if total_cases else 0.0
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    return SelectionMetrics(
        total_cases=total_cases,
        exact_matches=exact_matches,
        misses=misses,
        false_matches=false_matches,
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        exact_match_accuracy=accuracy,
        precision=float(true_positives / precision_denominator) if precision_denominator else 0.0,
        recall=float(true_positives / recall_denominator) if recall_denominator else 0.0,
    )


def evaluate_engine_scenarios(
    scenarios: list[EngineScenario],
) -> EngineAccuracyMetrics:
    successful_runs = 0
    expected_outcome_matches = 0
    total_requested_keys = 0
    attempted_keys = 0
    completed_keys = 0
    failed_keys = 0
    first_pass_verified_keys = 0
    retry_verified_keys = 0
    total_verify_attempts = 0

    for scenario in scenarios:
        engine = KeyboardDeciderEngine(
            code=scenario.code,
            strategy=SequentialCodeDecider(),
            backend=MockKeyPressBackend(verify_plan=scenario.verify_plan),
            layout=DEFAULT_KEY_LAYOUT,
            max_retries=scenario.max_retries,
        )
        result = engine.run()

        total_requested_keys += len(scenario.code.strip())
        completed_keys += len(result.completed_steps)
        attempted_keys += len(result.completed_steps) + int(result.failed_step is not None)
        successful_runs += int(result.success)
        first_pass_verified_keys += sum(int(step.attempts == 1) for step in result.completed_steps)
        retry_verified_keys += sum(int(step.attempts > 1) for step in result.completed_steps)
        total_verify_attempts += sum(step.attempts for step in result.completed_steps)
        if result.failed_step is not None:
            failed_keys += 1
            total_verify_attempts += result.failed_step.attempts

        matches_expectation = result.success == scenario.expected_success
        if scenario.expected_completed_keys is not None:
            matches_expectation = matches_expectation and (
                len(result.completed_steps) == scenario.expected_completed_keys
            )
        expected_outcome_matches += int(matches_expectation)

    total_runs = len(scenarios)
    failed_runs = total_runs - successful_runs
    return EngineAccuracyMetrics(
        total_runs=total_runs,
        successful_runs=successful_runs,
        failed_runs=failed_runs,
        expected_outcome_matches=expected_outcome_matches,
        total_requested_keys=total_requested_keys,
        attempted_keys=attempted_keys,
        completed_keys=completed_keys,
        failed_keys=failed_keys,
        first_pass_verified_keys=first_pass_verified_keys,
        retry_verified_keys=retry_verified_keys,
        total_verify_attempts=total_verify_attempts,
        run_success_rate=float(successful_runs / total_runs) if total_runs else 0.0,
        expected_outcome_accuracy=float(expected_outcome_matches / total_runs) if total_runs else 0.0,
        key_completion_rate=float(completed_keys / total_requested_keys) if total_requested_keys else 0.0,
        first_pass_completion_rate=(
            float(first_pass_verified_keys / total_requested_keys) if total_requested_keys else 0.0
        ),
        retry_recovery_rate=float(retry_verified_keys / total_requested_keys) if total_requested_keys else 0.0,
        average_attempts_per_attempted_key=(
            float(total_verify_attempts / attempted_keys) if attempted_keys else 0.0
        ),
        average_attempts_per_completed_key=(
            float(total_verify_attempts / completed_keys) if completed_keys else 0.0
        ),
    )
