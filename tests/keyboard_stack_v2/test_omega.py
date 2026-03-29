from __future__ import annotations

import sys
from pathlib import Path

import pytest


V2_DIR = Path(__file__).resolve().parents[2] / "src" / "rl_autonomy" / "rl_agent_pranav" / "keyboard_stack_v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from backends import MockKeyPressBackend
from layout import DEFAULT_KEY_LAYOUT
from omega import EngineState, KeyboardDeciderEngine, SequentialCodeDecider


def make_engine(code: str, verify_plan: dict[str, list[bool]] | None = None, max_retries: int = 1) -> KeyboardDeciderEngine:
    return KeyboardDeciderEngine(
        code=code,
        strategy=SequentialCodeDecider(),
        backend=MockKeyPressBackend(verify_plan=verify_plan),
        layout=DEFAULT_KEY_LAYOUT,
        max_retries=max_retries,
    )


def test_sequential_decode_preserves_code_order() -> None:
    targets = SequentialCodeDecider().decode("1203", DEFAULT_KEY_LAYOUT)
    assert [target.key_id for target in targets] == ["1", "2", "0", "3"]


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("", "Code must not be empty."),
        ("  ", "Code must not be empty."),
        ("12a", "Omega codes must contain only digits 0-9."),
    ],
)
def test_decode_rejects_invalid_codes(code: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SequentialCodeDecider().decode(code, DEFAULT_KEY_LAYOUT)


def test_engine_success_path_completes_all_keys() -> None:
    engine = make_engine("123")
    result = engine.run()

    assert result.success is True
    assert result.final_state == EngineState.COMPLETED.value
    assert [step.target.key_id for step in result.completed_steps] == ["1", "2", "3"]
    assert all(step.attempts == 1 for step in result.completed_steps)
    assert result.failed_step is None
    assert result.transitions[0].next_state == EngineState.DECODING.value
    assert result.transitions[-1].next_state == EngineState.COMPLETED.value


def test_engine_retries_and_then_succeeds() -> None:
    engine = make_engine("5", verify_plan={"5": [False, True]}, max_retries=1)
    result = engine.run()

    assert result.success is True
    assert result.final_state == EngineState.COMPLETED.value
    assert len(result.completed_steps) == 1
    assert result.completed_steps[0].attempts == 2
    assert "retry 5" in [transition.reason for transition in result.transitions]


def test_engine_fails_after_retry_budget_exhausted() -> None:
    engine = make_engine("7", verify_plan={"7": [False, False]}, max_retries=1)
    result = engine.run()

    assert result.success is False
    assert result.final_state == EngineState.FAILED.value
    assert result.failed_step is not None
    assert result.failed_step.target.key_id == "7"
    assert result.failed_step.attempts == 2
    assert result.failed_step.message == "verification failed after retry budget exhausted"
    assert [step.target.key_id for step in result.completed_steps] == []


def test_mock_backend_event_order_matches_engine_flow() -> None:
    engine = make_engine("8")
    result = engine.run()

    assert result.success is True
    assert engine.backend.events == [
        "move:8@(0.07, -0.04, 0.828)",
        "press:8",
        "verify:8",
    ]
