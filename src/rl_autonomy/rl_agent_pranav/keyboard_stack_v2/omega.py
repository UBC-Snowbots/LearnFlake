from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

try:
    from .backends import KeyPressBackend
    from .task_types import KeyLayout, RunResult, StateTransition, StepResult
except ImportError:
    from backends import KeyPressBackend
    from task_types import KeyLayout, RunResult, StateTransition, StepResult


class EngineState(str, Enum):
    IDLE = "idle"
    DECODING = "decoding"
    READY = "ready"
    MOVING = "moving"
    PRESSING = "pressing"
    VERIFYING = "verifying"
    ADVANCING = "advancing"
    COMPLETED = "completed"
    FAILED = "failed"


class SequentialCodeDecider:
    name = "sequence"

    def decode(self, code: str, layout: KeyLayout):
        normalized = code.strip()
        if not normalized:
            raise ValueError("Code must not be empty.")
        if not normalized.isdigit():
            raise ValueError("Omega codes must contain only digits 0-9.")
        unknown = [char for char in normalized if char not in layout]
        if unknown:
            symbols = ", ".join(sorted(set(unknown)))
            raise ValueError(f"Unsupported code symbols: {symbols}")
        return [layout[char] for char in normalized]


@dataclass
class KeyboardDeciderEngine:
    code: str
    strategy: SequentialCodeDecider
    backend: KeyPressBackend
    layout: KeyLayout
    max_retries: int = 1
    state: EngineState = EngineState.IDLE
    transitions: list[StateTransition] = field(default_factory=list)

    def run(self) -> RunResult:
        self._reset_run_state()

        self._transition(EngineState.DECODING, "decode startup code")
        targets = self.strategy.decode(self.code, self.layout)

        completed_steps: list[StepResult] = []
        failed_step: StepResult | None = None

        for index, target in enumerate(targets):
            self._transition(EngineState.READY, f"target {target.key_id} loaded")

            attempts = 0
            while True:
                attempts += 1

                self._transition(EngineState.MOVING, f"move to {target.key_id}")
                self.backend.move_to(target)

                self._transition(EngineState.PRESSING, f"press {target.key_id}")
                self.backend.press(target)

                self._transition(EngineState.VERIFYING, f"verify {target.key_id}")
                verified = bool(self.backend.verify(target))

                if verified:
                    step = StepResult(
                        target=target,
                        verified=True,
                        attempts=attempts,
                        message="press verified",
                    )
                    completed_steps.append(step)
                    self._transition(EngineState.ADVANCING, f"{target.key_id} verified")
                    break

                if attempts <= self.max_retries:
                    self._transition(EngineState.READY, f"retry {target.key_id}")
                    continue

                failed_step = StepResult(
                    target=target,
                    verified=False,
                    attempts=attempts,
                    message="verification failed after retry budget exhausted",
                )
                self._transition(EngineState.FAILED, f"{target.key_id} failed verification")
                return RunResult(
                    code=self.code,
                    strategy_name=self.strategy.name,
                    success=False,
                    final_state=self.state.value,
                    completed_steps=completed_steps,
                    failed_step=failed_step,
                    transitions=list(self.transitions),
                )

            if index == len(targets) - 1:
                continue

        self._transition(EngineState.COMPLETED, "all targets verified")
        return RunResult(
            code=self.code,
            strategy_name=self.strategy.name,
            success=True,
            final_state=self.state.value,
            completed_steps=completed_steps,
            failed_step=failed_step,
            transitions=list(self.transitions),
        )

    def _reset_run_state(self) -> None:
        self.transitions.clear()
        self.state = EngineState.IDLE

    def _transition(self, next_state: EngineState, reason: str) -> None:
        previous_state = self.state
        self.state = next_state
        self.transitions.append(
            StateTransition(
                previous_state=previous_state.value,
                next_state=next_state.value,
                reason=reason,
            )
        )
