from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from transitions import Machine

from .backends import KeyPressBackend
from .strategy import DeciderStrategy
from .types import KeyLayout, RunResult, StateTransition, StepResult


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


@dataclass
class DeciderEngine:
    code: str
    strategy: DeciderStrategy
    backend: KeyPressBackend
    layout: KeyLayout
    max_retries: int = 1
    state: EngineState = EngineState.IDLE
    transitions: List[StateTransition] = field(default_factory=list)
    machine: Machine = field(init=False, repr=False)
    _targets: List = field(init=False, repr=False, default_factory=list)
    _current_index: int = field(init=False, repr=False, default=0)
    _current_attempt: int = field(init=False, repr=False, default=0)
    _current_step: Optional[StepResult] = field(init=False, repr=False, default=None)
    _completed_steps: List[StepResult] = field(init=False, repr=False, default_factory=list)
    _failed_step: Optional[StepResult] = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.machine = Machine(
            model=self,
            states=[state.value for state in EngineState],
            initial=self.state.value,
            auto_transitions=False,
            send_event=True,
            queued=True,
            after_state_change="_record_transition",
        )
        self._build_machine()

    def run(self) -> RunResult:
        self._reset_run_state()
        self.start()
        return RunResult(
            code=self.code,
            strategy_name=self.strategy.name,
            success=self.state == EngineState.COMPLETED.value,
            final_state=self.state,
            completed_steps=list(self._completed_steps),
            failed_step=self._failed_step,
            transitions=list(self.transitions),
        )

    def _build_machine(self) -> None:
        self.machine.add_transition("start", EngineState.IDLE.value, EngineState.DECODING.value, after="_decode_code")
        self.machine.add_transition("decoded", EngineState.DECODING.value, EngineState.READY.value, after="_prepare_target")
        self.machine.add_transition("move", EngineState.READY.value, EngineState.MOVING.value, after="_move_to_target")
        self.machine.add_transition("press", EngineState.MOVING.value, EngineState.PRESSING.value, after="_press_target")
        self.machine.add_transition("verify", EngineState.PRESSING.value, EngineState.VERIFYING.value, after="_verify_target")
        self.machine.add_transition("advance", EngineState.VERIFYING.value, EngineState.ADVANCING.value, conditions="_last_verification_succeeded", after="_complete_current_step")
        self.machine.add_transition("retry", EngineState.VERIFYING.value, EngineState.READY.value, conditions="_can_retry", after="_prepare_retry")
        self.machine.add_transition("fail", EngineState.VERIFYING.value, EngineState.FAILED.value, unless=["_last_verification_succeeded", "_can_retry"], after="_mark_failed")
        self.machine.add_transition("next_target", EngineState.ADVANCING.value, EngineState.READY.value, conditions="_has_pending_targets", after="_prepare_target")
        self.machine.add_transition("finish", EngineState.ADVANCING.value, EngineState.COMPLETED.value, unless="_has_pending_targets")

    def _reset_run_state(self) -> None:
        self.transitions.clear()
        self._targets = []
        self._current_index = 0
        self._current_attempt = 0
        self._current_step = None
        self._completed_steps = []
        self._failed_step = None
        self.machine.set_state(EngineState.IDLE.value, model=self)

    def _decode_code(self, event) -> None:
        self._targets = self.strategy.decode(self.code, self.layout)
        self.decoded()

    def _prepare_target(self, event) -> None:
        self._current_attempt = 0
        self._current_step = None
        self.move()

    def _move_to_target(self, event) -> None:
        self._current_attempt += 1
        self.backend.move_to(self._current_target)
        self.press()

    def _press_target(self, event) -> None:
        self.backend.press(self._current_target)
        self.verify()

    def _verify_target(self, event) -> None:
        verified = self.backend.verify(self._current_target)
        self._current_step = StepResult(
            target=self._current_target,
            verified=verified,
            attempts=self._current_attempt,
            message="press verified" if verified else "verification failed",
        )

        if verified:
            self.advance()
        elif self._can_retry():
            self.retry()
        else:
            self.fail()

    def _complete_current_step(self, event) -> None:
        if self._current_step is not None:
            self._completed_steps.append(self._current_step)
        self._current_index += 1
        if self._has_pending_targets():
            self.next_target()
        else:
            self.finish()

    def _prepare_retry(self, event) -> None:
        self.move()

    def _mark_failed(self, event) -> None:
        if self._current_step is None:
            self._current_step = StepResult(
                target=self._current_target,
                verified=False,
                attempts=self._current_attempt,
                message="verification failed after retry budget exhausted",
            )
        else:
            self._current_step = StepResult(
                target=self._current_step.target,
                verified=False,
                attempts=self._current_step.attempts,
                message="verification failed after retry budget exhausted",
            )
        self._failed_step = self._current_step

    def _record_transition(self, event) -> None:
        transition = event.transition
        self.transitions.append(
            StateTransition(
                previous_state=str(transition.source),
                next_state=str(transition.dest),
                reason=self._describe_transition(str(transition.dest)),
            )
        )

    @property
    def _current_target(self):
        return self._targets[self._current_index]

    def _last_verification_succeeded(self, event=None) -> bool:
        return bool(self._current_step and self._current_step.verified)

    def _can_retry(self, event=None) -> bool:
        return self._current_attempt <= self.max_retries

    def _has_more_targets(self, event=None) -> bool:
        return self._current_index < len(self._targets) - 1

    def _has_pending_targets(self, event=None) -> bool:
        return self._current_index < len(self._targets)

    def _describe_transition(self, next_state: str) -> str:
        next_target = None
        if self._targets and self._current_index < len(self._targets):
            next_target = self._current_target.key_id

        active_target = self._current_step.target.key_id if self._current_step is not None else next_target

        if next_state == EngineState.DECODING.value:
            return "decode startup code"
        if next_state == EngineState.READY.value:
            if self._current_step is not None and not self._current_step.verified:
                return f"retry {active_target}"
            return f"target {next_target} loaded"
        if next_state == EngineState.MOVING.value:
            return f"move to {next_target}"
        if next_state == EngineState.PRESSING.value:
            return f"press {next_target}"
        if next_state == EngineState.VERIFYING.value:
            return f"verify {next_target}"
        if next_state == EngineState.ADVANCING.value:
            return f"{active_target} verified"
        if next_state == EngineState.COMPLETED.value:
            return "all targets verified"
        if next_state == EngineState.FAILED.value:
            return f"{active_target} failed verification"
        return next_state
