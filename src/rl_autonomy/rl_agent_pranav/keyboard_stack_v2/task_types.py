from __future__ import annotations

from dataclasses import dataclass, field


Position3D = tuple[float, float, float]


@dataclass(frozen=True)
class KeyTarget:
    key_id: str
    position: Position3D
    description: str = ""


@dataclass(frozen=True)
class StepResult:
    target: KeyTarget
    verified: bool
    attempts: int
    message: str = ""


@dataclass(frozen=True)
class StateTransition:
    previous_state: str
    next_state: str
    reason: str


@dataclass
class RunResult:
    code: str
    strategy_name: str
    success: bool
    final_state: str
    completed_steps: list[StepResult] = field(default_factory=list)
    failed_step: StepResult | None = None
    transitions: list[StateTransition] = field(default_factory=list)


KeyLayout = dict[str, KeyTarget]
