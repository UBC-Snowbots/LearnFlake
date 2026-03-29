from __future__ import annotations

from abc import ABC, abstractmethod

try:
    from .task_types import KeyTarget
except ImportError:
    from task_types import KeyTarget


class KeyPressBackend(ABC):
    @abstractmethod
    def move_to(self, target: KeyTarget) -> None:
        raise NotImplementedError

    @abstractmethod
    def press(self, target: KeyTarget) -> None:
        raise NotImplementedError

    @abstractmethod
    def verify(self, target: KeyTarget) -> bool:
        raise NotImplementedError


class MockKeyPressBackend(KeyPressBackend):
    """
    Mock backend for deterministic Omega tests.

    `verify_plan` maps key ids to a sequence of booleans. Each verify call pops the
    next outcome. If a key is not present, verification defaults to success.
    """

    def __init__(self, verify_plan: dict[str, list[bool]] | None = None):
        self.verify_plan = {str(key): list(values) for key, values in (verify_plan or {}).items()}
        self.events: list[str] = []

    def move_to(self, target: KeyTarget) -> None:
        self.events.append(f"move:{target.key_id}@{target.position}")

    def press(self, target: KeyTarget) -> None:
        self.events.append(f"press:{target.key_id}")

    def verify(self, target: KeyTarget) -> bool:
        self.events.append(f"verify:{target.key_id}")
        plan = self.verify_plan.get(str(target.key_id))
        if plan:
            return bool(plan.pop(0))
        return True
