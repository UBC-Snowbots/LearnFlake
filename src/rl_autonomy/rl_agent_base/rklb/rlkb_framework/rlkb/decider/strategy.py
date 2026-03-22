from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from .types import KeyLayout, KeyTarget


class DeciderStrategy(ABC):
    name: str

    @abstractmethod
    def decode(self, code: str, layout: KeyLayout) -> List[KeyTarget]:
        raise NotImplementedError


class SequentialCodeDecider(DeciderStrategy):
    name = "sequence"

    def decode(self, code: str, layout: KeyLayout) -> List[KeyTarget]:
        normalized = code.strip()
        if not normalized:
            raise ValueError("Code must not be empty.")
        if not normalized.isdigit():
            raise ValueError("Omega codes must contain only digits 0-9.")

        unknown = [char for char in normalized if char not in layout]
        if unknown:
            raise ValueError(f"Unsupported code symbols: {', '.join(sorted(set(unknown)))}")

        return [layout[char] for char in normalized]


class DeciderRegistry:
    def __init__(self):
        self._strategies: Dict[str, DeciderStrategy] = {}

    def register(self, strategy: DeciderStrategy) -> None:
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> DeciderStrategy:
        if name not in self._strategies:
            available = ", ".join(sorted(self._strategies))
            raise KeyError(f"Unknown strategy '{name}'. Available: {available}")
        return self._strategies[name]

    def names(self) -> List[str]:
        return sorted(self._strategies)


def build_default_registry() -> DeciderRegistry:
    registry = DeciderRegistry()
    registry.register(SequentialCodeDecider())
    return registry
