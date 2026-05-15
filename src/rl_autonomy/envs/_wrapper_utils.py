"""Helpers for walking wrapper stacks.

Every consumer that wants the inner KeyboardEnv or ObsAdapter ended up
re-implementing the same generator-style descent. This file is the
single source of truth.
"""
from __future__ import annotations

from typing import Any, Optional, TypeVar

import gymnasium as gym


T = TypeVar("T")


def _walk_inward(env: gym.Env):
    """Yield env → env.env → env.env.env → … (incl. starting env)."""
    cur: Any = env
    yield cur
    while True:
        nxt = getattr(cur, "env", None)
        if nxt is None or nxt is cur:
            return
        cur = nxt
        yield cur


def find_inner(env: gym.Env, cls: type[T]) -> Optional[T]:
    """Walk the wrapper stack and return the first instance of ``cls`` found,
    or None. Handles the convention where a wrapper exposes ``.underlying``
    pointing at the inner env (e.g. KeyboardGymEnv → KeyboardEnv)."""
    for cur in _walk_inward(env):
        if isinstance(cur, cls):
            return cur  # type: ignore[return-value]
        underlying = getattr(cur, "underlying", None)
        if underlying is not None and isinstance(underlying, cls):
            return underlying  # type: ignore[return-value]
    return None


def require_inner(env: gym.Env, cls: type[T]) -> T:
    """Like find_inner but raises if not found."""
    found = find_inner(env, cls)
    if found is None:
        raise RuntimeError(
            f"could not find {cls.__name__} in the wrapper stack rooted at "
            f"{type(env).__name__}"
        )
    return found
