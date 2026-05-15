"""Tests for envs._wrapper_utils — the single helper used to walk wrapper stacks."""
from __future__ import annotations

import warnings

import gymnasium as gym
import pytest

warnings.filterwarnings("ignore")


class _Inner(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (3,))
    action_space = gym.spaces.Box(-1, 1, (1,))

    def reset(self, **kw):
        import numpy as np
        return np.zeros(3), {}

    def step(self, a):
        import numpy as np
        return np.zeros(3), 0.0, False, False, {}


class _W1(gym.Wrapper):
    pass


class _W2(gym.Wrapper):
    pass


def test_find_inner_through_two_wrappers():
    from rl_autonomy.envs._wrapper_utils import find_inner
    inner = _Inner()
    stack = _W2(_W1(inner))
    assert find_inner(stack, _Inner) is inner
    assert find_inner(stack, _W1) is not None
    assert find_inner(stack, _W2) is stack  # the outermost wrapper itself


def test_find_inner_returns_none_when_missing():
    from rl_autonomy.envs._wrapper_utils import find_inner

    class _Unused(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (1,))
        action_space = gym.spaces.Box(-1, 1, (1,))
        def reset(self, **kw):
            import numpy as np
            return np.zeros(1), {}
        def step(self, a):
            import numpy as np
            return np.zeros(1), 0.0, False, False, {}

    stack = _W1(_Inner())
    assert find_inner(stack, _Unused) is None


def test_find_inner_follows_underlying_attribute():
    """Convention: a wrapper that holds the inner env on `.underlying` (not
    `.env`) should still be discoverable."""
    from rl_autonomy.envs._wrapper_utils import find_inner

    class _UnderlyingHolder(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.underlying = env._inner

    inner = _Inner()
    middle = _W1(inner)
    middle._inner = inner   # type: ignore[attr-defined]
    holder = _UnderlyingHolder(middle)
    # find_inner should still find _Inner by walking through .env
    assert find_inner(holder, _Inner) is inner


def test_require_inner_raises():
    from rl_autonomy.envs._wrapper_utils import require_inner

    class _Missing(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (1,))
        action_space = gym.spaces.Box(-1, 1, (1,))
        def reset(self, **kw):
            import numpy as np
            return np.zeros(1), {}
        def step(self, a):
            import numpy as np
            return np.zeros(1), 0.0, False, False, {}

    with pytest.raises(RuntimeError, match="could not find"):
        require_inner(_W1(_Inner()), _Missing)
