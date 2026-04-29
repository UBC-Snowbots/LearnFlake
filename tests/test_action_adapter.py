"""ActionAdapter masks correctly and the smoothing filter actually smooths."""
from __future__ import annotations

import warnings
import numpy as np

warnings.filterwarnings("ignore")


def test_approach_mode_zeros_solenoid():
    from rl_autonomy.envs import make_env

    # Use a fake-recording wrapper: peek what the inner env actually sees.
    env = make_env(mode="approach", frame_stack=1, domain_rand=False, horizon=5)
    # Reach into the wrapper stack to find the gym base env (KeyboardGymEnv).
    target = env
    while hasattr(target, "env") and not hasattr(target, "underlying"):
        target = target.env
    underlying = target.underlying  # robosuite KeyboardEnv

    # Patch step to capture the action passed in
    captured: list[np.ndarray] = []
    real_step = underlying.step
    def spy(action):
        captured.append(np.asarray(action, dtype=np.float32).copy())
        return real_step(action)
    underlying.step = spy

    env.reset(seed=0)
    env.step(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, +1.0], dtype=np.float32))
    assert captured, "no action recorded — step plumbing broken"
    a = captured[0]
    assert a.shape == (7,)
    # Approach must clamp solenoid to -1.0 regardless of policy command
    assert a[6] == -1.0


def test_strike_mode_zeros_joints():
    from rl_autonomy.envs import make_env
    env = make_env(mode="strike", frame_stack=1, domain_rand=False, horizon=5)
    target = env
    while hasattr(target, "env") and not hasattr(target, "underlying"):
        target = target.env
    underlying = target.underlying

    captured: list[np.ndarray] = []
    real_step = underlying.step
    def spy(action):
        captured.append(np.asarray(action, dtype=np.float32).copy())
        return real_step(action)
    underlying.step = spy

    env.reset(seed=0)
    env.step(np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, +1.0], dtype=np.float32))
    assert captured
    a = captured[0]
    assert a.shape == (7,)
    # Strike must zero the 6 joint dims
    assert np.allclose(a[:6], 0.0)
    # Solenoid pass-through
    assert abs(a[6] - 1.0) < 1e-6


def test_smoothing_attenuates_step_change():
    """A step input on a joint should be attenuated by alpha on the next step."""
    from rl_autonomy.envs.action_adapter import ActionAdapter
    import gymnasium as gym

    class FakeEnv(gym.Env):
        action_space = gym.spaces.Box(-1, 1, shape=(7,), dtype=np.float32)
        observation_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        last_action: np.ndarray | None = None
        def reset(self, **kw):
            self.last_action = None
            return np.zeros(1, dtype=np.float32), {}
        def step(self, a):
            self.last_action = np.asarray(a, dtype=np.float32).copy()
            return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

    inner = FakeEnv()
    env = ActionAdapter(inner, mode="approach", smooth_alpha=0.4)
    env.reset()
    # First step — buffer initialized to action[:6]
    a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    env.step(a)
    first = inner.last_action.copy()
    # Second step — different action, smoothed
    b = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    env.step(b)
    second = inner.last_action.copy()

    # On step 1, joints[0] should be 1.0 (initialized to first command, no prev)
    assert abs(first[0] - 1.0) < 1e-5
    # On step 2, joint[0] should be α * 1.0 + (1-α) * (-1.0) = 0.4 - 0.6 = -0.2
    assert abs(second[0] - (-0.2)) < 1e-5
