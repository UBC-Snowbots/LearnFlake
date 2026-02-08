from dataclasses import dataclass
from typing import Any, Optional, Tuple

import gymnasium as gym


@dataclass
class RoboSuiteConfig:
    env_name: str = "Lift"
    robots: Any = "Panda"
    controller: Any = "OSC_POSE"
    has_renderer: bool = False
    use_camera_obs: bool = False
    horizon: int = 200
    control_freq: int = 20
    reward_shaping: bool = True


class RoboSuiteGymEnv(gym.Env):
    """Lazy gym wrapper around RoboSuite environments."""

    def __init__(self, config: Optional[RoboSuiteConfig] = None):
        super().__init__()
        self.config = config or RoboSuiteConfig()

        # Lazy import inside initializer to avoid hard dependency at package import time.
        import robosuite as suite
        from robosuite.wrappers.gym_wrapper import GymWrapper

        self._env = GymWrapper(
            suite.make(
                env_name=self.config.env_name,
                robots=self.config.robots,
                controller_configs=self.config.controller,
                has_renderer=self.config.has_renderer,
                use_camera_obs=self.config.use_camera_obs,
                horizon=self.config.horizon,
                control_freq=self.config.control_freq,
                reward_shaping=self.config.reward_shaping,
            )
        )
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, dict]:
        if seed is not None:
            try:
                # GymWrapper may not support seed directly; fall back silently.
                self._env.seed(seed)
            except Exception:
                pass
        obs = self._env.reset()
        info = {} if not isinstance(obs, tuple) else (obs[1] if len(obs) > 1 else {})
        if isinstance(obs, tuple):
            obs = obs[0]
        return obs, info

    def step(self, action):
        return self._env.step(action)
