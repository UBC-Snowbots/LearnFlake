"""
Custom Gymnasium wrapper for Robosuite environments.
Provides a standardized interface for RL training with Stable Baselines3, etc.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from robosuite import make
from robosuite.wrappers import GymWrapper


class RobosuiteGymWrapper(gym.Env):
    """
    Generic wrapper for using robosuite environments with Gymnasium.

    Args:
        env_name (str): Robosuite environment name (e.g., "Lift", "Door", "Stack")
        robots (str or list): Robot(s) to use (e.g., "Panda", "Jaco", "Sawyer")
        has_renderer (bool): Enable on-screen rendering
        has_offscreen_renderer (bool): Enable offscreen rendering (None=auto, False=faster training)
        use_camera_obs (bool): Include camera observations
        reward_shaping (bool): Enable dense reward shaping
        control_freq (int): Control frequency in Hz
        **kwargs: Additional arguments passed to robosuite.make()

    Example:
        >>> from rl_autonomy import RobosuiteGymWrapper
        >>> env = RobosuiteGymWrapper("Lift", robots="Panda")
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    def __init__(
        self,
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=None,  # None = auto, True/False = override
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        **kwargs
    ):
        super().__init__()

        # Auto-set offscreen renderer if not specified
        if has_offscreen_renderer is None:
            has_offscreen_renderer = not has_renderer

        self.env = GymWrapper(
            make(
                env_name=env_name,
                robots=robots,
                has_renderer=has_renderer,
                has_offscreen_renderer=has_offscreen_renderer,
                use_camera_obs=use_camera_obs,
                reward_shaping=reward_shaping,
                control_freq=control_freq,
                **kwargs
            )
        )

        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.env.observation_space.shape,
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        Args:
            seed (int, optional): Random seed for reproducibility
            options (dict, optional): Additional reset options

        Returns:
            tuple: (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)

        obs = self.env.reset()

        # Handle both old gym (returns obs) and new gymnasium (returns obs, info)
        if isinstance(obs, tuple):
            return obs
        return obs, {}

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: Action to execute

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        result = self.env.step(action)

        # Convert old gym format (obs, reward, done, info) to gymnasium format
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, False, info
        return result

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Clean up resources."""
        self.env.close()
