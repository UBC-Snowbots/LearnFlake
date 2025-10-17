import robosuite as suite
from robosuite.wrappers import GymWrapper

# Importing rl_autonomy registers our custom manipulators with robosuite
from rl_autonomy import FaiveHand, SimpleArm  # noqa: F401


def make_env(env_name="Lift", robots="SimpleArm", render=False, controller_configs=None, **kwargs):
    """
    Convenience wrapper for robosuite.make that defaults to the custom SimpleArm robot.
    Pass robots=\"FaiveHand\" (or any other registered arm) to switch variants.
    """
    env = GymWrapper(
        suite.make(
            env_name=env_name,
            robots=robots,
            has_renderer=render,
            use_camera_obs=False,
            controller_configs=controller_configs,
            control_freq=20,
            **kwargs,
        )
    )
    return env
