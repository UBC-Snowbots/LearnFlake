import robosuite as suite
from robosuite.wrappers import GymWrapper

# Import your custom FaiveHand manipulator so robosuite knows about it
from rl_autonomy import FaiveHand  # noqa: F401

from robosuite.models.robots import robot_model
robot_model.REGISTERED_ROBOTS["FaiveHand"] = FaiveHand

from robosuite.models.robots.robot_model import register_robot
from rl_autonomy.assets.arms.faive_hand_p0 import FaiveHand

register_robot(FaiveHand)  # <-- this is the key



def make_env(
    env_name="Lift",
    robots="FaiveHand",
    render=False,
    controller_configs=None,
    render_offscreen=None,
    wrap_gym=True,
    **kwargs,
):
    """
    Convenience wrapper for robosuite.make that defaults to the custom FaiveHand robot.
    """

    # Match renderer settings between onscreen and offscreen if not explicitly set
    if render_offscreen is None:
        render_offscreen = render

    base_env = suite.make(
        env_name=env_name,
        robots=robots,
        has_renderer=render,
        has_offscreen_renderer=render_offscreen,
        use_camera_obs=False,
        controller_configs=controller_configs,
        control_freq=20,
        **kwargs,
    )

    if wrap_gym and GymWrapper is not None:
        return GymWrapper(base_env)
    return base_env
