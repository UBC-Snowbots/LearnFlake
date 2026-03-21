import os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite")

sys.path.insert(0, ROBO_PATH)

import time
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper

if __name__ == "__main__":

    if not os.path.exists("tmp/rover"):
        os.makedirs("tmp/rover")

    env_name = "Lift"

    env = suite.make(
        env_name,
        robots=["Rover2025"],
        controller_configs=suite.load_part_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=True,
        use_camera_obs=False,
        horizon=300,
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)

    critic_network = CriticNetwork([8], 8)
    # actor_network = ActorNetwork([8], 8)