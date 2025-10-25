import argparse
import copy
import json
import sys
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------------- #
# Project import path setup
# ----------------------------------------------------------------------------- #
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.append(str(_SRC_ROOT))

from rl_autonomy.env import make_env
from robosuite.controllers.parts.controller_factory import load_part_controller_config


# ----------------------------------------------------------------------------- #
# Controller loader
# ----------------------------------------------------------------------------- #
def _load_single_arm_controller() -> dict:
    """
    Load the default composite controller config and prune it to only include the right FaiveHand manipulator.
    """
    config_path = (
        _SRC_ROOT
        / "rl_autonomy"
        / "RoboSuite"
        / "robosuite"
        / "controllers"
        / "config"
        / "default"
        / "composite"
        / "basic.json"
    )

    with config_path.open() as f:
        config = json.load(f)

    right_part = load_part_controller_config(default_controller="JOINT_POSITION")
    right_part["gripper"] = {"type": "GRIP"}

    single_arm_config = {
        "type": "BASIC",
        "body_parts": {
            "right": right_part,
        },
    }

    if "composite_controller_specific_configs" in config:
        single_arm_config["composite_controller_specific_configs"] = copy.deepcopy(
            config["composite_controller_specific_configs"]
        )

    return single_arm_config


# ----------------------------------------------------------------------------- #
# Rollout helper
# ----------------------------------------------------------------------------- #
def _rollout(env, episode_length: int) -> None:
    """
    Sample random actions for a short rollout so assets can be visually inspected.
    Compatible with both gym-wrapped and native robosuite envs.
    """
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs, info = reset_out
    else:
        obs, info = reset_out, {}

    for _ in range(episode_length):
        if hasattr(env, "action_space"):
            action = env.action_space.sample()
        else:
            low, high = env.action_spec
            action = np.random.uniform(low, high)

        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
        else:
            obs, reward, done, info = step_out
            terminated, truncated = done, False

        # Telemetry: print Faive hand (palm / eef) pose to help debug positioning in headless runs
        robot = getattr(env, "robots", [None])[0]
        if robot is not None:
            palm_body = robot.sim.model.body_name2id(robot.robot_model.correct_naming("palm"))
            palm_pos = robot.sim.data.body_xpos[palm_body]
            eef_pos = robot.sim.data.site_xpos[robot.eef_site_id["right"]]
            print(f"FaiveHand palm @ {palm_pos} | eef @ {eef_pos}")

        if terminated or truncated:
            reset_out = env.reset()
            if isinstance(reset_out, tuple):
                obs, info = reset_out
            else:
                obs, info = reset_out, {}
    #env.close()
    input("Press Enter to exit...")


# ----------------------------------------------------------------------------- #
# Main entrypoint
# ----------------------------------------------------------------------------- #
def main(episode_length: int = 200, render: bool = True) -> None:
    """
    Instantiate the Lift task with the FaiveHand and run a short random-policy rollout.

    If GPU-based rendering fails to initialize, the script automatically retries without rendering so the
    environment can still be smoke-tested in headless setups.
    """
    try:
        controller_cfg = _load_single_arm_controller()
        env = make_env(
            env_name="Lift",
            robots="FaiveHand",
            render=render,
            render_offscreen=render,
            controller_configs=controller_cfg,
            wrap_gym=False,
        )
    except ImportError as exc:
        # Graceful fallback when EGL or GPU renderer isn't available
        if render and "EGL" in str(exc):
            print(
                "EGL renderer unavailable. Retrying in headless mode.\n"
                "Tip: Set MUJOCO_GL=osmesa for CPU rendering or run with --no-render."
            )
            env = make_env(
                env_name="Lift",
                robots="FaiveHand",
                render=False,
                render_offscreen=False,
                controller_configs=_load_single_arm_controller(),
                wrap_gym=False,
            )
        else:
            raise

    _rollout(env, episode_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaiveHand smoke-test runner.")
    parser.add_argument(
        "--episode-length",
        type=int,
        default=200,
        help="Number of environment steps to simulate while sampling random actions.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip visualization (useful on headless servers without EGL / GPU support).",
    )
    args = parser.parse_args()
    main(episode_length=args.episode_length, render=not args.no_render)
