# File is verified to work as intended therefore bugs are likely elsewhere

import os, sys
import argparse
import time
import numpy as np
from copy import deepcopy

# --- Path Setup ---
ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite")
sys.path.insert(0, ROBO_PATH)

import robosuite as suite
from robosuite.devices import Keyboard
from robosuite import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Updated default to Rover2025
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default=["Rover2025"], 
        help="Which robot(s) to use in the env",
    )
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--arm", type=str, default="right")
    parser.add_argument("--controller", type=str, default=None)
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0)
    parser.add_argument("--rot-sensitivity", type=float, default=1.0)
    parser.add_argument("--max_fr", default=20, type=int)
    parser.add_argument("--reverse_xy", type=bool, default=False)
    args = parser.parse_args()

    # Get controller config for Rover2025
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )

    env = VisualizationWrapper(env, indicator_configs=None)
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # Initialize device
    if args.device == "keyboard":
        device = Keyboard(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback(device.on_press)
    else:
        raise Exception(f"Device {args.device} not supported in this simplified script.")

    while True:
        obs = env.reset()
        env.render()
        device.start_control()
        print("------------")
        print("------------")
        print("------------")
        print("------------")
        print("------------")
        print("------------")
        print("------------")
        print("------------")
        print("------------")

        # Logic to track gripper states across all robots in the scene
        all_prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in env.robots
        ]

        while True:
            start = time.time()
            active_robot = env.robots[device.active_robot]
            input_ac_dict = device.input2action()

            if input_ac_dict is None:
                break

            action_dict = deepcopy(input_ac_dict)
            
            # Map input to controller types (Delta vs Absolute)
            for arm in active_robot.arms:
                if isinstance(active_robot.composite_controller, WholeBody):
                    input_type = active_robot.composite_controller.joint_action_policy.input_type
                else:
                    input_type = active_robot.part_controllers[arm].input_type

                if input_type == "delta":
                    action_dict[arm] = input_ac_dict.get(f"{arm}_delta", np.zeros(6))
                elif input_type == "absolute":
                    action_dict[arm] = input_ac_dict.get(f"{arm}_abs", np.zeros(6))


            # Construct the full environment action vector
            env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
            env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
            
            env_action = np.concatenate(env_action)
            for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

            
            # aaron's notes: use similar tactic to gripper.py to read eef and joint positions
            obs, reward, done, info = env.step(env_action)
            eef_pos = obs['robot0_eef_pos']
            joint_pos = obs['robot0_joint_pos']
            
            print(f"EEF Pos: {eef_pos} | Joints: {joint_pos}")
            
            env.render()

            if args.max_fr is not None:
                elapsed = time.time() - start
                diff = 1 / args.max_fr - elapsed
                if diff > 0:
                    time.sleep(diff)
