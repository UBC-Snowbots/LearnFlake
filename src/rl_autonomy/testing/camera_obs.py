import os, sys
import argparse
import time
import numpy as np
import cv2
from copy import deepcopy

# --- Path Setup ---
ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite")
sys.path.insert(0, ROBO_PATH)

import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper

def get_camera_image(obs, camera_name):
    """
    Retrieve and format camera image from observation.
    """
    # Try exact match first, then robot0 prefix
    keys = [f"{camera_name}_image", f"robot0_{camera_name}_image"]
    
    img = None
    for key in keys:
        if key in obs:
            img = obs[key]
            break
            
    if img is not None:
        # Flip vertically because Robosuite renders upside down for OpenCV sometimes
        # Actually Robosuite returns (H, W, C) in standard convention, but let's check.
        # It is usually RGB. OpenCV needs BGR.
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Flip if necessary (Robosuite renderer mimics OpenGL, bottom-left origin?)
        # Conventionally, we might need a flip. Let's try without first or flipUD.
        # Usually it is correct.
        return img
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    # Create environment with camera enabled
    # We requested "eye_in_hand"
    camera_name = "robot0_eye_in_hand"
    
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=True, # Needed for camera observations
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=True,
        camera_names=[camera_name],
        camera_heights=256,
        camera_widths=256,
        control_freq=20,
    )

    env = VisualizationWrapper(env, indicator_configs=None)
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # Initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard
        device = Keyboard(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback(device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse
        device = SpaceMouse(env=env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception(f"Device {args.device} not supported.")

    print(f"Observation keys: {env.reset().keys()}")

    while True:
        obs = env.reset()
        env.render()
        device.start_control()

        # Logic to track gripper states across all robots in the scene
        all_prev_gripper_actions = []
        for robot in env.robots:
            robot_gripper_dict = {}
            for arm in robot.arms:
                if robot.gripper[arm] is not None:
                    dof = robot.gripper[arm].dof
                    robot_gripper_dict[f"{arm}_gripper"] = np.zeros(dof)
            all_prev_gripper_actions.append(robot_gripper_dict)

        while True:
            start = time.time()
            active_robot = env.robots[device.active_robot]
            input_ac_dict = device.input2action()

            if input_ac_dict is None:
                break

            action_dict = deepcopy(input_ac_dict)
            
            # Map input to controller types
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
            env_action_list = []
            for i, robot in enumerate(env.robots):
                if i == device.active_robot:
                    env_action_list.append(robot.create_action_vector(action_dict))
                    for gripper_key in all_prev_gripper_actions[i]:
                        if gripper_key in action_dict:
                            all_prev_gripper_actions[i][gripper_key] = action_dict[gripper_key]
                else:
                    env_action_list.append(robot.create_action_vector(all_prev_gripper_actions[i]))

            obs, reward, done, info = env.step(np.concatenate(env_action_list))
            
            # Display Camera Feed
            cam_img = get_camera_image(obs, camera_name)
            if cam_img is not None:
                # Flip vertically because OpenGL renders from bottom-left
                cam_img = np.flipud(cam_img)
                cv2.imshow(f"{camera_name} view", cam_img)
                cv2.waitKey(1)
            
            env.render()

            if args.max_fr is not None:
                elapsed = time.time() - start
                time.sleep(max(0, (1.0 / args.max_fr) - elapsed))

    cv2.destroyAllWindows()
