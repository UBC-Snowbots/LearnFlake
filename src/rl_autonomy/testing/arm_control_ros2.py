"""
ROS2-RoboSuite Bridge for Arm Control

This script bridges your ROS2 IK stack to the RoboSuite simulation:
- Subscribes to /arm/command (ArmCommand) from your IK nodes
- Applies joint velocities to RoboSuite simulation
- Publishes joint feedback to /arm/feedback

Usage:
1. Start your IK stack (moveit_control, cbs_interface, etc.)
2. Run this script: python arm_control_ros2.py
3. Use keyboard_teleop.py or joystick to send commands

Requires: ros2, rclpy, rover_msgs
"""

import os
import sys
import time
import argparse
import numpy as np
import threading
from copy import deepcopy

# --- Path Setup ---
ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite")
sys.path.insert(0, ROBO_PATH)

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.wrappers import VisualizationWrapper

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float64MultiArray, Float64
from sensor_msgs.msg import JointState

# Try to import rover_msgs, fall back to Float64MultiArray if not available
try:
    from rover_msgs.msg import ArmCommand
    HAS_ROVER_MSGS = True
except ImportError:
    HAS_ROVER_MSGS = False
    print("[WARN] rover_msgs not found, using Float64MultiArray on /arm/sim_command")


class RoboSuiteBridge(Node):
    """ROS2 node that bridges commands to RoboSuite simulation."""
    
    def __init__(self):
        super().__init__('robosuite_bridge')
        
        self.num_joints = 6
        self.joint_velocities = np.zeros(self.num_joints)
        self.gripper_cmd = 0.0
        self.lock = threading.Lock()
        
        # Subscribe to arm commands
        if HAS_ROVER_MSGS:
            self.arm_sub = self.create_subscription(
                ArmCommand,
                '/arm/command',
                self.arm_command_callback,
                10
            )
            self.get_logger().info('Subscribed to /arm/command (ArmCommand)')
        
        # Also subscribe to sim_command (Float64MultiArray) as fallback
        self.sim_sub = self.create_subscription(
            Float64MultiArray,
            '/arm/sim_command',
            self.sim_command_callback,
            10
        )
        self.get_logger().info('Subscribed to /arm/sim_command (Float64MultiArray)')
        
        # Subscribe to gripper command separately
        self.gripper_sub = self.create_subscription(
            Float64,
            '/arm/sim_ee',
            self.gripper_callback,
            10
        )
        self.get_logger().info('Subscribed to /arm/sim_ee (Float64)')
        
        # Publisher for joint feedback
        self.joint_pub = self.create_publisher(JointState, '/arm/joint_states', 10)
        
        if HAS_ROVER_MSGS:
            self.feedback_pub = self.create_publisher(ArmCommand, '/arm/feedback', 10)
        
        self.get_logger().info('RoboSuite bridge node started')
    
    def arm_command_callback(self, msg):
        """Handle ArmCommand messages from IK stack."""
        with self.lock:
            # Extract velocities (6 joints)
            if len(msg.velocities) >= self.num_joints:
                self.joint_velocities = np.array(msg.velocities[:self.num_joints])
            # Gripper command
            self.gripper_cmd = msg.end_effector if hasattr(msg, 'end_effector') else 0.0
    
    def sim_command_callback(self, msg):
        """Handle Float64MultiArray messages (from sim_helper_node)."""
        with self.lock:
            if len(msg.data) >= self.num_joints:
                self.joint_velocities = np.array(msg.data[:self.num_joints])
    
    def gripper_callback(self, msg):
        """Handle gripper command."""
        with self.lock:
            self.gripper_cmd = msg.data
    
    def get_action(self):
        """Get current joint velocities for simulation step."""
        with self.lock:
            return self.joint_velocities.copy(), self.gripper_cmd
    
    def publish_feedback(self, joint_positions, joint_velocities, eef_pos):
        """Publish joint state feedback to ROS2."""
        # JointState message
        js_msg = JointState()
        js_msg.header.stamp = self.get_clock().now().to_msg()
        js_msg.name = [f'joint_{i}' for i in range(len(joint_positions))]
        js_msg.position = joint_positions.tolist()
        js_msg.velocity = joint_velocities.tolist() if joint_velocities is not None else []
        self.joint_pub.publish(js_msg)
        
        # ArmCommand feedback
        if HAS_ROVER_MSGS:
            fb_msg = ArmCommand()
            fb_msg.positions = joint_positions.tolist()
            fb_msg.velocities = joint_velocities.tolist() if joint_velocities is not None else [0.0] * len(joint_positions)
            self.feedback_pub.publish(fb_msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default=["Rover2026"])
    parser.add_argument("--max_fr", default=20, type=int, help="Max frame rate")
    parser.add_argument("--velocity-scale", type=float, default=0.1, 
                        help="Scale factor for joint velocities")
    args = parser.parse_args()

    # Initialize ROS2
    rclpy.init()
    bridge_node = RoboSuiteBridge()
    
    # ROS2 executor in separate thread
    executor = SingleThreadedExecutor()
    executor.add_node(bridge_node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()
    
    # Load JOINT_VELOCITY controller the proper way (like demo_control.py)
    arm_controller_config = suite.load_part_controller_config(default_controller="JOINT_VELOCITY")
    controller_config = refactor_composite_controller_config(
        arm_controller_config, args.robots[0], ["right"]
    )
    print(f"Using JOINT_VELOCITY controller config")
    
    # Create RoboSuite environment
    env = suite.make(
        env_name=args.environment,
        robots=args.robots,
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    env = VisualizationWrapper(env, indicator_configs=None)
    
    print("\n" + "="*60)
    print("RoboSuite-ROS2 Bridge Active")
    print("="*60)
    print(f"Listening on: /arm/command, /arm/sim_command")
    print(f"Publishing to: /arm/joint_states, /arm/feedback")
    print(f"Velocity scale: {args.velocity_scale}")
    print("="*60 + "\n")
    
    # Debug: print actual controller info
    robot = env.robots[0]
    print(f"Robot action_dim: {robot.action_dim}")
    print(f"Robot arms: {robot.arms}")
    for arm in robot.arms:
        ctrl = robot.part_controllers.get(arm)
        if ctrl:
            print(f"Controller for '{arm}': {ctrl.name}, input_type: {getattr(ctrl, 'input_type', 'N/A')}")
    
    try:
        while rclpy.ok():
            obs = env.reset()
            env.render()
            
            while rclpy.ok():
                start = time.time()
                
                # Get commands from ROS2
                joint_vel, gripper_cmd = bridge_node.get_action()
                
                # Scale velocities
                scaled_vel = joint_vel * args.velocity_scale
                
                # Debug: print when we receive non-zero commands
                if np.any(joint_vel != 0):
                    print(f"Raw vel: {joint_vel}, Scaled: {scaled_vel}, Gripper: {gripper_cmd}")
                
                # Build action vector: [joint_velocities, gripper]
                # For Rover2026 with JOINT_VELOCITY controller
                robot = env.robots[0]
                action_dim = robot.action_dim
                action = np.zeros(action_dim)
                
                # Set joint velocities (first 6 dims for 6-DOF arm)
                num_arm_joints = min(len(scaled_vel), 6)
                action[:num_arm_joints] = scaled_vel[:num_arm_joints]
                
                # Set gripper (last dim)
                if action_dim > num_arm_joints:
                    action[-1] = gripper_cmd
                
                # Debug: print action being sent
                if np.any(action != 0):
                    print(f"Action dim: {action_dim}, Action: {action}")
                
                # Step simulation
                obs, reward, done, info = env.step(action)
                
                # Get feedback
                joint_pos = obs.get('robot0_joint_pos', np.zeros(6))
                joint_vel_obs = obs.get('robot0_joint_vel', None)
                eef_pos = obs.get('robot0_eef_pos', np.zeros(3))
                
                # Publish feedback to ROS2
                bridge_node.publish_feedback(joint_pos, joint_vel_obs, eef_pos)
                
                env.render()
                
                # Frame rate limiting
                if args.max_fr is not None:
                    elapsed = time.time() - start
                    diff = 1.0 / args.max_fr - elapsed
                    if diff > 0:
                        time.sleep(diff)
                        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        env.close()
        bridge_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
