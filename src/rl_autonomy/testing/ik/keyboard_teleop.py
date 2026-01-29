#!/usr/bin/env python3
"""
Keyboard teleop for arm control via ROS2.
Direct joint velocity control - each key pair controls one joint.

Controls:
  Joint Controls (press and hold):
    Q/A  - Joint 1 (base rotation) +/-
    W/S  - Joint 2 (shoulder) +/-
    E/D  - Joint 3 (elbow) +/-
    R/F  - Joint 4 (wrist roll) +/-
    T/G  - Joint 5 (wrist pitch) +/-
    Y/H  - Joint 6 (wrist yaw) +/-
    
  Gripper:
    Space - Toggle gripper open/close
    O     - Open gripper
    P     - Close gripper
    
  Other:
    0     - Stop all movement
    Esc   - Quit
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64
import sys
import termios
import tty
import select
import numpy as np

# Try to import rover_msgs
try:
    from rover_msgs.msg import ArmCommand
    HAS_ROVER_MSGS = True
except ImportError:
    HAS_ROVER_MSGS = False
    print("[WARN] rover_msgs not found, using Float64MultiArray only")


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        
        # Publishers
        if HAS_ROVER_MSGS:
            self.arm_pub = self.create_publisher(ArmCommand, '/arm/command', 10)
        self.sim_pub = self.create_publisher(Float64MultiArray, '/arm/sim_command', 10)
        self.gripper_pub = self.create_publisher(Float64, '/arm/sim_ee', 10)
        
        # Control parameters - adjust these for your arm
        self.joint_scale = 1.0  # Velocity magnitude sent to simulation
        self.num_joints = 6
        
        # Gripper state: -1 = open, 1 = closed
        self.gripper_state = -1.0
        
        # Direct key -> (joint_index, direction) mapping
        self.joint_keys = {
            'q': (0, 1),   'a': (0, -1),   # Joint 1 (base)
            'w': (1, 1),   's': (1, -1),   # Joint 2 (shoulder)
            'e': (2, 1),   'd': (2, -1),   # Joint 3 (elbow)
            'r': (3, 1),   'f': (3, -1),   # Joint 4 (wrist roll)
            't': (4, 1),   'g': (4, -1),   # Joint 5 (wrist pitch)
            'y': (5, 1),   'h': (5, -1),   # Joint 6 (wrist yaw)
        }
        
        self.print_controls()
        
    def print_controls(self):
        print("\n" + "="*60)
        print("KEYBOARD ARM TELEOP - Direct Joint Control")
        print("="*60)
        print("Joint Controls (press and hold):")
        print("  Q/A  - Joint 1 (base rotation)")
        print("  W/S  - Joint 2 (shoulder)")
        print("  E/D  - Joint 3 (elbow)")
        print("  R/F  - Joint 4 (wrist roll)")
        print("  T/G  - Joint 5 (wrist pitch)")
        print("  Y/H  - Joint 6 (wrist yaw)")
        print("")
        print("Gripper:")
        print("  Space - Toggle gripper")
        print("  O     - Open gripper")
        print("  P     - Close gripper")
        print("")
        print("Other:")
        print("  0     - Stop all joints")
        print("  Esc   - Quit")
        print("="*60 + "\n")
        
    def get_key(self, timeout=0.05):
        """Get a single keypress with short timeout for responsiveness."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if rlist:
                key = sys.stdin.read(1)
            else:
                key = ''
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key
    
    def publish_command(self, velocities, gripper):
        """Publish joint velocities and gripper command."""
        # Float64MultiArray for joint velocities
        sim_msg = Float64MultiArray()
        sim_msg.data = velocities.tolist()
        self.sim_pub.publish(sim_msg)
        
        # Gripper as separate Float64 message
        gripper_msg = Float64()
        gripper_msg.data = gripper
        self.gripper_pub.publish(gripper_msg)
        
        # ArmCommand if rover_msgs available
        if HAS_ROVER_MSGS:
            arm_msg = ArmCommand()
            arm_msg.cmd_type = ord('V')
            arm_msg.velocities = velocities.tolist()
            arm_msg.positions = [0.0] * self.num_joints
            arm_msg.end_effector = gripper
            self.arm_pub.publish(arm_msg)
    
    def run(self):
        """Main control loop."""
        print("Ready for input...\n")
        
        # Persistent velocity state - doesn't reset each loop
        velocities = np.zeros(self.num_joints)
        
        try:
            while rclpy.ok():
                key = self.get_key()
                
                if key == '\x1b':  # Escape
                    print("\nExiting...")
                    break
                    
                elif key == ' ':  # Toggle gripper
                    self.gripper_state = 1.0 if self.gripper_state < 0 else -1.0
                    print(f"Gripper: {'CLOSED' if self.gripper_state > 0 else 'OPEN'}")
                    
                elif key == 'o':  # Open gripper
                    self.gripper_state = -1.0
                    print("Gripper: OPEN")
                    
                elif key == 'p':  # Close gripper
                    self.gripper_state = 1.0
                    print("Gripper: CLOSED")
                    
                elif key == '0':  # Emergency stop - zero all velocities
                    velocities = np.zeros(self.num_joints)
                    print("STOP ALL")
                    
                elif key == '':
                    # No key pressed - gradually decay velocities (smooth stop)
                    velocities *= 0.8  # Decay factor
                    if np.max(np.abs(velocities)) < 0.01:
                        velocities = np.zeros(self.num_joints)
                    
                elif key.lower() in self.joint_keys:
                    # Direct joint control - set velocity for this joint
                    joint_idx, direction = self.joint_keys[key.lower()]
                    velocities[joint_idx] = direction * self.joint_scale
                    print(f"Key '{key}' -> Joint {joint_idx}, vel={velocities[joint_idx]}")
                
                # Always publish current state
                self.publish_command(velocities, self.gripper_state)
                    
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Send stop command on exit
            self.publish_command(np.zeros(self.num_joints), self.gripper_state)


def main():
    rclpy.init()
    node = KeyboardTeleop()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()