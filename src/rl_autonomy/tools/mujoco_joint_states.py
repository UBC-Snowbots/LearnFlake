#!/usr/bin/env python3
"""
MuJoCo Joint State Echo

Lightweight ROS 2 subscriber that listens to /mujoco/joint_states
(published by the MuJoCo bridge in cartesian_control_ros.py) and
prints a formatted table to the terminal.

Usage:
  python3 mujoco_joint_states.py          # pretty-print at receive rate
  ros2 topic echo /mujoco/joint_states    # alternative, raw echo
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateEcho(Node):
    """Subscribes to /mujoco/joint_states and prints a table."""

    def __init__(self):
        super().__init__("mujoco_joint_state_echo")
        self.create_subscription(JointState, "/mujoco/joint_states", self._cb, 10)
        self._count = 0
        self.get_logger().info("Listening on /mujoco/joint_states …")

    def _cb(self, msg: JointState):
        self._count += 1
        # Print a header every 20 messages (~1 s at 20 Hz)
        if self._count % 20 == 1:
            print(f"\n{'Joint':<18} {'Pos (rad)':>10} {'Pos (deg)':>10} {'Vel (rad/s)':>12}")
            print("-" * 54)
        if self._count % 20 == 1:
            for name, pos, vel in zip(msg.name, msg.position, msg.velocity):
                deg = math.degrees(pos)
                print(f"  {name:<16} {pos:>10.4f} {deg:>10.2f} {vel:>12.4f}")


def main():
    rclpy.init()
    node = JointStateEcho()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
    main()
