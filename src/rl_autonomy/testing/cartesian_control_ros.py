#!/usr/bin/env python3
"""
MuJoCo Bridge Node — Keyboard Pipeline (Phase 3)

Runs KeyboardEnv in MuJoCo and bridges it to ROS2.  The arm is commanded
via IK-solved JointTrajectory from MoveIt Servo; the solenoid actuator is
commanded via /actuator/command.

Architecture:
    Joystick → /joy
        └─→ joy_arm_control → TwistStamped → /servo_node/delta_twist_cmds
                                                    │
                                               MoveIt Servo
                                                    │
                                /arm_controller/joint_trajectory
                               ╱                              ╲
                    ros2_control                          THIS NODE  ← ★
                   (RViz / hardware)                   (MuJoCo mirror)

Subscribes to:
    /arm_controller/joint_trajectory  — IK-solved arm motion from MoveIt Servo
    /actuator/command                 — solenoid command (Float32, >0=extend)
    /mujoco/set_target_key            — set current target key (String, e.g. "e")

Publishes:
    /mujoco/joint_states      — sensor_msgs/JointState  (6 arm joints + actuator)
    /mujoco/observations      — std_msgs/Float64MultiArray  (32-dim flat obs)
    /mujoco/actions           — std_msgs/Float64MultiArray  (7-dim action)
    /mujoco/eef_camera        — sensor_msgs/Image  (64×64 RGB from EEF cam)
    /mujoco/eef_rangefinder   — std_msgs/Float32  (distance below tip, metres)
    /mujoco/actuator_pos      — std_msgs/Float32  (solenoid extension, metres)
    /mujoco/actuator_torque   — std_msgs/Float32  (contact force proxy, N)
    /mujoco/contact_detected  — std_msgs/Bool     (torque+stall threshold)
    /mujoco/target_key        — std_msgs/String   (current target key name)

Usage:
    python3 cartesian_control_ros.py [--no-render] [--debug]
    python3 cartesian_control_ros.py --trajectory-topic /arm_controller/joint_trajectory
"""

import os
import sys
import time
import argparse
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray, Float32, Bool, String
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, ".."))
ROBO_PATH = os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite")
if os.path.exists(ROBO_PATH) and ROBO_PATH not in sys.path:
    sys.path.insert(0, ROBO_PATH)

from keyboard_env import (
    KeyboardEnv,
    CONTACT_FORCE_THRESHOLD,
    STALL_VEL_THRESHOLD,
)


# ============================================================================
# Bridge Node
# ============================================================================

class KeyboardBridgeNode(Node):

    TRAJECTORY_TOPIC  = "/arm_controller/joint_trajectory"
    ACTUATOR_TOPIC    = "/actuator/command"
    SET_KEY_TOPIC     = "/mujoco/set_target_key"

    OBS_TOPIC         = "/mujoco/observations"
    ACTION_TOPIC      = "/mujoco/actions"
    JOINT_STATE_TOPIC = "/mujoco/joint_states"
    EEF_CAM_TOPIC     = "/mujoco/eef_camera"
    RANGEFINDER_TOPIC = "/mujoco/eef_rangefinder"
    ACTUATOR_POS_TOPIC = "/mujoco/actuator_pos"
    ACTUATOR_TRQ_TOPIC = "/mujoco/actuator_torque"
    CONTACT_TOPIC     = "/mujoco/contact_detected"
    PRESSED_KEY_TOPIC = "/mujoco/pressed_key"
    TARGET_KEY_TOPIC  = "/mujoco/target_key"
    A5_TORQUE_TOPIC   = "/mujoco/a5_torque"         # synthetic Moteus id_5 torque
    A5_DELTA_TOPIC    = "/mujoco/a5_torque_delta"    # torque - baseline = contact signal

    URDF_JOINT_NAMES = [
        'shoulder_joint', 'link_1_joint', 'link1_link2',
        'a4_rotation', 'a5_rotation', 'a6_rotation',
    ]
    MUJOCO_JOINT_NAMES = [
        'robot0_shoulder_joint', 'robot0_link_1_joint', 'robot0_link1_link2',
        'robot0_a4_rotation', 'robot0_a5_rotation', 'robot0_a6_rotation',
    ]

    # EEF camera name after RoboSuite applies the gripper prefix
    EEF_CAM_NAME = "gripper0_right_eef_cam"
    CAM_W = 64
    CAM_H = 64

    def __init__(self, render: bool = True, debug: bool = False,
                 trajectory_topic: str | None = None):
        super().__init__('keyboard_bridge')

        self.debug = debug
        if trajectory_topic:
            self.TRAJECTORY_TOPIC = trajectory_topic

        # ---- Environment ----
        self.env = KeyboardEnv(
            render=render,
            use_camera_obs=False,   # we render the camera ourselves
            horizon=10_000,
        )
        self.render_enabled = render
        self.action_dim = self.env.action_dim

        sim = self.env.sim
        self._arm_qvel_addrs = np.array([
            sim.model.get_joint_qvel_addr(n) for n in self.MUJOCO_JOINT_NAMES
        ])
        self._arm_qpos_addrs = np.array([
            sim.model.get_joint_qpos_addr(n) for n in self.MUJOCO_JOINT_NAMES
        ])

        self._urdf_to_idx = {n: i for i, n in enumerate(self.URDF_JOINT_NAMES)}

        self.max_joint_velocities = np.array([1.0, 1.0, 0.8, 0.8, 0.8, 1.0])
        self.pos_fallback_kp = 10.0

        # ---- Synthetic Moteus a5 torque ----
        a5_mj_name = 'robot0_a5_rotation'
        self._a5_actuator_id = sim.model.actuator_name2id(a5_mj_name)
        # Noise parameters matching real Moteus noise floor
        self._a5_noise_std = 0.05        # Nm, Gaussian
        self._a5_outlier_prob = 0.01     # 1% chance of 3-sigma spike
        self._a5_drift = 0.0            # slow drift term (temperature sim)
        self._a5_drift_rate = 0.001     # Nm per step max drift
        # Baseline: captured at init, updated on reset
        self._a5_torque_baseline = float(sim.data.actuator_force[self._a5_actuator_id])

        # ---- Shared state ----
        self._joint_vel_cmd   = np.zeros(6, dtype=np.float64)
        self._joint_pos_target = None
        self._has_velocity    = False
        self._actuator_cmd    = 0.0   # -1=retract, >0=extend (passed as action[-1])
        self._last_traj_time  = 0.0
        self._traj_timeout    = 0.2
        self._lock            = threading.Lock()
        self._obs_raw: dict   = {}
        self._step_count      = 0

        # ---- Publishers ----
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.obs_pub        = self.create_publisher(Float64MultiArray, self.OBS_TOPIC, qos)
        self.action_pub     = self.create_publisher(Float64MultiArray, self.ACTION_TOPIC, qos)
        self.js_pub         = self.create_publisher(JointState, self.JOINT_STATE_TOPIC, 10)
        self.cam_pub        = self.create_publisher(Image, self.EEF_CAM_TOPIC, 10)
        self.range_pub      = self.create_publisher(Float32, self.RANGEFINDER_TOPIC, 10)
        self.act_pos_pub    = self.create_publisher(Float32, self.ACTUATOR_POS_TOPIC, 10)
        self.act_trq_pub    = self.create_publisher(Float32, self.ACTUATOR_TRQ_TOPIC, 10)
        self.contact_pub    = self.create_publisher(Bool, self.CONTACT_TOPIC, 10)
        self.pressed_pub    = self.create_publisher(String, self.PRESSED_KEY_TOPIC, 10)
        self.target_key_pub = self.create_publisher(String, self.TARGET_KEY_TOPIC, 10)
        self.a5_trq_pub     = self.create_publisher(Float32, self.A5_TORQUE_TOPIC, 10)
        self.a5_delta_pub   = self.create_publisher(Float32, self.A5_DELTA_TOPIC, 10)

        # ---- Subscriptions ----
        self.traj_sub    = self.create_subscription(JointTrajectory, self.TRAJECTORY_TOPIC, self._traj_cb, qos)
        self.act_sub     = self.create_subscription(Float32, self.ACTUATOR_TOPIC, self._actuator_cb, 10)
        self.set_key_sub = self.create_subscription(String, self.SET_KEY_TOPIC, self._set_key_cb, 10)

        self._timer = self.create_timer(1.0 / 20.0, self._step_tick)

        # ---- Initial reset ----
        self._obs_raw = self.env.reset()
        if render:
            self.env.render()

        self.get_logger().info(
            f"KeyboardBridge ONLINE | action_dim={self.action_dim} | "
            f"render={render} | debug={debug}"
        )
        self.get_logger().info(f"  obs_dim = {self.env.obs_dim}")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _traj_cb(self, msg: JointTrajectory):
        if not msg.points:
            return
        point = msg.points[-1]
        has_vel = len(point.velocities) > 0
        pos = np.zeros(6)
        vel = np.zeros(6)
        for i, name in enumerate(msg.joint_names):
            idx = self._urdf_to_idx.get(name)
            if idx is None:
                continue
            if has_vel and i < len(point.velocities):
                vel[idx] = point.velocities[i]
            if i < len(point.positions):
                pos[idx] = point.positions[i]
        with self._lock:
            self._joint_vel_cmd    = vel
            self._joint_pos_target = pos
            self._has_velocity     = has_vel
            self._last_traj_time   = time.monotonic()

    def _actuator_cb(self, msg: Float32):
        with self._lock:
            self._actuator_cmd = float(msg.data)

    def _set_key_cb(self, msg: String):
        key = msg.data.strip().lower()
        if key in self.env.AVAILABLE_KEYS:
            self.env.set_target_key(key)
            self.get_logger().info(f"Target key → '{key}'")
        else:
            self.get_logger().warn(
                f"Unknown key '{key}'. Available: {self.env.AVAILABLE_KEYS}"
            )

    # ------------------------------------------------------------------
    # Step loop (20 Hz)
    # ------------------------------------------------------------------

    def _step_tick(self):
        with self._lock:
            pos_target  = self._joint_pos_target
            vel_cmd     = self._joint_vel_cmd.copy()
            has_vel     = self._has_velocity
            traj_age    = time.monotonic() - self._last_traj_time
            actuator    = self._actuator_cmd

        # Arm velocity
        if pos_target is not None and traj_age < self._traj_timeout:
            if has_vel:
                joint_vel = vel_cmd / self.max_joint_velocities
            else:
                q_cur = self.env.sim.data.qpos[self._arm_qpos_addrs]
                joint_vel = self.pos_fallback_kp * (pos_target - q_cur) / self.max_joint_velocities
            joint_vel = np.clip(joint_vel, -1.0, 1.0)
        else:
            joint_vel = np.zeros(6)

        # Build 7-dim action: [6 arm joints, 1 solenoid]
        action = np.zeros(self.action_dim, dtype=np.float64)
        action[:6]  = joint_vel
        action[-1]  = actuator

        # Publish pre-step action
        act_msg = Float64MultiArray()
        act_msg.data = action.tolist()
        self.action_pub.publish(act_msg)

        # Step sim
        self._obs_raw, _, done, _ = self.env.step(action)
        self._step_count += 1

        # Read sensor values directly from sim for publishing
        sim = self.env.sim
        act_qpos = float(sim.data.qpos[self.env._actuator_qpos_addr])
        act_qvel = float(sim.data.qvel[self.env._actuator_qvel_addr])
        rangefinder = float(sim.data.sensordata[self.env._rangefinder_data_addr])
        force_vec = sim.data.cfrc_ext[self.env._actuator_body_id][3:]
        contact_force = float(np.linalg.norm(force_vec))
        contact_detected = (
            contact_force > CONTACT_FORCE_THRESHOLD
            and abs(act_qvel) < STALL_VEL_THRESHOLD
        )

        # ---- Synthetic Moteus a5 torque ----
        a5_raw = float(sim.data.actuator_force[self._a5_actuator_id])
        # Add Gaussian noise
        noise = np.random.normal(0.0, self._a5_noise_std)
        # Occasional outlier spike (1% chance)
        if np.random.rand() < self._a5_outlier_prob:
            noise += np.random.choice([-1, 1]) * 3.0 * self._a5_noise_std
        # Slow drift (temperature simulation)
        self._a5_drift += np.random.uniform(-self._a5_drift_rate, self._a5_drift_rate)
        self._a5_drift = np.clip(self._a5_drift, -0.1, 0.1)
        a5_noisy = a5_raw + noise + self._a5_drift
        a5_delta = a5_noisy - self._a5_torque_baseline

        now = self.get_clock().now().to_msg()

        # /mujoco/a5_torque and /mujoco/a5_torque_delta
        self.a5_trq_pub.publish(Float32(data=a5_noisy))
        self.a5_delta_pub.publish(Float32(data=a5_delta))

        # /mujoco/joint_states — 6 arm joints + solenoid
        qpos = sim.data.qpos[self._arm_qpos_addrs]
        qvel = sim.data.qvel[self._arm_qvel_addrs]
        js = JointState()
        js.header.stamp = now
        js.name     = list(self.URDF_JOINT_NAMES) + ['linear_actuator']
        js.position = qpos.tolist() + [act_qpos]
        js.velocity = qvel.tolist() + [act_qvel]
        self.js_pub.publish(js)

        # /mujoco/observations
        flat = self.env._flat_obs()
        obs_msg = Float64MultiArray()
        obs_msg.data = flat.tolist()
        self.obs_pub.publish(obs_msg)

        # /mujoco/eef_rangefinder
        self.range_pub.publish(Float32(data=rangefinder))

        # /mujoco/actuator_pos
        self.act_pos_pub.publish(Float32(data=act_qpos))

        # /mujoco/actuator_torque (synthetic contact force proxy)
        self.act_trq_pub.publish(Float32(data=contact_force))

        # /mujoco/contact_detected
        self.contact_pub.publish(Bool(data=contact_detected))

        # /mujoco/pressed_key — find which key the solenoid tip is closest to
        pressed_msg = String()
        if contact_detected:
            tip_pos = sim.data.site_xpos[
                sim.model.site_name2id("gripper0_right_actuator_tip_site")
            ]
            best_key = ""
            best_dist = float("inf")
            for name in self.env.AVAILABLE_KEYS:
                kpos = sim.data.body_xpos[self.env._key_body_ids[name]]
                d = float(np.linalg.norm(tip_pos[:2] - kpos[:2]))
                if d < best_dist:
                    best_dist = d
                    best_key = name
            pressed_msg.data = best_key
            if best_key != getattr(self, '_last_pressed', ''):
                self.get_logger().info(f"PRESSED: '{best_key}' (dist={best_dist*100:.1f}cm)")
                self._last_pressed = best_key
        else:
            pressed_msg.data = ""
            self._last_pressed = ""
        self.pressed_pub.publish(pressed_msg)

        # /mujoco/target_key
        tk_msg = String()
        tk_msg.data = self.env._target_key
        self.target_key_pub.publish(tk_msg)

        # /mujoco/eef_camera
        self._publish_camera(now)

        # Render
        if self.render_enabled:
            self.env.render()

        # Debug output
        if self.debug and self._step_count % 20 == 0:
            self._print_debug(joint_vel, act_qpos, rangefinder, contact_force,
                              contact_detected, a5_raw, a5_noisy, a5_delta)

        if done:
            self.get_logger().info("Episode done — resetting")
            self._obs_raw = self.env.reset()
            # Re-capture a5 torque baseline at rest pose
            self._a5_torque_baseline = float(
                self.env.sim.data.actuator_force[self._a5_actuator_id]
            )
            self._a5_drift = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _publish_camera(self, stamp):
        """Render EEF camera and publish as sensor_msgs/Image (rgb8, 64×64)."""
        try:
            frame = self.env.sim.render(
                width=self.CAM_W, height=self.CAM_H,
                camera_name=self.EEF_CAM_NAME,
                depth=False,
            )
            # RoboSuite returns (H, W, 3) with Y-axis flipped; flip back
            frame = frame[::-1, :, :]
            msg = Image()
            msg.header.stamp = stamp
            msg.height   = self.CAM_H
            msg.width    = self.CAM_W
            msg.encoding = 'rgb8'
            msg.step     = self.CAM_W * 3
            msg.data     = frame.flatten().tobytes()
            self.cam_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"Camera render failed: {e}", throttle_duration_sec=5.0)

    def _print_debug(self, joint_vel, act_qpos, rangefinder, contact_force,
                     contact_detected, a5_raw, a5_noisy, a5_delta):
        obs  = self._obs_raw
        pf   = self.env.robots[0].robot_model.naming_prefix
        eef  = np.array(obs.get(f"{pf}eef_pos", [0, 0, 0]))
        tkey = np.array(obs.get("target_key_pos", [0, 0, 0]))
        delta = np.array(obs.get("eef_to_key", [0, 0, 0]))
        jpos  = np.array(obs.get(f"{pf}joint_pos", np.zeros(6)))
        touch_str = "YES ←" if contact_detected else "no"
        contact_str = "CONTACT ←" if abs(a5_delta) > 0.5 else ""
        print(
            f"\n[t={self._step_count:05d}]"
            f"  EEF: [{eef[0]:.3f}, {eef[1]:.3f}, {eef[2]:.3f}]"
            f"  Actuator: {act_qpos:.4f}m"
            f"  Contact: {touch_str}"
        )
        print(
            f"           Target: key_{self.env._target_key}"
            f" @ [{tkey[0]:.3f}, {tkey[1]:.3f}, {tkey[2]:.3f}]"
            f"  Δ: [{delta[0]:.3f}, {delta[1]:.3f}, {delta[2]:.3f}]"
        )
        print(
            f"           Rangefinder: {rangefinder:.4f}m"
            f"  Force: {contact_force:.3f}N"
            f"  Joints: {np.round(jpos, 2).tolist()}"
        )
        print(
            f"           a5 torque: raw={a5_raw:.3f}Nm  "
            f"noisy={a5_noisy:.3f}Nm  "
            f"Δ={a5_delta:+.3f}Nm  "
            f"baseline={self._a5_torque_baseline:.3f}Nm  {contact_str}"
        )

    def destroy_node(self):
        self.env.close()
        super().destroy_node()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MuJoCo Keyboard Bridge Node")
    parser.add_argument('--no-render', action='store_true', help='Run headless')
    parser.add_argument('--debug', action='store_true',
                        help='Print formatted sensor table every second')
    parser.add_argument('--trajectory-topic', type=str, default=None,
                        help='Override JointTrajectory topic')
    args = parser.parse_args()

    rclpy.init()
    node = KeyboardBridgeNode(
        render=not args.no_render,
        debug=args.debug,
        trajectory_topic=args.trajectory_topic,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
