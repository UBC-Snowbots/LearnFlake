#!/usr/bin/env python3
"""
MuJoCo Bridge Node — Cartesian Twist → Jacobian IK

Subscribes to the SAME Cartesian twist topic that MoveIt Servo uses
(/arm_moveit_control/delta_twist_cmds) and performs Jacobian-based IK
locally in MuJoCo.  This drives both arms (RViz + MuJoCo) from the
same human intent — bypassing the firmware deg/s conversion entirely.

Also subscribes to /arm/ee_command/sim for gripper toggle commands.

Architecture:
    Joystick → moveit_control (joyCallback)
        ├─→ TwistStamped on /arm_moveit_control/delta_twist_cmds
        │      ├─→ MoveIt Servo → IK → RViz arm
        │      └─→ THIS NODE → Jacobian IK → MuJoCo arm  ← ★
        └─→ Gripper cmd on /arm/command → sim_helper_node
               └─→ /arm/ee_command/sim → THIS NODE        ← ★

Publishes:
    /mujoco/observations  — flat obs vector for demo_recorder / RL policy
    /mujoco/actions       — normalized [-1,1] action sent to env.step()

Usage:
    # Terminal 1: Launch RoverFlake2 arm teleop (joystick + moveit + RViz)
    # Terminal 2: Run this bridge node
    python3 cartesian_control_ros.py [--no-render] [--domain-rand]
    python3 cartesian_control_ros.py --linear-speed 0.4 --angular-speed 0.6
"""

import os
import sys
import time
import argparse
import threading
import numpy as np

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray, Float64
from geometry_msgs.msg import TwistStamped

# Path setup for RoboSuite
ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite")
if os.path.exists(ROBO_PATH) and ROBO_PATH not in sys.path:
    sys.path.insert(0, ROBO_PATH)

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
import mujoco


# ============================================================================
# Observation processing (matches train_lift_v2.py / RoboSuiteEnvV2 exactly)
# ============================================================================

OBS_KEYS = [
    'robot0_joint_pos',
    'robot0_joint_vel',
    'robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
    'cube_pos',
    'gripper_to_cube_pos',
]

NUM_PHASE_DIMS = 4  # reach, grasp, lift, hold (matches V2)


def compute_phase(obs: dict) -> np.ndarray:
    """Compute one-hot phase encoding identical to RoboSuiteEnvV2._compute_phase."""
    cube_pos = obs.get('cube_pos', [0, 0, 0])
    gripper_to_cube = obs.get('gripper_to_cube_pos', [0, 0, 0])
    gripper_qpos = obs.get('robot0_gripper_qpos', [0, 0])

    distance = np.linalg.norm(gripper_to_cube)
    gripper_closed = np.mean(gripper_qpos) < 0.02
    height_above_table = max(0, cube_pos[2] - 0.82)

    phase = np.zeros(NUM_PHASE_DIMS, dtype=np.float32)
    if height_above_table > 0.08:
        phase[3] = 1.0  # Hold
    elif height_above_table > 0.01 or (gripper_closed and distance < 0.1):
        phase[2] = 1.0  # Lift
    elif distance < 0.1:
        phase[1] = 1.0  # Grasp
    else:
        phase[0] = 1.0  # Reach
    return phase


def process_obs(obs: dict) -> np.ndarray:
    """Convert raw robosuite obs dict → flat float32 array (same as train_lift_v2)."""
    obs_list = []
    for key in OBS_KEYS:
        if key in obs:
            obs_list.append(np.array(obs[key]).flatten())
    base_obs = np.concatenate(obs_list).astype(np.float32)
    phase = compute_phase(obs)
    return np.concatenate([base_obs, phase]).astype(np.float32)


# ============================================================================
# MuJoCo Bridge Node — Jacobian IK version
# ============================================================================

class MujocoBridgeNode(Node):
    """
    ROS2 node that:
      1. Subscribes to /arm_moveit_control/delta_twist_cmds (TwistStamped)
         — the SAME Cartesian velocity the human commands via joystick
      2. Performs Jacobian IK in MuJoCo to convert Cartesian → joint velocities
         (same technique as the proven cartesian_control.py)
      3. Subscribes to /arm/ee_command/sim (Float64) for gripper open/close
      4. Steps the MuJoCo environment with the resulting normalized action
      5. Publishes processed observations and actions for the demo recorder
    """

    # Topic names
    TWIST_TOPIC = "/arm_moveit_control/delta_twist_cmds"
    EE_TOPIC = "/arm/ee_command/sim"
    OBS_TOPIC = "/mujoco/observations"
    ACTION_TOPIC = "/mujoco/actions"

    def __init__(self, render: bool = True, domain_rand: bool = False,
                 linear_speed: float = 0.3, angular_speed: float = 0.5):
        super().__init__('mujoco_bridge')

        # ---- RoboSuite environment (JOINT_VELOCITY controller) ----
        arm_ctrl = suite.load_part_controller_config(default_controller="JOINT_VELOCITY")
        ctrl_cfg = refactor_composite_controller_config(arm_ctrl, "Rover2026", ["right"])

        self.env = suite.make(
            env_name="Lift",
            robots=["Rover2026"],
            controller_configs=ctrl_cfg,
            has_renderer=render,
            has_offscreen_renderer=False,
            render_camera="agentview",
            ignore_done=True,
            use_camera_obs=False,
            control_freq=20,
            horizon=10_000,
            reward_shaping=True,
        )
        if render:
            self.env = VisualizationWrapper(self.env, indicator_configs=None)

        self.render_enabled = render
        self.domain_rand = domain_rand
        self.action_dim = self.env.action_dim  # 7 (6 joints + gripper)

        # ---- MuJoCo model references for Jacobian IK ----
        self.robot = self.env.robots[0]
        self.sim = self.env.sim
        self.model = self.sim.model
        self.data = self.sim.data

        # Find EE site (same lookup as cartesian_control.py)
        self.ee_site_name = "gripper0_grip_site"
        try:
            self.ee_site_id = self.model.site_name2id(self.ee_site_name)
        except Exception:
            for i in range(self.model.nsite):
                name = self.model.site_id2name(i)
                if "grip" in name.lower():
                    self.ee_site_name = name
                    self.ee_site_id = i
                    break
            else:
                raise ValueError("Could not find end-effector site!")

        self.get_logger().info(f"EE site: {self.ee_site_name} (id={self.ee_site_id})")

        # Arm joint info (6 arm joints, no gripper)
        self.arm_joint_names = [
            "robot0_shoulder_joint",
            "robot0_link_1_joint",
            "robot0_link1_link2",
            "robot0_a4_rotation",
            "robot0_a5_rotation",
            "robot0_a6_rotation",
        ]
        self.num_arm_joints = len(self.arm_joint_names)

        # Jacobian IK parameters (tuned from working cartesian_control.py)
        self.linear_speed_scale = linear_speed    # Scale joystick [-1,1] → m/s
        self.angular_speed_scale = angular_speed  # Scale joystick [-1,1] → rad/s
        self.damping = 0.01                       # Damped least-squares lambda

        # ---- Singularity protection (mirrors MoveIt Servo params) ----
        # From rover_servo_params_dev_arm.yaml:
        #   lower_singularity_threshold: 1000.0
        #   hard_stop_singularity_threshold: 5000.0
        #   leaving_singularity_threshold_multiplier: 2.0
        self.lower_singularity_threshold = 1000.0
        self.hard_stop_singularity_threshold = 5000.0
        self.leaving_singularity_multiplier = 2.0
        self._was_in_singularity = False

        # ---- Joint-limit margin deceleration (mirrors Servo joint_limit_margin) ----
        # Servo param: joint_limit_margin: 0.01 rad
        # We use a wider soft margin to start decelerating before hitting the hard limit
        self.joint_limit_margin = 0.01   # Hard stop margin (rad) — same as Servo
        self.joint_limit_decel_zone = 0.1  # Start decelerating this far from limit (rad)

        # Joint limits from URDF (matching robot.xml after our fix)
        # Format: (lower, upper) in radians.  None = unlimited
        self.joint_limits = [
            (-0.22, 5.5),       # shoulder_joint
            (-3.14, 0.0),       # link_1_joint
            (0.0,   3.14),      # link1_link2
            (-1.57, 1.57),      # a4_rotation
            (-3.14, 3.14),      # a5_rotation
            None,               # a6_rotation (continuous)
        ]

        # Max joint velocities from joint_limits.yaml (MoveIt)
        # These cap the normalized [-1,1] output velocity
        self.max_joint_velocities = [
            1.0,   # shoulder_joint
            1.0,   # link_1_joint
            0.8,   # link1_link2
            0.8,   # a4_rotation
            0.8,   # a5_rotation
            1.0,   # a6_rotation
        ]

        # ---- State ----
        self._twist_lin = np.zeros(3, dtype=np.float64)   # EE-frame linear vel
        self._twist_ang = np.zeros(3, dtype=np.float64)   # EE-frame angular vel
        self._gripper_state = -1.0  # -1 = open (MuJoCo default), +1 = closed
        self._lock = threading.Lock()
        self._obs_raw: dict = {}

        # ---- ROS2 plumbing ----
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # Subscribe to Cartesian twist (same topic MoveIt Servo receives)
        self.twist_sub = self.create_subscription(
            TwistStamped, self.TWIST_TOPIC, self._twist_callback, qos)

        # Subscribe to gripper commands
        self.ee_sub = self.create_subscription(
            Float64, self.EE_TOPIC, self._ee_callback, qos)

        # Publish observations for demo recorder
        self.obs_pub = self.create_publisher(
            Float64MultiArray, self.OBS_TOPIC, qos)

        # Publish actual actions sent to env.step() for demo recorder
        self.action_pub = self.create_publisher(
            Float64MultiArray, self.ACTION_TOPIC, qos)

        # Step timer at control_freq (20 Hz = 50 ms)
        self._timer = self.create_timer(1.0 / 20.0, self._step_tick)

        # ---- Initial reset ----
        self._obs_raw = self.env.reset()
        if self.domain_rand:
            self._randomize_cube()
            self._obs_raw = self.env._get_observations()
        if self.render_enabled:
            self.env.render()

        self.get_logger().info(
            f"MuJoCo bridge ONLINE (Jacobian IK)  |  action_dim={self.action_dim}  |  "
            f"render={render}  |  domain_rand={domain_rand}"
        )
        self.get_logger().info(f"  Twist topic:   {self.TWIST_TOPIC}")
        self.get_logger().info(f"  Gripper topic: {self.EE_TOPIC}")
        self.get_logger().info(f"  Obs topic:     {self.OBS_TOPIC}")
        self.get_logger().info(f"  Action topic:  {self.ACTION_TOPIC}")
        self.get_logger().info(
            f"  Speed scales:  linear={self.linear_speed_scale} m/s  "
            f"angular={self.angular_speed_scale} rad/s  damping={self.damping}"
        )

    # ------------------------------------------------------------------
    # ROS2 callbacks
    # ------------------------------------------------------------------

    def _twist_callback(self, msg: TwistStamped):
        """Receive Cartesian twist command (same input MoveIt Servo gets)."""
        with self._lock:
            self._twist_lin[0] = msg.twist.linear.x
            self._twist_lin[1] = msg.twist.linear.y
            self._twist_lin[2] = msg.twist.linear.z
            self._twist_ang[0] = msg.twist.angular.x
            self._twist_ang[1] = msg.twist.angular.y
            self._twist_ang[2] = msg.twist.angular.z

    def _ee_callback(self, msg: Float64):
        """Receive gripper command.
        RoverFlake2: GRIPPER_OPEN_VALUE=1.0, GRIPPER_CLOSE_VALUE=0.0
        RoboSuite:   -1.0 = open, +1.0 = closed
        """
        with self._lock:
            self._gripper_state = -1.0 if msg.data >= 0.5 else 1.0

    # ------------------------------------------------------------------
    # Jacobian IK (adapted from working cartesian_control.py)
    # ------------------------------------------------------------------

    def _get_ee_rotation(self) -> np.ndarray:
        """Get 3x3 rotation matrix of EE site in world frame."""
        return self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

    def _get_jacobian(self):
        """Compute position and rotation Jacobians at the EE site.
        Returns: (J_pos, J_rot) each shape (3, nv)
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model._model, self.data._data,
            jacp, jacr, self.ee_site_id
        )
        return jacp, jacr

    def _twist_to_joint_velocities(self, twist_lin_ee, twist_ang_ee) -> np.ndarray:
        """
        Convert EE-frame Cartesian twist → joint velocities via Jacobian IK.
        Includes singularity protection matching MoveIt Servo behavior.

        Steps:
            1. Scale joystick range [-1, 1] to physical velocities (m/s, rad/s)
            2. Rotate twist from EE frame to world frame using EE orientation
            3. Build 6D twist [linear; angular] in world frame
            4. Check singularity via Jacobian condition number → scale vel down
            5. Compute damped least-squares pseudoinverse of the arm Jacobian
            6. Apply joint-limit proximity deceleration
            7. Return joint velocities (num_arm_joints,)
        """
        # Scale joystick values to physical velocities
        cart_vel_ee = twist_lin_ee * self.linear_speed_scale
        ang_vel_ee = twist_ang_ee * self.angular_speed_scale

        # Transform from EE frame → world frame
        R = self._get_ee_rotation()
        cart_vel_world = R @ cart_vel_ee
        ang_vel_world = R @ ang_vel_ee

        # Stack into 6D twist [linear; angular]
        twist_world = np.concatenate([cart_vel_world, ang_vel_world])

        # Get full Jacobian at EE site (each 3 x nv)
        J_pos, J_rot = self._get_jacobian()
        J_full = np.vstack([J_pos, J_rot])  # (6, nv)

        # Extract columns for arm joints only (first num_arm_joints DoFs)
        J_arm = J_full[:, :self.num_arm_joints]  # (6, 6)

        # ---- Singularity protection (condition number check) ----
        # Mirrors MoveIt Servo: decelerate between lower and hard_stop thresholds
        singularity_scale = self._compute_singularity_scale(J_arm)
        twist_world *= singularity_scale

        # Damped least-squares: q̇ = Jᵀ (J Jᵀ + λ²I)⁻¹ ẋ
        JJT = J_arm @ J_arm.T
        damped = JJT + self.damping**2 * np.eye(6)
        try:
            joint_vel = J_arm.T @ np.linalg.solve(damped, twist_world)
        except np.linalg.LinAlgError:
            joint_vel = np.linalg.pinv(J_arm) @ twist_world

        # ---- Joint limit proximity deceleration ----
        joint_vel = self._apply_joint_limit_decel(joint_vel)

        # ---- Cap per-joint max velocity (from MoveIt joint_limits.yaml) ----
        for i in range(self.num_arm_joints):
            max_v = self.max_joint_velocities[i]
            joint_vel[i] = np.clip(joint_vel[i], -max_v, max_v)

        return joint_vel

    def _compute_singularity_scale(self, J_arm: np.ndarray) -> float:
        """
        Compute velocity scaling factor based on Jacobian condition number.
        Mirrors MoveIt Servo singularity handling:
          - Below lower_threshold: scale = 1.0 (no slowdown)
          - Between lower and hard_stop: linear ramp from 1.0 → 0.0
          - Above hard_stop: scale = 0.0 (full stop)
          - When leaving singularity: use multiplied threshold for hysteresis
        """
        try:
            sv = np.linalg.svd(J_arm, compute_uv=False)
            if sv[-1] < 1e-10:
                cond = float('inf')
            else:
                cond = sv[0] / sv[-1]
        except np.linalg.LinAlgError:
            cond = float('inf')

        # Hysteresis: when leaving singularity, use a wider threshold
        if self._was_in_singularity:
            effective_hard_stop = (self.hard_stop_singularity_threshold
                                   * self.leaving_singularity_multiplier)
        else:
            effective_hard_stop = self.hard_stop_singularity_threshold

        if cond >= effective_hard_stop:
            self._was_in_singularity = True
            self.get_logger().warn(
                f"SINGULARITY HARD STOP (cond={cond:.0f} >= {effective_hard_stop:.0f})",
                throttle_duration_sec=1.0,
            )
            return 0.0
        elif cond >= self.lower_singularity_threshold:
            # Linear ramp from 1.0 → 0.0
            t = ((cond - self.lower_singularity_threshold)
                 / (effective_hard_stop - self.lower_singularity_threshold))
            scale = 1.0 - t
            self.get_logger().info(
                f"Near singularity (cond={cond:.0f}, scale={scale:.2f})",
                throttle_duration_sec=1.0,
            )
            return max(scale, 0.0)
        else:
            self._was_in_singularity = False
            return 1.0

    def _apply_joint_limit_decel(self, joint_vel: np.ndarray) -> np.ndarray:
        """
        Decelerate joints approaching their limits.
        Mirrors MoveIt Servo joint_limit_margin behavior:
          - Within joint_limit_margin of limit: velocity set to 0 (hard stop)
          - Within joint_limit_decel_zone: linearly scale velocity toward 0
        Only affects velocity in the direction toward the limit.
        """
        # Get current joint positions from MuJoCo
        qpos = np.array([
            self.data.qpos[self.model.joint_name2id(name)]
            for name in self.arm_joint_names
        ])

        for i in range(self.num_arm_joints):
            limits = self.joint_limits[i]
            if limits is None:
                continue  # Unlimited joint

            lower, upper = limits

            # Check proximity to lower limit (only if moving toward it)
            if joint_vel[i] < 0:
                dist_to_lower = qpos[i] - lower
                if dist_to_lower <= self.joint_limit_margin:
                    joint_vel[i] = 0.0
                elif dist_to_lower <= self.joint_limit_decel_zone:
                    # Linear ramp: 0 at margin → 1 at decel_zone
                    t = ((dist_to_lower - self.joint_limit_margin)
                         / (self.joint_limit_decel_zone - self.joint_limit_margin))
                    joint_vel[i] *= t

            # Check proximity to upper limit (only if moving toward it)
            if joint_vel[i] > 0:
                dist_to_upper = upper - qpos[i]
                if dist_to_upper <= self.joint_limit_margin:
                    joint_vel[i] = 0.0
                elif dist_to_upper <= self.joint_limit_decel_zone:
                    t = ((dist_to_upper - self.joint_limit_margin)
                         / (self.joint_limit_decel_zone - self.joint_limit_margin))
                    joint_vel[i] *= t

        return joint_vel

    # ------------------------------------------------------------------
    # Step loop
    # ------------------------------------------------------------------

    def _step_tick(self):
        """Called at 20 Hz — Jacobian IK → step MuJoCo → publish obs + action."""
        with self._lock:
            twist_lin = self._twist_lin.copy()
            twist_ang = self._twist_ang.copy()
            gripper = self._gripper_state
            # Reset twist so arm stops when no new commands arrive
            self._twist_lin[:] = 0.0
            self._twist_ang[:] = 0.0

        # Jacobian IK: Cartesian twist → normalized joint velocities
        if np.any(twist_lin != 0) or np.any(twist_ang != 0):
            joint_vel = self._twist_to_joint_velocities(twist_lin, twist_ang)
            joint_vel = np.clip(joint_vel, -1.0, 1.0)
        else:
            joint_vel = np.zeros(self.num_arm_joints)

        # Build action vector: [joint_vels..., gripper]
        action = np.zeros(self.action_dim, dtype=np.float64)
        action[:self.num_arm_joints] = joint_vel
        action[-1] = gripper

        # Step MuJoCo
        self._obs_raw, reward, done, info = self.env.step(action)

        # Publish processed observation
        obs_flat = process_obs(self._obs_raw)
        obs_msg = Float64MultiArray()
        obs_msg.data = obs_flat.tolist()
        self.obs_pub.publish(obs_msg)

        # Publish action (the actual normalized [-1,1] sent to env.step)
        action_msg = Float64MultiArray()
        action_msg.data = action.tolist()
        self.action_pub.publish(action_msg)

        # Render
        if self.render_enabled:
            self.env.render()

        # Log EE position periodically when moving
        if np.any(joint_vel != 0):
            eef_pos = self._obs_raw.get('robot0_eef_pos', None)
            if eef_pos is not None:
                self.get_logger().info(
                    f"EE: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}] | "
                    f"twist_lin: [{twist_lin[0]:.2f}, {twist_lin[1]:.2f}, {twist_lin[2]:.2f}] | "
                    f"jvel: [{joint_vel[0]:.3f}, {joint_vel[1]:.3f}, {joint_vel[2]:.3f}, "
                    f"{joint_vel[3]:.3f}, {joint_vel[4]:.3f}, {joint_vel[5]:.3f}]",
                    throttle_duration_sec=0.5,
                )

        # Auto-reset on done
        if done:
            self.get_logger().info("Episode done — resetting environment")
            self._obs_raw = self.env.reset()
            if self.domain_rand:
                self._randomize_cube()
                self._obs_raw = self.env._get_observations()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _randomize_cube(self):
        """Randomize cube position (same ranges as train_lift_v2)."""
        try:
            cube_id = self.env.sim.model.body_name2id('cube_main')
            pos = self.env.sim.model.body_pos[cube_id].copy()
            pos[0] += np.random.uniform(-0.1, 0.1)
            pos[1] += np.random.uniform(-0.1, 0.1)
            self.env.sim.model.body_pos[cube_id] = pos
            self.env.sim.forward()
        except Exception:
            pass

    def destroy_node(self):
        self.env.close()
        super().destroy_node()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MuJoCo Jacobian-IK Bridge Node")
    parser.add_argument('--no-render', action='store_true', help='Run headless')
    parser.add_argument('--domain-rand', action='store_true', help='Randomize cube position')
    parser.add_argument('--linear-speed', type=float, default=0.3,
                        help='Max linear EE speed in m/s (default: 0.3)')
    parser.add_argument('--angular-speed', type=float, default=0.5,
                        help='Max angular EE speed in rad/s (default: 0.5)')
    args = parser.parse_args()

    rclpy.init()
    node = MujocoBridgeNode(
        render=not args.no_render,
        domain_rand=args.domain_rand,
        linear_speed=args.linear_speed,
        angular_speed=args.angular_speed,
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
