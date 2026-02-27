#!/usr/bin/env python3
"""
MuJoCo Bridge Node — Cartesian Twist → Jacobian IK

Two input modes (selected via --input-mode):

  twist  (default)
      Subscribes to the Cartesian twist topic that MoveIt Servo uses
      (/servo_node/delta_twist_cmds) for arm movement and to /joy for
      the gripper toggle button.  The C++ node (joy_arm_control) handles
      joystick → twist conversion; this node re-reads /joy only for
      the gripper since joy_arm_control doesn't publish gripper commands.

  joy
      Subscribes directly to /joy and converts raw button/axis data into
      Cartesian twist using controller_config.py profiles.  Supports both
      the Nintendo Switch Pro Controller and the Saitek Cyborg USB Stick.
      Auto-detects the controller from the first /joy message, or use
      --controller pro|cyborg to force one.

Both modes perform Jacobian-based IK locally in MuJoCo and publish the
same observation/action topics for demo_recorder and RL training.

Architecture (twist mode):
    Joystick → /joy
        ├─→ joy_arm_control → TwistStamped on /servo_node/delta_twist_cmds
        │      ├─→ MoveIt Servo → IK → RViz arm
        │      └─→ THIS NODE → Jacobian IK → MuJoCo arm       ← ★
        └─→ THIS NODE (reads /joy directly for gripper toggle) ← ★

Architecture (joy mode — new):
    Joystick → /joy
        └─→ THIS NODE (auto-detect Pro / Cyborg)
              ├─→ Cartesian twist → Jacobian IK → MuJoCo arm  ← ★
              └─→ Gripper toggle (controller button)           ← ★

Publishes:
    /mujoco/observations  — flat obs vector for demo_recorder / RL policy
    /mujoco/actions       — normalized [-1,1] action sent to env.step()

Usage:
    # Twist mode (default — requires RoverFlake2 arm launch running):
    python3 cartesian_control_ros.py [--no-render] [--domain-rand]
    python3 cartesian_control_ros.py --linear-speed 0.4 --angular-speed 1.0

    # Joy mode (direct joystick — works standalone):
    python3 cartesian_control_ros.py --input-mode joy
    python3 cartesian_control_ros.py --input-mode joy --controller pro
    python3 cartesian_control_ros.py --input-mode joy --controller cyborg
"""

# ---- Standard library ----
import os
import sys
import time
import argparse
import threading
import numpy as np

# ---- ROS2 core + message types ----
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TwistStamped   # Cartesian twist from MoveIt Servo pipeline
from sensor_msgs.msg import Joy               # Raw joystick input (both modes for gripper)

# ---- Dual-controller profiles (see controller_config.py) ----
from controller_config import (
    ControllerProfile, PRO_CONTROLLER, CYBORG_STICK,
    detect_controller, get_controller,
)

# ---- RoboSuite / MuJoCo path bootstrap ----
# RoboSuite lives outside the normal ROS workspace; add it to sys.path manually.
ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite")
if os.path.exists(ROBO_PATH) and ROBO_PATH not in sys.path:
    sys.path.insert(0, ROBO_PATH)

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
import mujoco  # Low-level API for Jacobian computation


# ============================================================================
# Observation processing (matches train_lift_v2.py / RoboSuiteEnvV2 exactly)
# ============================================================================

# Observation keys extracted from the RoboSuite obs dict, concatenated in this
# order to form the flat vector consumed by the RL policy and demo recorder.
# Must match the order used in train_lift_v2.py / RoboSuiteEnvV2.
OBS_KEYS = [
    'robot0_joint_pos',        # 6-dim joint angles
    'robot0_joint_vel',        # 6-dim joint velocities
    'robot0_eef_pos',          # 3-dim end-effector position
    'robot0_eef_quat',         # 4-dim end-effector quaternion
    'robot0_gripper_qpos',     # 2-dim gripper finger positions
    'cube_pos',                # 3-dim target object position
    'gripper_to_cube_pos',     # 3-dim relative vector (gripper → cube)
]

# One-hot phase encoding appended to observations (reach → grasp → lift → hold)
NUM_PHASE_DIMS = 4


def compute_phase(obs: dict) -> np.ndarray:
    """Compute one-hot phase encoding identical to RoboSuiteEnvV2._compute_phase.

    Phase progression: Reach → Grasp → Lift → Hold.  Determined by the
    gripper-to-cube distance, gripper closure, and cube height above table.
    The RL policy uses this as auxiliary input to know the current task stage.
    """
    cube_pos = obs.get('cube_pos', [0, 0, 0])
    gripper_to_cube = obs.get('gripper_to_cube_pos', [0, 0, 0])
    gripper_qpos = obs.get('robot0_gripper_qpos', [0, 0])

    distance = np.linalg.norm(gripper_to_cube)       # EE ↔ cube distance
    gripper_closed = np.mean(gripper_qpos) < 0.02    # fingers nearly shut
    height_above_table = max(0, cube_pos[2] - 0.82)  # table surface ≈ z=0.82

    phase = np.zeros(NUM_PHASE_DIMS, dtype=np.float32)
    if height_above_table > 0.08:
        phase[3] = 1.0  # Hold  — cube lifted and stable
    elif height_above_table > 0.01 or (gripper_closed and distance < 0.1):
        phase[2] = 1.0  # Lift  — cube rising or grasped near surface
    elif distance < 0.1:
        phase[1] = 1.0  # Grasp — close enough to grab
    else:
        phase[0] = 1.0  # Reach — still approaching cube
    return phase


def process_obs(obs: dict) -> np.ndarray:
    """Convert raw RoboSuite obs dict → flat float32 array.

    Concatenates the fields listed in OBS_KEYS (order matters!) followed
    by the 4-dim phase encoding.  Output shape must match the RL policy's
    expected input dimension.
    """
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
    ROS2 node that mirrors RViz arm motion in MuJoCo.

    Input modes:
      twist — subscribes to /servo_node/delta_twist_cmds (same topic MoveIt
              Servo reads) for arm movement, and to /joy for gripper toggle.
      joy   — subscribes only to /joy and converts raw button/axis data to
              Cartesian twist + gripper using controller_config.py profiles.

    Both modes:
      - Perform Jacobian IK in MuJoCo to convert Cartesian twist → joint vels
      - Step the MuJoCo environment with the resulting normalised action
      - Publish observations + actions for demo_recorder / RL training

    IMPORTANT — frame convention:
      Incoming twists are interpreted from TwistStamped.header.frame_id:
        - base/world frame id  -> use command directly as world-frame twist
        - ee_base_link frame   -> rotate EE-frame twist into world frame
      This lets button-based +/-X, +/-Y, +/-Z remain world-fixed and
      independent of current arm orientation when using base-frame commands.
    """

    # Topic names
    TWIST_TOPIC = "/servo_node/delta_twist_cmds"  # Same topic MoveIt Servo reads
    JOY_TOPIC = "/joy"                            # Raw gamepad input
    OBS_TOPIC = "/mujoco/observations"             # Flat obs for RL
    ACTION_TOPIC = "/mujoco/actions"               # Normalised [-1,1] actions
    BASE_FRAMES = {"base_link", "world", "map"}
    EE_FRAMES = {"ee_base_link"}

    def __init__(self, render: bool = True, domain_rand: bool = False,
                 linear_speed: float = 0.5, angular_speed: float = 1.2,
                 input_mode: str = "twist",
                 controller_profile: ControllerProfile | None = None):
        """Initialize the MuJoCo bridge node.

        Args:
            render:             Show MuJoCo viewer window (False for headless/CI).
            domain_rand:        Randomize cube position on reset for generalization.
            linear_speed:       Max Cartesian linear speed in m/s (scales [-1,1] input).
                                Default 0.5 matches MoveIt Servo scale.linear.
            angular_speed:      Max Cartesian angular speed in rad/s.
                                Default 1.2 matches MoveIt Servo scale.rotational.
            input_mode:         'twist' (C++ pipeline) or 'joy' (direct gamepad).
            controller_profile: Forced controller profile, or None to auto-detect from
                                the first /joy message.  Used in both modes (twist mode
                                needs it for the gripper button).
        """
        super().__init__('mujoco_bridge')

        # -- Input mode config --
        self.input_mode = input_mode
        self._controller_profile = controller_profile   # None = auto-detect
        self._controller_detected = controller_profile is not None
        self._prev_gripper_btn = False                  # edge-detect for toggle

        # ---- RoboSuite environment ----
        # We use JOINT_VELOCITY control: each action dimension is a normalized
        # joint velocity in [-1, 1].  This node converts Cartesian twist → joint
        # velocities via Jacobian IK, then feeds them as the action.
        arm_ctrl = suite.load_part_controller_config(default_controller="JOINT_VELOCITY")
        # Override output_max/min per joint so the normalised action [-1,1]
        # maps to the same physical velocity range as MoveIt joint_limits.yaml.
        # Without this, the default output_max=0.5 caps ALL joints at 0.5 rad/s.
        arm_ctrl["output_max"] = [1.0, 1.0, 0.8, 0.8, 0.8, 1.0]
        arm_ctrl["output_min"] = [-1.0, -1.0, -0.8, -0.8, -0.8, -1.0]
        ctrl_cfg = refactor_composite_controller_config(arm_ctrl, "Rover2026", ["right"])

        self.env = suite.make(
            env_name="Lift",              # Cube-lifting task
            robots=["Rover2026"],          # Custom robot matching physical arm
            controller_configs=ctrl_cfg,
            has_renderer=render,
            has_offscreen_renderer=False,   # No off-screen camera (saves GPU)
            render_camera="agentview",
            ignore_done=True,              # Keep running past horizon for teleop
            use_camera_obs=False,          # No pixel observations
            control_freq=20,               # 20 Hz — matches ROS timer + Servo
            horizon=10_000,                # Very long episode for human demos
            reward_shaping=True,           # Dense reward for RL training
        )
        if render:
            self.env = VisualizationWrapper(self.env, indicator_configs=None)

        self.render_enabled = render
        self.domain_rand = domain_rand
        self.action_dim = self.env.action_dim  # 7 (6 joints + gripper)

        # ---- MuJoCo model references for Jacobian IK ----
        # We access the raw MuJoCo model/data for Jacobian computation because
        # RoboSuite's high-level API doesn't expose this directly.
        self.robot = self.env.robots[0]
        self.sim = self.env.sim
        self.model = self.sim.model          # mujoco.MjModel wrapper
        self.data = self.sim.data            # mujoco.MjData wrapper

        # Locate the EE site used as the IK target.  "gripper0_grip_site" is
        # the point between the two finger pads — the natural grasp center.
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

        # The 6 arm DoFs (shoulder → wrist).  Gripper is handled separately
        # as a single binary action appended to the joint velocity vector.
        self.arm_joint_names = [
            "robot0_shoulder_joint",  # J0 — base rotation
            "robot0_link_1_joint",   # J1 — shoulder pitch
            "robot0_link1_link2",    # J2 — elbow
            "robot0_a4_rotation",    # J3 — wrist roll
            "robot0_a5_rotation",    # J4 — wrist pitch
            "robot0_a6_rotation",    # J5 — wrist yaw / EE rotation
        ]
        self.num_arm_joints = len(self.arm_joint_names)

        # Look up MuJoCo qvel/qpos addresses for proper Jacobian column extraction.
        # mj_jacSite returns a Jacobian with columns indexed by qvel (dof) address,
        # which may differ from simple [0,1,...,5] if multi-DoF joints (e.g. the
        # cube's free joint) or other elements precede arm joints in the model.
        # RoboSuite's own robot.py uses _ref_arm_joint_vel_indexes the same way.
        self._arm_qvel_addrs = np.array([
            self.sim.model.get_joint_qvel_addr(name)
            for name in self.arm_joint_names
        ])
        self._arm_qpos_addrs = np.array([
            self.sim.model.get_joint_qpos_addr(name)
            for name in self.arm_joint_names
        ])

        # ---- Jacobian IK tuning ----
        # These scale the unitless joystick input [-1,1] to physical velocities.
        # Increase for faster EE motion; decrease for finer control.
        self.linear_speed_scale = linear_speed    # m/s per unit input
        self.angular_speed_scale = angular_speed  # rad/s per unit input
        self.damping = 0.01  # λ for damped least-squares (J Jᵀ + λ²I)⁻¹

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

        # Joint limits from URDF (must match dev_arm.urdf / robot.xml)
        # Format: (lower, upper) in radians.  None = continuous (no limits).
        # Used by _apply_joint_limit_decel() to slow down near boundaries.
        self.joint_limits = [
            (-3.14, 3.14),      # J0 shoulder_joint
            (-3.14, 0.0),       # J1 link_1_joint
            (-3.14, 3.14),      # J2 link1_link2
            (-1.57, 3.14),      # J3 a4_rotation
            (-3.14, 3.14),      # J4 a5_rotation
            None,               # J5 a6_rotation (continuous)
        ]

        # Per-joint velocity caps from MoveIt's joint_limits.yaml.
        # Applied after Jacobian IK to prevent any single joint from
        # exceeding its rated speed, even if IK requests more.
        # These MUST match joint_limits.yaml so the IK → normalised-action
        # → RoboSuite output_max round-trip reproduces the correct rad/s.
        self.max_joint_velocities = [
            1.0,   # J0 shoulder_joint   (joint_limits.yaml: 1.0)
            1.0,   # J1 link_1_joint     (joint_limits.yaml: 1.0)
            0.8,   # J2 link1_link2      (joint_limits.yaml: 0.8)
            0.8,   # J3 a4_rotation      (joint_limits.yaml: 0.8)
            0.8,   # J4 a5_rotation      (joint_limits.yaml: 0.8)
            1.0,  # J5 a6_rotation      (joint_limits.yaml: 1.0)
        ]

        # ---- Shared state (written by callbacks, read by step timer) ----
        # Protected by _lock since ROS callbacks run on the executor thread
        # while _step_tick runs on the timer thread.
        self._twist_lin = np.zeros(3, dtype=np.float64)   # Cartesian linear  [x,y,z]
        self._twist_ang = np.zeros(3, dtype=np.float64)   # Cartesian angular [r,p,y]
        self._twist_frame = "base_link"
        self._gripper_state = -1.0  # -1 = open (RoboSuite convention), +1 = closed
        self._lock = threading.Lock()
        self._obs_raw: dict = {}    # Latest raw observation from env.step()

        # ---- ROS2 subscriptions & publishers ----
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        if self.input_mode == "twist":
            # Twist mode: arm movement comes from the same TwistStamped that
            # MoveIt Servo reads — both RViz and MuJoCo track the same input.
            self.twist_sub = self.create_subscription(
                TwistStamped, self.TWIST_TOPIC, self._twist_callback, qos)
            # Gripper: joy_arm_control.cpp does NOT publish gripper commands,
            # so we read /joy directly for the gripper toggle button only.
            self.gripper_joy_sub = self.create_subscription(
                Joy, self.JOY_TOPIC, self._gripper_joy_callback, 10)
        else:
            # Joy mode: we do the full joystick → twist + gripper conversion,
            # supporting both Pro Controller and Cyborg Stick profiles.
            self.joy_sub = self.create_subscription(
                Joy, self.JOY_TOPIC, self._joy_callback, 10)

        # demo_recorder.py subscribes to these topics to record demonstrations
        self.obs_pub = self.create_publisher(
            Float64MultiArray, self.OBS_TOPIC, qos)
        self.action_pub = self.create_publisher(
            Float64MultiArray, self.ACTION_TOPIC, qos)

        # Main control loop — 20 Hz matches RoboSuite's control_freq
        self._timer = self.create_timer(1.0 / 20.0, self._step_tick)

        # ---- Initial environment reset ----
        self._obs_raw = self.env.reset()  # Places robot at home pose
        if self.domain_rand:
            self._randomize_cube()                        # Move cube to random position
            self._obs_raw = self.env._get_observations()  # Re-observe after randomization
        if self.render_enabled:
            self.env.render()

        self.get_logger().info(
            f"MuJoCo bridge ONLINE (Jacobian IK)  |  action_dim={self.action_dim}  |  "
            f"render={render}  |  domain_rand={domain_rand}  |  input_mode={input_mode}"
        )
        self.get_logger().info(
            f"  Arm qvel addrs: {self._arm_qvel_addrs.tolist()}  "
            f"qpos addrs: {self._arm_qpos_addrs.tolist()}"
        )
        ctrl_name = self._controller_profile.name if self._controller_profile else "auto-detect"
        if self.input_mode == "twist":
            self.get_logger().info(f"  Twist topic:   {self.TWIST_TOPIC}")
            self.get_logger().info(f"  Gripper via:   {self.JOY_TOPIC} (button from {ctrl_name})")
        else:
            self.get_logger().info(f"  Joy topic:     {self.JOY_TOPIC}")
            self.get_logger().info(f"  Controller:    {ctrl_name}")
        self.get_logger().info(f"  Obs topic:     {self.OBS_TOPIC}")
        self.get_logger().info(f"  Action topic:  {self.ACTION_TOPIC}")
        self.get_logger().info(
            f"  Speed scales:  linear={self.linear_speed_scale} m/s  "
            f"angular={self.angular_speed_scale} rad/s  damping={self.damping}"
        )
        self.get_logger().info(
            f"  Twist frame handling: base/world = direct, ee_base_link = EE->world"
        )

    # ------------------------------------------------------------------
    # ROS2 callbacks
    # ------------------------------------------------------------------

    def _twist_callback(self, msg: TwistStamped):
        """Receive Cartesian twist from joy_arm_control.

        We store linear / angular components plus frame_id so IK can apply
        the correct frame conversion (base/world direct, EE frame rotated).
        """
        with self._lock:
            self._twist_lin[0] = msg.twist.linear.x
            self._twist_lin[1] = msg.twist.linear.y
            self._twist_lin[2] = msg.twist.linear.z
            self._twist_ang[0] = msg.twist.angular.x
            self._twist_ang[1] = msg.twist.angular.y
            self._twist_ang[2] = msg.twist.angular.z
            self._twist_frame = msg.header.frame_id or "base_link"

    def _gripper_joy_callback(self, msg: Joy):
        """Read gripper toggle from /joy (twist mode only).

        joy_arm_control.cpp publishes twist for the arm but does NOT publish
        gripper commands.  We subscribe directly to /joy to pick up the
        gripper button press using the active controller profile.
        """
        # Auto-detect controller on first message if not forced via CLI
        if not self._controller_detected:
            self._controller_profile = detect_controller(
                num_buttons=len(msg.buttons), num_axes=len(msg.axes)
            )
            self._controller_detected = True
            self.get_logger().info(
                f"Auto-detected controller: {self._controller_profile.name} "
                f"(buttons={len(msg.buttons)}, axes={len(msg.axes)})"
            )

        cfg = self._controller_profile
        btn_idx = cfg.btn_gripper_toggle
        gripper_btn = (0 <= btn_idx < len(msg.buttons)
                       and bool(msg.buttons[btn_idx]))
        axis_idx = cfg.axis_gripper_toggle
        if 0 <= axis_idx < len(msg.axes):
            gripper_btn = gripper_btn or (
                float(msg.axes[axis_idx]) < cfg.gripper_axis_pressed_threshold
            )

        # Edge-triggered toggle: -1 (open) ↔ +1 (closed)
        if gripper_btn and not self._prev_gripper_btn:
            with self._lock:
                self._gripper_state = -self._gripper_state
            state_str = "CLOSED" if self._gripper_state > 0 else "OPEN"
            self.get_logger().info(f"Gripper {state_str}")
        self._prev_gripper_btn = gripper_btn

    # ------------------------------------------------------------------
    # Joy callback (joy mode — direct gamepad input)
    # ------------------------------------------------------------------

    def _joy_callback(self, msg: Joy):
        """Process raw /joy message for both Pro Controller and Cyborg Stick.

        Auto-detects the controller from the first message if ``--controller``
        was not specified.  Converts buttons/axes → Cartesian twist + gripper
        toggle, mirroring the logic in RoverFlake2's joy_arm_control.cpp +
        controller_config.h.

        The resulting twist is written into ``_twist_lin`` / ``_twist_ang`` for
        the step timer to consume on the next 20 Hz tick.
        """
        # ---- Auto-detect controller on first message ----
        if not self._controller_detected:
            self._controller_profile = detect_controller(
                num_buttons=len(msg.buttons), num_axes=len(msg.axes)
            )
            self._controller_detected = True
            self.get_logger().info(
                f"Auto-detected controller: {self._controller_profile.name} "
                f"(buttons={len(msg.buttons)}, axes={len(msg.axes)})"
            )

        cfg = self._controller_profile

        # Helper: safe button read (out-of-range → False)
        def btn(idx: int) -> bool:
            return 0 <= idx < len(msg.buttons) and bool(msg.buttons[idx])

        # Helper: safe axis read with deadzone filtering
        def axis(idx: int) -> float:
            if idx < 0 or idx >= len(msg.axes):
                return 0.0
            val = float(msg.axes[idx])
            return val if abs(val) >= cfg.axis_deadzone else 0.0

        # ---- Cartesian translation ----
        # Two complementary methods: buttons (digital, Pro Controller face buttons)
        # and axes (analog, Cyborg stick).  Both contribute additively.
        lx, ly, lz = 0.0, 0.0, 0.0

        # Digital button translation (primarily used by Pro Controller)
        speed = cfg.cart_button_speed
        if btn(cfg.btn_cart_pos_x):  lx += speed
        if btn(cfg.btn_cart_neg_x):  lx -= speed
        if btn(cfg.btn_cart_pos_y):  ly += speed
        if btn(cfg.btn_cart_neg_y):  ly -= speed
        if btn(cfg.btn_cart_pos_z):  lz += speed
        if btn(cfg.btn_cart_neg_z):  lz -= speed

        # Analog axis translation (primarily used by Cyborg Stick)
        if cfg.axis_cart_x >= 0:
            v = axis(cfg.axis_cart_x) * cfg.cart_axis_speed
            lx += (-v if cfg.invert_cart_x else v)
        if cfg.axis_cart_y >= 0:
            v = axis(cfg.axis_cart_y) * cfg.cart_axis_speed
            ly += (-v if cfg.invert_cart_y else v)
        if cfg.axis_cart_z >= 0:
            v = axis(cfg.axis_cart_z) * cfg.cart_axis_speed
            lz += (-v if cfg.invert_cart_z else v)

        # ---- EE orientation from analog sticks (both controllers) ----
        rot_speed = cfg.rot_stick_speed
        ax_roll  = axis(cfg.axis_roll)  * rot_speed * (-1.0 if cfg.invert_roll  else 1.0)
        ax_pitch = axis(cfg.axis_pitch) * rot_speed * (-1.0 if cfg.invert_pitch else 1.0)
        ax_yaw   = axis(cfg.axis_yaw)   * rot_speed * (-1.0 if cfg.invert_yaw   else 1.0)

        # ---- Gripper toggle (edge-triggered) ----
        gripper_btn = btn(cfg.btn_gripper_toggle)
        if 0 <= cfg.axis_gripper_toggle < len(msg.axes):
            gripper_btn = gripper_btn or (
                float(msg.axes[cfg.axis_gripper_toggle]) < cfg.gripper_axis_pressed_threshold
            )
        if gripper_btn and not self._prev_gripper_btn:
            with self._lock:
                # Toggle: -1 ↔ +1
                self._gripper_state = -self._gripper_state
            state_str = "CLOSED" if self._gripper_state > 0 else "OPEN"
            self.get_logger().info(f"Gripper {state_str}")
        self._prev_gripper_btn = gripper_btn

        # ---- Write twist into shared state ----
        with self._lock:
            self._twist_lin[0] = lx
            self._twist_lin[1] = ly
            self._twist_lin[2] = lz
            self._twist_ang[0] = ax_roll
            self._twist_ang[1] = ax_pitch
            self._twist_ang[2] = ax_yaw
            self._twist_frame = cfg.cart_frame_id

    # ------------------------------------------------------------------
    # Jacobian IK (adapted from working cartesian_control.py)
    # ------------------------------------------------------------------

    def _get_jacobian(self):
        """Compute full 6×nv Jacobian at the EE site via MuJoCo's mj_jacSite.

        Returns:
            (J_pos, J_rot) — position and rotation Jacobians, each (3, nv).
            We later extract only the arm-joint columns to form a 6×6 matrix.
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model._model, self.data._data,
            jacp, jacr, self.ee_site_id
        )
        return jacp, jacr

    def _get_ee_rotation(self) -> np.ndarray:
        """Return the 3×3 rotation matrix of the EE site (world ← EE frame)."""
        # MuJoCo stores site orientations as 3×3 row-major rotation matrices
        return self.data.site_xmat[self.ee_site_id].reshape(3, 3)

    @staticmethod
    def _normalize_frame_id(frame_id: str | None) -> str:
        if not frame_id:
            return ""
        return frame_id.strip().lstrip("/").lower()

    def _twist_to_joint_velocities(self, twist_lin, twist_ang, twist_frame: str) -> np.ndarray:
        """
        Convert Cartesian twist → joint velocities via Jacobian IK.
        Base/world twists are consumed directly; EE-frame twists are rotated
        into world frame before solving the Jacobian.

        Steps:
            1. Scale input to physical velocities (m/s, rad/s)
            2. Frame conversion (base/world direct OR EE -> world rotation)
            3. Build 6D twist [linear; angular] in world frame
            4. Check singularity via Jacobian condition number → scale vel down
            5. Compute damped least-squares pseudoinverse of the arm Jacobian
            6. Apply joint-limit proximity deceleration
            7. Return joint velocities (num_arm_joints,)
        """
        # Scale unitless input to physical velocities
        cart_vel_cmd = twist_lin * self.linear_speed_scale
        ang_vel_cmd = twist_ang * self.angular_speed_scale

        # MuJoCo Jacobian is world-frame; convert only if command is EE-frame.
        frame_norm = self._normalize_frame_id(twist_frame)
        if frame_norm in self.EE_FRAMES:
            R_world_ee = self._get_ee_rotation()  # world ← EE
            cart_vel_world = R_world_ee @ cart_vel_cmd
            ang_vel_world = R_world_ee @ ang_vel_cmd
        else:
            cart_vel_world = cart_vel_cmd
            ang_vel_world = ang_vel_cmd

        # Stack into 6D twist [linear; angular]
        twist_world = np.concatenate([cart_vel_world, ang_vel_world])

        # Get full Jacobian at EE site (each 3 x nv)
        J_pos, J_rot = self._get_jacobian()
        J_full = np.vstack([J_pos, J_rot])  # (6, nv)

        # Extract columns for arm joints by their actual qvel (dof) addresses.
        # This is robust to joint ordering changes (e.g. cube free joint first).
        J_arm = J_full[:, self._arm_qvel_addrs]  # (6, 6)

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

        # ---- Proportional velocity scaling (preserves EE twist direction) ----
        # Scale ALL joints by the same factor if any single joint exceeds its
        # velocity limit.  Per-joint clipping would distort the Jacobian's
        # twist direction and cause unwanted EE rotation during translation.
        # MoveIt Servo handles this proportionally too.
        max_v = np.array(self.max_joint_velocities)
        ratios = np.abs(joint_vel) / max_v
        max_ratio = np.max(ratios)
        if max_ratio > 1.0:
            joint_vel /= max_ratio  # scale down proportionally

        # Normalise to [-1, 1] for RoboSuite JOINT_VELOCITY controller.
        # output_max must match max_joint_velocities for the round-trip:
        #   Jacobian rad/s → /max_v → [-1,1] → *output_max → rad/s
        joint_vel /= max_v

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
        # Get current joint positions using proper qpos addresses
        qpos = self.data.qpos[self._arm_qpos_addrs]

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
        """Main control loop tick (20 Hz).

        Pipeline: snapshot twist → Jacobian IK → env.step() → publish → render.
        The twist is consumed (zeroed) each tick so the arm stops when no new
        commands arrive — this matches MoveIt Servo's "command_in_type: unitless"
        behavior where input must be continuously refreshed.
        """
        # 1. Atomically snapshot and clear the latest twist command
        with self._lock:
            twist_lin = self._twist_lin.copy()
            twist_ang = self._twist_ang.copy()
            twist_frame = self._twist_frame
            gripper = self._gripper_state
            self._twist_lin[:] = 0.0
            self._twist_ang[:] = 0.0

        # 2. Jacobian IK: Cartesian twist → normalized joint velocities
        if np.any(twist_lin != 0) or np.any(twist_ang != 0):
            joint_vel = self._twist_to_joint_velocities(twist_lin, twist_ang, twist_frame)
            joint_vel = np.clip(joint_vel, -1.0, 1.0)  # RoboSuite expects [-1,1]
        else:
            joint_vel = np.zeros(self.num_arm_joints)

        # 3. Assemble action: [6 joint vels, 1 gripper] → 7-dim
        action = np.zeros(self.action_dim, dtype=np.float64)
        action[:self.num_arm_joints] = joint_vel
        action[-1] = gripper  # -1 open, +1 closed

        # 4. Publish *pre-step* (obs_t, action_t) for demonstration recording.
        #    This avoids recording a lagged mapping like (obs_{t+1}, action_{t-1}),
        #    which can hurt BC training quality.
        obs_t_flat = process_obs(self._obs_raw)
        action_msg = Float64MultiArray()
        action_msg.data = action.tolist()
        self.action_pub.publish(action_msg)

        obs_msg = Float64MultiArray()
        obs_msg.data = obs_t_flat.tolist()
        self.obs_pub.publish(obs_msg)

        # 5. Step the MuJoCo simulation to get obs_{t+1}
        self._obs_raw, reward, done, info = self.env.step(action)

        # 6. Render the viewer (if enabled)
        if self.render_enabled:
            self.env.render()

        # 7. Diagnostic: log EE position when arm is moving
        if np.any(joint_vel != 0):
            eef_pos = self._obs_raw.get('robot0_eef_pos', None)
            if eef_pos is not None:
                self.get_logger().info(
                    f"EE: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}] | "
                    f"frame: {twist_frame} | "
                    f"twist_lin: [{twist_lin[0]:.2f}, {twist_lin[1]:.2f}, {twist_lin[2]:.2f}] | "
                    f"jvel: [{joint_vel[0]:.3f}, {joint_vel[1]:.3f}, {joint_vel[2]:.3f}, "
                    f"{joint_vel[3]:.3f}, {joint_vel[4]:.3f}, {joint_vel[5]:.3f}]",
                    throttle_duration_sec=0.5,
                )

        # 8. Auto-reset when episode terminates (normally unreachable with
        #    ignore_done=True + long horizon, but kept as a safety net)
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
        """Randomize cube XY position by up to ±10 cm for domain randomization.

        Matches the randomization range used in train_lift_v2.py so that
        human demonstrations cover a similar distribution to the RL training.
        Silently no-ops if the cube body name doesn't exist.
        """
        try:
            cube_id = self.env.sim.model.body_name2id('cube_main')
            pos = self.env.sim.model.body_pos[cube_id].copy()
            pos[0] += np.random.uniform(-0.1, 0.1)  # ±10 cm in X
            pos[1] += np.random.uniform(-0.1, 0.1)  # ±10 cm in Y
            self.env.sim.model.body_pos[cube_id] = pos
            self.env.sim.forward()  # Recompute derived quantities
        except Exception:
            pass

    def destroy_node(self):
        self.env.close()
        super().destroy_node()


# ============================================================================
# Main
# ============================================================================

def main():
    """Entry point — parse CLI args, create the bridge node, and spin."""
    parser = argparse.ArgumentParser(description="MuJoCo Jacobian-IK Bridge Node")
    parser.add_argument('--no-render', action='store_true', help='Run headless')
    parser.add_argument('--domain-rand', action='store_true', help='Randomize cube position')
    parser.add_argument('--linear-speed', type=float, default=0.5,
                        help='Max linear EE speed in m/s (default: 0.5, matches Servo)')
    parser.add_argument('--angular-speed', type=float, default=1.2,
                        help='Max angular EE speed in rad/s (default: 1.2, matches Servo)')
    parser.add_argument('--input-mode', choices=['twist', 'joy'], default='twist',
                        help='twist = subscribe to pre-processed TwistStamped (drives '
                             'both RViz + MuJoCo from same input via MoveIt Servo), '
                             'joy = subscribe to /joy directly (supports Pro & Cyborg)')
    parser.add_argument('--controller', type=str, default='pro',
                        help='Controller profile: pro | cyborg (default: pro)')
    args = parser.parse_args()

    # Both modes need the controller profile: joy mode for full input mapping,
    # twist mode for the gripper toggle button.  None = auto-detect.
    controller_profile = get_controller(args.controller)

    rclpy.init()
    node = MujocoBridgeNode(
        render=not args.no_render,
        domain_rand=args.domain_rand,
        linear_speed=args.linear_speed,
        angular_speed=args.angular_speed,
        input_mode=args.input_mode,
        controller_profile=controller_profile,
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
