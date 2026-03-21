#!/usr/bin/env python3
"""
DAgger Data Collection — Policy Rollout with Expert Override

Runs a trained BC policy in MuJoCo (same environment as cartesian_control_ros.py)
and allows the human expert to override via joystick at any time.  Records the
resulting (obs, action) trajectories to HDF5 for DAgger re-training.

DAgger workflow:
    1. Record initial demos:     demo_recorder.py → demos/initial.hdf5
    2. Train BC:                 bc_train.py demos/*.hdf5
    3. Collect corrections:      THIS SCRIPT → demos/dagger_round1.hdf5
    4. Retrain on ALL data:      bc_train.py demos/*.hdf5
    5. Repeat 3-4 until good

Expert override:
    Hold the override button (Pro Controller: RB/R, Cyborg: trigger) to take
    manual control.  Release to let the policy resume.  The recorded action is
    always the one actually sent to the environment (expert when overriding,
    policy otherwise).  A boolean 'is_expert' flag is stored per timestep
    so the trainer can optionally weight expert vs. policy data.

Usage:
    # Default (auto-detect controller, render MuJoCo viewer):
    python3 dagger_collect.py --policy models/bc_best.pt

    # Custom output:
    python3 dagger_collect.py --policy models/bc_best.pt -o demos/dagger_round2.hdf5

    # No rendering (headless):
    python3 dagger_collect.py --policy models/bc_best.pt --no-render
"""

import os
import sys
import time
import argparse
import threading
import numpy as np
from datetime import datetime

# ---- ROS2 ----
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Joy

# ---- Controller profiles ----
from controller_config import (
    ControllerProfile, PRO_CONTROLLER, CYBORG_STICK,
    detect_controller, get_controller,
)

# ---- RoboSuite / MuJoCo path bootstrap ----
ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite")
if os.path.exists(ROBO_PATH) and ROBO_PATH not in sys.path:
    sys.path.insert(0, ROBO_PATH)

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
import mujoco

# ---- PyTorch ----
import torch
import torch.nn as nn

# ---- HDF5 ----
try:
    import h5py
except ImportError:
    print("ERROR: h5py required — pip install h5py")
    sys.exit(1)

# Reuse obs processing + BC model from sibling modules
from cartesian_control_ros import OBS_KEYS, NUM_PHASE_DIMS, compute_phase, process_obs
from bc_train import BCPolicy


# ============================================================================
# Episode buffer for DAgger data
# ============================================================================

class DAggerEpisodeBuffer:
    """Collect timesteps with an is_expert flag."""

    def __init__(self):
        self.obs_list = []
        self.action_list = []
        self.reward_list = []
        self.done_list = []
        self.phase_list = []
        self.is_expert_list = []  # True when human overrode the policy

    def add(self, obs, action, reward=0.0, done=False, phase=0, is_expert=False):
        self.obs_list.append(obs.copy())
        self.action_list.append(action.copy())
        self.reward_list.append(reward)
        self.done_list.append(done)
        self.phase_list.append(phase)
        self.is_expert_list.append(is_expert)

    def __len__(self):
        return len(self.obs_list)

    @property
    def is_empty(self):
        return len(self) == 0

    def to_arrays(self):
        return {
            'obs': np.array(self.obs_list, dtype=np.float32),
            'actions': np.array(self.action_list, dtype=np.float32),
            'rewards': np.array(self.reward_list, dtype=np.float32),
            'dones': np.array(self.done_list, dtype=bool),
            'phases': np.array(self.phase_list, dtype=np.int32),
            'is_expert': np.array(self.is_expert_list, dtype=bool),
        }


# ============================================================================
# DAgger Collector Node
# ============================================================================

class DAggerCollectorNode(Node):
    """
    Runs a trained BC policy in MuJoCo with expert override from /joy.

    - By default, the policy predicts actions from observations.
    - When the expert holds the override button, Jacobian IK from joystick
      commands is used instead (same as cartesian_control_ros.py).
    - All (obs, action) pairs are recorded, with an is_expert flag.
    """

    JOY_TOPIC = "/joy"
    OBS_TOPIC = "/mujoco/observations"
    ACTION_TOPIC = "/mujoco/actions"

    def __init__(self, policy_path: str, output_path: str,
                 render: bool = True, controller_profile=None):
        super().__init__('dagger_collector')

        self.output_path = output_path
        self.render_enabled = render

        # ---- Load trained BC policy ----
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        ckpt = torch.load(policy_path, map_location=device, weights_only=True)
        self.obs_dim = ckpt['obs_dim']
        self.action_dim_model = ckpt['action_dim']
        self.policy = BCPolicy(self.obs_dim, self.action_dim_model).to(device)
        self.policy.load_state_dict(ckpt['model'])
        self.policy.eval()
        self.get_logger().info(
            f"Loaded BC policy from {policy_path} "
            f"(obs={self.obs_dim}, act={self.action_dim_model}, "
            f"epoch={ckpt.get('epoch', '?')}, loss={ckpt.get('loss', '?'):.6f})"
        )

        # ---- RoboSuite environment (same config as cartesian_control_ros.py) ----
        arm_ctrl = suite.load_part_controller_config(default_controller="JOINT_VELOCITY")
        arm_ctrl["output_max"] = [1.0, 1.0, 0.8, 0.8, 0.8, 1.0]
        arm_ctrl["output_min"] = [-1.0, -1.0, -0.8, -0.8, -0.8, -1.0]
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

        self.env_action_dim = self.env.action_dim  # 7 (6 arm + 1 gripper)

        # ---- MuJoCo references (for Jacobian IK during expert override) ----
        self.robot = self.env.robots[0]
        self.sim = self.env.sim
        self.model = self.sim.model
        self.data = self.sim.data

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

        # Arm joint addressing (mirrors cartesian_control_ros.py)
        self.arm_joint_names = [
            "robot0_shoulder_joint", "robot0_link_1_joint",
            "robot0_link1_link2", "robot0_a4_rotation",
            "robot0_a5_rotation", "robot0_a6_rotation",
        ]
        self.num_arm_joints = 6
        self._arm_qvel_addrs = np.array([
            self.sim.model.get_joint_qvel_addr(n) for n in self.arm_joint_names
        ])

        # IK params (same as cartesian_control_ros.py)
        self.linear_speed_scale = 0.5
        self.angular_speed_scale = 1.2
        self.damping = 0.01
        self.max_joint_velocities = np.array([1.0, 1.0, 0.8, 0.8, 0.8, 1.0])

        # ---- Expert override state ----
        self._controller_profile = controller_profile
        self._controller_detected = controller_profile is not None
        self._expert_active = False  # True when override button is held
        self._prev_gripper_btn = False
        self._gripper_state = -1.0  # -1 open, +1 closed
        self._twist_lin = np.zeros(3)
        self._twist_ang = np.zeros(3)
        self._lock = threading.Lock()

        # ---- Recording state ----
        self._current_episode = DAggerEpisodeBuffer()
        self._saved_episodes = []
        self._recording = True
        self._step_count = 0

        # ---- ROS2 subscriptions ----
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.joy_sub = self.create_subscription(Joy, self.JOY_TOPIC, self._joy_cb, 10)
        self.obs_pub = self.create_publisher(Float64MultiArray, self.OBS_TOPIC, qos)
        self.action_pub = self.create_publisher(Float64MultiArray, self.ACTION_TOPIC, qos)

        # ---- Control loop ----
        self._timer = self.create_timer(1.0 / 20.0, self._step_tick)

        # ---- Initial reset ----
        self._obs_raw = self.env.reset()
        if render:
            self.env.render()

        self.get_logger().info("DAgger collector ONLINE")
        self.get_logger().info("  Hold override button (RB/R) → expert control")
        self.get_logger().info("  Release → policy control")
        self.get_logger().info("  Ctrl+C → save & exit")

        # Input thread for episode management (Enter = save, d = discard)
        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._input_thread.start()

    # ------------------------------------------------------------------
    # Joy callback
    # ------------------------------------------------------------------

    def _joy_cb(self, msg: Joy):
        if not self._controller_detected:
            self._controller_profile = detect_controller(
                num_buttons=len(msg.buttons), num_axes=len(msg.axes))
            self._controller_detected = True
            self.get_logger().info(
                f"Auto-detected: {self._controller_profile.name}")

        cfg = self._controller_profile

        def btn(idx):
            return 0 <= idx < len(msg.buttons) and bool(msg.buttons[idx])

        def axis(idx):
            if idx < 0 or idx >= len(msg.axes):
                return 0.0
            v = float(msg.axes[idx])
            return v if abs(v) >= cfg.axis_deadzone else 0.0

        # ---- Override button (RB / R trigger) ----
        # Use the same button as gripper toggle for simplicity,
        # or a dedicated button. Let's use btn_cart_pos_z as override indicator:
        # Expert is active if ANY directional button/axis is non-zero
        lx, ly, lz = 0.0, 0.0, 0.0
        speed = cfg.cart_button_speed
        if btn(cfg.btn_cart_pos_x): lx += speed
        if btn(cfg.btn_cart_neg_x): lx -= speed
        if btn(cfg.btn_cart_pos_y): ly += speed
        if btn(cfg.btn_cart_neg_y): ly -= speed
        if btn(cfg.btn_cart_pos_z): lz += speed
        if btn(cfg.btn_cart_neg_z): lz -= speed

        if cfg.axis_cart_x >= 0:
            v = axis(cfg.axis_cart_x) * cfg.cart_axis_speed
            lx += (-v if cfg.invert_cart_x else v)
        if cfg.axis_cart_y >= 0:
            v = axis(cfg.axis_cart_y) * cfg.cart_axis_speed
            ly += (-v if cfg.invert_cart_y else v)
        if cfg.axis_cart_z >= 0:
            v = axis(cfg.axis_cart_z) * cfg.cart_axis_speed
            lz += (-v if cfg.invert_cart_z else v)

        rot_speed = cfg.rot_stick_speed
        ax_roll  = axis(cfg.axis_roll)  * rot_speed * (-1 if cfg.invert_roll  else 1)
        ax_pitch = axis(cfg.axis_pitch) * rot_speed * (-1 if cfg.invert_pitch else 1)
        ax_yaw   = axis(cfg.axis_yaw)   * rot_speed * (-1 if cfg.invert_yaw   else 1)

        # Expert is active when ANY input is non-zero
        any_input = (abs(lx) + abs(ly) + abs(lz) +
                     abs(ax_roll) + abs(ax_pitch) + abs(ax_yaw)) > 0.001

        # Gripper toggle
        gripper_btn = btn(cfg.btn_gripper_toggle)
        if 0 <= cfg.axis_gripper_toggle < len(msg.axes):
            gripper_btn = gripper_btn or (
                float(msg.axes[cfg.axis_gripper_toggle]) < cfg.gripper_axis_pressed_threshold
            )
        if gripper_btn and not self._prev_gripper_btn:
            with self._lock:
                self._gripper_state = -self._gripper_state
        self._prev_gripper_btn = gripper_btn

        with self._lock:
            self._expert_active = any_input
            self._twist_lin[:] = [lx, ly, lz]
            self._twist_ang[:] = [ax_roll, ax_pitch, ax_yaw]

    # ------------------------------------------------------------------
    # Jacobian IK for expert override
    # ------------------------------------------------------------------

    def _expert_jacobian_ik(self, twist_lin, twist_ang):
        """Same Jacobian IK as cartesian_control_ros.py (with EE rotation)."""
        cart_vel = twist_lin * self.linear_speed_scale
        ang_vel = twist_ang * self.angular_speed_scale

        # Rotate EE → world
        R = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        cart_vel = R @ cart_vel
        ang_vel = R @ ang_vel

        twist = np.concatenate([cart_vel, ang_vel])

        # Jacobian
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model._model, self.data._data,
                          jacp, jacr, self.ee_site_id)
        J = np.vstack([jacp, jacr])
        J_arm = J[:, self._arm_qvel_addrs]

        # Damped least-squares
        JJT = J_arm @ J_arm.T + self.damping**2 * np.eye(6)
        try:
            joint_vel = J_arm.T @ np.linalg.solve(JJT, twist)
        except np.linalg.LinAlgError:
            joint_vel = np.linalg.pinv(J_arm) @ twist

        # Proportional scaling
        max_v = self.max_joint_velocities
        ratios = np.abs(joint_vel) / max_v
        max_ratio = np.max(ratios)
        if max_ratio > 1.0:
            joint_vel /= max_ratio
        joint_vel /= max_v

        return np.clip(joint_vel, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Step loop
    # ------------------------------------------------------------------

    def _step_tick(self):
        with self._lock:
            expert_active = self._expert_active
            twist_lin = self._twist_lin.copy()
            twist_ang = self._twist_ang.copy()
            gripper = self._gripper_state
            self._twist_lin[:] = 0.0
            self._twist_ang[:] = 0.0

        # Get current obs
        obs_flat = process_obs(self._obs_raw)

        if expert_active and (np.any(twist_lin != 0) or np.any(twist_ang != 0)):
            # ---- Expert override: Jacobian IK from joystick ----
            joint_vel = self._expert_jacobian_ik(twist_lin, twist_ang)
            is_expert = True
        else:
            # ---- Policy action ----
            with torch.no_grad():
                obs_t = torch.tensor(obs_flat, dtype=torch.float32,
                                     device=self.device).unsqueeze(0)
                action_t = self.policy(obs_t)
                policy_action = action_t.cpu().numpy().flatten()

            # policy_action has action_dim_model dims; we need env_action_dim
            joint_vel = policy_action[:self.num_arm_joints]
            joint_vel = np.clip(joint_vel, -1.0, 1.0)
            # If policy predicts gripper too, use it; else keep current
            if len(policy_action) > self.num_arm_joints:
                gripper = np.clip(policy_action[self.num_arm_joints], -1.0, 1.0)
            is_expert = False

        # Assemble action
        action = np.zeros(self.env_action_dim, dtype=np.float64)
        action[:self.num_arm_joints] = joint_vel
        action[-1] = gripper

        # Determine phase
        phase_onehot = obs_flat[-4:]
        phase = int(np.argmax(phase_onehot))

        # Record timestep
        self._current_episode.add(
            obs=obs_flat, action=action.astype(np.float32),
            reward=0.0, done=False, phase=phase, is_expert=is_expert,
        )
        self._step_count += 1

        # Step environment
        self._obs_raw, reward, done, info = self.env.step(action)

        # Publish for monitoring
        obs_msg = Float64MultiArray()
        obs_msg.data = obs_flat.tolist()
        self.obs_pub.publish(obs_msg)
        action_msg = Float64MultiArray()
        action_msg.data = action.tolist()
        self.action_pub.publish(action_msg)

        if self.render_enabled:
            self.env.render()

        # Progress logging
        if self._step_count % 20 == 0:
            n_saved = len(self._saved_episodes)
            n_cur = len(self._current_episode)
            mode = "EXPERT" if is_expert else "POLICY"
            phase_names = ['Reach', 'Grasp', 'Lift', 'Hold']
            p = phase_names[phase] if phase < len(phase_names) else '???'
            print(f"\r  [{mode:6s}] Saved: {n_saved} | "
                  f"Current: {n_cur} steps | Phase: {p}     ", end='', flush=True)

        if done:
            self._obs_raw = self.env.reset()

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def _finalize_episode(self, discard=False):
        if self._current_episode.is_empty:
            return
        if discard:
            n = len(self._current_episode)
            self._current_episode = DAggerEpisodeBuffer()
            print(f"\n  DISCARDED ({n} steps)")
            return

        arrays = self._current_episode.to_arrays()
        arrays['dones'][-1] = True
        self._saved_episodes.append(arrays)
        n = len(arrays['obs'])
        expert_pct = 100 * arrays['is_expert'].sum() / max(n, 1)
        total = len(self._saved_episodes)
        print(f"\n  SAVED episode #{total} ({n} steps, {expert_pct:.0f}% expert)")
        self._current_episode = DAggerEpisodeBuffer()

    def _input_loop(self):
        import termios, tty, select
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while self._recording:
                if select.select([sys.stdin], [], [], 0.2)[0]:
                    ch = sys.stdin.read(1)
                    if ch in ('\n', '\r'):
                        self._finalize_episode(discard=False)
                    elif ch.lower() == 'd':
                        self._finalize_episode(discard=True)
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    # ------------------------------------------------------------------
    # HDF5 saving
    # ------------------------------------------------------------------

    def save_hdf5(self):
        if not self._current_episode.is_empty:
            self._finalize_episode(discard=False)

        if not self._saved_episodes:
            print("\n  No data collected. Nothing to save.")
            return

        total = len(self._saved_episodes)
        print(f"\n  Writing {total} episodes to {self.output_path} ...")

        with h5py.File(self.output_path, 'w') as f:
            data_grp = f.create_group('data')
            for i, ep in enumerate(self._saved_episodes):
                demo_grp = data_grp.create_group(f'demo_{i}')
                for key, arr in ep.items():
                    demo_grp.create_dataset(key, data=arr, compression='gzip')

            meta = f.create_group('metadata')
            sample = self._saved_episodes[0]
            meta.attrs['obs_dim'] = sample['obs'].shape[1]
            meta.attrs['action_dim'] = sample['actions'].shape[1]
            meta.attrs['control_freq'] = 20
            meta.attrs['total_demos'] = total
            meta.attrs['env_name'] = 'Lift'
            meta.attrs['robot'] = 'Rover2026'
            meta.attrs['collection_mode'] = 'dagger'
            meta.attrs['recorded_at'] = datetime.now().isoformat()

        total_steps = sum(len(ep['obs']) for ep in self._saved_episodes)
        expert_steps = sum(ep['is_expert'].sum() for ep in self._saved_episodes)
        print(f"  Done! {total} episodes, {total_steps} steps "
              f"({int(expert_steps)} expert, {total_steps - int(expert_steps)} policy)")

    def destroy_node(self):
        self._recording = False
        self.env.close()
        super().destroy_node()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DAgger collection: policy + expert override")
    parser.add_argument('--policy', type=str, required=True,
                        help='Path to trained BC model (.pt)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output HDF5 path (default: demos/dagger_TIMESTAMP.hdf5)')
    parser.add_argument('--no-render', action='store_true')
    parser.add_argument('--controller', type=str, default='pro',
                        help='Controller profile: pro | cyborg')
    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        demo_dir = os.path.join(ROOT, 'demos')
        os.makedirs(demo_dir, exist_ok=True)
        args.output = os.path.join(demo_dir, f'dagger_{ts}.hdf5')

    controller = get_controller(args.controller)

    rclpy.init()
    node = DAggerCollectorNode(
        policy_path=args.policy,
        output_path=args.output,
        render=not args.no_render,
        controller_profile=controller,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_hdf5()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
