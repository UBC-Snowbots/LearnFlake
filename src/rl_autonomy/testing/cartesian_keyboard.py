#!/usr/bin/env python3
"""
Cartesian Keyboard Control

Uses the "Type" environment (60-key QWERTY keyboard on a table) with
Jacobian-based Cartesian end-effector control.  Press real keys to move
the stick clicker over the matching simulated key and press it.

Controls:
  W/S   → Move forward/backward  (X)
  A/D   → Move left/right        (Y)
  ;/.   → Move up/down            (Z)
  U/J   → Rotate wrist +/−
  R     → Reset
  Esc   → Quit

Contact detection prints which simulated key the stick tip is pressing.
"""

import os
import sys
import numpy as np
import termios
import tty
import select

# ── Path setup ──────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite")
sys.path.insert(0, ROBO_PATH)

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.wrappers import VisualizationWrapper


class CartesianKeyboardControl:
    """Cartesian stick control over the Type (keyboard) environment."""

    def __init__(self):
        # Control parameters
        self.linear_speed = 0.3     # m/s for XYZ movement
        self.angular_speed = 0.5    # rad/s for wrist rotation
        self.damping = 0.01         # Damping for Jacobian pseudoinverse

        # Build JOINT_VELOCITY controller config for Rover2026
        arm_controller_config = suite.load_part_controller_config(
            default_controller="JOINT_VELOCITY"
        )
        controller_config = refactor_composite_controller_config(
            arm_controller_config, "Rover2026", ["right"]
        )

        # Create the Type environment (keyboard on table)
        self.env = suite.make(
            env_name="Type",
            robots=["Rover2026"],
            controller_configs=controller_config,
            has_renderer=True,
            has_offscreen_renderer=False,
            render_camera="agentview",
            ignore_done=True,
            use_camera_obs=False,
            use_object_obs=True,
            control_freq=20,
        )
        self.env = VisualizationWrapper(self.env, indicator_configs=None)

        self.robot = self.env.robots[0]
        self.sim = self.env.sim
        self.model = self.sim.model
        self.data = self.sim.data

        # ── End-effector site ────────────────────────────────────────
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

        print(f"Using EE site: {self.ee_site_name} (id={self.ee_site_id})")

        # ── Arm joint names (same as cartesian_control_stick) ────────
        self.arm_joint_names = [
            "robot0_shoulder_joint",
            "robot0_link_1_joint",
            "robot0_link1_link2",
            "robot0_a4_rotation",
            "robot0_a5_rotation",
            "robot0_a6_rotation",
        ]
        self.arm_joint_ids = [
            self.model.joint_name2id(n) for n in self.arm_joint_names
        ]
        self.num_joints = len(self.arm_joint_ids)

        self.print_controls()

    # ─────────────────────────────────────────────────────────────────
    def print_controls(self):
        print("\n" + "=" * 60)
        print("  CARTESIAN KEYBOARD CONTROL  (Type env)")
        print("=" * 60)
        print("  W/S   → X forward / backward")
        print("  A/D   → Y left / right")
        print("  ;/.   → Z up / down")
        print("  U/J   → Wrist rotation +/−")
        print()
        print("  R     → Reset    Esc → Quit")
        print("=" * 60 + "\n")

    # ─────────────────────────────────────────────────────────────────
    def get_key(self, timeout=0.02):
        """Non-blocking single keypress reader."""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if rlist:
                ch = sys.stdin.read(1)
                if ch == '\x1b':
                    sys.stdin.read(1)   # '['
                    arrow = sys.stdin.read(1)
                    return f"ARROW_{arrow}"
                return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ''

    # ─────────────────────────────────────────────────────────────────
    def get_jacobian(self):
        """Position and rotation Jacobians for the EE site."""
        import mujoco
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model._model, self.data._data, jacp, jacr, self.ee_site_id
        )
        arm_jac_p = jacp[:, :self.num_joints]
        arm_jac_r = jacr[:, :self.num_joints]
        return arm_jac_p, arm_jac_r

    def cartesian_to_joint_velocities(self, cart_vel):
        """Damped-least-squares Jacobian IK."""
        J_pos, _ = self.get_jacobian()
        JJT = J_pos @ J_pos.T
        damped = JJT + self.damping ** 2 * np.eye(3)
        try:
            joint_vel = J_pos.T @ np.linalg.solve(damped, cart_vel)
        except np.linalg.LinAlgError:
            joint_vel = np.linalg.pinv(J_pos) @ cart_vel
        return joint_vel

    # ─────────────────────────────────────────────────────────────────
    def run(self):
        """Main control loop."""
        obs = self.env.reset()
        self.env.render()

        print("Ready! Move with WASD / ;.  |  Contacts printed below.\n")

        try:
            while True:
                key = self.get_key()

                cart_vel = np.zeros(3)

                if key == '\x1b':
                    print("\nExiting...")
                    break

                # WASD + arrows
                elif key.lower() == 'w' or key == 'ARROW_A':
                    cart_vel[0] = self.linear_speed
                elif key.lower() == 's' or key == 'ARROW_B':
                    cart_vel[0] = -self.linear_speed
                elif key.lower() == 'a' or key == 'ARROW_D':
                    cart_vel[1] = self.linear_speed
                elif key.lower() == 'd' or key == 'ARROW_C':
                    cart_vel[1] = -self.linear_speed

                # Z axis
                elif key == ';':
                    cart_vel[2] = self.linear_speed
                elif key == '.':
                    cart_vel[2] = -self.linear_speed

                # Reset
                elif key.lower() == 'r':
                    print("Resetting...")
                    obs = self.env.reset()
                    continue

                # Jacobian IK → joint velocities
                if np.any(cart_vel != 0):
                    joint_vel = self.cartesian_to_joint_velocities(cart_vel)
                    joint_vel = np.clip(joint_vel, -1.0, 1.0)
                else:
                    joint_vel = np.zeros(self.num_joints)

                # Action: [joint_vels]  (stick gripper has 0 DOF)
                action = np.zeros(self.env.action_dim)
                action[:self.num_joints] = joint_vel

                # Step
                obs, reward, done, info = self.env.step(action)

                # Print contacts from the Type environment
                pressed = info.get("pressed_keys", [])
                if pressed:
                    print(f"  ✓ Contact: {', '.join(pressed)}")

                # Print EE position when moving
                eef_pos = obs.get("robot0_eef_pos", None)
                if eef_pos is not None and np.any(cart_vel != 0):
                    print(
                        f"  EE: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, "
                        f"{eef_pos[2]:.3f}]"
                    )

                self.env.render()

        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self.env.close()


def main():
    controller = CartesianKeyboardControl()
    controller.run()


if __name__ == "__main__":
    main()
