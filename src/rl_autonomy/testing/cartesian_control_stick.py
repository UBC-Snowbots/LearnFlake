#!/usr/bin/env python3
"""
Cartesian End-Effector Control using Jacobian IK

This script provides Cartesian control of the arm's end-effector:
- Arrow keys: Move EE in X/Y plane (horizontal) without changing Z
- ;/.       : Move EE up/down (Z) without changing X/Y
- U/J       : Rotate wrist (optional)

The Jacobian is computed from MuJoCo to convert Cartesian velocities
to joint velocities, which are sent to the JOINT_VELOCITY controller.
"""

import os
import sys
import time
import numpy as np
import termios
import tty
import select

# Path setup
ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite")
sys.path.insert(0, ROBO_PATH)

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.wrappers import VisualizationWrapper


class CartesianArmControl:
    """Cartesian end-effector control using Jacobian-based IK."""
    
    def __init__(self):
        # Control parameters
        self.linear_speed = 0.3    # m/s for XYZ movement
        self.angular_speed = 0.5   # rad/s for rotation
        self.damping = 0.01
        
        # Create environment with JOINT_VELOCITY controller
        arm_controller_config = suite.load_part_controller_config(default_controller="JOINT_VELOCITY")
        controller_config = refactor_composite_controller_config(arm_controller_config, "Rover2026", ["right"])
        
        self.env = suite.make(
            env_name="Lift",
            robots=["Rover2026"],
            controller_configs=controller_config,
            has_renderer=True,
            has_offscreen_renderer=False,
            render_camera="agentview",
            ignore_done=True,
            use_camera_obs=False,
            control_freq=20,
        )
        self.env = VisualizationWrapper(self.env, indicator_configs=None)
        
        self.robot = self.env.robots[0]
        self.sim = self.env.sim
        self.model = self.sim.model
        self.data = self.sim.data
        
        # Get end-effector site ID
        self.ee_site_name = "gripper0_grip_site"  # RoboSuite naming
        try:
            self.ee_site_id = self.model.site_name2id(self.ee_site_name)
        except:
            # Fallback - find a grip site
            for i in range(self.model.nsite):
                name = self.model.site_id2name(i)
                if "grip" in name.lower():
                    self.ee_site_name = name
                    self.ee_site_id = i
                    break
            else:
                raise ValueError("Could not find end-effector site!")
        
        print(f"Using EE site: {self.ee_site_name} (id={self.ee_site_id})")
        
        # Get joint IDs for the arm (not gripper)
        self.arm_joint_names = [
            "robot0_shoulder_joint",
            "robot0_link_1_joint", 
            "robot0_link1_link2",
            "robot0_a4_rotation",
            "robot0_a5_rotation",
            "robot0_a6_rotation",
        ]
        self.arm_joint_ids = [self.model.joint_name2id(name) for name in self.arm_joint_names]
        self.num_joints = len(self.arm_joint_ids)
        
        self.print_controls()
    
    def print_controls(self):
        print("\n" + "="*60)
        print("CARTESIAN END-EFFECTOR CONTROL")
        print("="*60)
        print("Movement (press and hold):")
        print("  ↑/↓  - Move forward/backward (X)")
        print("  ←/→  - Move left/right (Y)")
        print("  ;/.  - Move up/down (Z)")
        print("")
        print("Rotation:")
        print("  U/J  - Rotate wrist +/-")
        print("")
        print("Other:")
        print("  R    - Reset environment")
        print("  Esc  - Quit")
        print("="*60 + "\n")
    
    def get_jacobian(self):
        """
        Get the Jacobian matrix for the end-effector.
        Maps joint velocities to end-effector Cartesian velocities.
        Returns: (3, num_joints) position Jacobian
        """
        # MuJoCo Jacobian computation
        jacp = np.zeros((3, self.model.nv))  # Position Jacobian
        jacr = np.zeros((3, self.model.nv))  # Rotation Jacobian
        
        # Get Jacobian at the EE site
        import mujoco
        mujoco.mj_jacSite(self.model._model, self.data._data, jacp, jacr, self.ee_site_id)
        
        # Extract columns for our arm joints only
        # Joint IDs map to velocity indices (assuming no floating base)
        arm_jac_p = jacp[:, :self.num_joints]
        arm_jac_r = jacr[:, :self.num_joints]
        
        return arm_jac_p, arm_jac_r
    
    # Navigate cartesian to joint_velocities through Rowan's moteus controller
    # (this is a temporary function)
    def cartesian_to_joint_velocities(self, cart_vel):
        """
        Convert Cartesian velocity (dx, dy, dz) to joint velocities using Jacobian pseudoinverse.
        
        Args:
            cart_vel: (3,) array of [vx, vy, vz] desired EE velocity
            
        Returns:
            (num_joints,) array of joint velocities
        """
        J_pos, J_rot = self.get_jacobian()
        
        # Use damped least squares (Levenberg-Marquardt) for stability
        # joint_vel = J^T * (J * J^T + lambda^2 * I)^(-1) * cart_vel
        JJT = J_pos @ J_pos.T
        damped = JJT + self.damping**2 * np.eye(3)
        
        try:
            joint_vel = J_pos.T @ np.linalg.solve(damped, cart_vel)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse
            joint_vel = np.linalg.pinv(J_pos) @ cart_vel
        
        return joint_vel
    
    def get_key(self, timeout=0.02):
        """Get a single keypress with timeout."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if rlist:
                key = sys.stdin.read(1)
                # Handle arrow keys (escape sequences)
                if key == '\x1b':
                    # Read the next two characters
                    sys.stdin.read(1)  # '['
                    arrow = sys.stdin.read(1)
                    return f"ARROW_{arrow}"
            else:
                key = ''
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key
    
    def run(self):
        """Main control loop."""
        obs = self.env.reset()
        self.env.render()
        
        print("Ready! Use arrow keys XY, ;/. for Z\n")
        
        # Velocity state (persists between frames for smooth control)
        cart_vel = np.zeros(3)  # [vx, vy, vz]
        
        try:
            while True:
                key = self.get_key()
                
                # Reset velocities each frame (direct control, not momentum)
                cart_vel = np.zeros(3)
                
                if key == '\x1b':  # esc for quit
                    print("\nExiting...")
                    break
                
                # Arrow keys
                elif key == 'ARROW_A':  # Up arrow
                    cart_vel[0] = self.linear_speed  # Forward (X+)
                elif key == 'ARROW_B':  # Down arrow
                    cart_vel[0] = -self.linear_speed  # Backward (X-)
                elif key == 'ARROW_D':  # Left arrow
                    cart_vel[1] = self.linear_speed  # Left (Y+)
                elif key == 'ARROW_C':  # Right arrow
                    cart_vel[1] = -self.linear_speed  # Right (Y-)
                
                # Vertical movement
                elif key == ';':
                    cart_vel[2] = self.linear_speed  # Up (Z+)
                elif key == '.':
                    cart_vel[2] = -self.linear_speed  # Down (Z-)
                
                # Reset
                elif key.lower() == 'r':
                    print("Resetting...")
                    obs = self.env.reset()
                    cart_vel = np.zeros(3)
                
                # Convert Cartesian velocity to joint velocities
                if np.any(cart_vel != 0):
                    joint_vel = self.cartesian_to_joint_velocities(cart_vel)
                    # Clamp joint velocities
                    joint_vel = np.clip(joint_vel, -1.0, 1.0)
                else:
                    joint_vel = np.zeros(self.num_joints)
                
                # Build action: [joint_vels] (no gripper — stick has 0 DOF)
                action = np.zeros(self.env.action_dim)
                action[:self.num_joints] = joint_vel
                
                # Step simulation
                obs, _, _, _ = self.env.step(action)
                
                # Print EE position occasionally
                eef_pos = obs.get('robot0_eef_pos', None)
                if eef_pos is not None and np.any(cart_vel != 0):
                    print(f"EE: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}] | "
                          f"Cmd: [{cart_vel[0]:.2f}, {cart_vel[1]:.2f}, {cart_vel[2]:.2f}]")
                
                self.env.render()
                
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self.env.close()


def main():
    controller = CartesianArmControl()
    controller.run()


if __name__ == '__main__':
    main()
