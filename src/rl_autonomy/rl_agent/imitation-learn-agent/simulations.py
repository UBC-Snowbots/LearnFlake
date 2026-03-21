#!/usr/bin/env python3
"""
Cartesian End-Effector Control using Jacobian IK

This script provides Cartesian control of the arm's end-effector:
- Arrow keys: Move EE in X/Y plane (horizontal) without changing Z
- ;/.       : Move EE up/down (Z) without changing X/Y
- U/J       : Rotate wrist (optional)

The Jacobian is computed from MuJoCo to convert Cartesian velocities
to joint velocities, which are sent to the JOINT_VELOCITY controller.

This gives you OSC_POSE-like control using your joint velocity infrastructure.
"""

import os
import sys
import time
import numpy as np
import termios
import tty
import select
import socket

def _is_display_usable(display):
    """Return True if DISPLAY appears reachable from inside this container."""
    if not display:
        return False

    # Local unix socket display (e.g. :0, :1)
    if display.startswith(":"):
        disp = display[1:].split(".", 1)[0]
        if not disp.isdigit():
            return False
        return os.path.exists(f"/tmp/.X11-unix/X{disp}")

    # TCP display (e.g. host.docker.internal:0.0)
    if ":" not in display:
        return False
    host, rest = display.split(":", 1)
    screen = rest.split(".", 1)[0]
    if not screen.isdigit():
        return False
    port = 6000 + int(screen)
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _auto_configure_display():
    """Auto-configure DISPLAY conservatively for X11 forwarding."""
    try:
        if os.environ.get("AUTO_CONFIGURE_DISPLAY", "1") in {"0", "false", "False"}:
            print("Skipping DISPLAY auto-config (AUTO_CONFIGURE_DISPLAY disabled)")
            return

        cur = os.environ.get("DISPLAY", "")
        prefer_local_x11 = os.environ.get("PREFER_LOCAL_X11", "1") not in {"0", "false", "False"}

        # Keep existing DISPLAY only if it is usable.
        if cur and _is_display_usable(cur):
            print(f"DISPLAY already set and reachable: {cur}")
            return
        if cur:
            print(f"DISPLAY set but unreachable: {cur}; searching alternatives...")

        # If host.docker.internal is set but not reachable, optionally switch to local X11.
        if cur.startswith("host.docker.internal") and prefer_local_x11:
            if os.path.exists("/tmp/.X11-unix/X0"):
                os.environ["DISPLAY"] = ":0"
                print(f"DISPLAY was {cur}; switched to :0 (local X0 socket detected)")
            else:
                print(f"Keeping DISPLAY={cur} (no local X0 socket; using host X server)")
            return

        # Typical Linux / host-network Docker case.
        if os.path.exists("/tmp/.X11-unix"):
            for sock in ("X0", "X1"):
                if os.path.exists(f"/tmp/.X11-unix/{sock}"):
                    disp = sock[1:]
                    os.environ["DISPLAY"] = f":{disp}"
                    print(f"Auto-set DISPLAY=:{disp} (local /tmp/.X11-unix/{sock} detected)")
                    return

        # Docker Desktop fallback.
        candidate = "host.docker.internal:0.0"
        if _is_display_usable(candidate):
            os.environ["DISPLAY"] = candidate
            print(f"Auto-set DISPLAY={candidate} (reachable)")
            return

        # Final fallback: Docker gateway IP from resolv.conf.
        try:
            with open("/etc/resolv.conf") as f:
                for line in f:
                    if line.startswith("nameserver"):
                        ip = line.split()[1].strip()
                        if ip:
                            candidate = f"{ip}:0.0"
                            if _is_display_usable(candidate):
                                os.environ["DISPLAY"] = candidate
                                print(f"Auto-set DISPLAY={candidate} (from /etc/resolv.conf)")
                                return
        except Exception:
            pass

        # Nothing usable found.
        os.environ.pop("DISPLAY", None)
        print("No reachable X11 display found; unset DISPLAY.")
    except Exception as e:
        print(f"Warning: failed to auto-configure DISPLAY: {e}")


_auto_configure_display()
force_gui = os.environ.get("FORCE_GUI", "0") in {"1", "true", "True"}
force_headless = os.environ.get("SIM_HEADLESS", "0") in {"1", "true", "True"}
display_value = os.environ.get("DISPLAY", "")
display_ok = _is_display_usable(display_value)

# Default to GLFW for interactive use, but avoid import-time GLFW errors when DISPLAY is invalid.
requested_gl = os.environ.get("MUJOCO_GL", "glfw").lower()
if force_gui and requested_gl != "glfw":
    os.environ["MUJOCO_GL"] = "glfw"
    print(f"FORCE_GUI enabled: switched MUJOCO_GL from {requested_gl} to glfw.")
elif force_headless and requested_gl == "glfw":
    os.environ["MUJOCO_GL"] = "egl"
    print("SIM_HEADLESS enabled: switched MUJOCO_GL from glfw to egl.")
elif (not force_gui) and requested_gl == "glfw" and not display_ok:
    os.environ["MUJOCO_GL"] = "egl"
    print(f"DISPLAY {display_value!r} unreachable: switched MUJOCO_GL from glfw to egl.")

print(f"Rendering config: MUJOCO_GL={os.environ.get('MUJOCO_GL')} DISPLAY={os.environ.get('DISPLAY')}")

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
        self.damping = 0.01        # Damping for pseudoinverse
        
        # Create environment with JOINT_VELOCITY controller
        arm_controller_config = suite.load_part_controller_config(default_controller="JOINT_VELOCITY")
        controller_config = refactor_composite_controller_config(arm_controller_config, "Rover2026", ["right"])

        display = os.environ.get("DISPLAY", "")
        force_headless = os.environ.get("SIM_HEADLESS", "0") in {"1", "true", "True"}
        force_gui = os.environ.get("FORCE_GUI", "0") in {"1", "true", "True"}
        display_ok = _is_display_usable(display)
        self.has_renderer = force_gui or ((not force_headless) and display_ok)
        self.has_offscreen_renderer = not self.has_renderer

        gl_backend = os.environ.get("MUJOCO_GL", "").lower()
        if self.has_renderer and gl_backend != "glfw":
            # This script is explicitly interactive; force GLFW when display is usable.
            os.environ["MUJOCO_GL"] = "glfw"
            print(f"Switched MUJOCO_GL from {gl_backend or 'unset'} to glfw for on-screen rendering.")
        elif not self.has_renderer and gl_backend == "glfw":
            # GLFW needs an X display; use EGL automatically when headless.
            os.environ["MUJOCO_GL"] = "egl"
            print("Switched MUJOCO_GL from glfw to egl (headless fallback).")

        if self.has_renderer:
            print(f"On-screen rendering enabled (DISPLAY={display}).")
        else:
            print(
                "Display unavailable or headless requested. "
                f"Using headless rendering (MUJOCO_GL={os.environ.get('MUJOCO_GL')}, DISPLAY={display!r})."
            )
            if display.startswith(":"):
                disp = display[1:].split(".", 1)[0]
                if disp.isdigit():
                    xsock = f"/tmp/.X11-unix/X{disp}"
                    print(f"X11 socket check: {xsock} exists={os.path.exists(xsock)}")
            if not display_ok:
                print("Set FORCE_GUI=1 to bypass display probe if your X11 setup is valid.")
        
        self.env = suite.make(
            env_name="Lift",
            robots=["Rover2026"],
            controller_configs=controller_config,
            has_renderer=self.has_renderer,
            has_offscreen_renderer=self.has_offscreen_renderer,
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
        
        # Gripper state
        self.gripper_state = -1.0  # -1 = open, 1 = closed
        
        self.print_controls()
    
    def print_controls(self):
        print("\n" + "="*60)
        print("CARTESIAN END-EFFECTOR CONTROL")
        print("="*60)
        print("Movement (press and hold):")
        print("  ↑/↓ (W/S) - Move forward/backward (X)")
        print("  ←/→ (A/D) - Move left/right (Y)")
        print("  ;/.       - Move up/down (Z)")
        print("")
        print("Rotation:")
        print("  U/J       - Rotate wrist +/-")
        print("")
        print("Gripper:")
        print("  Space     - Toggle gripper")
        print("  O/P       - Open/Close gripper")
        print("")
        print("Other:")
        print("  R         - Reset environment")
        print("  Esc       - Quit")
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
        if self.has_renderer:
            self.env.render()
        
        print("Ready! Use arrow keys or WASD for XY, ;/. for Z\n")
        
        # Velocity state (persists between frames for smooth control)
        cart_vel = np.zeros(3)  # [vx, vy, vz]
        
        try:
            while True:
                key = self.get_key()
                
                # Reset velocities each frame (direct control, not momentum)
                cart_vel = np.zeros(3)
                
                if key == '\x1b':  # Plain escape (quit)
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
                
                # WASD alternative
                elif key.lower() == 'w':
                    cart_vel[0] = self.linear_speed  # Forward
                elif key.lower() == 's':
                    cart_vel[0] = -self.linear_speed  # Backward
                elif key.lower() == 'a':
                    cart_vel[1] = self.linear_speed  # Left
                elif key.lower() == 'd':
                    cart_vel[1] = -self.linear_speed  # Right
                
                # Vertical movement
                elif key == ';':
                    cart_vel[2] = self.linear_speed  # Up (Z+)
                elif key == '.':
                    cart_vel[2] = -self.linear_speed  # Down (Z-)
                
                # Gripper
                elif key == ' ':
                    self.gripper_state = 1.0 if self.gripper_state < 0 else -1.0
                    print(f"Gripper: {'CLOSED' if self.gripper_state > 0 else 'OPEN'}")
                elif key.lower() == 'o':
                    self.gripper_state = -1.0
                    print("Gripper: OPEN")
                elif key.lower() == 'p':
                    self.gripper_state = 1.0
                    print("Gripper: CLOSED")
                
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
                
                # Build action: [joint_vels..., gripper]
                action = np.zeros(self.env.action_dim)
                action[:self.num_joints] = joint_vel
                action[-1] = self.gripper_state
                
                # Step simulation
                obs, _, _, _ = self.env.step(action)
                
                # Print EE position occasionally
                eef_pos = obs.get('robot0_eef_pos', None)
                if eef_pos is not None and np.any(cart_vel != 0):
                    print(f"EE: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}] | "
                          f"Cmd: [{cart_vel[0]:.2f}, {cart_vel[1]:.2f}, {cart_vel[2]:.2f}]")
                
                if self.has_renderer:
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
