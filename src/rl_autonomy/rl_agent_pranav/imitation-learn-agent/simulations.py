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
import json
import argparse
import numpy as np
import termios
import tty
import select
import socket
from pathlib import Path
from datetime import datetime

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
        # Velocity DOF indices in MuJoCo's nv-length velocity vector.
        # jnt_dofadr[joint_id] gives the column index in the Jacobian for that joint.
        self.arm_vel_indices = [int(self.model._model.jnt_dofadr[jid]) for jid in self.arm_joint_ids]
        print(f"Arm joint DOF indices (nv cols): {self.arm_vel_indices}")

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
        print("  Q / Ctrl+C - Quit")
        print("  --duration / --max-steps for timed run")
        print("  --record to save per-step dataset")
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
        
        # Extract columns corresponding to the arm joints' velocity DOFs.
        # arm_vel_indices are the correct nv column positions from jnt_dofadr.
        arm_jac_p = jacp[:, self.arm_vel_indices]
        arm_jac_r = jacr[:, self.arm_vel_indices]
        
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
                if key == '\x03':  # Ctrl+C in raw mode → treat as KeyboardInterrupt
                    raise KeyboardInterrupt
                # Handle arrow keys (escape sequences)
                if key == '\x1b':
                    # Check if more chars follow (arrow key) or bare Escape
                    rlist2, _, _ = select.select([sys.stdin], [], [], 0.15)
                    if not rlist2:
                        return '\x1b'  # bare Escape, don't block
                    sys.stdin.read(1)  # '['
                    arrow = sys.stdin.read(1)
                    return f"ARROW_{arrow}"
            else:
                key = ''
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key

    @staticmethod
    def _default_output_path(dataset_format):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(ROOT) / "data" / "imitation"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"teleop_{stamp}.{dataset_format}"

    @staticmethod
    def _save_records(records, out_path, dataset_format):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save metadata sidecar
        meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
        meta = {
            "num_rows": len(records),
            "created_at": datetime.now().isoformat(),
            "format": dataset_format,
            "schema_hint": list(records[0].keys()) if records else [],
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        if dataset_format == "jsonl":
            with out_path.open("w") as f:
                for row in records:
                    f.write(json.dumps(row) + "\n")
            return str(out_path), str(meta_path)

        if dataset_format == "csv":
            # Flatten list-like fields to JSON strings for CSV compatibility.
            import csv
            rows = []
            for row in records:
                flat = {}
                for k, v in row.items():
                    if isinstance(v, (list, dict)):
                        flat[k] = json.dumps(v)
                    else:
                        flat[k] = v
                rows.append(flat)
            with out_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
                if rows:
                    writer.writeheader()
                    writer.writerows(rows)
            return str(out_path), str(meta_path)

        # parquet (preferred): requires pandas + pyarrow/fastparquet
        try:
            import pandas as pd
            rows = []
            for row in records:
                flat = {}
                for k, v in row.items():
                    if isinstance(v, (list, dict)):
                        flat[k] = json.dumps(v)
                    else:
                        flat[k] = v
                rows.append(flat)
            df = pd.DataFrame(rows)
            df.to_parquet(out_path, index=False)
            return str(out_path), str(meta_path)
        except Exception as e:
            # Graceful fallback if parquet deps are unavailable.
            fallback = out_path.with_suffix(".jsonl")
            with fallback.open("w") as f:
                for row in records:
                    f.write(json.dumps(row) + "\n")
            print(f"Parquet save unavailable ({e}); fell back to JSONL: {fallback}")
            return str(fallback), str(meta_path)
    
    def run(self, duration_s=None, max_steps=None, record=False, out_path=None, dataset_format="parquet"):
        """Main control loop with optional timed run and per-step dataset recording."""
        obs = self.env.reset()
        if self.has_renderer:
            self.env.render()
        
        print("Ready! Use arrow keys or WASD for XY, ;/. for Z\n")
        
        # Velocity state (persists between frames for smooth control)
        cart_vel = np.zeros(3)  # [vx, vy, vz]
        run_start = time.time()
        step_count = 0
        records = []

        if record:
            if out_path is None or out_path.strip() == "":
                out_path = str(self._default_output_path(dataset_format))
            print(f"Recording enabled -> {out_path} (format={dataset_format})")

        # Flush any stale keystrokes buffered in stdin before entering the loop
        # (e.g. leftover Escape/arrow sequences from a previous hung session).
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)

        try:
            while True:
                if duration_s is not None and duration_s > 0 and (time.time() - run_start) >= duration_s:
                    print(f"\nTimed run finished at {duration_s:.2f}s")
                    break
                if max_steps is not None and max_steps > 0 and step_count >= max_steps:
                    print(f"\nMax steps reached: {max_steps}")
                    break

                key = self.get_key()
                if key:
                    print(f"  [DBG] {repr(key)}", flush=True)

                # Reset velocities each frame (direct control, not momentum)
                cart_vel = np.zeros(3)

                if key.lower() == 'q':  # Q to quit (Ctrl+C also works)
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
                obs, reward, done, info = self.env.step(action)
                step_count += 1

                if record:
                    eef_pos = obs.get('robot0_eef_pos', np.zeros(3))
                    row = {
                        "step": step_count,
                        "t": time.time() - run_start,
                        "key": key,
                        "cmd_vx": float(cart_vel[0]),
                        "cmd_vy": float(cart_vel[1]),
                        "cmd_vz": float(cart_vel[2]),
                        "gripper_cmd": float(self.gripper_state),
                        "joint_vel_cmd": [float(x) for x in joint_vel.tolist()],
                        "action": [float(x) for x in action.tolist()],
                        "reward": float(reward),
                        "done": bool(done),
                        "is_success": bool(info.get("is_success", False)) if isinstance(info, dict) else False,
                        "eef_pos": [float(x) for x in np.asarray(eef_pos).ravel().tolist()],
                    }
                    # Keep selected raw observations for imitation / debugging.
                    for k in (
                        "robot0_joint_pos",
                        "robot0_joint_vel",
                        "robot0_eef_pos",
                        "robot0_eef_quat",
                        "cube_pos",
                        "gripper_to_cube_pos",
                    ):
                        if k in obs:
                            row[k] = [float(x) for x in np.asarray(obs[k]).ravel().tolist()]
                    records.append(row)
                
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
            if record and records:
                saved_path, meta_path = self._save_records(records, out_path, dataset_format)
                print(f"Saved dataset: {saved_path}")
                print(f"Saved metadata: {meta_path}")
            self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Interactive Cartesian teleop with optional timed dataset recording.")
    parser.add_argument("--duration", type=float, default=0.0, help="Timed run length in seconds (0 = unlimited).")
    parser.add_argument("--max-steps", type=int, default=0, help="Maximum steps before auto-stop (0 = unlimited).")
    parser.add_argument("--record", action="store_true", help="Record per-step movement dataset.")
    parser.add_argument("--out", type=str, default="", help="Output dataset path. Default: data/imitation/teleop_<ts>.<format>")
    parser.add_argument("--format", choices=("parquet", "jsonl", "csv"), default="parquet", help="Dataset format.")
    args = parser.parse_args()

    controller = CartesianArmControl()
    controller.run(
        duration_s=args.duration if args.duration > 0 else None,
        max_steps=args.max_steps if args.max_steps > 0 else None,
        record=args.record,
        out_path=args.out,
        dataset_format=args.format,
    )


if __name__ == '__main__':
    main()
