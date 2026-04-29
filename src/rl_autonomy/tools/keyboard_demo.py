#!/usr/bin/env python3
"""
Interactive keyboard demo — Cartesian EEF control with MuJoCo rendering.

Usage
-----
Terminal 1 (this script, MuJoCo window pops up):
    python src/rl_autonomy/testing/keyboard_demo.py

Terminal 2 (live diagnostics):
    watch -n 0.1 cat /tmp/arm_state.txt

Controls (focus Terminal 1 to type)
------------------------------------
  W / S        : EEF  +X / -X  (forward / back)
  A / D        : EEF  +Y / -Y  (left / right)
  R / F        : EEF  +Z / -Z  (up / down)
  SPACE        : toggle solenoid (extend / retract)
  T            : cycle target key  (q → w → e → r → t → q …)
  0            : reset arm to home pose
  Q  / Ctrl-C  : quit
"""

import os, sys, time, threading, select, tty, termios, signal
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, ".."))
sys.path.insert(0, os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite"))

from keyboard_env import KeyboardEnv

# ── tunables ────────────────────────────────────────────────────────────────
CART_SPEED   = 0.08    # m/s for translational moves
CTRL_HZ      = 20      # must match env control_freq
STATE_FILE   = "/tmp/arm_state.txt"
KEY_CYCLE    = ["q", "w", "e", "r", "t"]
# ────────────────────────────────────────────────────────────────────────────


# ── non-blocking keyboard reader (runs in background thread) ─────────────────
class KeyReader(threading.Thread):
    """Reads single chars from stdin without blocking the main loop."""

    def __init__(self):
        super().__init__(daemon=True)
        self._pressed: set[str] = set()
        self._events:  list[str] = []   # one-shot events (space, t, 0, q)
        self._lock = threading.Lock()
        self._stop_flag = False

    def run(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not self._stop_flag:
                r, _, _ = select.select([sys.stdin], [], [], 0.02)
                if r:
                    ch = sys.stdin.read(1).lower()
                    with self._lock:
                        self._pressed.add(ch)
                        if ch in (" ", "t", "0", "q"):
                            self._events.append(ch)
                else:
                    with self._lock:
                        self._pressed.clear()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def pressed(self) -> set[str]:
        with self._lock:
            return set(self._pressed)

    def pop_events(self) -> list[str]:
        with self._lock:
            evts = list(self._events)
            self._events.clear()
        return evts

    def stop(self):
        self._stop_flag = True


# ── Jacobian-based Cartesian → joint-velocity conversion ────────────────────
def cartesian_to_joint_vel(sim, grip_site_name, arm_dof_indices, cart_vel_3d):
    """
    Convert a desired 3D Cartesian EEF velocity into 6 joint velocities
    using the translational Jacobian pseudo-inverse.
    """
    nv   = sim.model.nv
    jacp = sim.data.get_site_jacp(grip_site_name).reshape(3, nv)
    J    = jacp[:, arm_dof_indices]                 # 3 × 6
    J_pinv = np.linalg.pinv(J)                      # 6 × 3
    return J_pinv @ cart_vel_3d                     # 6D joint velocity


# ── state-file writer ────────────────────────────────────────────────────────
def write_state(env, target_key, solenoid_on, contact_detected, step):
    obs  = env.get_obs()
    sim  = env.sim

    jpos = np.array(obs.get("robot0_joint_pos", np.zeros(6)))
    jvel = np.array(obs.get("robot0_joint_vel", np.zeros(6)))
    epos = np.array(obs.get("robot0_eef_pos",   np.zeros(3)))
    kpos = np.array(obs.get("target_key_pos",   np.zeros(3)))
    dist = np.array(obs.get("eef_to_key",       np.zeros(3)))
    rf   = float(np.array(obs.get("rangefinder",  [0])).flat[0])
    cf   = float(np.array(obs.get("contact_force",[0])).flat[0])
    ae   = float(np.array(obs.get("actuator_extended",[0])).flat[0])
    aruco = np.array(obs.get("aruco_obs", np.zeros(3)))

    # contact geoms
    contacts = []
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        g1 = sim.model.geom_id2name(c.geom1) or f"geom{c.geom1}"
        g2 = sim.model.geom_id2name(c.geom2) or f"geom{c.geom2}"
        contacts.append(f"  {g1}  ↔  {g2}")

    lines = [
        f"╔══════════════════════════════════════════════╗",
        f"║  Rover2026 Arm State   step={step:>6d}        ║",
        f"╠══════════════════════════════════════════════╣",
        f"║  EEF position  : [{epos[0]:6.3f} {epos[1]:6.3f} {epos[2]:6.3f}]  ║",
        f"║  Target key    : {target_key}  @ [{kpos[0]:5.3f} {kpos[1]:5.3f} {kpos[2]:5.3f}]  ║",
        f"║  EEF→key vec   : [{dist[0]:6.3f} {dist[1]:6.3f} {dist[2]:6.3f}]  ║",
        f"║  |EEF→key|     : {np.linalg.norm(dist):7.4f} m                  ║",
        f"╠══════════════════════════════════════════════╣",
        f"║  Rangefinder   : {rf:7.4f} m (cutoff=0.30)        ║",
        f"║  Contact force : {cf:7.4f} N                       ║",
        f"║  Contact det.  : {'YES ◄◄◄' if contact_detected else 'no ':8s}                 ║",
        f"║  Solenoid      : {'EXTENDED' if ae > 0.5 else 'retracted':10s}  (cmd={'ON' if solenoid_on else 'off'})        ║",
        f"║  ArUco obs     : dx={aruco[0]:6.3f} dy={aruco[1]:6.3f} vis={aruco[2]:.0f}  ║",
        f"╠══════════════════════════════════════════════╣",
        f"║  Joint positions (rad):                      ║",
    ]
    jnames = ["shoulder","link_1 ","link1_2","a4_rot ","a5_rot ","a6_rot "]
    for name, p, v in zip(jnames, jpos, jvel):
        lines.append(f"║    {name}: pos={p:7.3f}  vel={v:7.3f}           ║")
    lines += [
        f"╠══════════════════════════════════════════════╣",
        f"║  Active contacts ({len(contacts)}):                       ║",
    ]
    for c in contacts[:4]:
        lines.append(f"║  {c[:44]:<44s}  ║")
    if not contacts:
        lines.append(f"║    (none)                                    ║")
    lines.append(f"╚══════════════════════════════════════════════╝")

    with open(STATE_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n  Rover2026 Keyboard Demo")
    print("  ─────────────────────────")
    print("  W/S A/D R/F  — EEF Cartesian translation")
    print("  SPACE        — toggle solenoid")
    print("  T            — cycle target key")
    print("  0            — reset to home")
    print("  Q / Ctrl-C   — quit")
    print(f"\n  Diagnostics: watch -n 0.1 cat {STATE_FILE}")
    print("\n  Loading environment …\n")

    env = KeyboardEnv(render=True, horizon=100_000)
    env.reset()

    # ── find grip site name (prefixed by RoboSuite) ─────────────────────────
    sim = env.sim
    grip_site = None
    for i in range(sim.model.nsite):
        name = sim.model.site_id2name(i)
        if "grip_site" in name and "cylinder" not in name:
            grip_site = name
            break
    assert grip_site, "grip_site not found"

    arm_dof_idx = list(env.robots[0]._ref_joint_vel_indexes)  # 6 arm DOF indices

    # ── key state ────────────────────────────────────────────────────────────
    key_idx     = 2           # start on 'e' (centre key)
    solenoid_on = False
    done        = False
    step        = 0

    env.set_target_key(KEY_CYCLE[key_idx])

    reader = KeyReader()
    reader.start()

    def _quit(*_):
        nonlocal done
        done = True

    signal.signal(signal.SIGINT, _quit)

    print("  Environment ready — focus this terminal and press keys.\n")

    dt = 1.0 / CTRL_HZ

    try:
        while not done:
            t0 = time.time()

            # ── handle one-shot events ──────────────────────────────────────
            for evt in reader.pop_events():
                if evt == "q":
                    done = True
                elif evt == " ":
                    solenoid_on = not solenoid_on
                elif evt == "t":
                    key_idx = (key_idx + 1) % len(KEY_CYCLE)
                    env.set_target_key(KEY_CYCLE[key_idx])
                    print(f"  → target key: {KEY_CYCLE[key_idx]}")
                elif evt == "0":
                    env.reset()
                    solenoid_on = False
                    print("  → reset")

            # ── build Cartesian velocity from held keys ─────────────────────
            held  = reader.pressed()
            cart  = np.zeros(3)
            if "w" in held: cart[0] += CART_SPEED
            if "s" in held: cart[0] -= CART_SPEED
            if "a" in held: cart[1] += CART_SPEED
            if "d" in held: cart[1] -= CART_SPEED
            if "r" in held: cart[2] += CART_SPEED
            if "f" in held: cart[2] -= CART_SPEED

            # ── joint velocity via Jacobian pseudo-inverse ──────────────────
            if np.any(cart != 0):
                jvel = cartesian_to_joint_vel(sim, grip_site, arm_dof_idx, cart)
                jvel = np.clip(jvel, -1.0, 1.0)
            else:
                jvel = np.zeros(6)

            action = np.append(jvel, 1.0 if solenoid_on else -1.0)

            obs, _, episode_done, _ = env.step(action)
            env.render()

            # ── contact detection ───────────────────────────────────────────
            cf = float(np.array(obs.get("contact_force", [0])).flat[0])
            av = float(np.array(obs.get("actuator_vel",  [0])).flat[0])
            contact_detected = (cf > 2.0) and (abs(av) < 0.005)

            # ── write diagnostics every 3 steps ────────────────────────────
            if step % 3 == 0:
                write_state(env, KEY_CYCLE[key_idx], solenoid_on,
                            contact_detected, step)

            if episode_done:
                env.reset()
                solenoid_on = False

            step += 1
            elapsed = time.time() - t0
            sleep   = dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

    finally:
        reader.stop()
        env.close()
        print("\n  Demo closed.")


if __name__ == "__main__":
    main()
