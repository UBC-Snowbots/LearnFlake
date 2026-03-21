#!/usr/bin/env python3
"""
Keyboard Typing Test (low-level MuJoCo, based on gripper.py pattern)

Builds the world manually: table + stick gripper + 60-key QWERTY keyboard.
Press keys on your real keyboard to move the stick over the matching
colour-coded simulated key and detect contacts.

12 control keys (colour-coded on the simulated keyboard):
  W/S   → x_pos / x_neg
  A/D   → y_pos / y_neg
  ;/.   → z_pos / z_neg
  U/J   → yaw_pos / yaw_neg
  O/P   → open_gripper / close_gripper
  Q/E   → none_0 / none_9

Other:
  R     → Reset    Esc → Quit
"""

import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
import termios
import tty
import select

# ── Path setup ───────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite")
sys.path.insert(0, ROBO_PATH)

from robosuite.models import MujocoWorldBase
from robosuite.models.arenas.table_arena import TableArena
from robosuite.models.grippers.stick import StickClickerBase
from robosuite.models.objects.simple_keyboard import SimpleKeyboardObject
from robosuite.renderers.viewer import OpenCVViewer
from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim
from robosuite.utils.mjcf_utils import new_actuator, new_joint

# ── Key layout: name → (x, y) offset in keyboard local frame ────────────
# Control keys are placed at their QWERTY positions.
# Row along X (back→front), column along Y (left→right), 19mm pitch.
KEY_POSITIONS = {
    # Row 1 (QWERTY row, X = +0.019)
    "none_0":         ( 0.019, -0.1045),  # Q
    "x_pos":          ( 0.019, -0.0855),  # W
    "none_9":         ( 0.019, -0.0665),  # E
    "yaw_pos":        ( 0.019,  0.0095),  # U
    "open_gripper":   ( 0.019,  0.0475),  # O
    "close_gripper":  ( 0.019,  0.0665),  # P
    # Row 2 (home row, X = 0.000)
    "y_pos":          ( 0.000, -0.1045),  # A
    "x_neg":          ( 0.000, -0.0855),  # S
    "y_neg":          ( 0.000, -0.0665),  # D
    "yaw_neg":        ( 0.000,  0.0095),  # J
    "z_pos":          ( 0.000,  0.0665),  # ;
    # Row 3 (bottom row, X = −0.019)
    "z_neg":          (-0.019,  0.0475),  # .
}

# Real key → simulated key name
CHAR_TO_KEY = {
    'w': 'x_pos',    's': 'x_neg',
    'a': 'y_pos',    'd': 'y_neg',
    ';': 'z_pos',    '.': 'z_neg',
    'u': 'yaw_pos',  'j': 'yaw_neg',
    'o': 'open_gripper', 'p': 'close_gripper',
    'q': 'none_0',   'e': 'none_9',
}

# ── World constants ─────────────────────────────────────────────────────
TABLE_OFFSET = np.array([0, 0, 1.1])
KEYBOARD_POS = np.array([0.0, 0.0, TABLE_OFFSET[2] + 0.025 + 0.003])
STICK_HOME   = np.array([0.0, 0.0, KEYBOARD_POS[2] + 0.20])

HOVER_Z  = KEYBOARD_POS[2] + 0.04   # float above key tops
PRESS_Z  = KEYBOARD_POS[2] + 0.005  # key top = base_z + 0.001 + 0.004


def get_key(timeout=0.02):
    """Non-blocking single keypress reader."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                sys.stdin.read(1)
                sys.stdin.read(1)
                return 'ESC'
            return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ''


def build_world():
    """Construct the MuJoCo world: table + stick + keyboard."""
    world = MujocoWorldBase()

    # ── Table ────────────────────────────────────────────────────────
    arena = TableArena(
        table_full_size=(0.8, 0.8, 0.05),
        table_offset=TABLE_OFFSET,
        has_legs=True,
    )
    world.merge(arena)

    # ── Stick gripper on XYZ slide joints ────────────────────────────
    stick = StickClickerBase()

    stick_body = ET.Element("body", name="stick_base")
    stick_body.set("pos", f"{STICK_HOME[0]} {STICK_HOME[1]} {STICK_HOME[2]}")
    stick_body.set("quat", "0 0 1 0")  # flip so stick points down

    stick_body.append(new_joint(name="stick_x", type="slide", axis="1 0 0", damping="50"))
    stick_body.append(new_joint(name="stick_y", type="slide", axis="0 1 0", damping="50"))
    stick_body.append(new_joint(name="stick_z", type="slide", axis="0 0 1", damping="50"))

    world.worldbody.append(stick_body)
    world.merge(stick, merge_body="stick_base")

    world.actuator.append(new_actuator(joint="stick_x", act_type="position", name="act_x", kp="500"))
    world.actuator.append(new_actuator(joint="stick_y", act_type="position", name="act_y", kp="500"))
    world.actuator.append(new_actuator(joint="stick_z", act_type="position", name="act_z", kp="500"))

    # ── Keyboard ─────────────────────────────────────────────────────
    keyboard = SimpleKeyboardObject(name="keyboard")
    keyboard_obj = keyboard.get_obj()
    keyboard_obj.set("pos", f"{KEYBOARD_POS[0]} {KEYBOARD_POS[1]} {KEYBOARD_POS[2]}")
    world.worldbody.append(keyboard_obj)

    return world


def main():
    world = build_world()

    # ── Start simulation ─────────────────────────────────────────────
    model = world.get_model(mode="mujoco")
    sim = MjSim(model)
    viewer = OpenCVViewer(sim)
    render_context = MjRenderContextOffscreen(sim, device_id=-1)
    sim.add_render_context(render_context)

    sim_state = sim.get_state()

    # Actuator IDs
    act_x_id = sim.model.actuator_name2id("act_x")
    act_y_id = sim.model.actuator_name2id("act_y")
    act_z_id = sim.model.actuator_name2id("act_z")

    # Gravity compensation joints
    gravity_joints = ["stick_x", "stick_y", "stick_z"]
    grav_idx = [sim.model.get_joint_qvel_addr(j) for j in gravity_joints]

    # Resolve key geom IDs for contact detection
    key_geom_ids = {}
    for key_name in KEY_POSITIONS:
        try:
            gid = sim.model.geom_name2id(f"keyboard_key_{key_name}")
            key_geom_ids[key_name] = gid
        except Exception:
            pass

    # Stick tip geom ID
    stick_tip_id = None
    for i in range(sim.model.ngeom):
        name = sim.model.geom_id2name(i)
        if name and "stick_tip" in name:
            stick_tip_id = i
            break

    # ── Print controls ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  KEYBOARD TYPING TEST")
    print("=" * 60)
    print("  W/S   → x_pos / x_neg")
    print("  A/D   → y_pos / y_neg")
    print("  ;/.   → z_pos / z_neg")
    print("  U/J   → yaw_pos / yaw_neg")
    print("  O/P   → open_gripper / close_gripper")
    print("  Q/E   → none_0 / none_9")
    print()
    print("  R     → Reset    Esc → Quit")
    print("=" * 60 + "\n")

    # ── Control state ────────────────────────────────────────────────
    target_key = None
    pressing = False
    target_offset = np.array([0.0, 0.0, 0.0])

    sim.set_state(sim_state)
    step = 0

    while True:
        ch = get_key()

        if ch == 'ESC' or ch == '\x1b':
            print("\nExiting...")
            break
        elif ch.lower() == 'r':
            print("Resetting...")
            sim.set_state(sim_state)
            target_key = None
            pressing = False
            target_offset = np.array([0.0, 0.0, 0.0])
        elif ch.lower() in CHAR_TO_KEY:
            target_key = CHAR_TO_KEY[ch.lower()]
            pressing = True
            print(f"  ▸ Pressing: {target_key}")
        else:
            pressing = False

        # Compute target position offset from home
        if target_key is not None:
            kx, ky = KEY_POSITIONS[target_key]
            key_world_x = KEYBOARD_POS[0] + kx
            key_world_y = KEYBOARD_POS[1] + ky
            key_world_z = PRESS_Z if pressing else HOVER_Z

            target_offset[0] = key_world_x - STICK_HOME[0]
            target_offset[1] = key_world_y - STICK_HOME[1]
            target_offset[2] = key_world_z - STICK_HOME[2]
        else:
            target_offset = np.array([0.0, 0.0, 0.0])

        # Set actuator targets
        sim.data.ctrl[act_x_id] = target_offset[0]
        sim.data.ctrl[act_y_id] = target_offset[1]
        sim.data.ctrl[act_z_id] = target_offset[2]

        # Step simulation
        sim.step()

        # Gravity compensation
        sim.data.qfrc_applied[grav_idx] = sim.data.qfrc_bias[grav_idx]

        # ── Contact detection ────────────────────────────────────────
        if stick_tip_id is not None and step % 10 == 0:
            contacts = set()
            for c_idx in range(sim.data.ncon):
                contact = sim.data.contact[c_idx]
                g1, g2 = contact.geom1, contact.geom2
                if stick_tip_id in (g1, g2):
                    other = g2 if g1 == stick_tip_id else g1
                    for kname, gid in key_geom_ids.items():
                        if other == gid:
                            contacts.add(kname)
            if contacts:
                print(f"  ✓ Contact: {', '.join(contacts)}")

        viewer.render()
        step += 1


if __name__ == "__main__":
    main()
