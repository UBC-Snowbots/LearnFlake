#!/usr/bin/env python3
"""
Controller Configuration for Arm Teleoperation (Python port)

Mirrors RoverFlake2/src/arm_control/include/controller_config.h so the
MuJoCo bridge can process raw /joy messages from either controller
without depending on which C++ node is running.

Supported controllers:
    1. Nintendo Switch Pro Controller  (hid-nintendo driver via joy node)
    2. Saitek Cyborg USB Stick

Auto-detection:
    The number of axes and buttons in a Joy message is used as a heuristic:
        Pro Controller  → 6 axes, ≥13 buttons
        Cyborg Stick    → 6 axes,  ≤6 buttons  (typical: 4 or 6)

    You can also force a controller via CLI flag --controller pro|cyborg.

Button/axis indices can be verified by running:
    ros2 topic echo /joy
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math


# ============================================================================
# Controller profiles
# ============================================================================

@dataclass(frozen=True)
class ControllerProfile:
    """Immutable description of a single controller's button/axis layout."""

    name: str

    # --- Cartesian translation buttons (index → direction) ---
    # Set to -1 to disable button-based Cartesian (e.g. Cyborg uses axes)
    btn_cart_pos_x: int = -1   # +X (forward)
    btn_cart_neg_x: int = -1   # -X (backward)
    btn_cart_pos_y: int = -1   # +Y (left)
    btn_cart_neg_y: int = -1   # -Y (right)
    btn_cart_pos_z: int = -1   # +Z (up)
    btn_cart_neg_z: int = -1   # -Z (down)

    # --- Cartesian translation axes (index → direction) ---
    # Set to -1 to disable axis-based Cartesian
    axis_cart_x: int = -1      # Axis controlling X
    axis_cart_y: int = -1      # Axis controlling Y
    axis_cart_z: int = -1      # Axis controlling Z
    invert_cart_x: bool = False
    invert_cart_y: bool = False
    invert_cart_z: bool = False

    # --- Orientation axes ---
    axis_roll:  int = -1
    axis_pitch: int = -1
    axis_yaw:   int = -1
    invert_roll:  bool = False
    invert_pitch: bool = False
    invert_yaw:   bool = False

    # --- Gripper toggle button ---
    btn_gripper_toggle: int = -1
    axis_gripper_toggle: int = -1
    gripper_axis_pressed_threshold: float = 0.0

    # --- Speed parameters ---
    cart_button_speed: float = 0.5   # unitless [0, 1] for button-based translation
    cart_axis_speed:   float = 1.0   # scale for axis-based translation (raw value * this)
    rot_stick_speed:   float = 0.6   # unitless [0, 1] for angular stick
    axis_deadzone:     float = 0.15

    # --- Frame for Cartesian twist commands ---
    cart_frame_id: str = "base_link"


# ────────────────────────────────────────────────────────────────────────────
#  Nintendo Switch Pro Controller
# ────────────────────────────────────────────────────────────────────────────
#  Buttons (hid-nintendo driver via joy node):
#    0 = B (east)        1 = A (south)
#    2 = Y (north)       3 = X (west)
#    4 = L shoulder      5 = R shoulder
#    6 = ZL trigger      7 = ZR trigger
#    8 = Minus           9 = Plus
#   10 = L stick press  11 = R stick press
#   12 = Home           13 = Capture
#
#  Axes:
#    0 = Left stick X    1 = Left stick Y
#    2 = Right stick X   3 = Right stick Y
#    4 = ZL analog       5 = ZR analog  (rest at 1.0, pressed = -1.0)
# ────────────────────────────────────────────────────────────────────────────

PRO_CONTROLLER = ControllerProfile(
    name="Nintendo Switch Pro Controller",

    # Face buttons → Cartesian translation
    #        Y (+X forward)
    #   X (+Y)         B (-Y)
    #        A (-X backward)
    #   R shoulder → +Z (up)      L shoulder → -Z (down)
    btn_cart_pos_x=2,    # Y (north)
    btn_cart_neg_x=1,    # A (south)
    btn_cart_pos_y=3,    # X (west)
    btn_cart_neg_y=0,    # B (east)
    btn_cart_pos_z=5,    # R shoulder
    btn_cart_neg_z=4,    # L shoulder

    # No axis-based Cartesian translation on Pro Controller
    axis_cart_x=-1,
    axis_cart_y=-1,
    axis_cart_z=-1,

    # EE orientation: left stick → pitch/yaw, right stick → roll
    axis_roll=2,         # Right stick X
    axis_pitch=1,        # Left stick Y
    axis_yaw=0,          # Left stick X
    invert_roll=False,
    invert_pitch=False,
    invert_yaw=True,     # push left = positive yaw

    # Gripper: axis[5] ("axis 6" in 1-based indexing)
    # Trigger axes on this driver rest at +1.0 and move toward -1.0 when pressed.
    btn_gripper_toggle=-1,
    axis_gripper_toggle=5,
    gripper_axis_pressed_threshold=0.0,

    cart_button_speed=0.5,
    rot_stick_speed=0.6,
    axis_deadzone=0.15,
    cart_frame_id="base_link",
)


# ────────────────────────────────────────────────────────────────────────────
#  Saitek Cyborg USB Stick
# ────────────────────────────────────────────────────────────────────────────
#  Axes:
#    0 = Stick X (left/right)
#    1 = Stick Y (forward/back)
#    2 = Throttle slider (rests non-zero — DO NOT USE for motion)
#    3 = Stick twist/rudder (rotation)
#    4 = Hat X
#    5 = Hat Y
#
#  Buttons:
#    0 = Trigger
# ────────────────────────────────────────────────────────────────────────────

CYBORG_STICK = ControllerProfile(
    name="Saitek Cyborg USB Stick",

    # No button-based Cartesian on Cyborg (uses analog axes instead)
    btn_cart_pos_x=-1,
    btn_cart_neg_x=-1,
    btn_cart_pos_y=-1,
    btn_cart_neg_y=-1,
    btn_cart_pos_z=-1,
    btn_cart_neg_z=-1,

    # Cyborg uses stick axes for Cartesian translation
    # Stick Y → X (forward/back), Stick X → Y (left/right), Hat Y → Z (up/down)
    axis_cart_x=1,       # Stick Y → linear X
    axis_cart_y=0,       # Stick X → linear Y
    axis_cart_z=5,       # Hat Y   → linear Z
    invert_cart_x=False,
    invert_cart_y=True,  # negate so stick-left = +Y (matches moveit_control.cpp)
    invert_cart_z=False,

    # Orientation: Hat X → angular X, stick twist → angular Y
    axis_roll=-1,        # not mapped
    axis_pitch=3,        # Stick twist/rudder → angular Y (pitch)
    axis_yaw=4,          # Hat X → angular X (yaw)
    invert_roll=False,
    invert_pitch=False,
    invert_yaw=False,

    # Gripper: trigger (button 0)
    btn_gripper_toggle=0,
    axis_gripper_toggle=-1,
    gripper_axis_pressed_threshold=0.0,

    cart_button_speed=0.3,
    cart_axis_speed=1.0,
    rot_stick_speed=1.0,
    axis_deadzone=0.15,
    cart_frame_id="base_link",
)


# ============================================================================
# Auto-detection
# ============================================================================

# Registry: controller profiles keyed by CLI name
CONTROLLERS: dict[str, ControllerProfile] = {
    "pro":    PRO_CONTROLLER,
    "cyborg": CYBORG_STICK,
}


def detect_controller(num_buttons: int, num_axes: int) -> ControllerProfile:
    """
    Heuristic auto-detection based on the shape of the first /joy message.

    Pro Controller:  6 axes, ≥13 buttons
    Cyborg Stick:    6 axes,  ≤6 buttons  (typically 4)
    """
    if num_buttons >= 10:
        return PRO_CONTROLLER
    else:
        return CYBORG_STICK


def get_controller(name: Optional[str] = None) -> Optional[ControllerProfile]:
    """Return a profile by CLI name, or None for auto-detect."""
    if name is None:
        return None
    key = name.strip().lower()
    if key in CONTROLLERS:
        return CONTROLLERS[key]
    raise ValueError(
        f"Unknown controller '{name}'. "
        f"Available: {', '.join(CONTROLLERS.keys())}"
    )
