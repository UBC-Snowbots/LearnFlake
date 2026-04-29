"""Redragon K552 TKL keyboard geometry + sensor constants.

The full 87-key layout is computed from a U-based grid (1 U = `KEY_PITCH`).
Coordinates are expressed in the keyboard body's local frame; the keyboard
is later placed on the table such that the space-bar row faces the arm.

This file is constants-only — no MuJoCo or gymnasium imports — so it can
be loaded by reward functions, observation builders, and tools without
pulling in heavy dependencies.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

KEY_PITCH = 0.019   # centre-to-centre key spacing (m), standard 19 mm
KEY_HALF = 0.009    # half side-length of a 1U key cap in XY (m)
KEY_H = 0.002       # key half-height (m)
KEY_GAP = 0.001     # visible gap between adjacent caps (m)

# TKL bounding box in U
_KB_WIDTH_U = 18.25     # total columns including nav cluster
_KB_HEIGHT_U = 6.5      # total rows (F-row + 0.5U gap + 5 main rows)
_KB_CENTER_COL = _KB_WIDTH_U / 2     # 9.125
_KB_CENTER_ROW = _KB_HEIGHT_U / 2    # 3.25


# ---------------------------------------------------------------------------
# ArUco synthesizer parameters
# ---------------------------------------------------------------------------
# Used by the env's aruco-synth observation. Values match the original
# `keyboard_env.py` so policies trained against the legacy env see the same
# noise model when ported.

ARUCO_NOISE_STD = 0.001     # ~1 mm position std at close range
ARUCO_VISIBLE_DIST = 0.05   # guaranteed visible within 5 cm XY
ARUCO_FALLOFF_DIST = 0.12   # fully invisible beyond 12 cm XY
ARUCO_MAX_TILT = 0.30       # rad (~17°) — beyond this, detection degrades


# ---------------------------------------------------------------------------
# Contact detection thresholds (mirrors synthetic-Moteus and real Moteus logic)
# ---------------------------------------------------------------------------

CONTACT_FORCE_THRESHOLD = 2.0     # N — actuator-tip contact force above this counts
STALL_VEL_THRESHOLD = 0.005       # m/s — solenoid velocity below this counts


# ---------------------------------------------------------------------------
# Layout builder
# ---------------------------------------------------------------------------

def _build_tkl_layout() -> list[tuple[str, float, float, float]]:
    """Compute the Redragon K552 TKL (87-key) layout.

    Returns a list of (name, x_local_m, y_local_m, width_u). Coordinates are
    in the keyboard body's local frame, with the keyboard rotated 180° about Z
    so the space-bar row faces the arm (negation of both axes vs. a standard
    top-down keyboard chart).
    """
    layout: list[tuple[str, float, float, float]] = []

    def add_row(row_u: float, keys: list[tuple[str | None, float]]) -> None:
        col = 0.0
        for name, w in keys:
            if name is not None:
                cx = -((row_u - _KB_CENTER_ROW) * KEY_PITCH)
                cy = -((col + w / 2.0 - _KB_CENTER_COL) * KEY_PITCH)
                layout.append((name, cx, cy, w))
            col += w

    # Row 0 — F-key row (centre at 0.5 U from top)
    add_row(0.5, [
        ("esc", 1), (None, 1),
        ("f1", 1), ("f2", 1), ("f3", 1), ("f4", 1), (None, 0.5),
        ("f5", 1), ("f6", 1), ("f7", 1), ("f8", 1), (None, 0.5),
        ("f9", 1), ("f10", 1), ("f11", 1), ("f12", 1), (None, 0.25),
        ("prtsc", 1), ("scrlk", 1), ("pause", 1),
    ])

    # Row 1 — number row
    add_row(2.0, [
        ("grave", 1), ("1", 1), ("2", 1), ("3", 1), ("4", 1), ("5", 1),
        ("6", 1), ("7", 1), ("8", 1), ("9", 1), ("0", 1),
        ("minus", 1), ("equal", 1), ("backspace", 2), (None, 0.25),
        ("ins", 1), ("home", 1), ("pgup", 1),
    ])

    # Row 2 — QWERTY
    add_row(3.0, [
        ("tab", 1.5), ("q", 1), ("w", 1), ("e", 1), ("r", 1), ("t", 1),
        ("y", 1), ("u", 1), ("i", 1), ("o", 1), ("p", 1),
        ("lbracket", 1), ("rbracket", 1), ("backslash", 1.5), (None, 0.25),
        ("del", 1), ("end", 1), ("pgdn", 1),
    ])

    # Row 3 — home row
    add_row(4.0, [
        ("caps", 1.75), ("a", 1), ("s", 1), ("d", 1), ("f", 1), ("g", 1),
        ("h", 1), ("j", 1), ("k", 1), ("l", 1),
        ("semicolon", 1), ("quote", 1), ("enter", 2.25),
    ])

    # Row 4 — shift row
    add_row(5.0, [
        ("lshift", 2.25), ("z", 1), ("x", 1), ("c", 1), ("v", 1), ("b", 1),
        ("n", 1), ("m", 1), ("comma", 1), ("period", 1), ("slash", 1),
        ("rshift", 2.75), (None, 1.25),
        ("up", 1),
    ])

    # Row 5 — space bar row
    add_row(6.0, [
        ("lctrl", 1.25), ("win", 1.25), ("lalt", 1.25),
        ("space", 6.25),
        ("ralt", 1.25), ("fn", 1.25), ("menu", 1.25), ("rctrl", 1.25),
        (None, 0.25),
        ("left", 1), ("down", 1), ("right", 1),
    ])

    return layout


KEYBOARD_LAYOUT: list[tuple[str, float, float, float]] = _build_tkl_layout()
AVAILABLE_KEYS: list[str] = [name for name, _, _, _ in KEYBOARD_LAYOUT]


# Sanity check — must run at import time so test_smoke catches malformed layouts
assert len(KEYBOARD_LAYOUT) == 87, (
    f"Redragon K552 TKL has 87 keys; layout produced {len(KEYBOARD_LAYOUT)}"
)
assert len(set(AVAILABLE_KEYS)) == len(AVAILABLE_KEYS), "duplicate key name in layout"


# Curriculum phases for TRACKER §7 — explicit lists used by the
# key_phase_curriculum to advance the target distribution over time.

PHASE_A_KEYS: list[str] = [
    "g", "h", "f", "j", "d", "k", "s", "l", "a", "semicolon",
    "t", "y", "r", "u", "e", "i", "w", "o", "q", "p",
]
"""Phase A — central alphanumeric keys reachable from the home pose."""

PHASE_B_KEYS: list[str] = sorted(set(
    PHASE_A_KEYS
    + list("zxcvbnm")
    + ["comma", "period", "slash"]
    + list("0123456789")
    + ["minus", "equal", "lbracket", "rbracket", "backslash", "quote", "grave"]
))
"""Phase B — full home + qwerty + bottom + number row."""

PHASE_C_KEYS: list[str] = AVAILABLE_KEYS
"""Phase C — every key on the keyboard."""
