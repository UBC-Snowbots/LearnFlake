"""
Keyboard layout data for the Redragon K552 TKL (87 keys).

Pure data module — no environment or MuJoCo dependencies.
All positions are in metres, relative to the keyboard body centre.

Coordinate convention (keyboard body local frame):
    X = row direction   (pos = front / space bar, neg = back / F-keys)
    Y = column direction (pos = right / nav cluster, neg = left / Esc side)
    Keyboard faces the robot (space bar row nearest to arm base).
"""

# ---------------------------------------------------------------------------
# Physical dimensions
# ---------------------------------------------------------------------------

KEY_PITCH = 0.019       # centre-to-centre spacing (m), standard 19 mm
KEY_HALF  = 0.009       # half-width of a 1U keycap in XY (m)
KEY_H     = 0.002       # keycap half-height (m)
KEY_GAP   = 0.001       # gap between adjacent keycaps (m)

# Pressable key mechanics
PRESS_DEPTH     = 0.004  # max key travel (m), ~4 mm like Cherry MX
PRESS_THRESHOLD = 0.002  # activation point (m), ~2 mm actuation distance
SPRING_STIFFNESS = 100.0 # N/m — gives ~0.4 N at full press
SPRING_DAMPING   = 1.0   # Ns/m — prevents bouncing

# TKL bounding box in U (1 U = KEY_PITCH)
KB_WIDTH_U  = 18.25     # total columns (with nav cluster)
KB_HEIGHT_U = 6.5       # total rows (F-row + 0.5U gap + 5 main rows)
KB_CENTER_COL = KB_WIDTH_U / 2     # 9.125
KB_CENTER_ROW = KB_HEIGHT_U / 2    # 3.25


# ---------------------------------------------------------------------------
# Layout builder
# ---------------------------------------------------------------------------

def build_tkl_layout():
    """Compute Redragon K552 TKL key positions.

    Returns
    -------
    list of (name, x_local_m, y_local_m, width_u)
        Each entry is one key: label, X position, Y position, width in U.
    """
    layout = []

    def add_row(row_u, keys):
        col = 0.0
        for name, w in keys:
            if name is not None:
                cx = -((row_u - KB_CENTER_ROW) * KEY_PITCH)
                cy = -((col + w / 2.0 - KB_CENTER_COL) * KEY_PITCH)
                layout.append((name, cx, cy, w))
            col += w

    # Row 0 — F-key row
    add_row(0.5, [
        ("esc", 1), (None, 1),
        ("f1", 1), ("f2", 1), ("f3", 1), ("f4", 1), (None, 0.5),
        ("f5", 1), ("f6", 1), ("f7", 1), ("f8", 1), (None, 0.5),
        ("f9", 1), ("f10", 1), ("f11", 1), ("f12", 1), (None, 0.25),
        ("prtsc", 1), ("scrlk", 1), ("pause", 1),
    ])

    # Row 1 — Number row
    add_row(2.0, [
        ("grave", 1), ("1", 1), ("2", 1), ("3", 1), ("4", 1), ("5", 1),
        ("6", 1), ("7", 1), ("8", 1), ("9", 1), ("0", 1),
        ("minus", 1), ("equal", 1), ("backspace", 2), (None, 0.25),
        ("ins", 1), ("home", 1), ("pgup", 1),
    ])

    # Row 2 — QWERTY row
    add_row(3.0, [
        ("tab", 1.5), ("q", 1), ("w", 1), ("e", 1), ("r", 1), ("t", 1),
        ("y", 1), ("u", 1), ("i", 1), ("o", 1), ("p", 1),
        ("lbracket", 1), ("rbracket", 1), ("backslash", 1.5), (None, 0.25),
        ("del", 1), ("end", 1), ("pgdn", 1),
    ])

    # Row 3 — Home row
    add_row(4.0, [
        ("caps", 1.75), ("a", 1), ("s", 1), ("d", 1), ("f", 1), ("g", 1),
        ("h", 1), ("j", 1), ("k", 1), ("l", 1),
        ("semicolon", 1), ("quote", 1), ("enter", 2.25),
    ])

    # Row 4 — Shift row
    add_row(5.0, [
        ("lshift", 2.25), ("z", 1), ("x", 1), ("c", 1), ("v", 1), ("b", 1),
        ("n", 1), ("m", 1), ("comma", 1), ("period", 1), ("slash", 1),
        ("rshift", 2.75), (None, 1.25),
        ("up", 1),
    ])

    # Row 5 — Space bar row
    add_row(6.0, [
        ("lctrl", 1.25), ("win", 1.25), ("lalt", 1.25),
        ("space", 6.25),
        ("ralt", 1.25), ("fn", 1.25), ("menu", 1.25), ("rctrl", 1.25),
        (None, 0.25),
        ("left", 1), ("down", 1), ("right", 1),
    ])

    return layout


# Pre-built layout — import this directly
TKL_KEYS = build_tkl_layout()
AVAILABLE_KEYS = [name for name, _, _, _ in TKL_KEYS]