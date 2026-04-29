"""Build the MuJoCo XML body for the Redragon K552 TKL keyboard.

Lifted (and cleaned up) from the legacy ``keyboard_env.py``. The geometry
matches the physical keyboard within ±0.5 mm. Returns a single ``<body>``
element that gets appended to the arena's worldbody.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

from .keyboard_layout import (
    KEYBOARD_LAYOUT,
    KEY_GAP,
    KEY_H,
    KEY_HALF,
    KEY_PITCH,
    _KB_HEIGHT_U,
    _KB_WIDTH_U,
)


def build_keyboard_body(
    offset_xy: tuple[float, float] = (-0.15, 0.0),
    height: float = 0.15,
) -> ET.Element:
    """Return the ``<body name='keyboard_base'>`` XML element.

    Args:
        offset_xy: (x, y) offset of the keyboard centre from the table
            centre, in metres. Default places the keyboard 15 cm in front of
            the arm base, which puts the home row inside the workspace.
        height: top of keyboard surface above the floor in metres.

    The returned element must be appended to a MuJoCo arena's worldbody.
    Each key is a child ``<body name='key_<name>'>`` whose XY position
    matches ``KEYBOARD_LAYOUT``.
    """
    base = ET.Element("body")
    base.set("name", "keyboard_base")
    base.set("pos", f"{offset_xy[0]:.4f} {offset_xy[1]:.4f} {height:.4f}")

    # Base slab — covers full TKL footprint with a small margin
    slab_half_x = _KB_HEIGHT_U / 2 * KEY_PITCH + KEY_HALF + 0.003
    slab_half_y = _KB_WIDTH_U / 2 * KEY_PITCH + KEY_HALF + 0.003
    slab = ET.SubElement(base, "geom")
    slab.set("name", "keyboard_surface")
    slab.set("type", "box")
    slab.set("size", f"{slab_half_x:.4f} {slab_half_y:.4f} {KEY_H:.4f}")
    slab.set("rgba", "0.15 0.15 0.15 1")
    slab.set("contype", "1")
    slab.set("conaffinity", "1")

    # Individual keys
    for key_name, x_local, y_local, width_u in KEYBOARD_LAYOUT:
        key_body = ET.SubElement(base, "body")
        key_body.set("name", f"key_{key_name}")
        key_body.set("pos", f"{x_local:.4f} {y_local:.4f} {KEY_H * 3:.4f}")

        half_col = (width_u * KEY_PITCH - KEY_GAP) / 2
        geom = ET.SubElement(key_body, "geom")
        geom.set("name", f"key_{key_name}_geom")
        geom.set("type", "box")
        geom.set("size", f"{KEY_HALF} {half_col:.4f} {KEY_H}")
        geom.set("rgba", "0.88 0.88 0.88 1")
        geom.set("contype", "1")
        geom.set("conaffinity", "1")

        site = ET.SubElement(key_body, "site")
        site.set("name", f"key_{key_name}_site")
        site.set("pos", f"0 0 {KEY_H:.4f}")
        site.set("size", "0.003")
        site.set("rgba", "0 1 0 0.4")
        site.set("group", "1")

    return base
