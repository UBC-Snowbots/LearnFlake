import numpy as np
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Tuple

from keyboard_layout import (
    TKL_KEYS, KEY_PITCH, KEY_HALF, KEY_H, KEY_GAP,
    KB_WIDTH_U, KB_HEIGHT_U,
    PRESS_DEPTH, PRESS_THRESHOLD, SPRING_STIFFNESS, SPRING_DAMPING,
)


@dataclass
class KeyboardConfig:
    """Configuration for keyboard appearance and physics.

    All colour fields are RGBA tuples (0-1).
    """
    # Colours
    base_color: Tuple[float, ...] = (0.12, 0.12, 0.14, 1.0)
    key_color:  Tuple[float, ...] = (0.28, 0.28, 0.32, 1.0)
    label_color: Tuple[float, ...] = (0.85, 0.85, 0.85, 1.0)
    target_highlight_color: Tuple[float, ...] = (0.1, 0.8, 0.2, 0.6)
    key_color_variance: float = 0.0  # per-key RGB noise (0 = uniform)

    # Spring mechanics
    spring_stiffness: float = SPRING_STIFFNESS
    spring_damping: float = SPRING_DAMPING
    press_depth: float = PRESS_DEPTH
    spring_stiffness_variance: float = 0.0  # per-key stiffness noise

    # Collision
    key_contype: int = 1
    key_conaffinity: int = 1


def _rgba_str(rgba):
    """Convert an RGBA tuple to a space-separated string."""
    return " ".join(f"{c:.3f}" for c in rgba)


def _jitter_color(base_rgba, variance, rng):
    """Add uniform noise to RGB channels, clamp to [0, 1]."""
    if variance <= 0:
        return base_rgba
    r, g, b, a = base_rgba
    noise = rng.uniform(-variance, variance, size=3)
    return (
        float(np.clip(r + noise[0], 0, 1)),
        float(np.clip(g + noise[1], 0, 1)),
        float(np.clip(b + noise[2], 0, 1)),
        a,
    )


def build_keyboard(
    offset: Tuple[float, float] = (-0.15, 0.0),
    height: float = 0.15,
    config: KeyboardConfig = None,
    seed: int = None,
) -> ET.Element:
    """Build a pressable TKL keyboard as a MuJoCo XML body.

    Parameters
    ----------
    offset : (x, y)
        World-frame XY position of the keyboard centre.
    height : float
        Z position of the keyboard surface above the floor.
    config : KeyboardConfig
        Appearance and physics settings. Uses defaults if None.
    seed : int or None
        RNG seed for domain randomization. None = no randomization.

    Returns
    -------
    ET.Element
        A ``<body>`` element ready to append to a worldbody.
    """
    if config is None:
        config = KeyboardConfig()

    rng = np.random.RandomState(seed)

    kx, ky = offset
    kz = height

    # Root body
    base = ET.Element("body")
    base.set("name", "keyboard_base")
    base.set("pos", f"{kx:.4f} {ky:.4f} {kz:.4f}")

    # Base plate
    slab_hx = KB_HEIGHT_U / 2 * KEY_PITCH + KEY_HALF + 0.003
    slab_hy = KB_WIDTH_U / 2 * KEY_PITCH + KEY_HALF + 0.003
    slab = ET.SubElement(base, "geom")
    slab.set("name", "keyboard_surface")
    slab.set("type", "box")
    slab.set("size", f"{slab_hx:.4f} {slab_hy:.4f} {KEY_H:.4f}")
    slab.set("rgba", _rgba_str(config.base_color))
    slab.set("contype", str(config.key_contype))
    slab.set("conaffinity", str(config.key_conaffinity))

    # Individual pressable keys
    for key_name, x_local, y_local, width_u in TKL_KEYS:
        key_color = _jitter_color(config.key_color, config.key_color_variance, rng)

        # Per-key stiffness variation
        stiffness = config.spring_stiffness
        if config.spring_stiffness_variance > 0:
            stiffness += rng.uniform(
                -config.spring_stiffness_variance,
                config.spring_stiffness_variance,
            )
            stiffness = max(10.0, stiffness)

        # Key body — positioned above the base plate
        key_body = ET.SubElement(base, "body")
        key_body.set("name", f"key_{key_name}")
        key_body.set("pos", f"{x_local:.4f} {y_local:.4f} {KEY_H * 3:.4f}")

        # Slide joint — allows key to depress along Z
        joint = ET.SubElement(key_body, "joint")
        joint.set("name", f"key_{key_name}_slide")
        joint.set("type", "slide")
        joint.set("axis", "0 0 1")
        joint.set("range", f"{-config.press_depth:.4f} 0")
        joint.set("stiffness", f"{stiffness:.1f}")
        joint.set("damping", f"{config.spring_damping:.1f}")
        joint.set("springref", "0")

        # Keycap geom
        half_col = (width_u * KEY_PITCH - KEY_GAP) / 2
        geom = ET.SubElement(key_body, "geom")
        geom.set("name", f"key_{key_name}_geom")
        geom.set("type", "box")
        geom.set("size", f"{KEY_HALF:.4f} {half_col:.4f} {KEY_H:.4f}")
        geom.set("rgba", _rgba_str(key_color))
        geom.set("contype", str(config.key_contype))
        geom.set("conaffinity", str(config.key_conaffinity))
        geom.set("mass", "0.005")

        # Label site — small dot on top of the keycap for visual identification
        site = ET.SubElement(key_body, "site")
        site.set("name", f"key_{key_name}_site")
        site.set("pos", f"0 0 {KEY_H:.4f}")
        site.set("size", "0.002")
        site.set("rgba", _rgba_str(config.label_color))
        site.set("group", "1")

    return base