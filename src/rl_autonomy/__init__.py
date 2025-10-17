"""
Top-level package initialization for rl_autonomy.

Ensures the vendored robosuite package is importable and registers custom robots.
"""

from pathlib import Path
import sys


def _ensure_robosuite_on_path():
    """
    Import robosuite if available, otherwise add the vendored copy to sys.path.
    """
    try:
        import robosuite  # noqa: F401
    except ModuleNotFoundError:  # pragma: no cover - executed only when robosuite isn't installed
        robo_root = Path(__file__).resolve().parent / "RoboSuite"
        if str(robo_root) not in sys.path:
            sys.path.append(str(robo_root))
        import robosuite  # noqa: F401,E402


_ensure_robosuite_on_path()

from .assets.arms.faive_hand_p0 import FaiveHand  # noqa: F401,E402
from .assets.arms.simple_arm import SimpleArm  # noqa: F401,E402

from robosuite.robots import ROBOT_CLASS_MAPPING, FixedBaseRobot

ROBOT_CLASS_MAPPING.setdefault("SimpleArm", FixedBaseRobot)
ROBOT_CLASS_MAPPING.setdefault("FaiveHand", FixedBaseRobot)

__all__ = ["SimpleArm", "FaiveHand"]
