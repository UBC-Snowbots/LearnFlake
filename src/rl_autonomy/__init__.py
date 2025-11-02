from .assets.arms.faive_hand_p0 import FaiveHand
from .assets.arms.simple_arm import SimpleArm

from robosuite.robots import ROBOT_CLASS_MAPPING, FixedBaseRobot

ROBOT_CLASS_MAPPING.setdefault("SimpleArm", FixedBaseRobot)
ROBOT_CLASS_MAPPING.setdefault("FaiveHand", FixedBaseRobot)

__all__ = ["SimpleArm", "FaiveHand"]