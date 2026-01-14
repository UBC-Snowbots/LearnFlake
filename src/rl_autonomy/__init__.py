from .assets.arms.really_simple_robot import ReallySimpleArm
from .assets.arms.simple_arm import SimpleArm

from robosuite.robots import ROBOT_CLASS_MAPPING, FixedBaseRobot

ROBOT_CLASS_MAPPING.setdefault("SimpleArm", FixedBaseRobot)
ROBOT_CLASS_MAPPING.setdefault("ReallySimpleArm", FixedBaseRobot)

__all__ = ["SimpleArm", "ReallySimpleArm"]