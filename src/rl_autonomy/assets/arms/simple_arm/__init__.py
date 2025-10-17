from pathlib import Path

import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel

_ASSET_PATH = Path(__file__).resolve().parent / "simple_arm.xml"


class SimpleArm(ManipulatorModel):
    """Minimal two-DOF planar arm backed by the simple_arm.xml asset."""

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(str(_ASSET_PATH), idn=idn)
        # Slight damping on joints to stabilize motion
        self.set_joint_attribute(attrib="damping", values=np.array((0.05, 0.05)))

    @property
    def default_base(self):
        return "NullMount"

    @property
    def default_gripper(self):
        return {"right": None}

    @property
    def default_controller_config(self):
        return {"right": "default_simplearm"}

    @property
    def init_qpos(self):
        return np.zeros(2)

    @property
    def base_xpos_offset(self):
        # Keeps the base centered regardless of arena preset
        return {
            "bins": (0.0, 0.0, 0.0),
            "empty": (0.0, 0.0, 0.0),
            "table": lambda table_length: (-table_length / 2 + 0.1, 0.0, 0.0),
        }

    @property
    def top_offset(self):
        return np.array((0.0, 0.0, 0.02))

    @property
    def _horizontal_radius(self):
        return 0.25

    @property
    def arm_type(self):
        return "single"
