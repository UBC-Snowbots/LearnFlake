import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel

_ASSET_DIR = Path(__file__).resolve().parent
_ASSET_PATH = _ASSET_DIR / "faive_hand.xml"
_STRUCTURE_PATH = _ASSET_DIR / "faive_structure.xml"
_METADATA_PATH = _ASSET_DIR / "faive_metadata.xml"
_RESOLVED_PATH = _ASSET_DIR / "_faive_hand_resolved.xml"


def _expand_includes(element: ET.Element, base_dir: Path):
    """
    Replace MuJoCo-style <include> tags with the contents of the referenced XML files so the robo-
    suite loader sees a single fully expanded document.
    """
    for idx, child in enumerate(list(element)):
        if child.tag == "include" and child.get("file"):
            include_path = (base_dir / child.get("file")).resolve()
            include_root = ET.parse(include_path).getroot()
            element.remove(child)
            for offset, sub in enumerate(list(include_root)):
                element.insert(idx + offset, sub)
                _expand_includes(sub, include_path.parent)
        else:
            _expand_includes(child, base_dir)


def _ensure_resolved_asset() -> Path:
    """
    Lazily materialize a flattened Faive Hand MJCF so ManipulatorModel can parse it.
    """
    deps = [_ASSET_PATH, _STRUCTURE_PATH, _METADATA_PATH]
    latest_src_mtime = max(path.stat().st_mtime for path in deps if path.exists())

    if _RESOLVED_PATH.exists() and _RESOLVED_PATH.stat().st_mtime >= latest_src_mtime:
        return _RESOLVED_PATH

    tree = ET.parse(_ASSET_PATH)
    root = tree.getroot()
    _expand_includes(root, _ASSET_DIR)
    tree.write(_RESOLVED_PATH, encoding="utf-8", xml_declaration=True)
    return _RESOLVED_PATH


class FaiveHand(ManipulatorModel):
    """Thin wrapper around the Faive hand MJCF so it behaves like other robosuite manipulators."""

    arms = ["right"]

    def __init__(self, idn=0):
        resolved_xml = _ensure_resolved_asset()
        super().__init__(str(resolved_xml), idn=idn)
        # Light damping to keep the many finger joints numerically stable
        self.set_joint_attribute(attrib="damping", values=np.full(self.dof, 0.05))

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
        return np.zeros(self.dof)

    @property
    def base_xpos_offset(self):
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

    @property
    def _eef_name(self):
        # Palm body acts as a proxy end-effector for robosuite's manipulator plumbing
        return {"right": "palm"}
