"""
KeypadLift -- Robosuite Lift environment with 10 static key objects on the table.

Extends the standard Lift env by adding physical key buttons at positions
matching Omega's DEFAULT_KEY_LAYOUT. Keys are static BoxObjects with
collision enabled for contact detection.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

from alpha_env_utils import ensure_alpha_import_paths

ensure_alpha_import_paths()

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask

from rlkb.decider.layouts import DEFAULT_KEY_LAYOUT

# Key visual appearance
KEY_COLORS: Dict[str, Tuple[float, float, float, float]] = {
    "0": (0.85, 0.85, 0.85, 1.0),  # silver
    "1": (0.9, 0.25, 0.25, 1.0),   # red
    "2": (0.25, 0.65, 0.9, 1.0),   # blue
    "3": (0.25, 0.85, 0.35, 1.0),  # green
    "4": (0.9, 0.75, 0.2, 1.0),    # gold
    "5": (0.7, 0.3, 0.85, 1.0),    # purple
    "6": (0.95, 0.5, 0.2, 1.0),    # orange
    "7": (0.3, 0.85, 0.85, 1.0),   # cyan
    "8": (0.9, 0.4, 0.6, 1.0),     # pink
    "9": (0.5, 0.5, 0.5, 1.0),     # grey
}

KEY_HALF_SIZE = (0.008, 0.008, 0.003)  # 16mm × 16mm × 6mm


class KeypadLift(Lift):
    """Lift env with 10 static keys at Omega layout positions."""

    def _load_model(self):
        # Let Lift set up arena, robot, cube, and placement sampler
        super()._load_model()

        # Table surface z = table_offset[2] + table_half_height
        table_surface_z = float(self.table_offset[2] + self.table_full_size[2] / 2.0)
        key_surface_z = table_surface_z + KEY_HALF_SIZE[2]  # key sits on table

        # Create 10 key objects
        self._key_objects: Dict[str, BoxObject] = {}
        self._key_positions: Dict[str, np.ndarray] = {}
        key_list: List[BoxObject] = []

        for key_id, target in sorted(DEFAULT_KEY_LAYOUT.items()):
            rgba = list(KEY_COLORS.get(key_id, (0.3, 0.3, 0.9, 1.0)))
            key_obj = BoxObject(
                name=f"key_{key_id}",
                size=list(KEY_HALF_SIZE),
                rgba=rgba,
                joints=None,     # static — no free joint
                obj_type="all",  # physics-enabled for contact detection
                density=8000,    # heavy so they don't budge
            )
            # Store world-frame position (xy from layout, z on table surface)
            pos = np.array([target.position[0], target.position[1], key_surface_z],
                           dtype=np.float64)
            self._key_objects[key_id] = key_obj
            self._key_positions[key_id] = pos
            key_list.append(key_obj)

        # Re-create the ManipulationTask with keys added alongside the cube
        # We need to grab the arena and robot models from the existing self.model
        self.model = ManipulationTask(
            mujoco_arena=self.model.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cube] + key_list,
        )

    def _reset_internal(self):
        super()._reset_internal()
        # Place each key at its fixed world-frame position
        for key_id, key_obj in self._key_objects.items():
            body_id = self.sim.model.body_name2id(key_obj.root_body)
            self.sim.model.body_pos[body_id] = self._key_positions[key_id]
        self.sim.forward()

    def get_key_position(self, key_id: str) -> np.ndarray:
        """Return the world-frame 3D position of a key."""
        return self._key_positions[key_id].copy()

    def get_all_key_positions(self) -> Dict[str, np.ndarray]:
        """Return all key positions."""
        return {k: v.copy() for k, v in self._key_positions.items()}
