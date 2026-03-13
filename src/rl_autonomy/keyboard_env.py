"""
KeyboardEnv — base MuJoCo environment for the keyboard typing pipeline.

Inherits from ManipulationEnv (RoboSuite) and adds:
  - Keyboard scene (5 test keys: Q W E R T) injected onto the table
  - Solenoid actuator observations (extended flag, velocity)
  - Rangefinder sensor
  - Synthesized ArUco observation (dx, dy, key_visible) — matches real pipeline
  - Contact force from cfrc_ext on actuator tip body
  - set_target_key(name) for the orchestrator / skill environments

Skill-specific reward functions and termination logic live in subclasses
(CoarseReachEnv, FineAlignEnv, PressKeyEnv) created in later phases.
"""

import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
from collections import OrderedDict

ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "external_pkgs", "RoboSuite")
if os.path.exists(ROBO_PATH) and ROBO_PATH not in sys.path:
    sys.path.insert(0, ROBO_PATH)

import robosuite as suite
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import EmptyArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)

# ---------------------------------------------------------------------------
# Keyboard layout (Phase 2: 5 test keys)
# ---------------------------------------------------------------------------
# ~70% of real keyboard scale.  Positions are relative to the keyboard base
# centre, in the table body frame.
# Full key layout can be expanded here without touching any other code.

KEY_PITCH = 0.019   # centre-to-centre spacing (m)
KEY_HALF  = 0.007   # key half-size in XY (m)
KEY_H     = 0.002   # key half-height (m)

# (name, x_offset_from_row_centre)
PHASE2_KEYS = [
    ("q", -2),
    ("w", -1),
    ("e",  0),
    ("r", +1),
    ("t", +2),
]

# ArUco synthesizer constants
ARUCO_NOISE_STD      = 0.001   # ~1 mm standard deviation
ARUCO_VISIBLE_DIST   = 0.05    # guaranteed visible within 5 cm XY
ARUCO_FALLOFF_DIST   = 0.12    # fully invisible beyond 12 cm XY
ARUCO_MAX_TILT       = 0.30    # radians; above this detection degrades

# Contact detection thresholds (mirrors PressKeyEnv and real Moteus logic)
CONTACT_FORCE_THRESHOLD = 2.0   # N
STALL_VEL_THRESHOLD     = 0.005 # m/s


class KeyboardEnv(ManipulationEnv):
    """
    Base keyboard environment.  Provides the full observation vector and
    sensor pipeline used by all skill subclasses.

    Args:
        keyboard_offset (2-tuple): (x, y) offset of the keyboard centre from
            the table centre (in table-body-frame XY, metres).
        render (bool): enable interactive viewer.
        use_camera_obs (bool): include rendered camera images in obs dict.
        horizon (int): max steps per episode.
    """

    # Keys available in Phase 2 (expanded to full layout later)
    AVAILABLE_KEYS = [k for k, _ in PHASE2_KEYS]

    def __init__(
        self,
        keyboard_offset=(0.15, 0.0),
        render=False,
        use_camera_obs=False,
        horizon=500,
        **kwargs,
    ):
        self.keyboard_offset = np.array(keyboard_offset)

        arm_cfg = suite.load_part_controller_config(default_controller="JOINT_VELOCITY")
        ctrl_cfg = refactor_composite_controller_config(arm_cfg, "Rover2026", ["right"])

        # Internal state
        self._target_key   = "e"          # default: centre key
        self._contact_steps = 0           # debounce counter for PressKey

        super().__init__(
            robots=["Rover2026"],
            controller_configs=ctrl_cfg,
            has_renderer=render,
            has_offscreen_renderer=True,  # always on — eef_cam may be inspected
            use_camera_obs=use_camera_obs,
            render_camera="frontview",
            control_freq=20,
            horizon=horizon,
            ignore_done=False,
            hard_reset=True,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_target_key(self, key_name: str):
        """Select which key the current skill should operate on."""
        assert key_name in self.AVAILABLE_KEYS, (
            f"Unknown key '{key_name}'. Available: {self.AVAILABLE_KEYS}"
        )
        self._target_key = key_name

    @property
    def obs_dim(self):
        """Flat observation vector length."""
        return len(self._flat_obs())

    def get_obs(self) -> dict:
        """Return the full labeled observation dictionary."""
        return self._get_observations(force_update=True)

    # ------------------------------------------------------------------
    # ManipulationEnv overrides
    # ------------------------------------------------------------------

    def _load_model(self):
        super()._load_model()

        mujoco_arena = EmptyArena()

        # White floor, no walls
        for geom in list(mujoco_arena.worldbody.findall("geom")):
            if "wall" in geom.get("name", ""):
                mujoco_arena.worldbody.remove(geom)
        floor = mujoco_arena.floor
        floor.attrib.pop("material", None)
        floor.set("rgba", "1 1 1 1")

        # Place robot base at floor level
        xpos = self.robots[0].robot_model.base_xpos_offset["empty"]
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Keyboard sits directly on the floor in world space
        keyboard_body = self._build_keyboard()
        mujoco_arena.worldbody.append(keyboard_body)

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=[],
        )

    def _setup_references(self):
        super()._setup_references()

        sim = self.sim

        # Key body IDs
        self._key_body_ids = {}
        for key_name, _ in PHASE2_KEYS:
            body_name = f"key_{key_name}"
            self._key_body_ids[key_name] = sim.model.body_name2id(body_name)

        # Solenoid / actuator
        actuator_body_name = None
        for i in range(sim.model.nbody):
            name = sim.model.body_id2name(i)
            if "linear_actuator" in name:
                actuator_body_name = name
                break
        assert actuator_body_name, "linear_actuator body not found in model"
        self._actuator_body_id = sim.model.body_name2id(actuator_body_name)

        # Solenoid joint address in qpos/qvel
        actuator_joint_name = None
        for i in range(sim.model.njnt):
            name = sim.model.joint_id2name(i)
            if "actuator_slide" in name:
                actuator_joint_name = name
                break
        assert actuator_joint_name, "actuator_slide joint not found in model"
        joint_id = sim.model.joint_name2id(actuator_joint_name)
        self._actuator_qpos_addr = sim.model.jnt_qposadr[joint_id]
        self._actuator_qvel_addr = sim.model.jnt_dofadr[joint_id]

        # Rangefinder sensor
        rangefinder_name = None
        for i in range(sim.model.nsensor):
            name = sim.model.sensor_id2name(i)
            if "eef_rangefinder" in name:
                rangefinder_name = name
                break
        assert rangefinder_name, "eef_rangefinder sensor not found in model"
        sensor_id = sim.model.sensor_name2id(rangefinder_name)
        # sensor_adr gives the correct offset into sensordata[], which is NOT
        # the same as the sensor id (multi-dim sensors like force/torque occupy
        # multiple consecutive slots).
        self._rangefinder_data_addr = sim.model.sensor_adr[sensor_id]

        # EEF site (for tip world position used in ArUco synthesis)
        self._eef_site_id = self.robots[0].eef_site_id["right"]

    def _setup_observables(self):
        observables = super()._setup_observables()

        pf = self.robots[0].robot_model.naming_prefix  # e.g. "robot0_"

        @sensor(modality="keyboard")
        def target_key_pos(obs_cache):
            return np.array(
                self.sim.data.body_xpos[self._key_body_ids[self._target_key]]
            )

        @sensor(modality="keyboard")
        def eef_to_key(obs_cache):
            eef_pos = obs_cache.get(f"{pf}eef_pos", np.zeros(3))
            key_pos = obs_cache.get("target_key_pos", np.zeros(3))
            return key_pos - eef_pos

        @sensor(modality="keyboard")
        def aruco_obs(obs_cache):
            return self._get_aruco_observation(obs_cache, pf)

        @sensor(modality="keyboard")
        def actuator_extended(obs_cache):
            # Binary: 0.0 = retracted, 1.0 = extended
            pos = self.sim.data.qpos[self._actuator_qpos_addr]
            return np.array([1.0 if pos > 0.02 else 0.0])

        @sensor(modality="keyboard")
        def actuator_vel(obs_cache):
            return np.array([self.sim.data.qvel[self._actuator_qvel_addr]])

        @sensor(modality="keyboard")
        def rangefinder(obs_cache):
            return np.array([self.sim.data.sensordata[self._rangefinder_data_addr]])

        @sensor(modality="keyboard")
        def contact_force(obs_cache):
            """
            cfrc_ext magnitude on actuator tip body.
            In the full pipeline this signal goes through synthetic_moteus_node
            (Phase 3).  Here it's read directly for Phase 2 obs validation.
            """
            force_vec = self.sim.data.cfrc_ext[self._actuator_body_id][3:]
            return np.array([float(np.linalg.norm(force_vec))])

        for obs_name, obs_sensor in [
            ("target_key_pos",  target_key_pos),
            ("eef_to_key",      eef_to_key),
            ("aruco_obs",       aruco_obs),
            ("actuator_extended", actuator_extended),
            ("actuator_vel",    actuator_vel),
            ("rangefinder",     rangefinder),
            ("contact_force",   contact_force),
        ]:
            observables[obs_name] = Observable(
                name=obs_name,
                sensor=obs_sensor,
                sampling_rate=self.control_freq,
            )

        return observables

    def reward(self, action=None):
        # Base class returns 0; skill subclasses override this.
        return 0.0

    def _check_success(self):
        return False

    def _reset_internal(self):
        super()._reset_internal()
        self._contact_steps = 0

    # ------------------------------------------------------------------
    # Sensor helpers
    # ------------------------------------------------------------------

    def _get_aruco_observation(self, obs_cache, pf) -> np.ndarray:
        """
        Synthesise the ArUco pipeline output: (dx, dy, key_visible).

        dx, dy are the X and Y components of the EEF-to-key vector expressed
        in the EEF frame — exactly what the real aruco_detector node gives
        (key position in camera frame projected to XY).

        Includes:
          - Realistic measurement noise (~1 mm std)
          - Detection failure when key is far away or EEF is too tilted
        """
        eef_pos  = obs_cache.get(f"{pf}eef_pos",  np.zeros(3))
        eef_quat = obs_cache.get(f"{pf}eef_quat", np.array([1, 0, 0, 0]))
        key_pos  = obs_cache.get("target_key_pos", np.zeros(3))

        # Vector from EEF to key in world frame
        diff_world = key_pos - eef_pos
        dist_xy    = np.linalg.norm(diff_world[:2])

        # Probability of successful detection
        eef_tilt   = self._eef_tilt_from_vertical(eef_quat)
        if dist_xy < ARUCO_VISIBLE_DIST and eef_tilt < ARUCO_MAX_TILT:
            p_visible = 1.0
        else:
            p_visible = max(0.0, 1.0 - dist_xy / ARUCO_FALLOFF_DIST)
            p_visible *= max(0.0, 1.0 - eef_tilt / (ARUCO_MAX_TILT * 2))

        key_visible = float(np.random.rand() < p_visible)

        if not key_visible:
            return np.array([0.0, 0.0, 0.0])

        # Rotate world-frame diff into EEF frame (w, x, y, z → rotation matrix)
        R_eef = self._quat_to_rot(eef_quat)          # world←EEF
        diff_eef = R_eef.T @ diff_world               # EEF frame

        # Add realistic ArUco noise
        noise = np.random.normal(0.0, ARUCO_NOISE_STD, size=2)
        return np.array([diff_eef[0] + noise[0], diff_eef[1] + noise[1], 1.0])

    @staticmethod
    def _eef_tilt_from_vertical(quat_wxyz: np.ndarray) -> float:
        """Return the angle (rad) between EEF Z-axis and world -Z (downward)."""
        R = KeyboardEnv._quat_to_rot(quat_wxyz)
        eef_z_world = R[:, 2]                         # EEF Z in world
        down = np.array([0, 0, -1.0])
        cos_a = np.clip(np.dot(eef_z_world, down), -1.0, 1.0)
        return float(np.arccos(cos_a))

    @staticmethod
    def _quat_to_rot(quat_wxyz: np.ndarray) -> np.ndarray:
        """Convert (w, x, y, z) quaternion to 3×3 rotation matrix."""
        w, x, y, z = quat_wxyz
        return np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)],
        ])

    def _flat_obs(self) -> np.ndarray:
        """Return the flat observation vector used by policies."""
        obs = self._get_observations(force_update=True)
        pf = self.robots[0].robot_model.naming_prefix
        parts = [
            obs.get(f"{pf}joint_pos",        np.zeros(6)),
            obs.get(f"{pf}joint_vel",        np.zeros(6)),
            obs.get(f"{pf}eef_pos",          np.zeros(3)),
            obs.get(f"{pf}eef_quat",         np.zeros(4)),
            obs.get("actuator_extended",     np.zeros(1)),
            obs.get("target_key_pos",        np.zeros(3)),
            obs.get("eef_to_key",            np.zeros(3)),
            obs.get("rangefinder",           np.zeros(1)),
            obs.get("contact_force",         np.zeros(1)),
            obs.get("actuator_vel",          np.zeros(1)),
            obs.get("aruco_obs",             np.zeros(3)),
        ]
        return np.concatenate([np.array(p).flatten() for p in parts])

    # ------------------------------------------------------------------
    # Keyboard XML builder
    # ------------------------------------------------------------------

    def _build_keyboard(self) -> ET.Element:
        """
        Build the keyboard body tree as ElementTree elements.
        Positions are in world space:
          - floor is at z = 0
          - keyboard_offset is the (x, y) world position of the keyboard centre
        """
        kx, ky = self.keyboard_offset
        kz = KEY_H  # keyboard base top flush with floor

        base = ET.Element("body")
        base.set("name", "keyboard_base")
        base.set("pos", f"{kx:.4f} {ky:.4f} {kz:.4f}")

        # Base slab (collision + visual)
        row_half_x = (len(PHASE2_KEYS) - 1) * KEY_PITCH / 2 + KEY_HALF + 0.003
        slab = ET.SubElement(base, "geom")
        slab.set("name", "keyboard_surface")
        slab.set("type", "box")
        slab.set("size",  f"{row_half_x:.4f} 0.014 {KEY_H:.4f}")
        slab.set("rgba",  "0.15 0.15 0.15 1")
        slab.set("contype", "1")
        slab.set("conaffinity", "1")

        # Individual keys
        for key_name, col_idx in PHASE2_KEYS:
            x_local = col_idx * KEY_PITCH

            key_body = ET.SubElement(base, "body")
            key_body.set("name", f"key_{key_name}")
            key_body.set("pos",  f"{x_local:.4f} 0 {KEY_H * 2:.4f}")

            geom = ET.SubElement(key_body, "geom")
            geom.set("name",        f"key_{key_name}_geom")
            geom.set("type",        "box")
            geom.set("size",        f"{KEY_HALF} {KEY_HALF} {KEY_H}")
            geom.set("rgba",        "0.88 0.88 0.88 1")
            geom.set("contype",     "1")
            geom.set("conaffinity", "1")

            site = ET.SubElement(key_body, "site")
            site.set("name",  f"key_{key_name}_site")
            site.set("pos",   f"0 0 {KEY_H:.4f}")
            site.set("size",  "0.003")
            site.set("rgba",  "0 1 0 0.4")
            site.set("group", "1")

        return base


# ===========================================================================
# Skill environments
# ===========================================================================

class CoarseReachEnv(KeyboardEnv):
    """
    Skill 1: Move EEF to within 3 cm XY and correct height above target key.

    Actuator is locked retracted throughout — only the 6 arm joints are used.
    Episode terminates on success or after `horizon` steps (default 300).
    """

    SUCCESS_XY   = 0.03   # m — XY tolerance for success
    SUCCESS_Z    = 0.015  # m — height error tolerance
    HOVER_HEIGHT = 0.05   # m — target hover height above key surface

    def __init__(self, random_key: bool = True, **kwargs):
        kwargs.setdefault('horizon', 300)
        self.random_key = random_key
        super().__init__(**kwargs)

    def _reset_internal(self):
        super()._reset_internal()
        if self.random_key:
            self.set_target_key(np.random.choice(self.AVAILABLE_KEYS))

    def reward(self, action=None):
        eef_pos = self.sim.data.site_xpos[self._eef_site_id]
        key_pos = self.sim.data.body_xpos[self._key_body_ids[self._target_key]]

        xy_dist = float(np.linalg.norm(eef_pos[:2] - key_pos[:2]))
        z_error = float(abs((eef_pos[2] - key_pos[2]) - self.HOVER_HEIGHT))

        if xy_dist < self.SUCCESS_XY and z_error < self.SUCCESS_Z:
            return 100.0

        r_reach  = 10.0 * np.exp(-5.0 * xy_dist)
        r_height = -2.0 * z_error
        return r_reach + r_height

    def _check_success(self):
        eef_pos = self.sim.data.site_xpos[self._eef_site_id]
        key_pos = self.sim.data.body_xpos[self._key_body_ids[self._target_key]]
        xy_dist = float(np.linalg.norm(eef_pos[:2] - key_pos[:2]))
        z_error = float(abs((eef_pos[2] - key_pos[2]) - self.HOVER_HEIGHT))
        return xy_dist < self.SUCCESS_XY and z_error < self.SUCCESS_Z
