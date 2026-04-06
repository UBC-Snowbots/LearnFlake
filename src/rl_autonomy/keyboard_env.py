"""
KeyboardEnv — base MuJoCo environment for the keyboard typing pipeline.

Inherits from ManipulationEnv (RoboSuite) and adds:
  - Keyboard scene (Redragon K552 TKL, 87 keys) injected onto the table
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
# Keyboard layout — Redragon K552 TKL (87 keys)
# ---------------------------------------------------------------------------
# Standard 19 mm key pitch.  Positions are relative to the keyboard body
# centre, computed from a U-based layout (1 U = KEY_PITCH).
#
# Coordinate mapping (keyboard body local frame):
#   X_local = row direction  (pos = back / F-keys, neg = front / space bar)
#   Y_local = column direction (pos = left / Esc side, neg = right / nav cluster)
#   Keyboard faces the robot (space bar row nearest to arm base).

KEY_PITCH = 0.019   # centre-to-centre spacing (m), standard 19 mm
KEY_HALF  = 0.009   # half-size of a 1 U key cap in XY (m)
KEY_H     = 0.002   # key half-height (m)
KEY_GAP   = 0.001   # gap between adjacent key caps (m)

# TKL bounding box in U
_KB_WIDTH_U  = 18.25   # total columns (with nav cluster)
_KB_HEIGHT_U = 6.5      # total rows (F-row + 0.5U gap + 5 main rows)
_KB_CENTER_COL = _KB_WIDTH_U / 2    # 9.125
_KB_CENTER_ROW = _KB_HEIGHT_U / 2   # 3.25


def _build_tkl_layout():
    """Compute Redragon K552 TKL layout.

    Returns list of (name, x_local_m, y_local_m, width_u).
    """
    layout = []

    def add_row(row_u, keys):
        col = 0.0
        for name, w in keys:
            if name is not None:
                # Negate both axes → 180° rotation so keyboard faces the arm
                cx = -((row_u - _KB_CENTER_ROW) * KEY_PITCH)
                cy = -((col + w / 2.0 - _KB_CENTER_COL) * KEY_PITCH)
                layout.append((name, cx, cy, w))
            col += w

    # Row 0 — F-key row (center at 0.5 U from top)
    add_row(0.5, [
        ("esc", 1), (None, 1),
        ("f1", 1), ("f2", 1), ("f3", 1), ("f4", 1), (None, 0.5),
        ("f5", 1), ("f6", 1), ("f7", 1), ("f8", 1), (None, 0.5),
        ("f9", 1), ("f10", 1), ("f11", 1), ("f12", 1), (None, 0.25),
        ("prtsc", 1), ("scrlk", 1), ("pause", 1),
    ])

    # Row 1 — Number row (2.0 U)
    add_row(2.0, [
        ("grave", 1), ("1", 1), ("2", 1), ("3", 1), ("4", 1), ("5", 1),
        ("6", 1), ("7", 1), ("8", 1), ("9", 1), ("0", 1),
        ("minus", 1), ("equal", 1), ("backspace", 2), (None, 0.25),
        ("ins", 1), ("home", 1), ("pgup", 1),
    ])

    # Row 2 — QWERTY row (3.0 U)
    add_row(3.0, [
        ("tab", 1.5), ("q", 1), ("w", 1), ("e", 1), ("r", 1), ("t", 1),
        ("y", 1), ("u", 1), ("i", 1), ("o", 1), ("p", 1),
        ("lbracket", 1), ("rbracket", 1), ("backslash", 1.5), (None, 0.25),
        ("del", 1), ("end", 1), ("pgdn", 1),
    ])

    # Row 3 — Home row (4.0 U)
    add_row(4.0, [
        ("caps", 1.75), ("a", 1), ("s", 1), ("d", 1), ("f", 1), ("g", 1),
        ("h", 1), ("j", 1), ("k", 1), ("l", 1),
        ("semicolon", 1), ("quote", 1), ("enter", 2.25),
    ])

    # Row 4 — Shift row (5.0 U)
    add_row(5.0, [
        ("lshift", 2.25), ("z", 1), ("x", 1), ("c", 1), ("v", 1), ("b", 1),
        ("n", 1), ("m", 1), ("comma", 1), ("period", 1), ("slash", 1),
        ("rshift", 2.75), (None, 1.25),
        ("up", 1),
    ])

    # Row 5 — Space bar row (6.0 U)
    add_row(6.0, [
        ("lctrl", 1.25), ("win", 1.25), ("lalt", 1.25),
        ("space", 6.25),
        ("ralt", 1.25), ("fn", 1.25), ("menu", 1.25), ("rctrl", 1.25),
        (None, 0.25),
        ("left", 1), ("down", 1), ("right", 1),
    ])

    return layout


TKL_KEYS = _build_tkl_layout()

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

    AVAILABLE_KEYS = [name for name, _, _, _ in TKL_KEYS]

    def __init__(
        self,
        keyboard_offset=(-0.15, 0.0),
        keyboard_height=0.15,
        render=False,
        use_camera_obs=False,
        horizon=500,
        **kwargs,
    ):
        self.keyboard_offset = np.array(keyboard_offset)
        self.keyboard_height = keyboard_height  # top of keyboard surface (m)

        arm_cfg = suite.load_part_controller_config(default_controller="JOINT_VELOCITY")
        ctrl_cfg = refactor_composite_controller_config(arm_cfg, "Rover2026", ["right"])

        # Internal state
        self._target_key   = "g"          # default: near keyboard centre
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

        # Keyboard at self.keyboard_height above the floor
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
        for key_name, _, _, _ in TKL_KEYS:
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
        """Return the angle (rad) between the solenoid push direction and world -Z.

        The Rover2026 solenoid pushes along EEF -Y in world frame (confirmed
        by grip_site → actuator_tip_site vector pointing along world -Z when
        the arm is in its nominal vertical configuration).
        """
        R = KeyboardEnv._quat_to_rot(quat_wxyz)
        push_dir = -R[:, 1]                           # EEF -Y in world
        down = np.array([0, 0, -1.0])
        cos_a = np.clip(np.dot(push_dir, down), -1.0, 1.0)
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
        Build the Redragon K552 TKL keyboard as MuJoCo XML elements.

        Local frame of the keyboard body:
          X = row direction  (neg = back / F-keys, pos = front / space)
          Y = column direction (neg = left / Esc, pos = right / nav cluster)
          Z = up
        """
        kx, ky = self.keyboard_offset
        kz = self.keyboard_height

        base = ET.Element("body")
        base.set("name", "keyboard_base")
        base.set("pos", f"{kx:.4f} {ky:.4f} {kz:.4f}")

        # Base slab — covers full TKL footprint with small margin
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
        for key_name, x_local, y_local, width_u in TKL_KEYS:
            key_body = ET.SubElement(base, "body")
            key_body.set("name", f"key_{key_name}")
            key_body.set("pos", f"{x_local:.4f} {y_local:.4f} {KEY_H * 3:.4f}")

            # Width varies per key; height is always 1 U
            half_col = (width_u * KEY_PITCH - KEY_GAP) / 2
            geom = ET.SubElement(key_body, "geom")
            geom.set("name",        f"key_{key_name}_geom")
            geom.set("type",        "box")
            geom.set("size",        f"{KEY_HALF} {half_col:.4f} {KEY_H}")
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
    Skill 1: Move EEF to within 3 cm XY and correct height above target key,
    with the EEF oriented perfectly horizontal (solenoid pointing straight down).

    Actuator is locked retracted throughout — only the 6 arm joints are used.
    Episode terminates on success or after `horizon` steps (default 300).
    """

    SUCCESS_XY   = 0.03   # m — XY tolerance for success
    SUCCESS_Z    = 0.015  # m — height error tolerance
    SUCCESS_TILT = 0.15   # rad (~8.6°) — max tilt from vertical for success
    HOVER_HEIGHT = 0.05   # m — target hover height above key surface

    # Joint config that places the EEF roughly vertical above the keyboard.
    # Used for a fraction of resets so the agent discovers what "good" looks like.
    ABOVE_KEYBOARD_QPOS = np.array([
        -2.3448,  # shoulder_joint
        -1.3110,  # link_1_joint
         0.4222,  # link1_link2
        -0.0107,  # a4_rotation
         1.7332,  # a5_rotation
         0.7774,  # a6_rotation
    ])

    def __init__(self, random_key: bool = True, easy_init_frac: float = 0.3,
                 **kwargs):
        kwargs.setdefault('horizon', 300)
        self.random_key = random_key
        self.easy_init_frac = easy_init_frac
        super().__init__(**kwargs)

    def _reset_internal(self):
        # Before super() resets the sim, optionally override init_qpos
        # so the arm starts near-vertical above the keyboard
        if np.random.rand() < self.easy_init_frac:
            robot = self.robots[0]
            robot.init_qpos = self.ABOVE_KEYBOARD_QPOS.copy()
            # Add small noise so it's not always identical
            robot.init_qpos += np.random.uniform(-0.05, 0.05, size=6)
        else:
            # Restore default so normal episodes still train generalization
            robot = self.robots[0]
            robot.init_qpos = robot.robot_model.init_qpos

        super()._reset_internal()
        if self.random_key:
            self.set_target_key(np.random.choice(self.AVAILABLE_KEYS))

    def reward(self, action=None):
        obs = self._get_observations(force_update=True)
        pf = self.robots[0].robot_model.naming_prefix

        eef_pos  = self.sim.data.site_xpos[self._eef_site_id]
        eef_quat = obs.get(f"{pf}eef_quat", np.array([1, 0, 0, 0]))
        key_pos  = self.sim.data.body_xpos[self._key_body_ids[self._target_key]]

        xy_dist = float(np.linalg.norm(eef_pos[:2] - key_pos[:2]))
        z_error = float(abs((eef_pos[2] - key_pos[2]) - self.HOVER_HEIGHT))
        tilt    = float(self._eef_tilt_from_vertical(eef_quat))

        # Success: position + orientation
        if (xy_dist < self.SUCCESS_XY and z_error < self.SUCCESS_Z
                and tilt < self.SUCCESS_TILT):
            return 100.0
        
        # ADD - jitter

        # Reach: exponential for close range + linear for long range gradient
        r_reach  = 10.0 * np.exp(-5.0 * xy_dist) - 15.0 * xy_dist
        r_height = -2.0 * z_error
        r_orient = -3.0 * tilt
        return r_reach + r_height + r_orient

    def _check_success(self):
        obs = self._get_observations(force_update=True)
        pf = self.robots[0].robot_model.naming_prefix

        eef_pos  = self.sim.data.site_xpos[self._eef_site_id]
        eef_quat = obs.get(f"{pf}eef_quat", np.array([1, 0, 0, 0]))
        key_pos  = self.sim.data.body_xpos[self._key_body_ids[self._target_key]]

        xy_dist = float(np.linalg.norm(eef_pos[:2] - key_pos[:2]))
        z_error = float(abs((eef_pos[2] - key_pos[2]) - self.HOVER_HEIGHT))
        tilt    = float(self._eef_tilt_from_vertical(eef_quat))
        return (xy_dist < self.SUCCESS_XY and z_error < self.SUCCESS_Z
                and tilt < self.SUCCESS_TILT)


class FineAlignEnv(KeyboardEnv):
    """
    Skill 2: Precisely center the actuator tip directly above the target key,
    within 5mm XY error, starting from CoarseReach's end state (~3cm away).

    Uses proprioceptive observations + synthesized ArUco (dx, dy, key_visible)
    for precision alignment. Actuator is locked retracted — only 6 arm joints.
    Episode terminates on success or after `horizon` steps (default 200).
    """

    SUCCESS_XY   = 0.005  # m — 5mm XY tolerance
    SUCCESS_Z    = 0.015  # m — height error tolerance
    SUCCESS_TILT = 0.15   # rad (~8.6°) — max tilt from vertical
    HOVER_HEIGHT = CoarseReachEnv.HOVER_HEIGHT  # same hover height

    # Start from known-good pose above keyboard with noise to simulate
    # CoarseReach output (~3cm scatter)
    ALIGNED_QPOS = CoarseReachEnv.ABOVE_KEYBOARD_QPOS.copy()

    def __init__(self, random_key: bool = True, init_noise: float = 0.03,
                 **kwargs):
        kwargs.setdefault('horizon', 200)
        self.random_key = random_key
        self.init_noise = init_noise  # joint noise to simulate CoarseReach scatter
        super().__init__(**kwargs)

    def _reset_internal(self):
        # Start near the aligned pose with small noise — CoarseReach
        # outputs a near-vertical pose so we don't add much perturbation
        robot = self.robots[0]
        robot.init_qpos = self.ALIGNED_QPOS.copy()
        robot.init_qpos += np.random.uniform(
            -self.init_noise, self.init_noise, size=6
        )

        super()._reset_internal()
        if self.random_key:
            self.set_target_key(np.random.choice(self.AVAILABLE_KEYS))

    def reward(self, action=None):
        obs = self._get_observations(force_update=True)
        pf = self.robots[0].robot_model.naming_prefix

        eef_pos  = self.sim.data.site_xpos[self._eef_site_id]
        eef_quat = obs.get(f"{pf}eef_quat", np.array([1, 0, 0, 0]))
        key_pos  = self.sim.data.body_xpos[self._key_body_ids[self._target_key]]

        xy_dist = float(np.linalg.norm(eef_pos[:2] - key_pos[:2]))
        z_diff  = float((eef_pos[2] - key_pos[2]) - self.HOVER_HEIGHT)  # positive = above, negative = below
        z_error = abs(z_diff)
        tilt    = float(self._eef_tilt_from_vertical(eef_quat))

        # Success — large terminal bonus + early-termination saves penalty
        if (xy_dist < self.SUCCESS_XY and z_error < self.SUCCESS_Z
                and tilt < self.SUCCESS_TILT):
            return 500.0

        # ArUco visibility bonus
        aruco = obs.get("aruco_obs", np.zeros(3))
        aruco_visible = float(aruco[2])

        # Tight exponential on XY — 50x decay rate (vs 5x for CoarseReach)
        # Peaks at 20.0 when perfectly aligned, decays to ~0 at 3cm
        r_align = 20.0 * np.exp(-50.0 * xy_dist)

        # Asymmetric height penalty — going below hover height risks collision
        # with keys, so penalize 5x harder than being too high
        if z_diff < 0:
            r_height = -15.0 * z_error   # below hover → collision danger
        else:
            r_height = -3.0 * z_error    # above hover → just inefficient

        # Orientation — light penalty; init state is already near-vertical
        # so this just prevents drift, not a hard shaping signal
        r_orient = -2.0 * tilt

        # Small time penalty to encourage finishing quickly
        r_time = -0.5

        # Bonus for ArUco visibility
        r_aruco = 2.0 * aruco_visible

        return r_align + r_height + r_orient + r_time + r_aruco

    def _check_success(self):
        obs = self._get_observations(force_update=True)
        pf = self.robots[0].robot_model.naming_prefix

        eef_pos  = self.sim.data.site_xpos[self._eef_site_id]
        eef_quat = obs.get(f"{pf}eef_quat", np.array([1, 0, 0, 0]))
        key_pos  = self.sim.data.body_xpos[self._key_body_ids[self._target_key]]

        xy_dist = float(np.linalg.norm(eef_pos[:2] - key_pos[:2]))
        z_error = float(abs((eef_pos[2] - key_pos[2]) - self.HOVER_HEIGHT))
        tilt    = float(self._eef_tilt_from_vertical(eef_quat))
        return (xy_dist < self.SUCCESS_XY and z_error < self.SUCCESS_Z
                and tilt < self.SUCCESS_TILT)


class PressKeyEnv(KeyboardEnv):
    """
    Skill 3: Extend the solenoid to press the target key and hold for 3 steps.

    Assumes the arm is already aligned above the key (FineAlign has run).
    Only the solenoid actuator (action[-1]) is active; arm joints are locked to 0.
    Episode terminates on 3-step contact hold (success) or timeout (50 steps).
    """

    HOLD_STEPS = 3   # consecutive contact steps required for success

    # Known-good joint config: arm vertical above keyboard centre.
    ALIGNED_QPOS = CoarseReachEnv.ABOVE_KEYBOARD_QPOS.copy()

    def __init__(self, random_key: bool = True, **kwargs):
        kwargs.setdefault('horizon', 50)
        self.random_key = random_key
        super().__init__(**kwargs)

    def _reset_internal(self):
        robot = self.robots[0]
        robot.init_qpos = self.ALIGNED_QPOS.copy()
        robot.init_qpos += np.random.uniform(-0.02, 0.02, size=6)

        super()._reset_internal()
        self._contact_steps = 0
        if self.random_key:
            self.set_target_key(np.random.choice(self.AVAILABLE_KEYS))

    def reward(self, action=None):
        act_pos = float(self.sim.data.qpos[self._actuator_qpos_addr])
        act_vel = float(self.sim.data.qvel[self._actuator_qvel_addr])
        force   = float(np.linalg.norm(
            self.sim.data.cfrc_ext[self._actuator_body_id][3:]
        ))

        contact = (force > CONTACT_FORCE_THRESHOLD
                   and abs(act_vel) < STALL_VEL_THRESHOLD)

        if contact:
            self._contact_steps += 1
            if self._contact_steps >= self.HOLD_STEPS:
                return 1000.0
            return 500.0
        else:
            self._contact_steps = 0

        r_extend   = 5.0 * act_pos
        r_stability = -2.0 * float(np.linalg.norm(action[:6])) if action is not None else 0.0
        return r_extend + r_stability

    def _check_success(self):
        return self._contact_steps >= self.HOLD_STEPS


class FinePressEnv(KeyboardEnv):
    """
    Fine press: precise center-of-key contact with force control.

    Tighter init from FineAlign output (±0.01 joint noise), requires 5
    consecutive hold steps, and penalizes off-center contact (XY drift
    during press) and force overshoot beyond the contact threshold.
    """

    HOLD_STEPS = 5
    SUCCESS_XY = 0.003     # m — must stay within 3mm of key center during press
    MAX_FORCE  = 8.0       # N — penalize force overshoot above this

    ALIGNED_QPOS = CoarseReachEnv.ABOVE_KEYBOARD_QPOS.copy()

    def __init__(self, random_key: bool = True, **kwargs):
        kwargs.setdefault('horizon', 80)
        self.random_key = random_key
        super().__init__(**kwargs)

    def _reset_internal(self):
        robot = self.robots[0]
        robot.init_qpos = self.ALIGNED_QPOS.copy()
        robot.init_qpos += np.random.uniform(-0.01, 0.01, size=6)

        super()._reset_internal()
        self._contact_steps = 0
        if self.random_key:
            self.set_target_key(np.random.choice(self.AVAILABLE_KEYS))

    def reward(self, action=None):
        act_pos = float(self.sim.data.qpos[self._actuator_qpos_addr])
        act_vel = float(self.sim.data.qvel[self._actuator_qvel_addr])
        force   = float(np.linalg.norm(
            self.sim.data.cfrc_ext[self._actuator_body_id][3:]
        ))

        # XY drift during press
        eef_pos = self.sim.data.site_xpos[self._eef_site_id]
        key_pos = self.sim.data.body_xpos[self._key_body_ids[self._target_key]]
        xy_drift = float(np.linalg.norm(eef_pos[:2] - key_pos[:2]))

        contact = (force > CONTACT_FORCE_THRESHOLD
                   and abs(act_vel) < STALL_VEL_THRESHOLD)

        if contact:
            self._contact_steps += 1
            if self._contact_steps >= self.HOLD_STEPS:
                # Bonus scaled by centering precision
                centering_bonus = 200.0 * np.exp(-100.0 * xy_drift)
                return 1000.0 + centering_bonus
            return 500.0
        else:
            self._contact_steps = 0

        r_extend    = 5.0 * act_pos
        r_stability = -2.0 * float(np.linalg.norm(action[:6])) if action is not None else 0.0
        r_center    = -20.0 * xy_drift  # penalize XY drift during approach
        r_force_overshoot = -2.0 * max(0.0, force - self.MAX_FORCE)

        return r_extend + r_stability + r_center + r_force_overshoot

    def _check_success(self):
        return self._contact_steps >= self.HOLD_STEPS


class RetractEnv(KeyboardEnv):
    """
    Skill 4: Retract solenoid and lift EEF back to hover height after pressing.

    Starts from a post-press state (solenoid extended, arm near key surface).
    Actions: 6 arm joints + 1 solenoid command (7-dim).
    Success: solenoid fully retracted AND EEF back at hover height AND minimal tilt.
    Episode terminates on success or after `horizon` steps (default 100).
    """

    SUCCESS_Z     = 0.015   # m — height error tolerance from hover
    SUCCESS_TILT  = 0.15    # rad — max tilt from vertical
    HOVER_HEIGHT  = CoarseReachEnv.HOVER_HEIGHT
    RETRACT_THRESHOLD = 0.005  # m — solenoid position below this = retracted

    ALIGNED_QPOS = CoarseReachEnv.ABOVE_KEYBOARD_QPOS.copy()

    def __init__(self, random_key: bool = True, **kwargs):
        kwargs.setdefault('horizon', 100)
        self.random_key = random_key
        super().__init__(**kwargs)

    def _reset_internal(self):
        robot = self.robots[0]
        robot.init_qpos = self.ALIGNED_QPOS.copy()
        # Small noise to simulate post-press arm variation
        robot.init_qpos += np.random.uniform(-0.02, 0.02, size=6)

        super()._reset_internal()
        self._contact_steps = 0

        # Simulate post-press state: extend the solenoid at reset
        self.sim.data.qpos[self._actuator_qpos_addr] = 0.03  # extended
        self.sim.forward()

        if self.random_key:
            self.set_target_key(np.random.choice(self.AVAILABLE_KEYS))

    def reward(self, action=None):
        obs = self._get_observations(force_update=True)
        pf = self.robots[0].robot_model.naming_prefix

        eef_pos  = self.sim.data.site_xpos[self._eef_site_id]
        eef_quat = obs.get(f"{pf}eef_quat", np.array([1, 0, 0, 0]))
        key_pos  = self.sim.data.body_xpos[self._key_body_ids[self._target_key]]

        act_pos = float(self.sim.data.qpos[self._actuator_qpos_addr])
        z_error = float(abs((eef_pos[2] - key_pos[2]) - self.HOVER_HEIGHT))
        tilt    = float(self._eef_tilt_from_vertical(eef_quat))
        retracted = act_pos < self.RETRACT_THRESHOLD

        # Success: solenoid retracted + back at hover height + upright
        if retracted and z_error < self.SUCCESS_Z and tilt < self.SUCCESS_TILT:
            return 200.0

        # Reward shaping
        r_retract = 10.0 * (0.03 - act_pos) / 0.03  # max 10 when fully retracted
        r_height  = -5.0 * z_error
        r_orient  = -2.0 * tilt
        r_time    = -0.5

        # Penalty for contact force — we want to lift off cleanly
        force = float(np.linalg.norm(
            self.sim.data.cfrc_ext[self._actuator_body_id][3:]
        ))
        r_contact = -3.0 * min(force / CONTACT_FORCE_THRESHOLD, 1.0)

        return r_retract + r_height + r_orient + r_time + r_contact

    def _check_success(self):
        obs = self._get_observations(force_update=True)
        pf = self.robots[0].robot_model.naming_prefix

        eef_pos  = self.sim.data.site_xpos[self._eef_site_id]
        eef_quat = obs.get(f"{pf}eef_quat", np.array([1, 0, 0, 0]))
        key_pos  = self.sim.data.body_xpos[self._key_body_ids[self._target_key]]

        act_pos = float(self.sim.data.qpos[self._actuator_qpos_addr])
        z_error = float(abs((eef_pos[2] - key_pos[2]) - self.HOVER_HEIGHT))
        tilt    = float(self._eef_tilt_from_vertical(eef_quat))

        return (act_pos < self.RETRACT_THRESHOLD
                and z_error < self.SUCCESS_Z
                and tilt < self.SUCCESS_TILT)


class FineRetractEnv(KeyboardEnv):
    """
    Fine retract: precise hover recovery with smooth, jerk-free motion.

    Tighter tolerances than RetractEnv (1cm height, 0.1 rad tilt), penalizes
    jerky motion (large action deltas), and requires near-zero EEF velocity
    at termination to ensure a stable handoff to Traverse.
    """

    SUCCESS_Z          = 0.010   # m — tighter than coarse (1cm vs 1.5cm)
    SUCCESS_TILT       = 0.10    # rad — tighter (~5.7° vs ~8.6°)
    SUCCESS_VEL        = 0.02    # m/s — EEF must be nearly stationary
    HOVER_HEIGHT       = CoarseReachEnv.HOVER_HEIGHT
    RETRACT_THRESHOLD  = 0.003   # m — stricter retraction check

    ALIGNED_QPOS = CoarseReachEnv.ABOVE_KEYBOARD_QPOS.copy()

    def __init__(self, random_key: bool = True, **kwargs):
        kwargs.setdefault('horizon', 120)
        self.random_key = random_key
        self._prev_action = None
        super().__init__(**kwargs)

    def _reset_internal(self):
        robot = self.robots[0]
        robot.init_qpos = self.ALIGNED_QPOS.copy()
        robot.init_qpos += np.random.uniform(-0.02, 0.02, size=6)

        super()._reset_internal()
        self._contact_steps = 0
        self._prev_action = None

        # Simulate post-press state
        self.sim.data.qpos[self._actuator_qpos_addr] = 0.03
        self.sim.forward()

        if self.random_key:
            self.set_target_key(np.random.choice(self.AVAILABLE_KEYS))

    def reward(self, action=None):
        obs = self._get_observations(force_update=True)
        pf = self.robots[0].robot_model.naming_prefix

        eef_pos  = self.sim.data.site_xpos[self._eef_site_id]
        eef_quat = obs.get(f"{pf}eef_quat", np.array([1, 0, 0, 0]))
        eef_vel  = obs.get(f"{pf}joint_vel", np.zeros(6))
        key_pos  = self.sim.data.body_xpos[self._key_body_ids[self._target_key]]

        act_pos  = float(self.sim.data.qpos[self._actuator_qpos_addr])
        z_error  = float(abs((eef_pos[2] - key_pos[2]) - self.HOVER_HEIGHT))
        tilt     = float(self._eef_tilt_from_vertical(eef_quat))
        retracted = act_pos < self.RETRACT_THRESHOLD
        vel_mag  = float(np.linalg.norm(eef_vel))

        # Success: retracted + hover + upright + stationary
        if (retracted and z_error < self.SUCCESS_Z
                and tilt < self.SUCCESS_TILT and vel_mag < self.SUCCESS_VEL):
            return 300.0

        r_retract = 10.0 * (0.03 - act_pos) / 0.03
        r_height  = -5.0 * z_error
        r_orient  = -3.0 * tilt
        r_time    = -0.5

        # Smoothness: penalize action jerk (difference from previous action)
        r_jerk = 0.0
        if self._prev_action is not None and action is not None:
            delta = np.array(action) - self._prev_action
            r_jerk = -1.0 * float(np.linalg.norm(delta))
        if action is not None:
            self._prev_action = np.array(action)

        # Penalize contact force
        force = float(np.linalg.norm(
            self.sim.data.cfrc_ext[self._actuator_body_id][3:]
        ))
        r_contact = -3.0 * min(force / CONTACT_FORCE_THRESHOLD, 1.0)

        # Settling bonus: reward low velocity near target state
        r_settle = -2.0 * vel_mag

        return r_retract + r_height + r_orient + r_time + r_jerk + r_contact + r_settle

    def _check_success(self):
        obs = self._get_observations(force_update=True)
        pf = self.robots[0].robot_model.naming_prefix

        eef_pos  = self.sim.data.site_xpos[self._eef_site_id]
        eef_quat = obs.get(f"{pf}eef_quat", np.array([1, 0, 0, 0]))
        eef_vel  = obs.get(f"{pf}joint_vel", np.zeros(6))
        key_pos  = self.sim.data.body_xpos[self._key_body_ids[self._target_key]]

        act_pos  = float(self.sim.data.qpos[self._actuator_qpos_addr])
        z_error  = float(abs((eef_pos[2] - key_pos[2]) - self.HOVER_HEIGHT))
        tilt     = float(self._eef_tilt_from_vertical(eef_quat))
        vel_mag  = float(np.linalg.norm(eef_vel))

        return (act_pos < self.RETRACT_THRESHOLD
                and z_error < self.SUCCESS_Z
                and tilt < self.SUCCESS_TILT
                and vel_mag < self.SUCCESS_VEL)


class TraverseEnv(KeyboardEnv):
    """
    Skill 5: Move laterally from one key to another at hover height.

    Starts hovering above a random source key, must reach hover position above
    a different random target key. This is faster than CoarseReach because
    the arm is already at the correct height and orientation — it just needs
    to slide laterally.

    Actions: 6 arm joints (solenoid locked retracted).
    Success: EEF within 3cm XY of target key at correct hover height.
    The subsequent FineAlign skill handles the last cm of precision.
    Episode terminates on success or after `horizon` steps (default 200).
    """

    SUCCESS_XY   = 0.03    # m — hand off to FineAlign at this tolerance
    SUCCESS_Z    = 0.015   # m — height error tolerance
    SUCCESS_TILT = 0.15    # rad — max tilt from vertical
    HOVER_HEIGHT = CoarseReachEnv.HOVER_HEIGHT
    MIN_Z        = 0.03    # m — minimum height above key (collision safety)

    ALIGNED_QPOS = CoarseReachEnv.ABOVE_KEYBOARD_QPOS.copy()

    def __init__(self, random_key: bool = True, **kwargs):
        kwargs.setdefault('horizon', 200)
        self.random_key = random_key
        self._source_key = "g"
        super().__init__(**kwargs)

    def _reset_internal(self):
        robot = self.robots[0]
        robot.init_qpos = self.ALIGNED_QPOS.copy()
        robot.init_qpos += np.random.uniform(-0.03, 0.03, size=6)

        super()._reset_internal()

        if self.random_key:
            # Pick two different keys: source (where we are) and target (where to go)
            keys = self.AVAILABLE_KEYS
            src, tgt = np.random.choice(keys, size=2, replace=False)
            self._source_key = src
            self.set_target_key(tgt)

    def reward(self, action=None):
        obs = self._get_observations(force_update=True)
        pf = self.robots[0].robot_model.naming_prefix

        eef_pos  = self.sim.data.site_xpos[self._eef_site_id]
        eef_quat = obs.get(f"{pf}eef_quat", np.array([1, 0, 0, 0]))
        key_pos  = self.sim.data.body_xpos[self._key_body_ids[self._target_key]]

        xy_dist = float(np.linalg.norm(eef_pos[:2] - key_pos[:2]))
        z_above_key = float(eef_pos[2] - key_pos[2])
        z_error = float(abs(z_above_key - self.HOVER_HEIGHT))
        tilt    = float(self._eef_tilt_from_vertical(eef_quat))

        # Success
        if (xy_dist < self.SUCCESS_XY and z_error < self.SUCCESS_Z
                and tilt < self.SUCCESS_TILT):
            return 100.0

        # XY approach reward — exponential + linear gradient
        r_reach  = 10.0 * np.exp(-5.0 * xy_dist) - 10.0 * xy_dist

        # Height maintenance — penalize deviations from hover, especially dipping
        if z_above_key < self.MIN_Z:
            r_height = -20.0 * (self.MIN_Z - z_above_key)  # harsh collision penalty
        else:
            r_height = -3.0 * z_error

        r_orient = -2.0 * tilt
        r_time   = -0.3

        return r_reach + r_height + r_orient + r_time

    def _check_success(self):
        obs = self._get_observations(force_update=True)
        pf = self.robots[0].robot_model.naming_prefix

        eef_pos  = self.sim.data.site_xpos[self._eef_site_id]
        eef_quat = obs.get(f"{pf}eef_quat", np.array([1, 0, 0, 0]))
        key_pos  = self.sim.data.body_xpos[self._key_body_ids[self._target_key]]

        xy_dist = float(np.linalg.norm(eef_pos[:2] - key_pos[:2]))
        z_error = float(abs((eef_pos[2] - key_pos[2]) - self.HOVER_HEIGHT))
        tilt    = float(self._eef_tilt_from_vertical(eef_quat))
        return (xy_dist < self.SUCCESS_XY and z_error < self.SUCCESS_Z
                and tilt < self.SUCCESS_TILT)


class FineTraverseEnv(KeyboardEnv):
    """
    Fine traverse: precise key-to-key movement with tight tolerances.

    Success within 1cm XY (vs 3cm coarse), tighter height/tilt, and rewards
    smooth lateral movement. Penalizes jerk to produce trajectories that
    hand off cleanly to FineAlign.
    """

    SUCCESS_XY   = 0.010   # m — 1cm (vs 3cm coarse)
    SUCCESS_Z    = 0.010   # m
    SUCCESS_TILT = 0.10    # rad
    HOVER_HEIGHT = CoarseReachEnv.HOVER_HEIGHT
    MIN_Z        = 0.03    # m — collision safety floor

    ALIGNED_QPOS = CoarseReachEnv.ABOVE_KEYBOARD_QPOS.copy()

    def __init__(self, random_key: bool = True, **kwargs):
        kwargs.setdefault('horizon', 250)
        self.random_key = random_key
        self._source_key = "g"
        self._prev_action = None
        super().__init__(**kwargs)

    def _reset_internal(self):
        robot = self.robots[0]
        robot.init_qpos = self.ALIGNED_QPOS.copy()
        robot.init_qpos += np.random.uniform(-0.02, 0.02, size=6)

        super()._reset_internal()
        self._prev_action = None

        if self.random_key:
            keys = self.AVAILABLE_KEYS
            src, tgt = np.random.choice(keys, size=2, replace=False)
            self._source_key = src
            self.set_target_key(tgt)

    def reward(self, action=None):
        obs = self._get_observations(force_update=True)
        pf = self.robots[0].robot_model.naming_prefix

        eef_pos  = self.sim.data.site_xpos[self._eef_site_id]
        eef_quat = obs.get(f"{pf}eef_quat", np.array([1, 0, 0, 0]))
        key_pos  = self.sim.data.body_xpos[self._key_body_ids[self._target_key]]

        xy_dist     = float(np.linalg.norm(eef_pos[:2] - key_pos[:2]))
        z_above_key = float(eef_pos[2] - key_pos[2])
        z_error     = float(abs(z_above_key - self.HOVER_HEIGHT))
        tilt        = float(self._eef_tilt_from_vertical(eef_quat))

        if (xy_dist < self.SUCCESS_XY and z_error < self.SUCCESS_Z
                and tilt < self.SUCCESS_TILT):
            return 200.0

        r_reach = 15.0 * np.exp(-10.0 * xy_dist) - 10.0 * xy_dist

        if z_above_key < self.MIN_Z:
            r_height = -20.0 * (self.MIN_Z - z_above_key)
        else:
            r_height = -5.0 * z_error

        r_orient = -3.0 * tilt
        r_time   = -0.3

        r_jerk = 0.0
        if self._prev_action is not None and action is not None:
            delta = np.array(action) - self._prev_action
            r_jerk = -0.5 * float(np.linalg.norm(delta))
        if action is not None:
            self._prev_action = np.array(action)

        return r_reach + r_height + r_orient + r_time + r_jerk

    def _check_success(self):
        obs = self._get_observations(force_update=True)
        pf = self.robots[0].robot_model.naming_prefix

        eef_pos  = self.sim.data.site_xpos[self._eef_site_id]
        eef_quat = obs.get(f"{pf}eef_quat", np.array([1, 0, 0, 0]))
        key_pos  = self.sim.data.body_xpos[self._key_body_ids[self._target_key]]

        xy_dist = float(np.linalg.norm(eef_pos[:2] - key_pos[:2]))
        z_error = float(abs((eef_pos[2] - key_pos[2]) - self.HOVER_HEIGHT))
        tilt    = float(self._eef_tilt_from_vertical(eef_quat))
        return (xy_dist < self.SUCCESS_XY and z_error < self.SUCCESS_Z
                and tilt < self.SUCCESS_TILT)


# ===========================================================================
# Domain Randomization Wrapper
# ===========================================================================

class DomainRandWrapper:
    """
    Wraps any KeyboardEnv subclass and randomizes physics/sensor parameters
    on each reset for sim-to-real transfer.

    Randomized parameters:
      - keyboard_offset: XY position shift (+-2cm)
      - keyboard_height: surface height (+-1cm)
      - observation_noise: additive Gaussian scaled by a random factor
      - action_scale: multiplicative scaling of actions (motor gain variation)
      - gravity: +-5% Z-axis variation
      - key_friction: surface friction coefficient randomization
      - control_latency: 0-2 step action delay

    Usage:
        base_env = CoarseReachEnv(render=False)
        env = DomainRandWrapper(base_env)
        obs = env.reset()        # randomizes params then resets
        obs, r, done, info = env.step(action)
    """

    KB_OFFSET_RANGE    = 0.02          # m, +-2cm keyboard XY shift
    KB_HEIGHT_RANGE    = 0.01          # m, +-1cm keyboard Z shift
    OBS_NOISE_RANGE    = (0.5, 2.0)    # multiplier on base noise std
    ACTION_SCALE_RANGE = (0.85, 1.15)  # motor gain variation
    GRAVITY_RANGE      = (9.61, 10.01) # m/s^2, ~+-2.5% of 9.81
    FRICTION_RANGE     = (0.3, 1.5)    # friction coefficient scale
    LATENCY_STEPS      = (0, 3)        # action delay in sim steps

    def __init__(self, env: KeyboardEnv):
        self.env = env
        self._action_buffer = []
        self._latency = 0
        self._action_scale = 1.0
        self._obs_noise_scale = 1.0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        kx_base, ky_base = -0.15, 0.0
        self.env.keyboard_offset = np.array([
            kx_base + np.random.uniform(-self.KB_OFFSET_RANGE, self.KB_OFFSET_RANGE),
            ky_base + np.random.uniform(-self.KB_OFFSET_RANGE, self.KB_OFFSET_RANGE),
        ])
        self.env.keyboard_height = 0.15 + np.random.uniform(
            -self.KB_HEIGHT_RANGE, self.KB_HEIGHT_RANGE
        )

        self._action_scale = np.random.uniform(*self.ACTION_SCALE_RANGE)
        self._obs_noise_scale = np.random.uniform(*self.OBS_NOISE_RANGE)
        self._latency = np.random.randint(self.LATENCY_STEPS[0], self.LATENCY_STEPS[1])
        self._action_buffer = []

        result = self.env.reset(**kwargs)

        # Randomize gravity post-reset
        g = np.random.uniform(*self.GRAVITY_RANGE)
        self.env.sim.model.opt.gravity[2] = -g

        # Randomize key friction
        friction_scale = np.random.uniform(*self.FRICTION_RANGE)
        for i in range(self.env.sim.model.ngeom):
            name = self.env.sim.model.geom_id2name(i)
            if name and name.startswith("key_"):
                self.env.sim.model.geom_friction[i, 0] *= friction_scale

        return result

    def step(self, action):
        scaled_action = np.array(action) * self._action_scale

        self._action_buffer.append(scaled_action)
        if len(self._action_buffer) > self._latency:
            delayed_action = self._action_buffer.pop(0)
        else:
            delayed_action = np.zeros_like(scaled_action)

        obs, reward, done, info = self.env.step(delayed_action)
        info['dr_action_scale'] = self._action_scale
        info['dr_obs_noise'] = self._obs_noise_scale
        info['dr_latency'] = self._latency
        return obs, reward, done, info

    def _flat_obs(self):
        obs = self.env._flat_obs()
        noise = np.random.normal(0, 0.001 * self._obs_noise_scale, size=obs.shape)
        return (obs + noise).astype(np.float32)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()
