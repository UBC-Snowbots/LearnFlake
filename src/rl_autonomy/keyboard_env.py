"""
KeyboardEnv — base MuJoCo environment for the keyboard typing pipeline.

Inherits from ManipulationEnv (RoboSuite) and provides:
    - Pressable keyboard (spring-loaded slide joints per key)
    - Joint-based press detection (displacement past activation threshold)
    - Per-key contact detection with terminal logging
    - Solenoid actuator observations (extended flag, velocity)
    - Rangefinder sensor
    - Synthesized ArUco observation (dx, dy, key_visible)
    - Contact force from cfrc_ext on actuator tip body
    - Domain randomization (keyboard position, key colours, spring stiffness)
    - set_target_key(name) for the orchestrator / skill environments

Skill-specific reward functions live in skill_envs.py.
"""

import os
import sys
import numpy as np
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

from keyboard_layout import (
    TKL_KEYS, AVAILABLE_KEYS, PRESS_DEPTH, PRESS_THRESHOLD,
)
from keyboard_builder import build_keyboard, KeyboardConfig

# ---------------------------------------------------------------------------
# ArUco synthesizer constants
# ---------------------------------------------------------------------------
ARUCO_NOISE_STD    = 0.001   # ~1 mm standard deviation
ARUCO_VISIBLE_DIST = 0.05    # guaranteed visible within 5 cm XY
ARUCO_FALLOFF_DIST = 0.12    # fully invisible beyond 12 cm XY
ARUCO_MAX_TILT     = 0.30    # rad; above this detection degrades

# Contact detection thresholds
CONTACT_FORCE_THRESHOLD = 2.0   # N
STALL_VEL_THRESHOLD     = 0.005 # m/s


class KeyboardEnv(ManipulationEnv):
    """Base keyboard environment with pressable keys and RL-ready observations.

    Parameters
    ----------
    keyboard_offset : tuple of float
        (x, y) offset of the keyboard centre from the table centre (metres).
    keyboard_height : float
        Height of the keyboard surface above the floor (metres).
    render : bool
        Enable interactive MuJoCo viewer.
    use_camera_obs : bool
        Include rendered camera images in observation dict.
    keyboard_config : KeyboardConfig or None
        Keyboard appearance / physics. None uses defaults.
    randomize_keyboard_pos : bool
        Randomize keyboard (x, y) position on each reset.
    keyboard_pos_range : tuple of float
        (dx, dy) max offset from nominal position for randomization (metres).
    randomize_appearance : bool
        Randomize key colours and spring stiffness on each reset.
    log_contacts : bool
        Print key press events to terminal.
    horizon : int
        Max steps per episode.
    """

    AVAILABLE_KEYS = AVAILABLE_KEYS

    def __init__(
        self,
        keyboard_offset=(-0.15, 0.0),
        keyboard_height=0.15,
        render=False,
        use_camera_obs=False,
        keyboard_config=None,
        randomize_keyboard_pos=False,
        keyboard_pos_range=(0.05, 0.05),
        randomize_appearance=False,
        log_contacts=True,
        horizon=500,
        **kwargs,
    ):
        self.keyboard_offset_nominal = np.array(keyboard_offset)
        self.keyboard_offset = self.keyboard_offset_nominal.copy()
        self.keyboard_height = keyboard_height
        self.keyboard_config = keyboard_config or KeyboardConfig()
        self.randomize_keyboard_pos = randomize_keyboard_pos
        self.keyboard_pos_range = np.array(keyboard_pos_range)
        self.randomize_appearance = randomize_appearance
        self.log_contacts = log_contacts

        arm_cfg = suite.load_part_controller_config(
            default_controller="JOINT_VELOCITY"
        )
        ctrl_cfg = refactor_composite_controller_config(
            arm_cfg, "Rover2026", ["right"]
        )

        # Internal state
        self._target_key = "g"
        self._pressed_keys = set()

        super().__init__(
            robots=["Rover2026"],
            controller_configs=ctrl_cfg,
            has_renderer=render,
            has_offscreen_renderer=True,
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
        """Select which key the current skill should target."""
        assert key_name in self.AVAILABLE_KEYS, (
            f"Unknown key '{key_name}'. Available: {self.AVAILABLE_KEYS}"
        )
        self._target_key = key_name

    @property
    def target_key(self) -> str:
        return self._target_key

    @property
    def obs_dim(self) -> int:
        """Flat observation vector length."""
        return len(self._flat_obs())

    def get_obs(self) -> dict:
        """Return the full labeled observation dictionary."""
        return self._get_observations(force_update=True)

    def get_key_displacement(self, key_name: str) -> float:
        """Return how far a key is depressed (positive = deeper, in metres)."""
        addr = self._key_joint_addrs[key_name]
        return -float(self.sim.data.qpos[addr])

    def is_key_pressed(self, key_name: str) -> bool:
        """True if key is depressed past the activation threshold."""
        return self.get_key_displacement(key_name) > PRESS_THRESHOLD

    def get_pressed_keys(self) -> set:
        """Return set of all currently pressed key names."""
        return {
            name for name in self.AVAILABLE_KEYS
            if self.is_key_pressed(name)
        }

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

        # Build the keyboard
        keyboard_body = build_keyboard(
            offset=tuple(self.keyboard_offset),
            height=self.keyboard_height,
            config=self.keyboard_config,
        )
        mujoco_arena.worldbody.append(keyboard_body)

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=[],
        )

    def _setup_references(self):
        super()._setup_references()
        sim = self.sim

        # Key body IDs and joint addresses
        self._key_body_ids = {}
        self._key_joint_addrs = {}
        for key_name, _, _, _ in TKL_KEYS:
            body_name = f"key_{key_name}"
            self._key_body_ids[key_name] = sim.model.body_name2id(body_name)

            joint_name = f"key_{key_name}_slide"
            jid = sim.model.joint_name2id(joint_name)
            self._key_joint_addrs[key_name] = sim.model.jnt_qposadr[jid]

        # Solenoid / actuator body
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
        jid = sim.model.joint_name2id(actuator_joint_name)
        self._actuator_qpos_addr = sim.model.jnt_qposadr[jid]
        self._actuator_qvel_addr = sim.model.jnt_dofadr[jid]

        # Rangefinder sensor
        rangefinder_name = None
        for i in range(sim.model.nsensor):
            name = sim.model.sensor_id2name(i)
            if "eef_rangefinder" in name:
                rangefinder_name = name
                break
        assert rangefinder_name, "eef_rangefinder sensor not found in model"
        sid = sim.model.sensor_name2id(rangefinder_name)
        self._rangefinder_data_addr = sim.model.sensor_adr[sid]

        # EEF site (for tip world position)
        self._eef_site_id = self.robots[0].eef_site_id["right"]

    def _setup_observables(self):
        observables = super()._setup_observables()
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality="keyboard")
        def target_key_pos(obs_cache):
            return np.array(
                self.sim.data.body_xpos[self._key_body_ids[self._target_key]]
            )

        @sensor(modality="keyboard")
        def eef_to_key(obs_cache):
            eef = obs_cache.get(f"{pf}eef_pos", np.zeros(3))
            key = obs_cache.get("target_key_pos", np.zeros(3))
            return key - eef

        @sensor(modality="keyboard")
        def aruco_obs(obs_cache):
            return self._synthesize_aruco(obs_cache, pf)

        @sensor(modality="keyboard")
        def actuator_extended(obs_cache):
            pos = self.sim.data.qpos[self._actuator_qpos_addr]
            return np.array([1.0 if pos > 0.02 else 0.0])

        @sensor(modality="keyboard")
        def actuator_vel(obs_cache):
            return np.array([self.sim.data.qvel[self._actuator_qvel_addr]])

        @sensor(modality="keyboard")
        def rangefinder(obs_cache):
            return np.array(
                [self.sim.data.sensordata[self._rangefinder_data_addr]]
            )

        @sensor(modality="keyboard")
        def contact_force(obs_cache):
            force_vec = self.sim.data.cfrc_ext[self._actuator_body_id][3:]
            return np.array([float(np.linalg.norm(force_vec))])

        @sensor(modality="keyboard")
        def target_key_press_depth(obs_cache):
            """Normalized press depth of the target key (0 = up, 1 = fully pressed)."""
            displacement = self.get_key_displacement(self._target_key)
            return np.array([np.clip(displacement / PRESS_DEPTH, 0.0, 1.0)])

        for name, obs_sensor in [
            ("target_key_pos",        target_key_pos),
            ("eef_to_key",            eef_to_key),
            ("aruco_obs",             aruco_obs),
            ("actuator_extended",     actuator_extended),
            ("actuator_vel",          actuator_vel),
            ("rangefinder",           rangefinder),
            ("contact_force",         contact_force),
            ("target_key_press_depth", target_key_press_depth),
        ]:
            observables[name] = Observable(
                name=name,
                sensor=obs_sensor,
                sampling_rate=self.control_freq,
            )

        return observables

    def reward(self, action=None):
        return 0.0

    def _check_success(self):
        return False

    def _reset_internal(self):
        super()._reset_internal()
        self._pressed_keys = set()

        # Randomize keyboard position
        if self.randomize_keyboard_pos:
            dx = np.random.uniform(
                -self.keyboard_pos_range[0], self.keyboard_pos_range[0]
            )
            dy = np.random.uniform(
                -self.keyboard_pos_range[1], self.keyboard_pos_range[1]
            )
            self.keyboard_offset = self.keyboard_offset_nominal + np.array([dx, dy])
            kb_id = self.sim.model.body_name2id("keyboard_base")
            self.sim.model.body_pos[kb_id][0] = self.keyboard_offset[0]
            self.sim.model.body_pos[kb_id][1] = self.keyboard_offset[1]
            self.sim.forward()

    def step(self, action):
        """Step the environment and detect key presses via joint displacement."""
        obs, reward, done, info = super().step(action)

        # Detect presses from joint displacement (not geom contacts)
        currently_pressed = self.get_pressed_keys()
        newly_pressed = currently_pressed - self._pressed_keys
        newly_released = self._pressed_keys - currently_pressed

        if self.log_contacts:
            for key in newly_pressed:
                depth_mm = self.get_key_displacement(key) * 1000
                print(f"[KEY DOWN] {key}  ({depth_mm:.1f} mm)")
            for key in newly_released:
                print(f"[KEY UP]   {key}")

        self._pressed_keys = currently_pressed

        info["pressed_keys"] = currently_pressed
        info["target_pressed"] = self._target_key in currently_pressed
        info["target_press_depth"] = self.get_key_displacement(self._target_key)

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # ArUco synthesis
    # ------------------------------------------------------------------

    def _synthesize_aruco(self, obs_cache, pf) -> np.ndarray:
        """Synthesize ArUco pipeline output: (dx, dy, key_visible).

        dx, dy are the EEF-to-key vector in EEF frame (what the real
        aruco_detector node outputs). Includes measurement noise and
        detection failure at distance / tilt.
        """
        eef_pos  = obs_cache.get(f"{pf}eef_pos",  np.zeros(3))
        eef_quat = obs_cache.get(f"{pf}eef_quat", np.array([1, 0, 0, 0]))
        key_pos  = obs_cache.get("target_key_pos", np.zeros(3))

        diff_world = key_pos - eef_pos
        dist_xy = np.linalg.norm(diff_world[:2])

        # Detection probability
        tilt = self._eef_tilt_from_vertical(eef_quat)
        if dist_xy < ARUCO_VISIBLE_DIST and tilt < ARUCO_MAX_TILT:
            p_visible = 1.0
        else:
            p_visible = max(0.0, 1.0 - dist_xy / ARUCO_FALLOFF_DIST)
            p_visible *= max(0.0, 1.0 - tilt / (ARUCO_MAX_TILT * 2))

        if np.random.rand() >= p_visible:
            return np.array([0.0, 0.0, 0.0])

        # Rotate into EEF frame and add noise
        R_eef = self._quat_to_rot(eef_quat)
        diff_eef = R_eef.T @ diff_world
        noise = np.random.normal(0.0, ARUCO_NOISE_STD, size=2)
        return np.array([diff_eef[0] + noise[0], diff_eef[1] + noise[1], 1.0])

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _eef_tilt_from_vertical(quat_wxyz: np.ndarray) -> float:
        """Angle (rad) between solenoid push direction and world -Z."""
        R = KeyboardEnv._quat_to_rot(quat_wxyz)
        push_dir = -R[:, 1]  # EEF -Y in world
        cos_a = np.clip(np.dot(push_dir, [0, 0, -1]), -1.0, 1.0)
        return float(np.arccos(cos_a))

    @staticmethod
    def _quat_to_rot(quat_wxyz: np.ndarray) -> np.ndarray:
        """Convert (w, x, y, z) quaternion to 3x3 rotation matrix."""
        w, x, y, z = quat_wxyz
        return np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
        ])

    def _flat_obs(self) -> np.ndarray:
        """Flat observation vector used by policies."""
        obs = self._get_observations(force_update=True)
        pf = self.robots[0].robot_model.naming_prefix
        parts = [
            obs.get(f"{pf}joint_pos",          np.zeros(6)),
            obs.get(f"{pf}joint_vel",          np.zeros(6)),
            obs.get(f"{pf}eef_pos",            np.zeros(3)),
            obs.get(f"{pf}eef_quat",           np.zeros(4)),
            obs.get("actuator_extended",       np.zeros(1)),
            obs.get("target_key_pos",          np.zeros(3)),
            obs.get("eef_to_key",              np.zeros(3)),
            obs.get("rangefinder",             np.zeros(1)),
            obs.get("contact_force",           np.zeros(1)),
            obs.get("actuator_vel",            np.zeros(1)),
            obs.get("aruco_obs",               np.zeros(3)),
            obs.get("target_key_press_depth",  np.zeros(1)),
        ]
        return np.concatenate([np.array(p).flatten() for p in parts])