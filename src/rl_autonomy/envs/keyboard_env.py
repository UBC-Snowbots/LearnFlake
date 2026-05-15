"""KeyboardEnv — single mode-switched robosuite env for keyboard typing.

Replaces the legacy 9-class hierarchy in `src/rl_autonomy/keyboard_env.py`
with one class parameterized by `mode='approach'|'strike'`. See TRACKER §6
(action space), §5 (reward), §9 (observation) for the design.

This module exposes:
    KeyboardEnv            -- robosuite-native ManipulationEnv subclass
    KeyboardGymEnv         -- gymnasium.Env wrapper over KeyboardEnv
    make_env(mode, ...)    -- factory returning a fully-wrapped gym.Env
                              (frame-stack, action smoothing, optional DR)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

# robosuite must be on the path before this module is imported. The package
# does that via `src/external_pkgs/RoboSuite` being a submodule; pip-installing
# rl_autonomy doesn't pull robosuite in (it's a vendored dep), so we add it
# to sys.path here for safety in standalone scripts.
_ROBO_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "external_pkgs", "RoboSuite")
)
if os.path.exists(_ROBO_PATH):
    import sys as _sys
    if _ROBO_PATH not in _sys.path:
        _sys.path.insert(0, _ROBO_PATH)

import robosuite as suite  # noqa: E402
from robosuite.environments.manipulation.manipulation_env import (  # noqa: E402
    ManipulationEnv,
)
from robosuite.models.arenas import EmptyArena  # noqa: E402
from robosuite.models.tasks import ManipulationTask  # noqa: E402
from robosuite.utils.observables import Observable, sensor  # noqa: E402

from ..configs import CONTROLLER_JP_PATH
from .keyboard_layout import (
    ARUCO_FALLOFF_DIST,
    ARUCO_MAX_TILT,
    ARUCO_NOISE_STD,
    ARUCO_VISIBLE_DIST,
    AVAILABLE_KEYS,
    CONTACT_FORCE_THRESHOLD,
    KEYBOARD_LAYOUT,
    STALL_VEL_THRESHOLD,
)
from .keyboard_mjcf import build_keyboard_body
from .rewards import (
    approach_potential,
    approach_reward,
    approach_success,
    pbrs_term,
    strike_reward,
)


Mode = Literal["approach", "strike"]


# Joint config that places the EEF roughly vertical above the keyboard,
# 5 cm above the home row. Used for default reset; per-key randomization
# is layered on top by the curriculum (see TRACKER §7).
ABOVE_KEYBOARD_QPOS = np.array([
    -2.3448,   # shoulder_joint
    -1.3110,   # link_1_joint
     0.4222,   # link1_link2
    -0.0107,   # a4_rotation
     1.7332,   # a5_rotation
     0.7774,   # a6_rotation
])

HOVER_HEIGHT = 0.05  # m above key surface where the EEF parks for a strike


# ---------------------------------------------------------------------------
# Helpers — rotation conversions kept here (no internal state) so they're
# easy to unit-test in isolation.
# ---------------------------------------------------------------------------

def quat_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    """(w,x,y,z) quaternion → 3x3 rotation matrix."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_6d(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → 6-D continuous representation (Zhou+ CVPR 2019).

    First two columns of R, flattened. The third column is recovered by
    cross product + Gram-Schmidt orthonormalization in any consumer that
    needs the full matrix back. Continuous, no double-cover issue.
    """
    return R[:, :2].T.reshape(6)


def eef_tilt_from_vertical(q_wxyz: np.ndarray) -> float:
    """Angle (rad) between the solenoid push direction and world -Z.

    Matches the legacy convention: actuator pushes along EEF -Y when the
    arm is in its nominal vertical configuration (verified against the
    Rover2026 model). Returns 0 when push direction is exactly down.
    """
    R = quat_to_rot(q_wxyz)
    push_dir = -R[:, 1]                     # EEF -Y axis in world frame
    cos_a = np.clip(np.dot(push_dir, np.array([0.0, 0.0, -1.0])), -1.0, 1.0)
    return float(np.arccos(cos_a))


# ---------------------------------------------------------------------------
# Robosuite-native env
# ---------------------------------------------------------------------------

class KeyboardEnv(ManipulationEnv):
    """Single-class env for the keyboard typing task.

    The mode parameter changes:
      - episode horizon
      - which dimensions of the action vector the env executes (others are
        zeroed by the action wrapper, but the env doesn't enforce that —
        that's the wrapper's job)
      - reward function
      - termination criterion

    The simulation, observation pipeline, and PBRS bookkeeping are common.
    """

    AVAILABLE_KEYS = AVAILABLE_KEYS

    def __init__(
        self,
        mode: Mode = "approach",
        keyboard_offset: tuple[float, float] = (-0.15, 0.0),
        keyboard_height: float = 0.15,
        random_key: bool = True,
        horizon: int | None = None,
        render: bool = False,
        use_camera_obs: bool = False,
        gamma: float = 0.99,
        **kwargs: Any,
    ) -> None:
        if mode not in ("approach", "strike"):
            raise ValueError(f"mode must be 'approach' or 'strike', got {mode!r}")

        self.mode: Mode = mode
        self.keyboard_offset = np.array(keyboard_offset)
        self.keyboard_height = float(keyboard_height)
        self.random_key = bool(random_key)
        self._gamma = float(gamma)

        self._target_key = "g"            # default; randomized in reset() if random_key
        self._contact_steps = 0           # used by Strike
        self._prev_action: np.ndarray | None = None
        self._prev_potential: float | None = None
        self._collision_flag = False

        if horizon is None:
            horizon = 200 if mode == "approach" else 50

        ctrl_cfg = suite.load_composite_controller_config(controller=CONTROLLER_JP_PATH)

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
    # Public API used by tests and tools
    # ------------------------------------------------------------------

    def set_target_key(self, key_name: str) -> None:
        if key_name not in AVAILABLE_KEYS:
            raise ValueError(f"unknown key {key_name!r}")
        self._target_key = key_name

    @property
    def target_key(self) -> str:
        return self._target_key

    @property
    def hover_height(self) -> float:
        return HOVER_HEIGHT

    def get_obs_dict(self) -> dict[str, np.ndarray]:
        """Return the labelled observation dict — see envs/obs_adapter.py."""
        return self._build_obs_dict()

    # ------------------------------------------------------------------
    # ManipulationEnv overrides
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        super()._load_model()

        arena = EmptyArena()

        # White floor, no walls
        for geom in list(arena.worldbody.findall("geom")):
            if "wall" in geom.get("name", ""):
                arena.worldbody.remove(geom)
        floor = arena.floor
        floor.attrib.pop("material", None)
        floor.set("rgba", "1 1 1 1")

        # Place robot at floor level
        xpos = self.robots[0].robot_model.base_xpos_offset["empty"]
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Inject keyboard
        arena.worldbody.append(
            build_keyboard_body(offset_xy=tuple(self.keyboard_offset), height=self.keyboard_height)
        )

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=[],
        )

    def _setup_references(self) -> None:
        super()._setup_references()
        sim = self.sim
        robot = self.robots[0]

        self._key_body_ids: dict[str, int] = {
            name: sim.model.body_name2id(f"key_{name}") for name, *_ in KEYBOARD_LAYOUT
        }

        # Solenoid actuator — find the body and joint by name pattern
        self._actuator_body_id = self._find_body_by_substring(sim, "linear_actuator")
        self._actuator_qpos_addr, self._actuator_qvel_addr = (
            self._find_joint_addr_by_substring(sim, "actuator_slide")
        )

        # Rangefinder
        self._rangefinder_data_addr = self._find_sensor_data_addr(sim, "eef_rangefinder")

        # EEF site
        self._eef_site_id = robot.eef_site_id["right"]

        # Cache the arm-joint qpos/qvel indices (numpy fancy-indexing arrays)
        # so _build_obs_dict can read sim.data directly without paying for
        # robosuite's observable pipeline. Profile showed _get_observations
        # was 3-5% of training wallclock; this cuts it out.
        self._arm_qpos_idx = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
        self._arm_qvel_idx = np.asarray(robot._ref_joint_vel_indexes, dtype=np.int64)

        # Note: we read EEF orientation from site_xmat (and convert to quat
        # via robosuite's transform_utils.mat2quat) instead of body_xquat.
        # body_xquat would be a hair faster but it ignores any rotation
        # offset the site has w.r.t. its parent body, which silently breaks
        # if the MJCF ever adds an <euler> to the eef site. The site_xmat
        # path is universally correct.
        import robosuite.utils.transform_utils as _T  # noqa: F401
        self._mat2quat = _T.mat2quat

        # For checking collisions with the keyboard surface
        self._keyboard_geom_ids: set[int] = {
            sim.model.geom_name2id("keyboard_surface"),
        }
        for name, *_ in KEYBOARD_LAYOUT:
            try:
                self._keyboard_geom_ids.add(sim.model.geom_name2id(f"key_{name}_geom"))
            except Exception:
                pass

    @staticmethod
    def _find_body_by_substring(sim, needle: str) -> int:
        for i in range(sim.model.nbody):
            name = sim.model.body_id2name(i)
            if name is not None and needle in name:
                return sim.model.body_name2id(name)
        raise RuntimeError(f"body containing {needle!r} not found in model")

    @staticmethod
    def _find_joint_addr_by_substring(sim, needle: str) -> tuple[int, int]:
        for i in range(sim.model.njnt):
            name = sim.model.joint_id2name(i)
            if name is not None and needle in name:
                jid = sim.model.joint_name2id(name)
                return sim.model.jnt_qposadr[jid], sim.model.jnt_dofadr[jid]
        raise RuntimeError(f"joint containing {needle!r} not found in model")

    @staticmethod
    def _find_sensor_data_addr(sim, needle: str) -> int:
        for i in range(sim.model.nsensor):
            name = sim.model.sensor_id2name(i)
            if name is not None and needle in name:
                return sim.model.sensor_adr[sim.model.sensor_name2id(name)]
        raise RuntimeError(f"sensor containing {needle!r} not found in model")

    def _setup_observables(self) -> dict:
        # Keep robosuite's default observables but DON'T add custom ones —
        # we build the full observation dict in `_build_obs_dict()` directly.
        # That keeps the obs schema in one place (this module) instead of
        # spread across robosuite's Observable framework.
        return super()._setup_observables()

    def _reset_internal(self) -> None:
        # Default init pose is "above keyboard"; curricula override via
        # robot.init_qpos before super()._reset_internal() runs.
        # Init perturbation ±0.02 rad (≈±1.1°) per joint keeps the EEF tilt
        # under 4° at reset, which is well inside the 5° success threshold so
        # a position-only controller can converge. The state-replay curriculum
        # (TRACKER §7) reintroduces wider diversity from demo states later.
        robot = self.robots[0]
        robot.init_qpos = ABOVE_KEYBOARD_QPOS.copy()
        robot.init_qpos += np.random.uniform(-0.02, 0.02, size=6)

        super()._reset_internal()

        self._contact_steps = 0
        self._prev_action = None
        self._prev_potential = None
        self._collision_flag = False
        if self.random_key:
            self.set_target_key(np.random.choice(AVAILABLE_KEYS))

    # ------------------------------------------------------------------
    # Reward + termination
    # ------------------------------------------------------------------

    def reward(self, action: Optional[np.ndarray] = None) -> float:
        if self.mode == "approach":
            return self._approach_reward(action)
        return self._strike_reward(action)

    def _approach_reward(self, action: Optional[np.ndarray]) -> float:
        xy_dist, z_error, tilt = self._compute_approach_errors()
        action_arr = np.zeros(7) if action is None else np.asarray(action, dtype=np.float64)
        action_delta = (
            float(np.linalg.norm(action_arr - self._prev_action))
            if self._prev_action is not None else 0.0
        )
        success = approach_success(xy_dist, z_error, tilt)
        collision = self._detect_collision()
        components = approach_reward(
            xy_dist=xy_dist,
            z_error=z_error,
            tilt=tilt,
            action_delta=action_delta,
            success=success,
            collision=collision,
        )

        # PBRS layered on top of tolerance shaping (TRACKER §5.3).
        phi_now = approach_potential(xy_dist, z_error, tilt)
        if self._prev_potential is None:
            r_pbrs = 0.0
        else:
            r_pbrs = pbrs_term(phi_s=self._prev_potential, phi_s_prime=phi_now, gamma=self._gamma)
        self._prev_potential = phi_now
        self._prev_action = action_arr
        self._collision_flag = collision

        return float(components.total + r_pbrs)

    def _strike_reward(self, action: Optional[np.ndarray]) -> float:
        in_contact = self._in_contact()
        if action is None:
            extending = False
        else:
            extending = float(action[-1]) > 0.0
        actuator_ext = float(self.sim.data.qpos[self._actuator_qpos_addr])

        components, done, new_hold = strike_reward(
            in_contact=in_contact,
            hold_counter=self._contact_steps,
            actuator_extension=actuator_ext,
            extending=extending,
        )
        self._contact_steps = new_hold
        # Note: `done` is consumed via `_check_success` rather than mutating
        # `self.done` directly; robosuite checks success after step()'s reward.
        return float(components.total)

    def _check_success(self) -> bool:
        if self.mode == "approach":
            xy_dist, z_error, tilt = self._compute_approach_errors()
            return approach_success(xy_dist, z_error, tilt)
        # Strike: success is encoded by hold counter reaching threshold
        return self._contact_steps >= 3

    # ------------------------------------------------------------------
    # Internal observation pieces
    # ------------------------------------------------------------------

    def _compute_approach_errors(self) -> tuple[float, float, float]:
        """Errors used for reward + success — measured in **world** frame.

        The OBSERVATION uses EEF-frame target offset (TRACKER §9.1 #6) for
        translation invariance, but the success criterion is "the actuator tip
        is X mm above the key" which is a world-frame question. Mixing the
        two introduces a phantom error of ~hover_height·sin(tilt) on each
        axis when the EEF is tilted, which makes a 5° tilt produce a 4.4 mm
        ghost Z error even when the tip is perfectly placed.
        """
        eef_pos, eef_quat, key_pos = self._eef_and_target()
        xy_dist = float(np.linalg.norm(eef_pos[:2] - key_pos[:2]))
        z_error = float(abs((eef_pos[2] - key_pos[2]) - HOVER_HEIGHT))
        tilt = eef_tilt_from_vertical(eef_quat)
        return xy_dist, z_error, tilt

    def _eef_and_target(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Direct sim.data reads — skip the observable pipeline entirely.

        sim.step() already calls mj_forward, so site_xpos / site_xmat are
        current. Going through self._get_observations(force_update=False)
        is functionally equivalent but adds robosuite's per-observable
        bookkeeping cost on each call. Profile showed this mattered.
        """
        sim = self.sim
        eef_pos = sim.data.site_xpos[self._eef_site_id]
        eef_quat = self._mat2quat(sim.data.site_xmat[self._eef_site_id].reshape(3, 3))
        key_pos = sim.data.body_xpos[self._key_body_ids[self._target_key]]
        return eef_pos, eef_quat, key_pos

    def _in_contact(self) -> bool:
        force = float(np.linalg.norm(self.sim.data.cfrc_ext[self._actuator_body_id][3:]))
        vel = float(self.sim.data.qvel[self._actuator_qvel_addr])
        return force > CONTACT_FORCE_THRESHOLD and abs(vel) < STALL_VEL_THRESHOLD

    def _detect_collision(self) -> bool:
        """True iff the actuator tip body is in contact with any keyboard geom."""
        sim = self.sim
        n_contact = sim.data.ncon
        for i in range(n_contact):
            c = sim.data.contact[i]
            if c.geom1 in self._keyboard_geom_ids or c.geom2 in self._keyboard_geom_ids:
                # Filter for contacts involving the actuator body — tip touching
                # the keyboard surface counts; key body touching key body doesn't.
                b1 = sim.model.geom_bodyid[c.geom1]
                b2 = sim.model.geom_bodyid[c.geom2]
                if b1 == self._actuator_body_id or b2 == self._actuator_body_id:
                    return True
        return False

    # ------------------------------------------------------------------
    # Full observation dict (consumed by obs_adapter)
    # ------------------------------------------------------------------

    def _build_obs_dict(self) -> dict[str, np.ndarray]:
        """Single source of truth for the observation schema.

        Optimized: reads sim.data directly (no robosuite observable pipeline).
        Profile showed _get_observations(force_update=True) was a 3–5% hit on
        training wallclock through _update_observables. sim.step() already
        ran mj_forward, so all sim.data fields below are up-to-date.

        Keys (all np.float32 arrays):
            joint_pos              (6,)
            joint_vel              (6,)
            eef_pos                (3,)         in world frame
            eef_quat               (4,)         (w, x, y, z)
            actuator_extended      (1,)         binary 0.0 / 1.0
            actuator_pos           (1,)         continuous (privileged)
            actuator_vel           (1,)
            target_key_pos_world   (3,)
            target_offset_eef      (3,)         goal vector in EEF frame
            aruco_obs              (3,)         (Δx, Δy, visible)
            rangefinder            (1,)
            contact_force_norm     (1,)         scalar magnitude
            contact_force_vec      (3,)         (privileged)
            tilt_rad               (1,)         angle from vertical
        """
        sim = self.sim
        d = sim.data

        # Direct sim.data reads — no robosuite observable bookkeeping.
        joint_pos = d.qpos[self._arm_qpos_idx].astype(np.float32, copy=False)
        joint_vel = d.qvel[self._arm_qvel_idx].astype(np.float32, copy=False)
        eef_pos = d.site_xpos[self._eef_site_id].astype(np.float32, copy=False)
        eef_quat = self._mat2quat(d.site_xmat[self._eef_site_id].reshape(3, 3)).astype(np.float32)

        actuator_pos = float(d.qpos[self._actuator_qpos_addr])
        actuator_vel = float(d.qvel[self._actuator_qvel_addr])
        actuator_extended = 1.0 if actuator_pos > 0.02 else 0.0

        key_pos_w = d.body_xpos[self._key_body_ids[self._target_key]].astype(np.float32, copy=False)
        offset_world = key_pos_w - eef_pos
        R = quat_to_rot(eef_quat)
        offset_eef = (R.T @ offset_world).astype(np.float32)

        aruco_obs = self._aruco_synth(eef_pos, eef_quat, key_pos_w)

        rangefinder = float(d.sensordata[self._rangefinder_data_addr])

        contact_vec = d.cfrc_ext[self._actuator_body_id][3:].astype(np.float32, copy=False)
        contact_norm = float(np.linalg.norm(contact_vec))

        tilt = eef_tilt_from_vertical(eef_quat)

        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
            "actuator_extended": np.array([actuator_extended], dtype=np.float32),
            "actuator_pos": np.array([actuator_pos], dtype=np.float32),
            "actuator_vel": np.array([actuator_vel], dtype=np.float32),
            "target_key_pos_world": key_pos_w,
            "target_offset_eef": offset_eef,
            "aruco_obs": aruco_obs,
            "rangefinder": np.array([rangefinder], dtype=np.float32),
            "contact_force_norm": np.array([contact_norm], dtype=np.float32),
            "contact_force_vec": contact_vec,
            "tilt_rad": np.array([tilt], dtype=np.float32),
        }

    def _aruco_synth(
        self,
        eef_pos: np.ndarray,
        eef_quat: np.ndarray,
        key_pos: np.ndarray,
    ) -> np.ndarray:
        """Synthesize the (Δx, Δy, visible) signal an aruco detector would produce.

        See TRACKER §11.4 / `documentation/keyboard_typing_pipeline.md`. Visibility
        is a Bernoulli draw whose probability falls off with distance + tilt.
        """
        diff_world = key_pos - eef_pos
        dist_xy = float(np.linalg.norm(diff_world[:2]))
        tilt = eef_tilt_from_vertical(eef_quat)

        if dist_xy < ARUCO_VISIBLE_DIST and tilt < ARUCO_MAX_TILT:
            p_visible = 1.0
        else:
            p_visible = max(0.0, 1.0 - dist_xy / ARUCO_FALLOFF_DIST)
            p_visible *= max(0.0, 1.0 - tilt / (ARUCO_MAX_TILT * 2))

        visible = float(np.random.rand() < p_visible)
        if not visible:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        R = quat_to_rot(eef_quat)
        diff_eef = R.T @ diff_world
        noise = np.random.normal(0.0, ARUCO_NOISE_STD, size=2)
        return np.array([diff_eef[0] + noise[0], diff_eef[1] + noise[1], 1.0],
                        dtype=np.float32)


# ---------------------------------------------------------------------------
# Gymnasium wrapper + factory — populated when the obs_adapter / action_adapter
# / domain_rand modules are imported at the top of __init__.py.
# ---------------------------------------------------------------------------

def make_env(
    mode: Mode = "approach",
    *,
    render: bool = False,
    random_key: bool = True,
    domain_rand: bool = False,
    frame_stack: int = 3,
    horizon: int | None = None,
    seed: int | None = None,
):
    """Build a fully-wrapped gym.Env ready for training.

    Wrapper stack (outer → inner):
        DomainRandWrapper (optional) →
            ObsAdapter (dict obs → flat actor + critic arrays, RunningMeanStd) →
                FrameStackWrapper (k=frame_stack) →
                    ActionAdapter (mode-aware masking + first-order smoothing) →
                        KeyboardGymEnv (gymnasium.Env) →
                            KeyboardEnv (robosuite-native)
    """
    # Local imports avoid circular references between wrappers and the env.
    from .action_adapter import ActionAdapter
    from .obs_adapter import KeyboardGymEnv, ObsAdapter, FrameStackWrapper
    from .domain_rand import DomainRandWrapper

    base = KeyboardEnv(
        mode=mode,
        render=render,
        random_key=random_key,
        horizon=horizon,
    )
    gym_env = KeyboardGymEnv(base, mode=mode, seed=seed)
    env = ActionAdapter(gym_env, mode=mode)
    env = FrameStackWrapper(env, k=frame_stack)
    env = ObsAdapter(env)
    if domain_rand:
        env = DomainRandWrapper(env)
    return env
