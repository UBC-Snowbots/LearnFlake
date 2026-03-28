"""
Skill environments for the HRL keyboard-typing pipeline.

Each skill is a subclass of KeyboardEnv with its own reward function,
success condition, and reset logic.

    CoarseReachEnv — move EEF within 3 cm XY of target key
    FineAlignEnv   — precision align to 5 mm using ArUco observations
    PressKeyEnv    — depress target key past activation threshold and hold
"""

import numpy as np

from keyboard_env import KeyboardEnv, CONTACT_FORCE_THRESHOLD, STALL_VEL_THRESHOLD
from keyboard_layout import PRESS_DEPTH, PRESS_THRESHOLD


# ---------------------------------------------------------------------------
# Skill 1: Coarse Reach
# ---------------------------------------------------------------------------

class CoarseReachEnv(KeyboardEnv):
    """Move EEF to within 3 cm XY and correct height above target key,
    with the EEF oriented near-vertical (solenoid pointing straight down).

    Actuator is locked retracted — only the 6 arm joints are used.
    """

    SUCCESS_XY   = 0.03   # m
    SUCCESS_Z    = 0.015  # m
    SUCCESS_TILT = 0.15   # rad (~8.6 deg)
    HOVER_HEIGHT = 0.05   # m above key surface

    # Joint config that places the EEF roughly vertical above the keyboard.
    ABOVE_KEYBOARD_QPOS = np.array([
        -2.3448, -1.3110, 0.4222, -0.0107, 1.7332, 0.7774,
    ])

    def __init__(self, random_key=True, easy_init_frac=0.3, **kwargs):
        kwargs.setdefault("horizon", 300)
        self.random_key = random_key
        self.easy_init_frac = easy_init_frac
        super().__init__(**kwargs)

    def _reset_internal(self):
        robot = self.robots[0]
        if np.random.rand() < self.easy_init_frac:
            robot.init_qpos = self.ABOVE_KEYBOARD_QPOS.copy()
            robot.init_qpos += np.random.uniform(-0.05, 0.05, size=6)
        else:
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

        if (xy_dist < self.SUCCESS_XY and z_error < self.SUCCESS_Z
                and tilt < self.SUCCESS_TILT):
            return 100.0

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


# ---------------------------------------------------------------------------
# Skill 2: Fine Align
# ---------------------------------------------------------------------------

class FineAlignEnv(KeyboardEnv):
    """Precisely center the actuator tip above the target key within 5 mm XY,
    starting from CoarseReach's end state (~3 cm away).

    Uses proprioceptive + synthesized ArUco observations.
    Actuator is locked retracted — only 6 arm joints.
    """

    SUCCESS_XY   = 0.005  # m
    SUCCESS_Z    = 0.015  # m
    SUCCESS_TILT = 0.15   # rad
    HOVER_HEIGHT = CoarseReachEnv.HOVER_HEIGHT

    ALIGNED_QPOS = CoarseReachEnv.ABOVE_KEYBOARD_QPOS.copy()

    def __init__(self, random_key=True, init_noise=0.03, **kwargs):
        kwargs.setdefault("horizon", 200)
        self.random_key = random_key
        self.init_noise = init_noise
        super().__init__(**kwargs)

    def _reset_internal(self):
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
        z_diff  = float((eef_pos[2] - key_pos[2]) - self.HOVER_HEIGHT)
        z_error = abs(z_diff)
        tilt    = float(self._eef_tilt_from_vertical(eef_quat))

        if (xy_dist < self.SUCCESS_XY and z_error < self.SUCCESS_Z
                and tilt < self.SUCCESS_TILT):
            return 500.0

        aruco = obs.get("aruco_obs", np.zeros(3))
        aruco_visible = float(aruco[2])

        r_align  = 20.0 * np.exp(-50.0 * xy_dist)
        r_height = -15.0 * z_error if z_diff < 0 else -3.0 * z_error
        r_orient = -2.0 * tilt
        r_time   = -0.5
        r_aruco  = 2.0 * aruco_visible

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


# ---------------------------------------------------------------------------
# Skill 3: Press Key
# ---------------------------------------------------------------------------

class PressKeyEnv(KeyboardEnv):
    """Extend the solenoid to press the target key and hold contact.

    Success = target key depressed past activation threshold for HOLD_STEPS
    consecutive steps. Reward is shaped by key displacement so the policy
    learns to push harder, not just touch.

    Assumes the arm is already aligned above the key (FineAlign has run).
    Only the solenoid actuator (action[-1]) is active; arm joints are locked to 0.
    """

    HOLD_STEPS = 3

    ALIGNED_QPOS = CoarseReachEnv.ABOVE_KEYBOARD_QPOS.copy()

    def __init__(self, random_key=True, **kwargs):
        kwargs.setdefault("horizon", 50)
        self.random_key = random_key
        self._hold_counter = 0
        super().__init__(**kwargs)

    def _reset_internal(self):
        robot = self.robots[0]
        robot.init_qpos = self.ALIGNED_QPOS.copy()
        robot.init_qpos += np.random.uniform(-0.02, 0.02, size=6)

        super()._reset_internal()
        self._hold_counter = 0
        if self.random_key:
            self.set_target_key(np.random.choice(self.AVAILABLE_KEYS))

    def reward(self, action=None):
        displacement = self.get_key_displacement(self._target_key)
        normalized_depth = np.clip(displacement / PRESS_DEPTH, 0.0, 1.0)
        pressed = displacement > PRESS_THRESHOLD

        if pressed:
            self._hold_counter += 1
            if self._hold_counter >= self.HOLD_STEPS:
                return 1000.0
            # Bonus for each consecutive hold step
            return 200.0 + 100.0 * self._hold_counter
        else:
            self._hold_counter = 0

        # Continuous shaping: reward proportional to how deep the key is pushed
        r_depth = 50.0 * normalized_depth

        # Penalize arm movement — only the solenoid should move
        r_stability = 0.0
        if action is not None:
            r_stability = -2.0 * float(np.linalg.norm(action[:6]))

        return r_depth + r_stability

    def _check_success(self):
        return self._hold_counter >= self.HOLD_STEPS