"""
Options-framework HRL environment for keyboard typing.

The high-level policy selects which frozen low-level skill to execute.
Each high-level step runs the chosen skill for up to `max_skill_steps`
low-level sim steps (or until the skill self-terminates). The high-level
then observes the resulting state and picks the next skill.

High-level action space:  Discrete(5)
    0 = CoarseReach   — move to ~3cm above target key
    1 = FineAlign      — sub-cm precision alignment
    2 = Press          — extend solenoid, make contact, hold
    3 = Retract        — retract solenoid, return to hover
    4 = Traverse       — lateral move to the NEXT key in the target string

High-level observation:  42-dim
    [0:32]  base env obs (joint pos/vel, eef, aruco, etc.)
    [32:37] one-hot active skill (5)
    [37]    current skill succeeded (0/1)
    [38]    fraction of target string completed (0..1)
    [39]    number of keys remaining (normalized by string length)
    [40]    timesteps since last successful press (normalized)
    [41]    total episode progress (current step / horizon, 0..1)

High-level reward:
    +100  per correct key press
    +500  typing the full target string
    -0.1  per high-level step (encourages speed)
    -5.0  selecting a clearly wrong skill (e.g. Press before aligned)
"""

import os
import sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ROBO_PATH = os.path.join(ROOT, "..", "external_pkgs", "RoboSuite")
if os.path.exists(ROBO_PATH) and ROBO_PATH not in sys.path:
    sys.path.insert(0, ROBO_PATH)

from stable_baselines3 import SAC

from keyboard_env import (
    KeyboardEnv, CoarseReachEnv, FineAlignEnv, PressKeyEnv,
    RetractEnv, TraverseEnv,
    CONTACT_FORCE_THRESHOLD, STALL_VEL_THRESHOLD,
)

# Skill IDs
COARSE_REACH = 0
FINE_ALIGN   = 1
PRESS        = 2
RETRACT      = 3
TRAVERSE     = 4
NUM_SKILLS   = 5

SKILL_NAMES = ["coarse_reach", "fine_align", "press", "retract", "traverse"]

# Default checkpoint paths (best models from domain-rand stage)
DEFAULT_CHECKPOINTS = {
    COARSE_REACH: os.path.join(ROOT, "checkpoints", "reach_dr",    "best_model.zip"),
    FINE_ALIGN:   os.path.join(ROOT, "checkpoints", "reach_fine",   "best_model.zip"),
    PRESS:        os.path.join(ROOT, "checkpoints", "press_dr",     "best_model.zip"),
    RETRACT:      os.path.join(ROOT, "checkpoints", "retract_dr",   "best_model.zip"),
    TRAVERSE:     os.path.join(ROOT, "checkpoints", "traverse_dr",  "best_model.zip"),
}

# Action dims that each frozen skill policy outputs
SKILL_ACTION_DIMS = {
    COARSE_REACH: 6,
    FINE_ALIGN:   6,
    PRESS:        1,
    RETRACT:      7,
    TRAVERSE:     6,
}

# Thresholds for determining skill readiness (used for reward shaping)
COARSE_REACH_XY = CoarseReachEnv.SUCCESS_XY       # 0.03m
FINE_ALIGN_XY   = FineAlignEnv.SUCCESS_XY          # 0.005m
HOVER_HEIGHT    = CoarseReachEnv.HOVER_HEIGHT       # 0.05m
TILT_THRESHOLD  = CoarseReachEnv.SUCCESS_TILT       # 0.15 rad


# Target string to key name mapping
_CHAR_TO_KEY = {
    ' ': 'space', '\t': 'tab', '\n': 'enter',
    '`': 'grave', '-': 'minus', '=': 'equal',
    '[': 'lbracket', ']': 'rbracket', '\\': 'backslash',
    ';': 'semicolon', "'": 'quote', ',': 'comma',
    '.': 'period', '/': 'slash',
}
# Single alphanumeric characters map directly
for c in 'abcdefghijklmnopqrstuvwxyz0123456789':
    _CHAR_TO_KEY[c] = c


def string_to_keys(text: str) -> list[str]:
    """Convert a target string into a list of key names."""
    keys = []
    for ch in text.lower():
        key = _CHAR_TO_KEY.get(ch)
        if key and key in KeyboardEnv.AVAILABLE_KEYS:
            keys.append(key)
    return keys


class SkillRunner:
    """
    Manages a single frozen low-level skill policy.

    Loads a trained SAC checkpoint and provides a `run()` method that
    executes the skill for up to `max_steps`, returning the cumulative
    low-level reward, number of steps taken, and whether the skill
    self-reported success.
    """

    def __init__(self, skill_id: int, checkpoint_path: str):
        self.skill_id = skill_id
        self.action_dim = SKILL_ACTION_DIMS[skill_id]
        self.model = SAC.load(checkpoint_path)
        print(f"  Loaded {SKILL_NAMES[skill_id]} from {checkpoint_path}")

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get the frozen policy's action (deterministic) for the base env obs."""
        # The frozen policy expects the same 32-dim obs it was trained on
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def to_env_action(self, action: np.ndarray) -> np.ndarray:
        """Expand skill-specific action to 7-dim env action."""
        if self.skill_id in (COARSE_REACH, FINE_ALIGN, TRAVERSE):
            # 6-dim arm -> append 0 solenoid
            scaled = action * (0.3 if self.skill_id == FINE_ALIGN else 1.0)
            return np.append(scaled, 0.0)
        elif self.skill_id == PRESS:
            # 1-dim solenoid -> pad 0s for arm
            full = np.zeros(7)
            full[-1] = float(action[0])
            return full
        else:
            # RETRACT: 7-dim direct
            return action


class HRLEnv:
    """
    Options-framework HRL environment for multi-key typing.

    Each high-level step:
      1. High-level policy picks a skill ID (Discrete 0-4)
      2. The frozen low-level policy runs for up to `max_skill_steps`
      3. The resulting state becomes the next high-level observation
      4. High-level reward is computed based on typing progress

    The episode ends when the full target string is typed or the
    high-level horizon is reached.
    """

    def __init__(
        self,
        target_string: str = "hello",
        checkpoints: dict[int, str] | None = None,
        max_skill_steps: int = 200,
        high_level_horizon: int = 100,
        render: bool = False,
    ):
        """
        Args:
            target_string: String to type (characters mapped to key names).
            checkpoints: {skill_id: path} overrides for frozen policies.
            max_skill_steps: Max low-level steps per skill invocation.
            high_level_horizon: Max high-level decisions per episode.
            render: Enable MuJoCo rendering.
        """
        self.target_keys = string_to_keys(target_string)
        assert len(self.target_keys) > 0, (
            f"No typeable keys in '{target_string}'. Available chars: "
            f"{sorted(_CHAR_TO_KEY.keys())}"
        )
        self.max_skill_steps = max_skill_steps
        self.high_level_horizon = high_level_horizon

        # Base environment — use KeyboardEnv directly (no skill-specific subclass)
        # since the frozen policies handle their own reward/termination logic
        self._env = KeyboardEnv(render=render, horizon=max_skill_steps)

        # Load frozen skill policies
        ckpts = {**DEFAULT_CHECKPOINTS, **(checkpoints or {})}
        print("Loading frozen skill policies:")
        self._skills = {}
        for sid in range(NUM_SKILLS):
            path = ckpts[sid]
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Checkpoint not found for {SKILL_NAMES[sid]}: {path}\n"
                    f"Train the skill first or provide --checkpoints"
                )
            self._skills[sid] = SkillRunner(sid, path)

        # Episode state
        self._key_idx = 0              # index into target_keys
        self._active_skill = -1        # last skill selected
        self._skill_succeeded = False
        self._hl_step = 0              # high-level step counter
        self._steps_since_press = 0    # low-level steps since last press
        self._total_ll_steps = 0       # total low-level steps this episode
        self._keys_pressed = 0         # keys successfully pressed

        # State flags tracking where we are in the per-key cycle
        self._reached = False          # CoarseReach succeeded for current key
        self._aligned = False          # FineAlign succeeded for current key
        self._pressed = False          # Press succeeded for current key
        self._retracted = False        # Retract succeeded after press

    @property
    def obs_dim(self):
        return 42  # 32 base + 5 one-hot + 5 meta

    def _get_obs(self) -> np.ndarray:
        """Build the 42-dim high-level observation."""
        base_obs = self._env._flat_obs().astype(np.float32)  # 32

        skill_onehot = np.zeros(NUM_SKILLS, dtype=np.float32)
        if 0 <= self._active_skill < NUM_SKILLS:
            skill_onehot[self._active_skill] = 1.0

        n_keys = len(self.target_keys)
        meta = np.array([
            1.0 if self._skill_succeeded else 0.0,
            self._key_idx / max(n_keys, 1),           # fraction completed
            (n_keys - self._key_idx) / max(n_keys, 1), # fraction remaining
            min(self._steps_since_press / 500.0, 1.0),  # normalized time since press
            self._hl_step / max(self.high_level_horizon, 1),  # episode progress
        ], dtype=np.float32)

        return np.concatenate([base_obs, skill_onehot, meta])

    def _assess_state(self) -> dict:
        """Evaluate current EEF state relative to target key."""
        obs = self._env._get_observations(force_update=True)
        pf = self._env.robots[0].robot_model.naming_prefix

        eef_pos  = self._env.sim.data.site_xpos[self._env._eef_site_id]
        eef_quat = obs.get(f"{pf}eef_quat", np.array([1, 0, 0, 0]))
        key_pos  = self._env.sim.data.body_xpos[
            self._env._key_body_ids[self._env._target_key]
        ]
        act_pos = float(self._env.sim.data.qpos[self._env._actuator_qpos_addr])
        force   = float(np.linalg.norm(
            self._env.sim.data.cfrc_ext[self._env._actuator_body_id][3:]
        ))
        act_vel = float(self._env.sim.data.qvel[self._env._actuator_qvel_addr])

        xy_dist = float(np.linalg.norm(eef_pos[:2] - key_pos[:2]))
        z_error = float(abs((eef_pos[2] - key_pos[2]) - HOVER_HEIGHT))
        tilt    = float(KeyboardEnv._eef_tilt_from_vertical(eef_quat))
        contact = force > CONTACT_FORCE_THRESHOLD and abs(act_vel) < STALL_VEL_THRESHOLD

        return {
            'xy_dist': xy_dist,
            'z_error': z_error,
            'tilt': tilt,
            'act_pos': act_pos,
            'force': force,
            'contact': contact,
            'coarse_reached': xy_dist < COARSE_REACH_XY and z_error < 0.015 and tilt < TILT_THRESHOLD,
            'fine_aligned': xy_dist < FINE_ALIGN_XY and z_error < 0.015 and tilt < TILT_THRESHOLD,
            'retracted': act_pos < 0.005,
        }

    def reset(self, target_string: str | None = None) -> np.ndarray:
        """
        Reset the environment for a new typing episode.

        Args:
            target_string: Optionally change the target string.
        """
        if target_string is not None:
            self.target_keys = string_to_keys(target_string)

        self._env.reset()
        self._key_idx = 0
        self._active_skill = -1
        self._skill_succeeded = False
        self._hl_step = 0
        self._steps_since_press = 0
        self._total_ll_steps = 0
        self._keys_pressed = 0
        self._reached = False
        self._aligned = False
        self._pressed = False
        self._retracted = False

        # Set first target key
        self._env.set_target_key(self.target_keys[0])

        return self._get_obs()

    def step(self, skill_id: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute one high-level step.

        Args:
            skill_id: Which frozen skill to run (0-4).

        Returns:
            (obs, reward, done, info)
        """
        assert 0 <= skill_id < NUM_SKILLS, f"Invalid skill: {skill_id}"

        self._active_skill = skill_id
        self._hl_step += 1
        skill = self._skills[skill_id]

        # ---- Run the frozen low-level policy ----
        skill_steps = 0
        skill_reward_sum = 0.0
        skill_done = False

        # Determine max steps based on skill type
        if skill_id == PRESS:
            step_limit = min(80, self.max_skill_steps)
        elif skill_id == RETRACT:
            step_limit = min(120, self.max_skill_steps)
        else:
            step_limit = self.max_skill_steps

        for _ in range(step_limit):
            base_obs = self._env._flat_obs().astype(np.float32)
            action = skill.get_action(base_obs)
            env_action = skill.to_env_action(action)

            _, ll_reward, ll_done, _ = self._env.step(env_action)
            skill_reward_sum += ll_reward
            skill_steps += 1
            self._total_ll_steps += 1
            self._steps_since_press += 1

            if ll_done:
                skill_done = True
                break

        # ---- Assess result ----
        state = self._assess_state()
        self._skill_succeeded = False
        hl_reward = -0.1  # time penalty per high-level step

        # Update phase tracking based on what happened
        if skill_id == COARSE_REACH:
            if state['coarse_reached']:
                self._skill_succeeded = True
                self._reached = True
                hl_reward += 5.0
            else:
                hl_reward += -1.0  # failed to reach

        elif skill_id == FINE_ALIGN:
            if state['fine_aligned']:
                self._skill_succeeded = True
                self._aligned = True
                hl_reward += 10.0
            elif not self._reached:
                # Tried to fine-align without coarse reach — bad sequencing
                hl_reward += -5.0

        elif skill_id == PRESS:
            if state['contact']:
                self._skill_succeeded = True
                self._pressed = True
                self._keys_pressed += 1
                self._steps_since_press = 0
                hl_reward += 100.0  # big reward for correct press
            elif not self._aligned:
                hl_reward += -5.0  # tried to press without alignment

        elif skill_id == RETRACT:
            if state['retracted'] and state['z_error'] < 0.015:
                self._skill_succeeded = True
                self._retracted = True
                hl_reward += 5.0
            elif not self._pressed:
                hl_reward += -5.0  # retract without having pressed

        elif skill_id == TRAVERSE:
            if self._pressed and self._retracted:
                # Advance to next key
                self._key_idx += 1
                if self._key_idx < len(self.target_keys):
                    next_key = self.target_keys[self._key_idx]
                    self._env.set_target_key(next_key)
                    # Reset per-key phase tracking
                    self._reached = False
                    self._aligned = False
                    self._pressed = False
                    self._retracted = False

                    # Check if traverse got us close
                    post_state = self._assess_state()
                    if post_state['coarse_reached']:
                        self._skill_succeeded = True
                        self._reached = True
                        hl_reward += 10.0
                    else:
                        hl_reward += 2.0  # partial credit for advancing
            else:
                hl_reward += -5.0  # traverse before completing press+retract cycle

        # ---- Check episode termination ----
        done = False
        info = {
            'skill': SKILL_NAMES[skill_id],
            'skill_succeeded': self._skill_succeeded,
            'skill_steps': skill_steps,
            'keys_pressed': self._keys_pressed,
            'keys_total': len(self.target_keys),
            'key_idx': self._key_idx,
            'current_target': self.target_keys[min(self._key_idx, len(self.target_keys) - 1)],
            'total_ll_steps': self._total_ll_steps,
            **{f'state_{k}': v for k, v in state.items()},
        }

        # Full string typed
        if self._keys_pressed >= len(self.target_keys):
            hl_reward += 500.0
            done = True
            info['success'] = True

        # High-level horizon exceeded
        elif self._hl_step >= self.high_level_horizon:
            done = True
            info['success'] = False

        return self._get_obs(), hl_reward, done, info

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()
