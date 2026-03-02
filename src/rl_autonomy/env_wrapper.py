import os
import sys
import numpy as np
import multiprocessing as mp
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.utils.placement_samplers import UniformRandomSampler

# Path setup for RoboSuite if not installed as package
ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "external_pkgs", "RoboSuite")
if os.path.exists(ROBO_PATH) and ROBO_PATH not in sys.path:
    sys.path.insert(0, ROBO_PATH)


# ============================================================================
# Curriculum Ranges
# ============================================================================
# Cube placement ranges per curriculum level (x_range, y_range)
CURRICULUM_RANGES = {
    0: ((-0.08, 0.08), (-0.08, 0.08)),   # Easy: near centre
    1: ((-0.15, 0.15), (-0.15, 0.15)),   # Medium: wider spread
    2: ((-0.25, 0.25), (-0.20, 0.20)),   # Hard: full table
}

PERTURBATION_STRENGTH = {0: 0.02, 1: 0.04, 2: 0.06}
FORCED_DROP_PROB      = {0: 0.0,  1: 0.005, 2: 0.01}

# Target height for the lift task (metres above the table).
# This is the "setpoint" analogous to an impedance controller's reference.
# The reward peaks here; going higher yields diminishing / negative returns.
TARGET_LIFT_HEIGHT = 0.15

# Tolerance band (σ of the Gaussian bell-curve reward).
# Within ±1σ the agent receives >60 % of peak reward — this is the
# "compliant region" where position maintenance is considered successful.
LIFT_HEIGHT_SIGMA  = 0.04


class RoboSuiteEnvV3:
    """
    Enhanced Lift wrapper with curriculum-aware domain randomization.

    Key features:
      - Cube placement randomization via RoboSuite's native placement_initializer
        (guarantees the cube actually spawns at the sampled location).
      - Drop detection, recovery rewards, and mid-episode perturbations.
      - 5-phase encoding: Reach, Grasp, Lift, Hold, Recover.
    """

    CAMERA_NAME = "robot0_eye_in_hand"

    def __init__(
        self,
        render=False,
        domain_randomization=True,
        perturbation_prob=0.02,
        wide_randomization=True,
        curriculum_level=0,
    ):
        self.domain_randomization = domain_randomization
        self.perturbation_prob = perturbation_prob
        self.wide_randomization = wide_randomization
        self.curriculum_level = curriculum_level
        self.horizon = 400

        # Forced-drop probability (set by curriculum)
        self.forced_drop_prob = FORCED_DROP_PROB.get(curriculum_level, 0.0)
        self.perturbation_strength = PERTURBATION_STRENGTH.get(curriculum_level, 0.02)

        # Build controller
        arm_controller_config = suite.load_part_controller_config(
            default_controller="JOINT_VELOCITY"
        )
        controller_config = refactor_composite_controller_config(
            arm_controller_config, "Rover2026", ["right"]
        )

        # Build a placement initializer with the desired cube range.
        # RoboSuite's Lift._reset_internal() will call placement_initializer.sample()
        # on every reset, so the cube *actually* ends up at the sampled position.
        placement_init = self._build_placement_initializer()

        self.env = suite.make(
            env_name="Lift",
            robots=["Rover2026"],
            controller_configs=controller_config,
            has_renderer=render,
            has_offscreen_renderer=False,
            ignore_done=False,
            use_camera_obs=False,
            control_freq=20,
            horizon=self.horizon,
            reward_shaping=True,
            placement_initializer=placement_init,
        )

        self.render_enabled = render
        self._setup_spaces()
        
        # Episode state tracking
        self._was_lifted = False
        self._drop_detected = False
        self._recovery_count = 0
        self._max_height_achieved = 0.0
        self._initial_cube_pos = None
        self._steps_since_drop = 0

        # Drop / recovery tracking
        self._drop_count = 0
        self._failed_recoveries = 0
        self._recovery_deadline = 80
        self._last_lift_height = 0.0
        self._drop_penalty_applied = False
        self._holding_cube = False

        # One-time milestone flags (prevent per-step success_reward explosion)
        self._milestone_04 = False   # 4 cm
        self._milestone_08 = False   # 8 cm
        self._milestone_target = False  # TARGET_LIFT_HEIGHT

        # Smoothness tracking
        self._prev_action = None
        self._prev_joint_vel = None
        self._prev_eef_pos = None

    # ------------------------------------------------------------------
    # Placement initializer helpers
    # ------------------------------------------------------------------
    def _build_placement_initializer(self):
        """Create a UniformRandomSampler with curriculum-appropriate ranges.

        When domain_randomization is disabled, the cube spawns at a fixed
        position (x_range and y_range are both [0, 0]).
        """
        if self.domain_randomization:
            x_range, y_range = CURRICULUM_RANGES.get(
                self.curriculum_level, CURRICULUM_RANGES[2]
            )
        else:
            x_range = (0.0, 0.0)
            y_range = (0.0, 0.0)

        return UniformRandomSampler(
            name="ObjectSampler",
            x_range=list(x_range),
            y_range=list(y_range),
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.env.table_offset if hasattr(self, 'env') else (0, 0, 0.8),
            z_offset=0.01,
        )

    def _rebuild_placement_initializer(self):
        """Rebuild and hot-swap the placement initializer on a live env.

        Called when the curriculum level changes so that subsequent resets
        sample cube positions from the updated range.
        """
        new_sampler = self._build_placement_initializer()
        # Re-register the cube object with the new sampler
        new_sampler.reset()
        new_sampler.add_objects(self.env.cube)
        self.env.placement_initializer = new_sampler
    
    def set_curriculum_level(self, level):
        """Update curriculum difficulty and rebuild the placement initializer."""
        self.curriculum_level = min(2, max(0, level))
        self.forced_drop_prob = FORCED_DROP_PROB.get(self.curriculum_level, 0.01)
        self.perturbation_strength = PERTURBATION_STRENGTH.get(self.curriculum_level, 0.06)
        self._rebuild_placement_initializer()
    
    def _setup_spaces(self):
        obs = self.env.reset()
        
        self.obs_keys = [
            'robot0_joint_pos',
            'robot0_joint_vel',
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
            'cube_pos',
            'gripper_to_cube_pos',
        ]
        
        # +5 for phase one-hot encoding (reach, grasp, lift, hold, recover)
        # +3 for extra state info (was_lifted, drop_detected, recovery_progress)
        self.obs_dim = sum(len(np.array(obs[key]).flatten()) for key in self.obs_keys if key in obs) + 5 + 3
        self.action_dim = self.env.action_dim
    
    def _compute_phase(self, obs):
        """Compute current task phase including recovery - IMPROVED DROP DETECTION."""
        cube_pos = obs.get('cube_pos', [0, 0, 0])
        gripper_to_cube = obs.get('gripper_to_cube_pos', [0, 0, 0])
        gripper_qpos = obs.get('robot0_gripper_qpos', [0, 0])
        
        distance = np.linalg.norm(gripper_to_cube)
        gripper_closed = np.mean(gripper_qpos) < 0.02
        height_above_table = max(0, cube_pos[2] - 0.82)
        
        # =================================================================
        # IMPROVED DROP DETECTION - More sensitive and accurate!
        # =================================================================
        
        # Track holding state (gripper closed + cube close + cube lifted)
        currently_holding = gripper_closed and distance < 0.12 and height_above_table > 0.02
        
        # Detect DROP: was holding, now not holding AND cube is low
        # This catches: gripper opening, cube slipping, cube falling
        if self._holding_cube and not currently_holding:
            if height_above_table < 0.03:  # Cube is back on/near table
                if not self._drop_detected:  # New drop!
                    self._drop_detected = True
                    self._drop_count += 1
                    self._steps_since_drop = 0
                    self._last_lift_height = self._max_height_achieved
                    self._drop_penalty_applied = False
        
        # Update holding state
        self._holding_cube = currently_holding
        
        # Track steps since drop
        if self._drop_detected:
            self._steps_since_drop += 1
            
            # Check for RECOVERY FAILURE (didn't re-grasp in time)
            if self._steps_since_drop > self._recovery_deadline:
                if not self._drop_penalty_applied:
                    # Mark this drop as a failed recovery
                    self._failed_recoveries += 1
                    self._drop_penalty_applied = True
                    # Note: We keep drop_detected=True, agent must still try to recover
        
        # Check for SUCCESSFUL RECOVERY (re-lifted after drop)
        if self._drop_detected and height_above_table > 0.04 and currently_holding:
            self._drop_detected = False
            self._recovery_count += 1
            self._drop_penalty_applied = False
            # Don't reset max_height - we want to track overall progress
        
        # Update max height only when actually holding
        if currently_holding and height_above_table > 0:
            self._max_height_achieved = max(self._max_height_achieved, height_above_table)
            self._was_lifted = True
        
        # =================================================================
        # Phase determination (5 phases: Reach, Grasp, Lift, Hold, Recover)
        # =================================================================
        phase = np.zeros(5, dtype=np.float32)
        
        if self._drop_detected:
            phase[4] = 1.0  # RECOVER phase - highest priority!
        elif height_above_table > TARGET_LIFT_HEIGHT * 0.5 and currently_holding:
            phase[3] = 1.0  # Hold (maintain at target height)
        elif height_above_table > 0.01 or (gripper_closed and distance < 0.1):
            phase[2] = 1.0  # Lift
        elif distance < 0.12:
            phase[1] = 1.0  # Grasp
        else:
            phase[0] = 1.0  # Reach
        
        return phase, distance, height_above_table, gripper_closed
    
    def _process_obs(self, obs):
        obs_list = []
        for key in self.obs_keys:
            if key in obs:
                obs_list.append(np.array(obs[key]).flatten())
        base_obs = np.concatenate(obs_list).astype(np.float32)
        
        # Phase info (6-dim)
        phase, distance, height, gripper_closed = self._compute_phase(obs)
        
        # Extra state info (3-dim)
        extra_state = np.array([
            float(self._was_lifted),
            float(self._drop_detected),
            min(1.0, self._steps_since_drop / 100.0) if self._drop_detected else 0.0,
        ], dtype=np.float32)
        
        return np.concatenate([base_obs, phase, extra_state]).astype(np.float32)

    def _apply_perturbation(self):
        """Apply a random velocity impulse to the cube mid-episode.

        Uses MuJoCo's qvel (generalised velocity) to give the cube a
        physical push rather than teleporting it, so the physics engine
        handles the resulting motion naturally.
        """
        if np.random.random() > self.perturbation_prob:
            return False

        try:
            # The cube has a free joint — its qvel entries are
            # [vx, vy, vz, wx, wy, wz] (linear + angular velocity).
            cube_joint_name = self.env.cube.joints[0]
            joint_id = self.env.sim.model.joint_name2id(cube_joint_name)
            qvel_addr = self.env.sim.model.jnt_dofadr[joint_id]

            impulse = np.random.uniform(
                -self.perturbation_strength, self.perturbation_strength, size=3
            )
            self.env.sim.data.qvel[qvel_addr:qvel_addr + 3] += impulse
            return True
        except Exception:
            return False
    
    def reset(self):
        """Reset the environment.

        RoboSuite's internal reset already calls placement_initializer.sample()
        which places the cube at a random position within the curriculum range.
        No additional cube-teleporting is needed.
        """
        obs = self.env.reset()

        # Reset episode state
        self._was_lifted = False
        self._drop_detected = False
        self._recovery_count = 0
        self._max_height_achieved = 0.0
        self._steps_since_drop = 0
        self._drop_count = 0
        self._failed_recoveries = 0
        self._last_lift_height = 0.0
        self._drop_penalty_applied = False
        self._holding_cube = False

        # Reset milestone flags
        self._milestone_04 = False
        self._milestone_08 = False
        self._milestone_target = False

        # Reset smoothness tracking
        self._prev_action = None
        self._prev_joint_vel = None
        self._prev_eef_pos = None

        # Record where the cube actually spawned (for logging / debugging)
        cube_pos = obs.get('cube_pos', None)
        if cube_pos is not None:
            self._initial_cube_pos = np.array(cube_pos).copy()

        return self._process_obs(obs)
    
    def step(self, action, skill=None):
        # Optionally apply perturbation
        perturbation_applied = self._apply_perturbation()
        
        # NEW: Forced Drop Logic (The "Clumsy" Mechanism)
        # If we are holding the block high enough, randomly click gripper open
        forced_drop_event = False
        if self._holding_cube and self._max_height_achieved > 0.15:
            if np.random.random() < self.forced_drop_prob:
                # OVERRIDE actions to force gripper open
                # Action index 6 is gripper (-1 = open, 1 = closed usually)
                # We set it to -1.0 to force open
                action = action.copy()
                if len(action) > 6:
                    action[6] = -1.0 
                    forced_drop_event = True
                    # We also want to stop lifting for a moment so gravity takes over
                    action[2] = -0.5  # slight downward velocity
        
        obs, base_reward, done, info = self.env.step(action)
        
        # Get positions
        cube_pos = obs.get('cube_pos', [0, 0, 0])
        gripper_to_cube = obs.get('gripper_to_cube_pos', [0, 0, 0])
        gripper_qpos = obs.get('robot0_gripper_qpos', [0, 0])
        
        distance = np.linalg.norm(gripper_to_cube)
        gripper_closed = np.mean(gripper_qpos) < 0.02
        cube_height = cube_pos[2]
        table_height = 0.82
        height_above_table = max(0, cube_height - table_height)
        
        # Track max height
        if gripper_closed and distance < 0.1:
            self._max_height_achieved = max(self._max_height_achieved, height_above_table)
        
        # =================================================================
        # V3 REWARD STRUCTURE: Recovery-aware
        # =================================================================
        
        # Phase 1: REACH
        reach_reward = 5.0 * np.exp(-5.0 * distance)
        
        # Phase 2: GRASP — reward for getting close and closing gripper.
        # Keep moderate so it doesn't become a local optimum vs lifting.
        # At max (~40/step) over 400 steps = 16K, which is less than
        # lift+success (~30K at target + 1550 milestone).
        grasp_reward = 0.0
        if distance < 0.15:
            grasp_reward = 8.0 * (1.0 - distance / 0.15)
            if distance < 0.05:
                grasp_reward += 10.0
            if gripper_closed and distance < 0.08:
                grasp_reward += 20.0
        
        # Phase 3: LIFT — bell-curve reward centred on TARGET_LIFT_HEIGHT
        # ────────────────────────────────────────────────────────────────
        # Analogous to an impedance controller's setpoint: the reward
        # defines WHERE the cube should be, not HOW to hold it there.
        # The JointVelocityController's PID loop naturally produces
        # braking torques when the agent outputs zero-velocity commands,
        # so we don't need to explicitly reward "being still".
        #
        # Below the target: monotonically increasing (agent is encouraged
        #   to lift higher).  Steeper near the target to pull it up.
        # At the target: moderate per-step reward (not overwhelming).
        # Above the target: Gaussian roll-off.
        #
        # IMPORTANT: Per-step rewards during Hold must stay moderate
        # (~80-100/step) or they dominate Q-values and cause skill collapse.
        lift_reward = 0.0
        if gripper_closed and distance < 0.1:
            if height_above_table > 0:
                # --- Below target: quadratic ramp (steepens near target) ---
                if height_above_table <= TARGET_LIFT_HEIGHT:
                    progress = height_above_table / TARGET_LIFT_HEIGHT
                    # Ramp: 20 (initial contact) + up to 80 (at target) = 100 peak
                    lift_reward = 20.0 + 80.0 * (progress ** 1.5)
                else:
                    # --- Above target: Gaussian tolerance band ---
                    overshoot = height_above_table - TARGET_LIFT_HEIGHT
                    bell = np.exp(-0.5 * (overshoot / LIFT_HEIGHT_SIGMA) ** 2)
                    # Peak is 100 at exactly TARGET_LIFT_HEIGHT, rolls off above
                    lift_reward = 100.0 * bell
        
        # Phase 4: SUCCESS milestones — ONE-TIME sparse bonuses.
        # These fire once per episode when the height is first reached.
        # Without this, the agent gets 1000/step during Hold → 300K total
        # → massive Q-value imbalance → skill collapse.
        success_reward = 0.0
        if gripper_closed and distance < 0.1:
            if height_above_table > 0.04 and not self._milestone_04:
                success_reward += 150.0
                self._milestone_04 = True
            if height_above_table > 0.08 and not self._milestone_08:
                success_reward += 400.0
                self._milestone_08 = True
            if height_above_table > TARGET_LIFT_HEIGHT - 0.02 and not self._milestone_target:
                success_reward += 1000.0
                self._milestone_target = True
        
        # Phase 5: RECOVERY - STRONG incentives to pick up dropped cube!
        recovery_reward = 0.0
        drop_penalty = 0.0
        
        if self._drop_detected:
            # =================================================================
            # RECOVERY REWARDS - Make recovery VERY attractive
            # =================================================================
            
            # Continuous reward for approaching the dropped cube
            recovery_reward += 30.0 * np.exp(-3.0 * distance)
            
            # Bonus for getting close to dropped cube
            if distance < 0.15:
                recovery_reward += 20.0 * (1.0 - distance / 0.15)
            
            # BIG bonus for re-grasping dropped cube!
            if gripper_closed and distance < 0.08:
                recovery_reward += 150.0  # Re-grasped!
                info['recovery_grasp'] = True
            
            # HUGE bonus for successfully lifting after drop
            if height_above_table > 0.05 and gripper_closed and distance < 0.1:
                recovery_reward += 500.0  # Full recovery!
                info['full_recovery'] = True
            
            # =================================================================
            # FAILURE PENALTIES - Punish NOT attempting recovery
            # =================================================================
            
            # Time-based urgency: penalty increases the longer you don't recover
            urgency_penalty = -0.5 * min(self._steps_since_drop, 100)  # Up to -50 per step
            
            # Penalty for moving AWAY from dropped cube during recovery
            if self._prev_eef_pos is not None:
                # If we're in recovery and not getting closer, penalize
                if distance > 0.15 and self._steps_since_drop > 10:
                    drop_penalty -= 10.0  # Should be approaching!
            
            # BIG penalty once recovery deadline passes
            if self._steps_since_drop > self._recovery_deadline:
                if height_above_table < 0.03:  # Still haven't recovered
                    # Scaled by how high the cube was before drop
                    height_multiplier = max(1.0, self._last_lift_height * 10)
                    drop_penalty -= 100.0 * height_multiplier  # MAJOR penalty
                    info['recovery_failed'] = True
            
            recovery_reward += urgency_penalty
        
        # Initial drop penalty (when drop first detected)
        if self._drop_count > 0 and not self._drop_penalty_applied:
            # Mild penalty for dropping, but recoverable
            # If it was a forced drop (training drill), DON'T penalize the initial drop
            # Only penalize if they fail to recover
            if not forced_drop_event:
                drop_penalty -= 30.0 * self._drop_count
        
        # Compute phase ONCE and reuse (avoid repeated _compute_phase calls)
        phase, _, _, _ = self._compute_phase(obs)
        actual_phase = np.argmax(phase)

        # =================================================================
        # POSITION MAINTENANCE — impedance-style setpoint tracking
        # =================================================================
        # Instead of rewarding "stay still" (which fights the controller's
        # job), we reward the OUTCOME: the cube staying near the target
        # height.  This is what impedance control does — define a reference
        # position and let the low-level controller handle stiffness /
        # damping.  The JointVelocityController's PID + gravity comp
        # naturally holds the arm when the agent outputs near-zero commands,
        # so the agent just needs a reason to keep the cube at the target
        # rather than keep climbing.
        position_maintenance_reward = 0.0
        joint_vel = obs.get('robot0_joint_vel', np.zeros(7))
        eef_pos = obs.get('robot0_eef_pos', np.zeros(3))
        vel_magnitude = np.linalg.norm(joint_vel)
        action_magnitude = np.linalg.norm(action[:6])  # Exclude gripper

        task_achieved = (gripper_closed and distance < 0.1
                         and height_above_table > 0.05)

        if task_achieved:
            # How close is the cube to the target height?
            height_error = abs(height_above_table - TARGET_LIFT_HEIGHT)
            # Gaussian reward: peaks at target, decays away from it
            position_maintenance_reward = 30.0 * np.exp(
                -0.5 * (height_error / LIFT_HEIGHT_SIGMA) ** 2
            )
            # Small additional bonus for being very close (within 1 cm)
            if height_error < 0.01:
                position_maintenance_reward += 10.0

        # With the bell-curve lift reward, the Hold phase no longer needs
        # a special cap — the reward naturally plateaus at TARGET_LIFT_HEIGHT.
        
        # Skill bonus — must be large enough relative to other rewards
        # to give the skill selector a meaningful gradient signal.
        # (Reuses `phase` and `actual_phase` computed above.)
        skill_bonus = 0.0
        if skill is not None:
            if skill == actual_phase:
                skill_bonus = 25.0   # Correct skill for this phase
            else:
                skill_bonus = -10.0  # Wrong skill — mild penalty
        
        # =================================================================
        # SMOOTHNESS PENALTIES - Penalize jerky, unnatural motion only
        # =================================================================
        # No positive bonuses — those incentivize "do nothing" which
        # competes with "do the task".  We only penalize BAD motion.
        smoothness_penalty = 0.0

        # Get current velocities and positions
        eef_pos = obs.get('robot0_eef_pos', np.zeros(3))

        # 1. Penalize sudden action changes (jerk)
        if self._prev_action is not None:
            action_delta = np.linalg.norm(action - self._prev_action)
            if action_delta > 0.8:
                smoothness_penalty -= 5.0 * (action_delta - 0.8)

        # 2. Penalize sudden acceleration
        if self._prev_joint_vel is not None:
            vel_delta = np.linalg.norm(joint_vel - self._prev_joint_vel)
            if vel_delta > 1.5:
                smoothness_penalty -= 4.0 * (vel_delta - 1.5)

        # 3. Penalize erratic high-speed end-effector motion
        if self._prev_eef_pos is not None:
            eef_delta = eef_pos - self._prev_eef_pos
            eef_speed = np.linalg.norm(eef_delta)
            if eef_speed > 0.05:
                smoothness_penalty -= 2.0 * (eef_speed - 0.05)

        # 4. Penalize excessively high joint velocities
        if vel_magnitude > 3.0:
            smoothness_penalty -= 3.0 * (vel_magnitude - 3.0)
        
        # =================================================================
        # EFFICIENCY PENALTIES - Penalize wasteful motion only
        # =================================================================
        efficiency_penalty = 0.0

        joint_pos = obs.get('robot0_joint_pos', np.zeros(7))

        # 1. Penalize excessive joint movement
        total_joint_movement = np.sum(np.abs(action[:6]))
        if total_joint_movement > 3.0:
            efficiency_penalty -= 3.0 * (total_joint_movement - 3.0)

        # 2. Penalize excessive action magnitude
        if action_magnitude > 2.0:
            efficiency_penalty -= 2.0 * (action_magnitude - 2.0)

        # 3. Penalize joint limit approach (safety concern)
        for jp in joint_pos[:6]:
            if abs(jp) > 0.9:
                efficiency_penalty -= 5.0 * (abs(jp) - 0.9)
            elif abs(jp) > 0.75:
                efficiency_penalty -= 1.5 * (abs(jp) - 0.75)

        # 4. Penalize extreme wrist motion
        if len(action) >= 6:
            wrist_movement = np.linalg.norm(action[4:6])
            if wrist_movement > 1.0:
                efficiency_penalty -= 2.0 * (wrist_movement - 1.0)
        
        # Update previous values for next step
        self._prev_action = action.copy()
        self._prev_joint_vel = joint_vel.copy() if hasattr(joint_vel, 'copy') else np.array(joint_vel)
        self._prev_eef_pos = eef_pos.copy() if hasattr(eef_pos, 'copy') else np.array(eef_pos)
        
        # Legacy action magnitude penalty (reduced, efficiency handles this better now)
        action_penalty = -0.01 * np.sum(action ** 2)
        
        shaped_reward = (
            base_reward + 
            reach_reward + 
            grasp_reward + 
            lift_reward + 
            success_reward + 
            recovery_reward + 
            drop_penalty + 
            skill_bonus + 
            position_maintenance_reward +
            smoothness_penalty +
            efficiency_penalty +
            action_penalty
        )
        
        # Info for logging
        phase_names = ['reach', 'grasp', 'lift', 'hold', 'recover']
        info['phase'] = phase_names[actual_phase]
        info['actual_phase'] = actual_phase
        info['height'] = height_above_table
        info['distance'] = distance
        info['max_height'] = self._max_height_achieved
        info['drop_detected'] = self._drop_detected
        info['recovery_count'] = self._recovery_count
        info['perturbation'] = perturbation_applied
        info['forced_drop'] = forced_drop_event
        info['smoothness'] = smoothness_penalty
        info['position_maintenance'] = position_maintenance_reward
        info['efficiency'] = efficiency_penalty
        # NEW: Enhanced drop/recovery tracking for logging
        info['drop_count'] = self._drop_count
        info['failed_recoveries'] = self._failed_recoveries
        info['steps_since_drop'] = self._steps_since_drop if self._drop_detected else 0
        info['recovery_reward'] = recovery_reward
        info['drop_penalty'] = drop_penalty
        
        state = self._process_obs(obs)
        return state, shaped_reward, done, info
    
    def render(self):
        if self.render_enabled:
            self.env.render()
    
    def close(self):
        self.env.close()


def worker_v3(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                action, skill = data
                state, reward, done, info = env.step(action, skill=skill)
                if done:
                    state = env.reset()
                remote.send((state, reward, done, info))
            elif cmd == 'reset':
                state = env.reset()
                remote.send(state)
            elif cmd == 'get_spaces':
                remote.send((env.obs_dim, env.action_dim))
            elif cmd == 'set_curriculum':
                env.set_curriculum_level(data)
                remote.send(True)
            elif cmd == 'close':
                env.close()
                remote.close()
                break
    except EOFError:
        pass


class SubprocVecEnvV3:
    """Parallel environments with curriculum support."""
    
    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.ps = [mp.Process(target=worker_v3, args=(wr, r, fn))
                   for wr, r, fn in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for wr in self.work_remotes:
            wr.close()
        
        self.remotes[0].send(('get_spaces', None))
        self.obs_dim, self.action_dim = self.remotes[0].recv()
    
    def step(self, actions, skills):
        for remote, action, skill in zip(self.remotes, actions, skills):
            remote.send(('step', (action, skill)))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        return np.stack(obs), np.array(rewards), np.array(dones), infos
    
    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])
    
    def set_curriculum(self, level):
        for remote in self.remotes:
            remote.send(('set_curriculum', level))
        for remote in self.remotes:
            remote.recv()
    
    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
