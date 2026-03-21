import os
import sys
import numpy as np
import multiprocessing as mp
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config

# Path setup for RoboSuite if not installed as package
ROOT = os.path.dirname(os.path.abspath(__file__))
# ../external_pkgs/RoboSuite relative to src/rl_autonomy
ROBO_PATH = os.path.join(ROOT, "..", "external_pkgs", "RoboSuite")
if os.path.exists(ROBO_PATH) and ROBO_PATH not in sys.path:
    sys.path.insert(0, ROBO_PATH)


class RoboSuiteEnvV3:
    """
    Enhanced environment with:
    - Wide domain randomization (full table)
    - Drop detection & recovery rewards
    - Mid-episode perturbations
    - 6-phase encoding (includes Recovery & Return)
    """
    
    CAMERA_NAME = "robot0_eye_in_hand"
    
    def __init__(
        self,
        render=False,
        domain_randomization=True,
        perturbation_prob=0.02,  # 2% chance per step
        wide_randomization=True,  # Full table coverage
        curriculum_level=0,  # 0=easy, 1=medium, 2=hard
    ):
        arm_controller_config = suite.load_part_controller_config(default_controller="JOINT_VELOCITY")
        controller_config = refactor_composite_controller_config(arm_controller_config, "Rover2026", ["right"])
        
        self.domain_randomization = domain_randomization
        self.perturbation_prob = perturbation_prob
        self.wide_randomization = wide_randomization
        self.curriculum_level = curriculum_level
        
        # Longer horizon for recovery practice
        self.horizon = 400
        
        # NEW: Forced drop probability (Curriculum based)
        # We will intentionally open gripper mid-air to force recovery practice
        self.forced_drop_prob = 0.0  # Set in update_randomization_bounds
        
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
        )
        
        self.render_enabled = render
        self._setup_spaces()
        
        # Domain randomization bounds (curriculum-based)
        self._update_randomization_bounds()
        
        # Episode state tracking
        self._was_lifted = False
        self._drop_detected = False
        self._recovery_count = 0
        self._max_height_achieved = 0.0
        self._initial_cube_pos = None
        self._steps_since_drop = 0
        
        # NEW: Enhanced drop/recovery tracking
        self._drop_count = 0  # Total drops this episode
        self._failed_recoveries = 0  # Failed recovery attempts
        self._recovery_deadline = 80  # Steps allowed to recover before penalty
        self._last_lift_height = 0.0  # Height before drop (for penalty scaling)
        self._drop_penalty_applied = False  # Track if we already penalized this drop
        self._holding_cube = False  # Currently holding the cube
        
        # Smoothness tracking for fluid motion rewards
        self._prev_action = None
        self._prev_joint_vel = None
        self._prev_eef_pos = None
    
    def _update_randomization_bounds(self):
        """Update randomization based on curriculum level."""
        if self.curriculum_level == 0:
            # Easy: small range
            self.cube_x_range = (-0.08, 0.08)
            self.cube_y_range = (-0.08, 0.08)
            self.perturbation_strength = 0.02
            self.forced_drop_prob = 0.0  # No forced drops yet
        elif self.curriculum_level == 1:
            # Medium: wider range
            self.cube_x_range = (-0.15, 0.15)
            self.cube_y_range = (-0.15, 0.15)
            self.perturbation_strength = 0.04
            self.forced_drop_prob = 0.005  # 0.5% chance per step to force drop if holding
        else:
            # Hard: full table
            self.cube_x_range = (-0.25, 0.25)
            self.cube_y_range = (-0.20, 0.20)
            self.perturbation_strength = 0.06
            self.forced_drop_prob = 0.01  # 1% chance per step to force drop if holding
    
    def set_curriculum_level(self, level):
        """Update curriculum difficulty."""
        self.curriculum_level = min(2, max(0, level))
        self._update_randomization_bounds()
    
    def _setup_spaces(self):
        obs = self.env.reset()
        
        # Store initial joint pose for "return home" skill
        self._initial_joint_pos = obs['robot0_joint_pos'].copy()
        
        self.obs_keys = [
            'robot0_joint_pos',
            'robot0_joint_vel',
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
            'cube_pos',
            'gripper_to_cube_pos',
        ]
        
        # +6 for phase one-hot encoding (reach, grasp, lift, hold, RECOVER, RETURN)
        # +3 for extra state info (was_lifted, drop_detected, recovery_progress)
        self.obs_dim = sum(len(np.array(obs[key]).flatten()) for key in self.obs_keys if key in obs) + 6 + 3
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
        # Phase determination
        # =================================================================
        phase = np.zeros(6, dtype=np.float32)
        
        if self._drop_detected:
            phase[4] = 1.0  # RECOVER phase - highest priority!
        # If successfully lifted high enough, start RETURN phase (NEW!)
        elif height_above_table > 0.20 and currently_holding:
            phase[5] = 1.0  # RETURN phase - bring it home!
        elif height_above_table > 0.08 and currently_holding:
            phase[3] = 1.0  # Hold (stabilize before return)
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
    
    def _randomize_cube_position(self):
        """Randomize cube position."""
        if not self.domain_randomization:
            return
        
        try:
            cube_body_id = self.env.sim.model.body_name2id('cube_main')
            base_pos = self.env.sim.model.body_pos[cube_body_id].copy()
            
            if self.wide_randomization:
                base_pos[0] += np.random.uniform(*self.cube_x_range)
                base_pos[1] += np.random.uniform(*self.cube_y_range)
            else:
                base_pos[0] += np.random.uniform(-0.1, 0.1)
                base_pos[1] += np.random.uniform(-0.1, 0.1)
            
            self.env.sim.model.body_pos[cube_body_id] = base_pos
            self.env.sim.forward()
            self._initial_cube_pos = base_pos.copy()
        except Exception:
            pass
    
    def _apply_perturbation(self):
        """Apply random perturbation to cube (simulates accidental bump)."""
        if np.random.random() > self.perturbation_prob:
            return False
        
        try:
            cube_body_id = self.env.sim.model.body_name2id('cube_main')
            # Apply random velocity to cube
            cube_joint_id = self.env.sim.model.body_jntadr[cube_body_id]
            if cube_joint_id >= 0:
                # Random velocity impulse
                vel = np.random.uniform(-self.perturbation_strength, self.perturbation_strength, 3)
                # This is a simplification - actual perturbation would need qvel modification
                # For now, just note that perturbation happened
                return True
        except Exception:
            pass
        return False
    
    def reset(self):
        obs = self.env.reset()
        
        # Reset episode state
        self._was_lifted = False
        self._drop_detected = False
        self._recovery_count = 0
        self._max_height_achieved = 0.0
        self._steps_since_drop = 0
        
        # Reset NEW enhanced tracking
        self._drop_count = 0
        self._failed_recoveries = 0
        self._last_lift_height = 0.0
        self._drop_penalty_applied = False
        self._holding_cube = False
        
        # Reset smoothness tracking
        self._prev_action = None
        self._prev_joint_vel = None
        self._prev_eef_pos = None
        
        self._randomize_cube_position()
        obs = self.env._get_observations()
        
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
        
        # Phase 2: GRASP
        grasp_reward = 0.0
        if distance < 0.15:
            grasp_reward = 10.0 * (1.0 - distance / 0.15)
            if distance < 0.05:
                grasp_reward += 15.0
            if gripper_closed and distance < 0.08:
                grasp_reward += 30.0
        
        # Phase 3: LIFT
        lift_reward = 0.0
        if gripper_closed and distance < 0.1:
            if height_above_table > 0:
                lift_reward = 100.0 * height_above_table
                lift_reward += 500.0 * (height_above_table ** 2)
                lift_reward += 50.0
        
        # Phase 4: SUCCESS
        success_reward = 0.0
        if height_above_table > 0.05:
            success_reward = 200.0
        if height_above_table > 0.1:
            success_reward = 500.0
        if height_above_table > 0.15:
            success_reward = 1000.0
        
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
        
        # Phase 6: RETURN HOME (NEW!)
        # Reward for returning joints to initial state while holding cube
        return_reward = 0.0
        phase, _, _, _ = self._compute_phase(obs)
        actual_phase = np.argmax(phase)
        if actual_phase == 5:  # RETURN phase
            # Calculate joint error relative to initial pose
            current_joints = obs.get('robot0_joint_pos', np.zeros(7))
            # Only care about first 4 joints (shoulder/elbow) for coarse return
            joint_error = np.linalg.norm(current_joints[:4] - self._initial_joint_pos[:4])
            
            # Continuous reward for getting closer to home
            if joint_error < 0.5:
                return_reward = 20.0 * (1.0 - joint_error / 0.5)
            
            # Bonus for being home
            if joint_error < 0.2:
                return_reward += 50.0
            
            # Massive bonus for STAYING home
            if joint_error < 0.1:
                return_reward += 100.0
        
        # Skill bonus
        skill_bonus = 0.0
        phase, _, _, _ = self._compute_phase(obs)
        actual_phase = np.argmax(phase)
        if skill is not None and skill == actual_phase:
            skill_bonus = 3.0
        
        # =================================================================
        # SMOOTHNESS REWARDS - Encourage fluid, natural motion
        # =================================================================
        smoothness_reward = 0.0
        
        # Get current velocities and positions
        joint_vel = obs.get('robot0_joint_vel', np.zeros(7))
        eef_pos = obs.get('robot0_eef_pos', np.zeros(3))
        
        # 1. Action smoothness: penalize sudden action changes (jerk in command space)
        action_smoothness = 0.0
        if self._prev_action is not None:
            action_delta = np.linalg.norm(action - self._prev_action)
            # Reward small changes, penalize large ones
            # Threshold: delta < 0.3 is smooth, > 0.8 is jerky
            if action_delta < 0.3:
                action_smoothness = 2.0 * (1.0 - action_delta / 0.3)  # Bonus up to +2
            elif action_delta > 0.8:
                action_smoothness = -3.0 * (action_delta - 0.8)  # Penalty for jerk
        
        # 2. Velocity smoothness: penalize sudden velocity changes (acceleration)
        velocity_smoothness = 0.0
        if self._prev_joint_vel is not None:
            vel_delta = np.linalg.norm(joint_vel - self._prev_joint_vel)
            # Smooth acceleration is vel_delta < 0.5
            if vel_delta < 0.5:
                velocity_smoothness = 1.5 * (1.0 - vel_delta / 0.5)  # Bonus up to +1.5
            elif vel_delta > 1.5:
                velocity_smoothness = -2.0 * (vel_delta - 1.5)  # Penalty for jerky motion
        
        # 3. End-effector path smoothness: reward straight-line motion toward goal
        path_smoothness = 0.0
        if self._prev_eef_pos is not None:
            eef_delta = eef_pos - self._prev_eef_pos
            eef_speed = np.linalg.norm(eef_delta)
            
            # During reach phase, reward moving toward cube
            if actual_phase == 0 and distance > 0.1:  # Reach phase
                # Direction toward cube
                to_cube = gripper_to_cube / (np.linalg.norm(gripper_to_cube) + 1e-6)
                # Direction of motion
                motion_dir = eef_delta / (eef_speed + 1e-6)
                # Alignment (dot product): 1.0 = moving straight to cube
                alignment = np.dot(to_cube, motion_dir)
                if alignment > 0.5:
                    path_smoothness = 2.0 * alignment  # Bonus for efficient path
            
            # Penalize erratic high-speed motion
            if eef_speed > 0.05:  # Moving fast
                path_smoothness -= 0.5 * max(0, eef_speed - 0.05)
        
        # 4. Low velocity magnitude bonus (controlled, deliberate motion)
        vel_magnitude = np.linalg.norm(joint_vel)
        controlled_motion_bonus = 0.0
        if vel_magnitude < 1.0:
            controlled_motion_bonus = 0.5 * (1.0 - vel_magnitude)  # Small bonus for slow control
        elif vel_magnitude > 3.0:
            controlled_motion_bonus = -1.0 * (vel_magnitude - 3.0)  # Penalty for flailing
        
        smoothness_reward = (
            action_smoothness + 
            velocity_smoothness + 
            path_smoothness + 
            controlled_motion_bonus
        )
        
        # =================================================================
        # EFFICIENCY REWARDS - Minimal movement, low effort, natural poses
        # =================================================================
        efficiency_reward = 0.0
        
        # Get joint positions for posture analysis
        joint_pos = obs.get('robot0_joint_pos', np.zeros(7))
        
        # 1. Minimal joint movement: reward small total joint displacement
        #    Encourages efficient paths that don't contort unnecessarily
        joint_movement_reward = 0.0
        total_joint_movement = np.sum(np.abs(action[:6]))  # Exclude gripper (last action)
        if total_joint_movement < 1.0:
            joint_movement_reward = 1.5 * (1.0 - total_joint_movement)  # Bonus for minimal movement
        elif total_joint_movement > 3.0:
            joint_movement_reward = -1.0 * (total_joint_movement - 3.0)  # Penalty for excessive movement
        
        # 2. Low torque/effort: penalize large action magnitudes (proxy for torque)
        #    Encourages the arm to work "smarter not harder"
        effort_penalty = 0.0
        action_magnitude = np.linalg.norm(action[:6])  # Exclude gripper
        if action_magnitude < 0.5:
            effort_penalty = 1.0  # Bonus for low effort
        elif action_magnitude > 2.0:
            effort_penalty = -0.5 * (action_magnitude - 2.0)  # Penalty for high effort
        
        # 3. Natural pose reward: prefer joint angles near center of range
        #    Avoids extreme joint positions (contorted poses)
        #    Assumes joints are normalized to [-1, 1] range
        pose_reward = 0.0
        joint_extremity = np.mean(np.abs(joint_pos[:6]))  # How far from center (0)
        if joint_extremity < 0.5:
            pose_reward = 1.0 * (1.0 - joint_extremity / 0.5)  # Bonus for centered joints
        elif joint_extremity > 0.8:
            pose_reward = -1.5 * (joint_extremity - 0.8)  # Penalty for extreme poses
        
        # 4. Joint limit avoidance: strong penalty near joint limits
        #    Prevents the arm from reaching awkward configurations
        joint_limit_penalty = 0.0
        for i, jp in enumerate(joint_pos[:6]):
            if abs(jp) > 0.9:  # Very close to limit
                joint_limit_penalty -= 2.0 * (abs(jp) - 0.9)
            elif abs(jp) > 0.75:  # Approaching limit
                joint_limit_penalty -= 0.5 * (abs(jp) - 0.75)
        
        # 5. Symmetric arm preference: slight bonus for balanced left-right movement
        #    Encourages more natural, human-like reaching
        symmetry_bonus = 0.0
        # ... logic skipped in source ...
        
        # 6. Wrist orientation stability: penalize excessive wrist rotation
        #    Encourages stable end-effector orientation
        wrist_stability = 0.0
        if len(action) >= 6:
            wrist_actions = action[4:6] if len(action) >= 6 else []
            if len(wrist_actions) > 0:
                wrist_movement = np.linalg.norm(wrist_actions)
                if wrist_movement < 0.3:
                    wrist_stability = 0.5 * (1.0 - wrist_movement / 0.3)
                elif wrist_movement > 1.0:
                    wrist_stability = -0.5 * (wrist_movement - 1.0)
        
        efficiency_reward = (
            joint_movement_reward +
            effort_penalty +
            pose_reward +
            joint_limit_penalty +
            wrist_stability
        )
        
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
            return_reward +  # Include return reward
            drop_penalty + 
            skill_bonus + 
            smoothness_reward +
            efficiency_reward +
            action_penalty
        )
        
        # Info for logging
        phase_names = ['reach', 'grasp', 'lift', 'hold', 'recover', 'return']
        info['phase'] = phase_names[actual_phase]
        info['actual_phase'] = actual_phase
        info['height'] = height_above_table
        info['distance'] = distance
        info['max_height'] = self._max_height_achieved
        info['drop_detected'] = self._drop_detected
        info['recovery_count'] = self._recovery_count
        info['perturbation'] = perturbation_applied
        info['forced_drop'] = forced_drop_event  # Log forced drops
        info['smoothness'] = smoothness_reward
        info['efficiency'] = efficiency_reward
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
