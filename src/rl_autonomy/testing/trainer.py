import os
import sys
import time
import argparse
import numpy as np
from collections import deque
from datetime import datetime
import torch

from .config import Config
from .memory import GPUReplayBuffer
from .agent import HierarchicalSACAgentV3
from .networks import SkillSelectorV3
from .env_wrapper import RoboSuiteEnvV3, SubprocVecEnvV3
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config

def train_v3(args):
    """Main training loop with curriculum learning."""
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"\n{'='*60}")
    print(f"  Hierarchical SAC V3 - Robust Recovery Training")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Curriculum: Enabled (3 levels)")
    print(f"  Skills: {SkillSelectorV3.SKILL_NAMES}")
    print(f"  Episode length: 400 steps")
    print(f"{'='*60}\n")
    
    # Create save directory
    # Save checkpoints inside rl_agent/checkpoints
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ROOT = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(ROOT, "checkpoints", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"  Checkpoints: {save_dir}\n")
    
    # Create parallel environments
    num_envs = args.num_envs
    env_fns = [lambda: RoboSuiteEnvV3(
        render=False,
        domain_randomization=True,
        perturbation_prob=0.02,
        wide_randomization=True,
        curriculum_level=0,  # Start easy
    ) for _ in range(num_envs)]
    
    vec_env = SubprocVecEnvV3(env_fns)
    obs_dim = vec_env.obs_dim
    action_dim = vec_env.action_dim
    
    print(f"  Environments: {num_envs} parallel")
    print(f"  Observation dim: {obs_dim} (27 base + 5 phase + 3 state)")
    print(f"  Action dim: {action_dim}")
    
    # Create agent
    agent = HierarchicalSACAgentV3(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        use_amp=args.cuda,
    )
    
    # Resume OR load pre-trained (not both!)
    start_episode = 0
    if args.resume:
        # RESUME: Load full V3 checkpoint (ignores pretrained)
        agent.load(args.resume, resume_training=True)
        # Try to extract episode number from filename
        import re
        match = re.search(r'checkpoint_ep(\d+)', args.resume)
        if match:
            start_episode = int(match.group(1))
        print(f"  ✓ Resumed from: {args.resume} (Episode {start_episode})")
    elif args.pretrained:
        # TRANSFER: Load pre-trained skills
        agent.load_pretrained_skills(
            args.pretrained, 
            freeze_old_skills=args.freeze_old_skills
        )
        if args.differential_lr:
            agent.set_differential_lr(
                old_skill_lr=args.old_skill_lr,
                new_skill_lr=args.new_skill_lr
            )
    
    # Create replay buffer
    buffer = GPUReplayBuffer(
        capacity=args.buffer_size,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )
    
    # Training state
    total_steps = 0
    best_reward = -float('inf')
    reward_history = deque(maxlen=100)
    recovery_history = deque(maxlen=100)
    drop_history = deque(maxlen=100)  # NEW: Track drops per episode
    failed_recovery_history = deque(maxlen=100)  # NEW: Track failed recoveries
    
    # Curriculum schedule
    curriculum_thresholds = [
        (0, 0),       # Start at level 0
        (100, 1),     # After 100 episodes, level 1
        (300, 2),     # After 300 episodes, level 2
    ]
    current_curriculum = 0
    
    obs = vec_env.reset()
    episode_rewards = np.zeros(num_envs)
    episode_recoveries = np.zeros(num_envs)
    episode_drops = np.zeros(num_envs)  # NEW: Track drops per env
    episode_failed_recoveries = np.zeros(num_envs)  # NEW: Track failed recoveries per env
    skill_counts = np.zeros(SkillSelectorV3.NUM_SKILLS)
    
    # Per-env agent state
    agent_states = [{'skill': None, 'steps': 0} for _ in range(num_envs)]
    
    print(f"\n  Starting training from episode {start_episode}...\n")
    
    # Track if we need to unfreeze skills later
    skills_unfrozen = not args.pretrained or not args.freeze_old_skills
    
    for episode in range(start_episode, args.episodes):
        # Curriculum update
        for threshold, level in curriculum_thresholds:
            if episode >= threshold and current_curriculum < level:
                current_curriculum = level
                vec_env.set_curriculum(level)
                print(f"\n  📈 Curriculum level increased to {level}!")
        
        # Unfreeze old skills after initial training on Recovery
        if not skills_unfrozen and episode >= args.unfreeze_episode:
            agent.unfreeze_all_skills()
            # Restore normal learning rate
            agent.set_differential_lr(old_skill_lr=1e-4, new_skill_lr=3e-4)
            skills_unfrozen = True
        
        episode_start = time.time()
        episode_steps = 0
        agent.reset()
        
        while episode_steps < 400:  # Max steps per episode
            # Get actions
            actions = []
            skills = []
            
            for i in range(num_envs):
                action, skill = agent.get_action(obs[i], deterministic=False)
                actions.append(action)
                skills.append(skill)
                skill_counts[skill] += 1
            
            actions = np.array(actions)
            skills = np.array(skills)
            
            # Step environments
            next_obs, rewards, dones, infos = vec_env.step(actions, skills)
            
            # Store transitions
            for i in range(num_envs):
                buffer.add(
                    obs[i], actions[i], rewards[i], next_obs[i], dones[i], skills[i]
                )
                episode_rewards[i] += rewards[i]
                
                # Track recoveries and drops
                if infos[i].get('recovery_count', 0) > 0:
                    episode_recoveries[i] = infos[i]['recovery_count']
                
                # Track drops for logging
                if infos[i].get('drop_count', 0) > episode_drops[i]:
                    episode_drops[i] = infos[i]['drop_count']
                if infos[i].get('failed_recoveries', 0) > episode_failed_recoveries[i]:
                    episode_failed_recoveries[i] = infos[i]['failed_recoveries']
                
                if dones[i]:
                    reward_history.append(episode_rewards[i])
                    recovery_history.append(episode_recoveries[i])
                    drop_history.append(episode_drops[i])
                    failed_recovery_history.append(episode_failed_recoveries[i])
                    episode_rewards[i] = 0
                    episode_recoveries[i] = 0
                    episode_drops[i] = 0
                    episode_failed_recoveries[i] = 0
                    agent.reset()
            
            obs = next_obs
            total_steps += num_envs
            episode_steps += 1
            
            # Update agent
            if len(buffer) >= args.batch_size:
                for _ in range(args.updates_per_step):
                    metrics = agent.update(buffer, args.batch_size)
        
        # Logging
        if (episode + 1) % 10 == 0 and len(reward_history) > 0:
            avg_reward = np.mean(reward_history)
            avg_recovery = np.mean(recovery_history)
            avg_drops = np.mean(drop_history) if len(drop_history) > 0 else 0
            avg_failed = np.mean(failed_recovery_history) if len(failed_recovery_history) > 0 else 0
            skill_dist = skill_counts / (skill_counts.sum() +  1e-8) * 100
            skill_str = " ".join([f"{n[0]}:{d:.0f}%" for n, d in zip(SkillSelectorV3.SKILL_NAMES, skill_dist)])
            
            # Show drops and recoveries prominently
            drop_rec_str = f"D:{avg_drops:.1f}/R:{avg_recovery:.1f}"
            if avg_failed > 0:
                drop_rec_str += f"/F:{avg_failed:.1f}"
            
            print(f"  Ep {episode+1:4d} | R: {avg_reward:8.1f} | "
                  f"{drop_rec_str} | Cur: L{current_curriculum} | "
                  f"Steps: {total_steps//1000}K | {skill_str}")
            
            skill_counts = np.zeros(SkillSelectorV3.NUM_SKILLS)
            
            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save(os.path.join(save_dir, "best_model.pt"))
                print(f"        ✓ New best: {best_reward:.1f}")
        
        # Periodic checkpoints
        if (episode + 1) % 100 == 0:
            agent.save(os.path.join(save_dir, f"checkpoint_ep{episode+1}.pt"))
    
    # Final save
    agent.save(os.path.join(save_dir, "final_model.pt"))
    vec_env.close()
    
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best reward: {best_reward:.1f}")
    print(f"  Model saved: {save_dir}")
    print(f"{'='*60}\n")


def evaluate_v3(args):
    """Evaluate with visualization."""
    import cv2
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    
    # Create environment
    arm_controller_config = suite.load_part_controller_config(default_controller="JOINT_VELOCITY")
    controller_config = refactor_composite_controller_config(arm_controller_config, "Rover2026", ["right"])
    
    env = suite.make(
        env_name="Lift",
        robots=["Rover2026"],
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera="frontview",
        camera_names=["robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
        ignore_done=False,
        use_camera_obs=True,
        control_freq=20,
        horizon=400,
        reward_shaping=True,
    )
    
    obs_keys = [
        'robot0_joint_pos', 'robot0_joint_vel', 'robot0_eef_pos',
        'robot0_eef_quat', 'robot0_gripper_qpos', 'cube_pos', 'gripper_to_cube_pos',
    ]
    
    # Episode state (mirror RoboSuiteEnvV3)
    was_lifted = False
    holding_cube = False
    drop_detected = False
    max_height = 0.0
    steps_since_drop = 0
    recovery_count = 0
    
    def process_obs(obs_dict):
        nonlocal was_lifted, holding_cube, drop_detected, max_height, steps_since_drop, recovery_count
        
        obs_list = [np.array(obs_dict[k]).flatten() for k in obs_keys if k in obs_dict]
        base_obs = np.concatenate(obs_list).astype(np.float32)
        
        # Compute phase (Logic mirrors RoboSuiteEnvV3)
        cube_pos = obs_dict.get('cube_pos', [0, 0, 0])
        gripper_to_cube = obs_dict.get('gripper_to_cube_pos', [0, 0, 0])
        gripper_qpos = obs_dict.get('robot0_gripper_qpos', [0, 0])
        
        distance = np.linalg.norm(gripper_to_cube)
        gripper_closed = np.mean(gripper_qpos) < 0.02
        height = max(0, cube_pos[2] - 0.82)
        
        # Track holding
        currently_holding = gripper_closed and distance < 0.12 and height > 0.02
        
        # Drop detection
        if holding_cube and not currently_holding:
            if height < 0.03:
                if not drop_detected:
                    drop_detected = True
                    steps_since_drop = 0
        
        holding_cube = currently_holding
        
        if drop_detected:
            steps_since_drop += 1
        
        # Successful Recovery
        if drop_detected and height > 0.04 and currently_holding:
            drop_detected = False
            recovery_count += 1
        
        if currently_holding and height > 0:
            max_height = max(max_height, height)
            was_lifted = True
            
        # Phase encoding (6-dim)
        phase = np.zeros(6, dtype=np.float32)
        
        if drop_detected:
            phase[4] = 1.0  # Recover
        elif height > 0.20 and currently_holding:
            phase[5] = 1.0  # Return (NEW)
        elif height > 0.08 and currently_holding:
            phase[3] = 1.0  # Hold
        elif height > 0.01 or (gripper_closed and distance < 0.1):
            phase[2] = 1.0  # Lift
        elif distance < 0.1:
            phase[1] = 1.0  # Grasp
        else:
            phase[0] = 1.0  # Reach
        
        # Extra state (3-dim)
        extra = np.array([
            float(was_lifted),
            float(drop_detected),
            min(1.0, steps_since_drop / 100.0) if drop_detected else 0.0,
        ], dtype=np.float32)
        
        return np.concatenate([base_obs, phase, extra]).astype(np.float32), distance, height
    
    def randomize_cube(level=2):
        if not args.domain_rand:
            return
        ranges = [(-0.08, 0.08), (-0.15, 0.15), (-0.25, 0.25)]
        r = ranges[min(level, 2)]
        try:
            cube_id = env.sim.model.body_name2id('cube_main')
            pos = env.sim.model.body_pos[cube_id].copy()
            pos[0] += np.random.uniform(*r)
            pos[1] += np.random.uniform(*r)
            env.sim.model.body_pos[cube_id] = pos
            env.sim.forward()
        except:
            pass
    
    # Get obs_dim
    test_obs = env.reset()
    obs, _, _ = process_obs(test_obs)
    obs_dim = len(obs)
    
    # Create agent
    agent = HierarchicalSACAgentV3(
        obs_dim=obs_dim,
        action_dim=env.action_dim,
        device=device,
        use_amp=False,
    )
    agent.load(args.eval)
    
    print(f"\n{'='*60}")
    print(f"  EVALUATING V3: {args.eval}")
    print(f"{'='*60}")
    print(f"  Domain Randomization: {args.domain_rand}")
    print(f"  Skills: {SkillSelectorV3.SKILL_NAMES}")
    print(f"{'='*60}\n")
    
    cv2.namedWindow("Gripper Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gripper Camera", 400, 400)
    
    for episode in range(args.eval_episodes):
        raw_obs = env.reset()
        
        # Reset state
        was_lifted = False
        holding_cube = False
        drop_detected = False
        max_height = 0.0
        steps_since_drop = 0
        recovery_count = 0
        
        randomize_cube(args.difficulty)
        raw_obs = env._get_observations()
        
        obs, distance, height = process_obs(raw_obs)
        agent.reset()
        episode_reward = 0
        done = False
        skill_seq = []
        
        while not done:
            action, skill = agent.get_action(obs, deterministic=True)
            skill_seq.append(skill)
            
            raw_obs, reward, done, info = env.step(action)
            obs, distance, height = process_obs(raw_obs)
            episode_reward += reward
            
            env.render()
            
            # Show gripper camera
            img_key = "robot0_eye_in_hand_image"
            if img_key in raw_obs:
                img = cv2.cvtColor(raw_obs[img_key], cv2.COLOR_RGB2BGR)
                cv2.putText(img, f"Skill: {SkillSelectorV3.SKILL_NAMES[skill]}", 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img, f"Dist: {distance:.3f}m | H: {height:.3f}m", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                if drop_detected:
                    cv2.putText(img, "DROP DETECTED - RECOVERING", 
                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow("Gripper Camera", img)
                cv2.waitKey(1)
            
            time.sleep(0.02)
        
        from collections import Counter
        skill_counts = Counter([SkillSelectorV3.SKILL_NAMES[s] for s in skill_seq])
        skill_str = ", ".join([f"{k}: {v}" for k, v in skill_counts.most_common()])
        
        print(f"  Ep {episode+1}: R={episode_reward:.1f} | MaxH={max_height:.3f}m | "
              f"Recoveries={recovery_count} | Skills: {skill_str}")
    
    cv2.destroyAllWindows()
    env.close()
    print(f"\n  Evaluation complete!")
