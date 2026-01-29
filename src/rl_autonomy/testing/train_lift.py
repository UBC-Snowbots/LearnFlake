#!/usr/bin/env python3
"""
SAC (Soft Actor-Critic) agent for training Rover2026 arm to lift objects.

This implements:
- SAC algorithm with automatic entropy tuning
- Actor-Critic neural networks
- Replay buffer
- Environment wrapper for RoboSuite

Usage:
    python train_lift.py --train          # Train a new agent
    python train_lift.py --eval model.pt  # Evaluate a trained model
"""

import os
import sys
import time
import argparse
import numpy as np
from collections import deque
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# Path setup
ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite")
sys.path.insert(0, ROBO_PATH)

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config


# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """Experience replay buffer for off-policy learning with GPU optimization."""
    
    def __init__(self, capacity, obs_dim, action_dim, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate memory with pinned memory for faster GPU transfer
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        # Pre-allocated tensors for batch sampling (avoids repeated allocation)
        self._batch_obs = None
        self._batch_size = 0
    
    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        # Use torch.from_numpy (zero-copy) + non_blocking transfer for speed
        return (
            torch.from_numpy(self.obs[idxs]).to(self.device, non_blocking=True),
            torch.from_numpy(self.actions[idxs]).to(self.device, non_blocking=True),
            torch.from_numpy(self.rewards[idxs]).to(self.device, non_blocking=True),
            torch.from_numpy(self.next_obs[idxs]).to(self.device, non_blocking=True),
            torch.from_numpy(self.dones[idxs]).to(self.device, non_blocking=True),
        )


# ============================================================================
# Neural Networks
# ============================================================================

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    """Build a multi-layer perceptron."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
        else:
            layers.append(output_activation())
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """
    Gaussian policy network for continuous action spaces.
    Outputs mean and log_std for each action dimension.
    """
    
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=(256, 256)):
        super().__init__()
        
        self.net = mlp([obs_dim] + list(hidden_sizes), activation=nn.ReLU)
        self.mu_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dim)
    
    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        
        # Sample action
        dist = Normal(mu, std)
        if deterministic:
            action = mu
        else:
            action = dist.rsample()  # Reparameterization trick
        
        # Compute log probability
        if with_logprob:
            # Apply tanh squashing correction
            logprob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            logprob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(dim=-1, keepdim=True)
        else:
            logprob = None
        
        # Squash action to [-1, 1]
        action = torch.tanh(action)
        
        return action, logprob
    
    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            action, _ = self.forward(obs, deterministic=deterministic, with_logprob=False)
            return action.cpu().numpy()[0]


class Critic(nn.Module):
    """Q-function critic network."""
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.q = mlp([obs_dim + action_dim] + list(hidden_sizes) + [1])
    
    def forward(self, obs, action):
        return self.q(torch.cat([obs, action], dim=-1))


class DoubleCritic(nn.Module):
    """Twin Q-networks for SAC (reduces overestimation bias)."""
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.q1 = Critic(obs_dim, action_dim, hidden_sizes)
        self.q2 = Critic(obs_dim, action_dim, hidden_sizes)
    
    def forward(self, obs, action):
        return self.q1(obs, action), self.q2(obs, action)


# ============================================================================
# SAC Agent
# ============================================================================

class SACAgent:
    """Soft Actor-Critic agent with automatic entropy tuning."""
    
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_sizes=(256, 256),
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        device='cpu',
    ):
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.action_dim = action_dim
        
        # Networks
        self.actor = GaussianActor(obs_dim, action_dim, hidden_sizes).to(device)
        self.critic = DoubleCritic(obs_dim, action_dim, hidden_sizes).to(device)
        self.critic_target = DoubleCritic(obs_dim, action_dim, hidden_sizes).to(device)
        
        # Copy parameters to target
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Entropy tuning
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -action_dim  # Heuristic: -dim(A)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
    
    def get_action(self, obs, deterministic=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        return self.actor.get_action(obs, deterministic)
    
    def update(self, replay_buffer, batch_size=256):
        obs, actions, rewards, next_obs, dones = replay_buffer.sample(batch_size)
        
        # ---- Critic update ----
        with torch.no_grad():
            next_actions, next_logprobs = self.actor(next_obs)
            q1_target, q2_target = self.critic_target(next_obs, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_logprobs
            td_target = rewards + self.gamma * (1 - dones) * q_target
        
        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---- Actor update ----
        new_actions, logprobs = self.actor(obs)
        q1_new, q2_new = self.critic(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * logprobs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ---- Alpha (entropy temperature) update ----
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (logprobs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # Clamp alpha to prevent collapse (keep exploring)
            with torch.no_grad():
                self.log_alpha.clamp_(min=np.log(0.01), max=np.log(1.0))
            
            self.alpha = self.log_alpha.exp().item()
        
        # ---- Soft update target networks ----
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
        }
    
    def save(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_alpha else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.auto_alpha else None,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath, resume_training=False):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        
        if resume_training:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            if self.auto_alpha and checkpoint.get('log_alpha') is not None:
                self.log_alpha.data.copy_(checkpoint['log_alpha'].data)
                self.alpha = self.log_alpha.exp().item()
                if checkpoint.get('alpha_optimizer') is not None:
                    self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            print(f"Resumed training from {filepath}")
        else:
            print(f"Model loaded from {filepath} (eval mode)")


# ============================================================================
# Environment Wrapper
# ============================================================================

class RoboSuiteEnv:
    """Wrapper for RoboSuite environment with observation processing."""
    
    def __init__(self, render=False):
        # Load JOINT_VELOCITY controller (which we know works with Rover2026)
        arm_controller_config = suite.load_part_controller_config(default_controller="JOINT_VELOCITY")
        controller_config = refactor_composite_controller_config(arm_controller_config, "Rover2026", ["right"])
        
        self.env = suite.make(
            env_name="Lift",
            robots=["Rover2026"],
            controller_configs=controller_config,
            has_renderer=render,
            has_offscreen_renderer=False,
            render_camera="agentview",
            ignore_done=False,
            use_camera_obs=False,
            control_freq=20,
            horizon=200,  # Max steps per episode
            reward_shaping=True,  # Dense rewards for easier learning
        )
        
        self.render_enabled = render
        
        # Determine observation and action dimensions
        self._setup_spaces()
    
    def _setup_spaces(self):
        """Setup observation and action space dimensions."""
        # Get a sample observation
        obs = self.env.reset()
        
        # Select relevant observations for learning
        self.obs_keys = [
            'robot0_joint_pos',      # 6D - joint positions
            'robot0_joint_vel',      # 6D - joint velocities  
            'robot0_eef_pos',        # 3D - end effector position
            'robot0_eef_quat',       # 4D - end effector orientation
            'robot0_gripper_qpos',   # 2D - gripper position
            'cube_pos',              # 3D - cube position
            'gripper_to_cube_pos',   # 3D - relative position
        ]
        
        # Calculate observation dimension
        self.obs_dim = sum(len(np.array(obs[key]).flatten()) for key in self.obs_keys if key in obs)
        
        # Action dimension: 6 joint velocities + 1 gripper
        self.action_dim = self.env.action_dim
        
        print(f"Observation dim: {self.obs_dim}")
        print(f"Action dim: {self.action_dim}")
    
    def _process_obs(self, obs):
        """Convert observation dict to flat numpy array."""
        obs_list = []
        for key in self.obs_keys:
            if key in obs:
                obs_list.append(np.array(obs[key]).flatten())
        return np.concatenate(obs_list).astype(np.float32)
    
    def reset(self):
        obs = self.env.reset()
        return self._process_obs(obs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Get positions
        eef_pos = obs.get('robot0_eef_pos', [0, 0, 0])
        cube_pos = obs.get('cube_pos', [0, 0, 0])
        gripper_to_cube = obs.get('gripper_to_cube_pos', [0, 0, 0])
        
        # Distance-based shaping (guide arm toward cube)
        distance = np.linalg.norm(gripper_to_cube)
        reach_reward = 1.0 - np.tanh(5.0 * distance)  # 0-1 based on proximity
        
        # Height bonus (encourage lifting)
        cube_height = cube_pos[2]
        lift_reward = 0.0
        if cube_height > 0.82:  # Table height ~0.8
            lift_reward = 10.0 * (cube_height - 0.82)
        
        # Grasp bonus
        grasp_reward = 0.0
        if distance < 0.05:  # Very close to cube
            grasp_reward = 5.0
        
        # Combine rewards
        shaped_reward = reward + reach_reward + lift_reward + grasp_reward
        
        return self._process_obs(obs), shaped_reward, done, info
    
    def render(self):
        if self.render_enabled:
            self.env.render()
    
    def close(self):
        self.env.close()


# ============================================================================
# Training Loop
# ============================================================================

def train(args):
    """Main training function."""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment
    env = RoboSuiteEnv(render=args.render)
    
    # Create agent
    agent = SACAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_sizes=(256, 256),
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        device=device,
    )
    
    # Resume from checkpoint if specified
    start_episode = 1
    if args.resume:
        if os.path.exists(args.resume):
            agent.load(args.resume, resume_training=True)
            # Try to extract episode number from filename
            basename = os.path.basename(args.resume)
            if 'ep' in basename:
                try:
                    start_episode = int(basename.split('ep')[1].split('.')[0]) + 1
                    print(f"Resuming from episode {start_episode}")
                except:
                    pass
        else:
            print(f"Warning: Resume checkpoint {args.resume} not found, starting fresh")
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        capacity=args.buffer_size,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=device,
    )
    
    # Training metrics
    episode_rewards = deque(maxlen=100)
    best_avg_reward = -float('inf')
    
    # Create save directory (use existing if resuming, else new)
    if args.resume and os.path.dirname(args.resume):
        save_dir = os.path.dirname(args.resume)
    else:
        save_dir = os.path.join(ROOT, "checkpoints", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Starting SAC Training for Rover2026 Lift Task")
    print("="*60)
    print(f"Episodes: {start_episode} -> {args.episodes}")
    print(f"Buffer size: {args.buffer_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Save directory: {save_dir}")
    if args.resume:
        print(f"Resumed from: {args.resume}")
    print("="*60 + "\n")
    
    # Adjust total_steps if resuming (approximate)
    total_steps = (start_episode - 1) * 200  # Approximate based on horizon
    
    for episode in range(start_episode, args.episodes + 1):
        obs = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            # Select action (skip warmup if resuming with trained model)
            if total_steps < args.warmup_steps and not args.resume:
                # Random action during warmup
                action = np.random.uniform(-1, 1, env.action_dim)
            else:
                action = agent.get_action(obs, deterministic=False)
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            
            # Store transition
            replay_buffer.add(obs, action, reward, next_obs, float(done))
            
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # Render if enabled
            if args.render:
                env.render()
            
            # Update agent (start immediately if resuming)
            should_update = (total_steps >= args.warmup_steps) or (args.resume and replay_buffer.size >= args.batch_size)
            if should_update and replay_buffer.size >= args.batch_size and total_steps % args.update_freq == 0:
                for _ in range(args.updates_per_step):
                    metrics = agent.update(replay_buffer, args.batch_size)
        
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards)
        
        # Logging
        if episode % args.log_freq == 0:
            print(f"Episode {episode:5d} | "
                  f"Steps: {episode_steps:3d} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Avg(100): {avg_reward:8.2f} | "
                  f"Alpha: {agent.alpha:.4f}")
        
        # Save best model
        if avg_reward > best_avg_reward and episode > start_episode + 50:
            best_avg_reward = avg_reward
            agent.save(os.path.join(save_dir, "best_model.pt"))
        
        # Periodic save
        if episode % args.save_freq == 0:
            agent.save(os.path.join(save_dir, f"model_ep{episode}.pt"))
    
    # Final save
    agent.save(os.path.join(save_dir, "final_model.pt"))
    env.close()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best average reward: {best_avg_reward:.2f}")
    print(f"Models saved to: {save_dir}")
    print("="*60)


def evaluate(args):
    """Evaluate a trained model."""
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    
    # Create environment with rendering
    env = RoboSuiteEnv(render=True)
    
    # Create and load agent
    agent = SACAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=device,
    )
    agent.load(args.eval)
    
    print("\n" + "="*60)
    print("Evaluating trained agent")
    print("="*60 + "\n")
    
    for episode in range(args.eval_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            env.render()
            time.sleep(0.02)  # Slow down for visualization
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    env.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAC Training for Rover2026 Lift Task")
    
    # Mode
    parser.add_argument('--train', action='store_true', help='Train a new agent')
    parser.add_argument('--eval', type=str, default=None, help='Path to model for evaluation')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    
    # Training hyperparameters
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--buffer_size', type=int, default=1000000, help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for updates (larger = better GPU utilization)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Random actions before training')
    parser.add_argument('--update_freq', type=int, default=1, help='Steps between updates')
    parser.add_argument('--updates_per_step', type=int, default=4, help='Gradient updates per step (higher = more GPU work)')
    
    # Logging and saving
    parser.add_argument('--log_freq', type=int, default=10, help='Episodes between logging')
    parser.add_argument('--save_freq', type=int, default=100, help='Episodes between saves')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Episodes for evaluation')
    
    # Environment
    parser.add_argument('--render', action='store_true', help='Render during training')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    
    args = parser.parse_args()
    
    if args.eval:
        evaluate(args)
    elif args.train:
        train(args)
    else:
        print("Please specify --train or --eval <model_path>")
        print("Example: python train_lift.py --train")
        print("Example: python train_lift.py --eval checkpoints/best_model.pt")
