#!/usr/bin/env python3
"""
SAC (Soft Actor-Critic) Training for Rover2026 Arm

Hardware Optimizations:
- RTX 5070 Ti: BF16 mixed precision, torch.compile(), fused AdamW, GPU-resident replay buffer
- Ryzen 9 9900X: Parallel environment sampling across 12 cores

Usage:
    python train_lift.py --train --cuda                    # Train with defaults
    python train_lift.py --train --cuda --num_envs 12      # Custom parallelism  
    python train_lift.py --eval checkpoints/best_model.pt  # Evaluate
"""

import os
import sys
import time
import argparse
import numpy as np
import multiprocessing as mp
from collections import deque
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.cuda.amp import GradScaler

# Path setup
ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite")
sys.path.insert(0, ROBO_PATH)

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config


# ============================================================================
# GPU-Resident Replay Buffer (Minimizes PCIe Bottleneck)
# ============================================================================

class GPUReplayBuffer:
    """
    Replay buffer that lives entirely on GPU VRAM.
    Batched CPU->GPU transfers reduce PCIe overhead.
    """
    
    def __init__(self, capacity, obs_dim, action_dim, device='cuda'):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate GPU tensors
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        
        # CPU staging buffer with pinned memory for async transfers
        self._buffer_size = 256
        self._buffer_ptr = 0
        self._obs_buf = torch.zeros((self._buffer_size, obs_dim), dtype=torch.float32, pin_memory=True)
        self._act_buf = torch.zeros((self._buffer_size, action_dim), dtype=torch.float32, pin_memory=True)
        self._rew_buf = torch.zeros((self._buffer_size, 1), dtype=torch.float32, pin_memory=True)
        self._next_buf = torch.zeros((self._buffer_size, obs_dim), dtype=torch.float32, pin_memory=True)
        self._done_buf = torch.zeros((self._buffer_size, 1), dtype=torch.float32, pin_memory=True)
        
        self._stream = torch.cuda.Stream(device=device)
    
    def add(self, obs, action, reward, next_obs, done):
        """Stage transition in CPU buffer, flush to GPU when full."""
        idx = self._buffer_ptr
        self._obs_buf[idx] = torch.from_numpy(obs) if isinstance(obs, np.ndarray) else obs
        self._act_buf[idx] = torch.from_numpy(action) if isinstance(action, np.ndarray) else action
        self._rew_buf[idx, 0] = reward
        self._next_buf[idx] = torch.from_numpy(next_obs) if isinstance(next_obs, np.ndarray) else next_obs
        self._done_buf[idx, 0] = done
        self._buffer_ptr += 1
        
        if self._buffer_ptr >= self._buffer_size:
            self._flush()
    
    def _flush(self):
        """Batch transfer staged data to GPU."""
        if self._buffer_ptr == 0:
            return
        n = self._buffer_ptr
        
        with torch.cuda.stream(self._stream):
            if self.ptr + n <= self.capacity:
                end = self.ptr + n
                self.obs[self.ptr:end] = self._obs_buf[:n].to(self.device, non_blocking=True)
                self.actions[self.ptr:end] = self._act_buf[:n].to(self.device, non_blocking=True)
                self.rewards[self.ptr:end] = self._rew_buf[:n].to(self.device, non_blocking=True)
                self.next_obs[self.ptr:end] = self._next_buf[:n].to(self.device, non_blocking=True)
                self.dones[self.ptr:end] = self._done_buf[:n].to(self.device, non_blocking=True)
            else:
                # Handle wrap-around
                first = self.capacity - self.ptr
                self.obs[self.ptr:] = self._obs_buf[:first].to(self.device, non_blocking=True)
                self.obs[:n-first] = self._obs_buf[first:n].to(self.device, non_blocking=True)
                self.actions[self.ptr:] = self._act_buf[:first].to(self.device, non_blocking=True)
                self.actions[:n-first] = self._act_buf[first:n].to(self.device, non_blocking=True)
                self.rewards[self.ptr:] = self._rew_buf[:first].to(self.device, non_blocking=True)
                self.rewards[:n-first] = self._rew_buf[first:n].to(self.device, non_blocking=True)
                self.next_obs[self.ptr:] = self._next_buf[:first].to(self.device, non_blocking=True)
                self.next_obs[:n-first] = self._next_buf[first:n].to(self.device, non_blocking=True)
                self.dones[self.ptr:] = self._done_buf[:first].to(self.device, non_blocking=True)
                self.dones[:n-first] = self._done_buf[first:n].to(self.device, non_blocking=True)
        
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)
        self._buffer_ptr = 0
    
    def sample(self, batch_size):
        """Sample directly from GPU memory - zero PCIe overhead."""
        if self._buffer_ptr > 0:
            self._flush()
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.obs[idxs], self.actions[idxs], self.rewards[idxs], self.next_obs[idxs], self.dones[idxs]


# ============================================================================
# Neural Networks (Optimized for RTX 5070 Ti)
# ============================================================================

def mlp(sizes, activation=nn.SiLU, output_activation=nn.Identity):
    """MLP with LayerNorm for mixed-precision stability."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(activation())
        else:
            layers.append(output_activation())
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Gaussian policy for continuous actions with tanh squashing."""
    
    LOG_STD_MIN, LOG_STD_MAX = -20, 2
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=(512, 512, 256)):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes))
        self.mu_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dim)
    
    @torch.compile()
    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = torch.clamp(self.log_std_layer(net_out), self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        
        dist = Normal(mu, std)
        action = mu if deterministic else dist.rsample()
        
        logprob = None
        if with_logprob:
            logprob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            logprob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(dim=-1, keepdim=True)
        
        return torch.tanh(action), logprob
    
    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            action, _ = self.forward(obs, deterministic=deterministic, with_logprob=False)
            return action.cpu().numpy()[0]


class Critic(nn.Module):
    """Q-function network."""
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=(512, 512, 256)):
        super().__init__()
        self.q = mlp([obs_dim + action_dim] + list(hidden_sizes) + [1])
    
    @torch.compile()
    def forward(self, obs, action):
        return self.q(torch.cat([obs, action], dim=-1))


class DoubleCritic(nn.Module):
    """Twin Q-networks for reduced overestimation bias."""
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=(512, 512, 256)):
        super().__init__()
        self.q1 = Critic(obs_dim, action_dim, hidden_sizes)
        self.q2 = Critic(obs_dim, action_dim, hidden_sizes)
    
    def forward(self, obs, action):
        return self.q1(obs, action), self.q2(obs, action)


# ============================================================================
# SAC Agent (GPU-Optimized)
# ============================================================================

class SACAgent:
    """
    Soft Actor-Critic agent with automatic entropy tuning.
    
    GPU Optimizations:
    - Mixed precision training (BF16 on RTX 5070 Ti)
    - Fused AdamW optimizer
    - torch.compile() on networks
    - Gradient scaling for FP16 stability
    """
    
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_sizes=(512, 512, 256),  # Larger networks for GPU
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        device='cuda',
        use_amp=True,  # Automatic mixed precision
    ):
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.action_dim = action_dim
        self.use_amp = use_amp and device != 'cpu'
        
        # Determine best dtype for this GPU (BF16 preferred on Ampere+)
        if self.use_amp:
            # RTX 5070 Ti (Blackwell) supports BF16 natively
            self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print(f"Using mixed precision with {self.amp_dtype}")
        
        # Networks (larger for better GPU utilization)
        self.actor = GaussianActor(obs_dim, action_dim, hidden_sizes).to(device)
        self.critic = DoubleCritic(obs_dim, action_dim, hidden_sizes).to(device)
        self.critic_target = DoubleCritic(obs_dim, action_dim, hidden_sizes).to(device)
        
        # Copy parameters to target
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Fused optimizers (fewer kernel launches, better GPU efficiency)
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(), 
            lr=lr, 
            fused=True if device != 'cpu' else False,
            weight_decay=1e-5
        )
        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(), 
            lr=lr, 
            fused=True if device != 'cpu' else False,
            weight_decay=1e-5
        )
        
        # Gradient scaler for mixed precision (only needed for FP16)
        self.scaler = GradScaler(enabled=(self.use_amp and self.amp_dtype == torch.float16))
        
        # Entropy tuning
        self.auto_alpha = auto_alpha
        if auto_alpha:
            # Lower target entropy = less exploration, more exploitation
            # Use -0.5 * dim(A) for faster convergence after initial exploration
            self.target_entropy = -0.5 * action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
    
    def get_action(self, obs, deterministic=False):
        # Mark step for CUDA graphs if they are used
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
            
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.actor.get_action(obs, deterministic)
    
    def update(self, replay_buffer, batch_size=1024):
        """Update networks with mixed precision for GPU efficiency."""
        # Mark step for CUDA graphs stability
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
            
        obs, actions, rewards, next_obs, dones = replay_buffer.sample(batch_size)
        
        # ---- Critic update with AMP ----
        with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            with torch.no_grad():
                next_actions, next_logprobs = self.actor(next_obs)
                # Clone outputs to prevent CUDA graph overwriting issues when passing to another module
                next_actions = next_actions.clone()
                next_logprobs = next_logprobs.clone()
                
                q1_target, q2_target = self.critic_target(next_obs, next_actions)
                q_target = torch.min(q1_target, q2_target) - self.alpha * next_logprobs
                td_target = rewards + self.gamma * (1 - dones) * q_target
            
            q1, q2 = self.critic(obs, actions)
            critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        
        self.critic_optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # Gradient clipping
        self.scaler.step(self.critic_optimizer)
        
        # ---- Actor update with AMP ----
        with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            # Mark step again before second model invocation block
            if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
                
            new_actions, logprobs = self.actor(obs)
            q1_new, q2_new = self.critic(obs, new_actions)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha * logprobs - q_new).mean()
        
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.scaler.step(self.actor_optimizer)
        
        # Update scaler
        self.scaler.update()
        
        # ---- Alpha (entropy temperature) update ----
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (logprobs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # Clamp alpha to prevent collapse (keep exploring)
            with torch.no_grad():
                self.log_alpha.clamp_(min=np.log(0.01), max=np.log(1.0))
            
            self.alpha = self.log_alpha.exp().item()
        
        # ---- Soft update target networks (in-place for efficiency) ----
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.lerp_(param.data, self.tau)
        
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
            'scaler': self.scaler.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath, resume_training=False):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
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
            if checkpoint.get('scaler') is not None:
                self.scaler.load_state_dict(checkpoint['scaler'])
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
            render_camera="frontview",  # Better view of the arm
            ignore_done=False,
            use_camera_obs=False,
            control_freq=20,
            horizon=200,  # Max steps per episode
            reward_shaping=True,  # Dense rewards for easier learning
        )
        
        self.render_enabled = render
        self._camera_initialized = False
        
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
        # Note: Don't print here - SubprocVecEnv prints once for all envs
    
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
        cube_pos = obs.get('cube_pos', [0, 0, 0])
        gripper_to_cube = obs.get('gripper_to_cube_pos', [0, 0, 0])
        gripper_qpos = obs.get('robot0_gripper_qpos', [0, 0])
        
        # Distance to cube
        distance = np.linalg.norm(gripper_to_cube)
        
        # Phase 1: Reach toward cube (0-2 reward)
        reach_reward = 2.0 * (1.0 - np.tanh(3.0 * distance))
        
        # Phase 2: Grasp reward - encourage closing gripper when close
        grasp_reward = 0.0
        gripper_closed = np.mean(gripper_qpos) < 0.02  # Gripper is closing
        if distance < 0.08:  # Close to cube
            grasp_reward = 3.0
            if gripper_closed:
                grasp_reward = 8.0  # Big bonus for grasping
        
        # Phase 3: Lift reward - MUCH stronger incentive
        cube_height = cube_pos[2]
        table_height = 0.82
        lift_reward = 0.0
        if cube_height > table_height:
            height_above_table = cube_height - table_height
            # Exponential reward for lifting - gets very large for high lifts
            lift_reward = 50.0 * height_above_table + 100.0 * (height_above_table ** 2)
            
            # Bonus for sustained lift while gripper closed
            if gripper_closed and height_above_table > 0.05:
                lift_reward += 20.0
        
        # Success bonus
        success_reward = 0.0
        if cube_height > table_height + 0.1:  # Lifted 10cm
            success_reward = 100.0
        
        # Combine rewards (base reward from env + shaped rewards)
        shaped_reward = reward + reach_reward + grasp_reward + lift_reward + success_reward
        
        return self._process_obs(obs), shaped_reward, done, info
    
    def render(self):
        if self.render_enabled:
            self.env.render()
            # Adjust camera on first render
            if not self._camera_initialized:
                try:
                    # Try to access the MuJoCo viewer and adjust camera
                    viewer = self.env.viewer
                    if hasattr(viewer, 'viewer') and viewer.viewer is not None:
                        viewer.viewer.cam.distance = 2.5
                        viewer.viewer.cam.elevation = -25
                        viewer.viewer.cam.azimuth = 135
                        self._camera_initialized = True
                except Exception:
                    pass  # Camera adjustment not supported, use default view
    
    def close(self):
        self.env.close()


def worker(remote, parent_remote, env_fn):
    """Worker process for SubprocVecEnv."""
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                remote.send((obs, reward, done, info))
            elif cmd == 'reset':
                obs = env.reset()
                remote.send(obs)
            elif cmd == 'get_spaces':
                remote.send((env.obs_dim, env.action_dim))
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError
    except EOFError:
        pass


class SubprocVecEnv:
    """Parallel environments using multiprocessing."""
    
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote, env_fn))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
            
        self.remotes[0].send(('get_spaces', None))
        self.obs_dim, self.action_dim = self.remotes[0].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.array(rews), np.array(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


# ============================================================================
# Training Loop (GPU-Optimized)
# ============================================================================

def train(args):
    """Main training function with GPU and CPU optimizations."""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    
    if args.cuda and device.type == 'cpu':
        print("\n" + "!"*60)
        print("CRITICAL ERROR: CUDA REQUESTED BUT NO GPU DETECTED!")
        print("Please check if nvidia-container-toolkit is installed and ")
        print("verify you ran docker with --gpus all.")
        print("!"*60 + "\n")
        sys.exit(1)
        
    print(f"Using device: {device}")
    
    # GPU-specific optimizations
    if device.type == 'cuda':
        # Enable TF32 for faster matmuls on Ampere+ GPUs (RTX 30xx, 40xx, 50xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # Optimize convolution algorithms
        
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
    # CPU Optimization for Ryzen 9 9900X (24 threads)
    # Set intra-op threads to allow envs to have more CPU slices
    torch.set_num_threads(max(1, min(8, mp.cpu_count() // args.num_envs)))
    print(f"CPU Threads per Torch Op: {torch.get_num_threads()}")
    
    # Create parallel environments
    def make_env():
        return RoboSuiteEnv(render=False)
    
    print(f"Launching {args.num_envs} parallel environments on Ryzen 9 9900X...")
    env = SubprocVecEnv([make_env for _ in range(args.num_envs)])
    
    # Create agent with larger networks for GPU
    agent = SACAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_sizes=(512, 512, 256),  # Larger network for better GPU utilization
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        device=device,
        use_amp=args.cuda,  # Enable mixed precision on GPU
    )
    
    # Resume from checkpoint if specified
    start_episode = 1
    if args.resume:
        if os.path.exists(args.resume):
            agent.load(args.resume, resume_training=True)
            basename = os.path.basename(args.resume)
            if 'ep' in basename:
                try:
                    start_episode = int(basename.split('ep')[1].split('.')[0]) + 1
                except:
                    pass
        else:
            print(f"[!] Checkpoint {args.resume} not found, starting fresh")
    
    # Create GPU-resident replay buffer
    replay_buffer = GPUReplayBuffer(
        capacity=args.buffer_size,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=device,
    )
    
    # Training metrics
    episode_rewards = np.zeros(args.num_envs)
    episode_steps = np.zeros(args.num_envs)
    all_episode_rewards = deque(maxlen=100)
    best_avg_reward = -float('inf')
    
    # Create save directory
    if args.resume and os.path.dirname(args.resume):
        save_dir = os.path.dirname(args.resume)
    else:
        save_dir = os.path.join(ROOT, "checkpoints", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    
    # Print training configuration
    print("\n" + "="*70)
    print("  SAC TRAINING - Rover2026 Lift Task")
    print("="*70)
    print(f"  Hardware:")
    print(f"    GPU: {torch.cuda.get_device_name(0)} (BF16 AMP enabled)")
    print(f"    CPU: {mp.cpu_count()} threads, {args.num_envs} parallel envs")
    print(f"  Training:")
    print(f"    Episodes: {args.episodes} | Batch: {args.batch_size} | Updates/step: {args.updates_per_step}")
    print(f"    Buffer: {args.buffer_size:,} | Warmup: {args.warmup_steps:,} steps")
    print(f"  Save: {save_dir}")
    print("="*70)
    print("\n  [Warmup] Collecting random samples..." if not args.resume else f"\n  [Resume] Starting from episode {start_episode}")
    
    # Timing and metrics
    total_steps = (start_episode - 1) * 200
    update_times = deque(maxlen=100)
    train_start = time.time()
    
    obs = env.reset()
    for episode in range(start_episode, args.episodes + 1):
        while True:
            # Select actions
            if total_steps < args.warmup_steps and not args.resume:
                actions = np.random.uniform(-1, 1, (args.num_envs, env.action_dim))
            else:
                with torch.no_grad():
                    if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                        torch.compiler.cudagraph_mark_step_begin()
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    actions_t, _ = agent.actor(obs_t, deterministic=False)
                    actions = actions_t.cpu().numpy()
                    actions = actions_t.cpu().numpy()
            
            # Step all environments
            next_obs, rewards, dones, infos = env.step(actions)
            
            # Store transitions
            for i in range(args.num_envs):
                replay_buffer.add(obs[i], actions[i], rewards[i], next_obs[i], float(dones[i]))
                episode_rewards[i] += rewards[i]
                episode_steps[i] += 1
                
                if dones[i]:
                    all_episode_rewards.append(episode_rewards[i])
                    episode_rewards[i] = 0
                    episode_steps[i] = 0
            
            obs = next_obs
            total_steps += args.num_envs
            
            # Update agent (GPU-intensive SAC updates)
            should_update = (total_steps >= args.warmup_steps) or (args.resume and replay_buffer.size >= args.batch_size)
            if should_update and replay_buffer.size >= args.batch_size:
                update_start = time.perf_counter()
                for _ in range(args.updates_per_step):
                    metrics = agent.update(replay_buffer, args.batch_size)
                torch.cuda.synchronize()
                update_times.append(time.perf_counter() - update_start)
                
            if any(dones):
                break
        
        avg_reward = np.mean(all_episode_rewards) if all_episode_rewards else 0
        
        # Logging (only at log_freq intervals)
        if episode % args.log_freq == 0:
            avg_update_time = np.mean(update_times) * 1000 if update_times else 0
            elapsed = time.time() - train_start
            sps = total_steps / elapsed if elapsed > 0 else 0
            
            print(f"  Ep {episode:5d} | Steps {total_steps:8,} | "
                  f"Reward {avg_reward:7.1f} | Alpha {agent.alpha:.3f} | "
                  f"Update {avg_update_time:5.1f}ms | SPS {sps:5.0f}")
            
            # Save best model (only check when logging to avoid spam)
            if avg_reward > best_avg_reward and episode > start_episode + 50:
                best_avg_reward = avg_reward
                agent.save(os.path.join(save_dir, "best_model.pt"))
        
        # Periodic checkpoint
        if episode % args.save_freq == 0:
            agent.save(os.path.join(save_dir, f"checkpoint_ep{episode}.pt"))
    
    # Training complete
    agent.save(os.path.join(save_dir, "final_model.pt"))
    env.close()
    
    elapsed = time.time() - train_start
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    
    print("\n" + "="*70)
    print("  TRAINING COMPLETE")
    print("="*70)
    print(f"  Time: {elapsed/60:.1f} min | Steps: {total_steps:,} | Best Reward: {best_avg_reward:.1f}")
    print(f"  Peak GPU Memory: {peak_mem:.2f} GB")
    print(f"  Models saved to: {save_dir}")
    print("="*70 + "\n")


def evaluate(args):
    """Evaluate a trained model with visualization."""
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    env = RoboSuiteEnv(render=True)
    
    agent = SACAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_sizes=(512, 512, 256),
        device=device,
        use_amp=False,
    )
    agent.load(args.eval)
    
    print("\n" + "="*70)
    print("  EVALUATION MODE")
    print("="*70)
    print(f"  Model: {args.eval}")
    print(f"  Episodes: {args.eval_episodes}")
    print("="*70 + "\n")
    
    rewards = []
    for episode in range(args.eval_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            env.render()
            time.sleep(0.02)
        
        rewards.append(episode_reward)
        print(f"  Episode {episode + 1:3d}: {episode_reward:7.1f}")
    
    env.close()
    print(f"\n  Mean Reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAC Training for Rover2026 Lift Task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    mode = parser.add_argument_group("Mode")
    mode.add_argument('--train', action='store_true', help='Train a new agent')
    mode.add_argument('--eval', type=str, default=None, metavar='PATH', help='Evaluate model at PATH')
    mode.add_argument('--resume', type=str, default=None, metavar='PATH', help='Resume training from checkpoint')
    
    # Training hyperparameters  
    train_args = parser.add_argument_group("Training")
    train_args.add_argument('--episodes', type=int, default=1000, help='Training episodes')
    train_args.add_argument('--buffer_size', type=int, default=1_000_000, help='Replay buffer size')
    train_args.add_argument('--batch_size', type=int, default=1024, help='Batch size (larger = better GPU util)')
    train_args.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    train_args.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    train_args.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient')
    train_args.add_argument('--warmup_steps', type=int, default=5000, help='Random exploration steps')
    train_args.add_argument('--updates_per_step', type=int, default=8, help='Gradient updates per env step')
    
    # Logging/Saving
    log_args = parser.add_argument_group("Logging")
    log_args.add_argument('--log_freq', type=int, default=10, help='Episodes between logs')
    log_args.add_argument('--save_freq', type=int, default=100, help='Episodes between checkpoints')
    log_args.add_argument('--eval_episodes', type=int, default=10, help='Evaluation episodes')
    
    # Hardware
    hw_args = parser.add_argument_group("Hardware")
    hw_args.add_argument('--cuda', action='store_true', help='Use CUDA GPU acceleration')
    hw_args.add_argument('--num_envs', type=int, default=16, help='Parallel environments (8-16 for Ryzen 9)')
    hw_args.add_argument('--render', action='store_true', help='Render during training')
    
    args = parser.parse_args()
    
    if args.eval:
        evaluate(args)
    elif args.train:
        train(args)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python train_lift.py --train --cuda")
        print("  python train_lift.py --eval checkpoints/best_model.pt --cuda")
