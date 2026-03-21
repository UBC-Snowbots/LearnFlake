#!/usr/bin/env python3
"""
Hierarchical SAC Training for Rover2026 Arm - Version 2

Improvements over V1:
- Camera-based observations (CNN encoder for visual input)
- Domain randomization (random cube positions, lighting, textures)
- Hierarchical policy (high-level skill selector + low-level motor control)
- Smoother actions via action smoothing/filtering

Hardware Optimizations (preserved from V1):
- RTX 5070 Ti: BF16 mixed precision, torch.compile(), fused AdamW, GPU-resident buffer
- Ryzen 9 9900X: Parallel environment sampling across 12 cores

Usage:
    python train_lift_v2.py --train --cuda                    # Train with defaults
    python train_lift_v2.py --train --cuda --use_camera       # Train with camera obs
    python train_lift_v2.py --eval checkpoints/best_model.pt  # Evaluate
"""

import os
import sys
import time
import argparse
import numpy as np
import multiprocessing as mp
from collections import deque
from datetime import datetime
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler
from torch.distributions import Normal, Categorical

# Path setup
ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite")
sys.path.insert(0, ROBO_PATH)

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config


# ============================================================================
# GPU-Resident Replay Buffer (Supports both state and image observations)
# ============================================================================

class GPUReplayBuffer:
    """
    GPU-resident replay buffer supporting both state and image observations.

    Function:
    - Stores transitions for off-policy SAC training.
    - Keeps state/action/reward/done on GPU for fast sampling.
    - Keeps images on CPU to avoid GPU memory pressure.

    Inputs (stored per transition):
    - obs: float32 state vector of shape [obs_dim]
    - action: float32 action vector of shape [action_dim]
    - reward: float scalar
    - next_obs: float32 state vector
    - done: float/bool terminal flag
    - image / next_image: uint8 CHW image if use_images=True

    Outputs:
    - sample(batch_size) returns a dict of torch tensors on GPU:
      obs, actions, rewards, next_obs, dones, and optional images.
    """
    
    def __init__(self, capacity, obs_dim, action_dim, device='cuda',
                 use_images=False, image_shape=(3, 84, 84),
                 her_ratio=0.8, her_success_reward=100.0):
        """
        Args:
            capacity: max number of transitions for state/action buffers.
            obs_dim: length of state observation vector.
            action_dim: length of action vector.
            device: torch device for GPU-resident tensors.
            use_images: whether to store and sample image observations.
            image_shape: CHW shape for images (uint8 on CPU).
        """
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.use_images = use_images
        self.image_shape = image_shape
        self.her_ratio = float(her_ratio)
        self.her_success_reward = float(her_success_reward)
        # Ring-buffer pointers:
        # - ptr: next write location
        # - size: number of valid transitions currently stored (<= capacity)
        
        # Pre-allocate GPU tensors for state observations
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        
        # Images stored on CPU (too large for GPU) with memory-mapped numpy arrays
        if use_images:
            # Use smaller capacity for images to save memory
            self.image_capacity = min(capacity, 100_000)  # Max 100k images
            self.images = np.zeros((self.image_capacity, *image_shape), dtype=np.uint8)
            self.next_images = np.zeros((self.image_capacity, *image_shape), dtype=np.uint8)
            self.image_ptr = 0
            self.image_size = 0
        
        # CPU staging buffer with pinned memory
        # We stage multiple transitions on CPU first, then copy in a single batched transfer.
        # This amortizes host->device overhead compared to copying one sample at a time.
        self._buffer_size = 256
        self._buffer_ptr = 0
        self._obs_buf = torch.zeros((self._buffer_size, obs_dim), dtype=torch.float32, pin_memory=True)
        self._act_buf = torch.zeros((self._buffer_size, action_dim), dtype=torch.float32, pin_memory=True)
        self._rew_buf = torch.zeros((self._buffer_size, 1), dtype=torch.float32, pin_memory=True)
        self._next_buf = torch.zeros((self._buffer_size, obs_dim), dtype=torch.float32, pin_memory=True)
        self._done_buf = torch.zeros((self._buffer_size, 1), dtype=torch.float32, pin_memory=True)
        
        if use_images:
            self._img_buf = np.zeros((self._buffer_size, *image_shape), dtype=np.uint8)
            self._next_img_buf = np.zeros((self._buffer_size, *image_shape), dtype=np.uint8)
        
        self._stream = torch.cuda.Stream(device=device)
    
    def add(self, obs, action, reward, next_obs, done, image=None, next_image=None):
        """
        Stage a transition in the CPU staging buffer.

        Inputs:
        - obs, action, next_obs: np.ndarray or torch.Tensor
        - reward, done: scalars
        - image, next_image: optional uint8 CHW arrays (if use_images)
        """
        idx = self._buffer_ptr
        # Accept both numpy and torch inputs from callers; normalize to torch staging tensors.
        self._obs_buf[idx] = torch.from_numpy(obs) if isinstance(obs, np.ndarray) else obs
        self._act_buf[idx] = torch.from_numpy(action) if isinstance(action, np.ndarray) else action
        self._rew_buf[idx, 0] = reward
        self._next_buf[idx] = torch.from_numpy(next_obs) if isinstance(next_obs, np.ndarray) else next_obs
        self._done_buf[idx, 0] = done
        
        if self.use_images and image is not None:
            # Images are kept on CPU to avoid large VRAM pressure for replay storage.
            self._img_buf[idx] = image if isinstance(image, np.ndarray) else image.numpy()
            self._next_img_buf[idx] = next_image if isinstance(next_image, np.ndarray) else next_image.numpy()
        
        self._buffer_ptr += 1
        if self._buffer_ptr >= self._buffer_size:
            self._flush()
    
    def _flush(self):
        """
        Flush staged CPU data to GPU tensors in a single batch.

        Outputs:
        - Updates internal GPU buffers and pointers; no return value.
        """
        if self._buffer_ptr == 0:
            return
        n = self._buffer_ptr

        
        # Copy staged data to GPU asynchronously on a dedicated CUDA stream.
        with torch.cuda.stream(self._stream):
            end = (self.ptr + n) % self.capacity
            if self.ptr + n <= self.capacity:
                self.obs[self.ptr:self.ptr + n] = self._obs_buf[:n].to(self.device, non_blocking=True)
                self.actions[self.ptr:self.ptr + n] = self._act_buf[:n].to(self.device, non_blocking=True)
                self.rewards[self.ptr:self.ptr + n] = self._rew_buf[:n].to(self.device, non_blocking=True)
                self.next_obs[self.ptr:self.ptr + n] = self._next_buf[:n].to(self.device, non_blocking=True)
                self.dones[self.ptr:self.ptr + n] = self._done_buf[:n].to(self.device, non_blocking=True)
            else:
                # Handle wrap-around
                # When we cross end-of-buffer, split write into tail + head slices.
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
        
        # Store images on CPU (separate pointer for smaller image buffer)
        if self.use_images:
            for i in range(n):
                img_idx = (self.image_ptr + i) % self.image_capacity
                self.images[img_idx] = self._img_buf[i]
                self.next_images[img_idx] = self._next_img_buf[i]
            self.image_ptr = (self.image_ptr + n) % self.image_capacity
            self.image_size = min(self.image_size + n, self.image_capacity)
        
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)
        self._buffer_ptr = 0
    
    def sample(self, batch_size):
        """
        Sample a batch for training.

        Args:
            batch_size: number of transitions to sample.

        Returns:
            dict with tensors on GPU:
            - obs: [B, obs_dim]
            - actions: [B, action_dim]
            - rewards: [B, 1]
            - next_obs: [B, obs_dim]
            - dones: [B, 1]
            - images / next_images: [B, C, H, W] if use_images
        """
        if self._buffer_ptr > 0:
            # Ensure latest staged transitions are visible to sampling.
            self._flush()
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        # Clone to avoid mutating replay history when applying HER relabeling.
        obs = self.obs[idxs].clone()
        actions = self.actions[idxs].clone()
        rewards = self.rewards[idxs].clone()
        next_obs = self.next_obs[idxs].clone()
        dones = self.dones[idxs].clone()

        # Future-strategy HER (GPU-native):
        # - achieved goal: next_obs[..., -6:-3]
        # - desired goal:  obs/next_obs[..., -3:]
        # Relabel a random subset of batch transitions.
        her_mask = (torch.rand((batch_size,), device=self.device) < self.her_ratio)
        if her_mask.any():
            achieved_goal = next_obs[her_mask, -6:-3]
            obs[her_mask, -3:] = achieved_goal
            next_obs[her_mask, -3:] = achieved_goal
            rewards[her_mask, 0] = self.her_success_reward
            dones[her_mask, 0] = 1.0

        batch = {
            'obs': obs,
            'actions': actions,
            'rewards': rewards,
            'next_obs': next_obs,
            'dones': dones,
        }
        
        if self.use_images:
            # Sample from image buffer (may have different indices due to smaller capacity)
            # NOTE: this is intentionally separate from state/action indexing in current design.
            img_idxs = torch.randint(0, self.image_size, (batch_size,)).numpy()
            # Transfer batch to GPU
            imgs = torch.from_numpy(self.images[img_idxs]).float().to(self.device) / 255.0
            next_imgs = torch.from_numpy(self.next_images[img_idxs]).float().to(self.device) / 255.0
            batch['images'] = imgs
            batch['next_images'] = next_imgs
        
        return batch


# ============================================================================
# Neural Network Components
# ============================================================================

def mlp(sizes, activation=nn.SiLU, output_activation=nn.Identity):
    """
    Build a LayerNorm-MLP.

    Inputs:
    - sizes: list like [in, h1, h2, ..., out]
    - activation: hidden layer activation class
    - output_activation: output layer activation class

    Output:
    - nn.Sequential MLP with LayerNorm on hidden layers
    """
    layers = []
    # All hidden layers use LayerNorm to reduce scale drift during long training.
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(activation())
        else:
            layers.append(output_activation())
    return nn.Sequential(*layers)


class CNNEncoder(nn.Module):
    """CNN encoder for image observations (DrQ-style)."""
    
    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(256, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)
    
    def forward(self, x):
        """
        Inputs:
        - x: float tensor of shape [B, C, H, W] in [0, 1]

        Outputs:
        - feature vector [B, feature_dim]
        """
        # Conv stack extracts spatial features; LN on final FC stabilizes feature scale.
        h = self.conv(x)
        h = self.ln(self.fc(h))
        return h


class GaussianActor(nn.Module):
    """
    Gaussian policy with optional image encoder.

    Function:
    - Produces a squashed (tanh) continuous action distribution.
    """
    
    LOG_STD_MIN, LOG_STD_MAX = -20, 2
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=(512, 512, 256),
                 use_images=False, image_feature_dim=256):
        super().__init__()
        self.use_images = use_images
        
        if use_images:
            self.encoder = CNNEncoder(feature_dim=image_feature_dim)
            input_dim = obs_dim + image_feature_dim
        else:
            input_dim = obs_dim
        
        self.net = mlp([input_dim] + list(hidden_sizes))
        self.mu_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dim)
    
    def forward(self, obs, images=None, deterministic=False, with_logprob=True):
        """
        Inputs:
        - obs: [B, obs_dim] float tensor
        - images: optional [B, C, H, W] float tensor in [0, 1]
        - deterministic: if True, return mean action
        - with_logprob: if True, return log-prob under squashed Gaussian

        Outputs:
        - action: [B, action_dim] in [-1, 1] after tanh
        - logprob: [B, 1] or None
        """
        if self.use_images and images is not None:
            img_features = self.encoder(images)
            x = torch.cat([obs, img_features], dim=-1)
        else:
            x = obs
        
        net_out = self.net(x)
        mu = self.mu_layer(net_out)
        log_std = torch.clamp(self.log_std_layer(net_out), self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        
        dist = Normal(mu, std)
        action = mu if deterministic else dist.rsample()
        
        logprob = None
        if with_logprob:
            logprob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            # Tanh-squash correction term for log-probability.
            # Without this correction SAC entropy term is biased.
            logprob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(dim=-1, keepdim=True)
        
        return torch.tanh(action), logprob


class Critic(nn.Module):
    """
    Q-function with optional image encoder.

    Inputs:
    - obs: [B, obs_dim]
    - action: [B, action_dim]
    - images: optional [B, C, H, W]

    Output:
    - Q-value tensor [B, 1]
    """
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=(512, 512, 256),
                 use_images=False, image_feature_dim=256):
        super().__init__()
        self.use_images = use_images
        
        if use_images:
            self.encoder = CNNEncoder(feature_dim=image_feature_dim)
            input_dim = obs_dim + action_dim + image_feature_dim
        else:
            input_dim = obs_dim + action_dim
        
        self.q = mlp([input_dim] + list(hidden_sizes) + [1])
    
    def forward(self, obs, action, images=None):
        # Q(s, a) optionally conditioned on visual features.
        if self.use_images and images is not None:
            img_features = self.encoder(images)
            x = torch.cat([obs, action, img_features], dim=-1)
        else:
            x = torch.cat([obs, action], dim=-1)
        return self.q(x)


class DoubleCritic(nn.Module):
    """
    Twin Q-networks (Q1, Q2) for clipped double Q-learning.

    Output:
    - tuple (q1, q2), each [B, 1]
    """
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=(512, 512, 256),
                 use_images=False, image_feature_dim=256):
        super().__init__()
        self.q1 = Critic(obs_dim, action_dim, hidden_sizes, use_images, image_feature_dim)
        self.q2 = Critic(obs_dim, action_dim, hidden_sizes, use_images, image_feature_dim)
    
    def forward(self, obs, action, images=None):
        return self.q1(obs, action, images), self.q2(obs, action, images)


# ============================================================================
# Hierarchical Policy Components
# ============================================================================

class SkillSelector(nn.Module):
    """
    High-level policy that selects discrete skills.
    Skills: 0=Reach, 1=Grasp, 2=Lift, 3=Hold
    """
    
    NUM_SKILLS = 4
    SKILL_NAMES = ['Reach', 'Grasp', 'Lift', 'Hold']
    
    def __init__(self, obs_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [self.NUM_SKILLS])
    
    def forward(self, obs, deterministic=False):
        """
        Inputs:
        - obs: [B, obs_dim]
        - deterministic: if True, pick argmax skill

        Outputs:
        - skill: [B] int64 skill indices
        - log_prob: [B, 1] or None (if deterministic)
        - logits: [B, num_skills]
        """
        logits = self.net(obs)
        
        if deterministic:
            # Greedy skill selection for evaluation.
            skill = logits.argmax(dim=-1)
            log_prob = None
        else:
            # Stochastic skill sampling for exploration during training.
            dist = Categorical(logits=logits)
            skill = dist.sample()
            log_prob = dist.log_prob(skill).unsqueeze(-1)
        
        return skill, log_prob, logits


class SkillConditionedActor(nn.Module):
    """
    Low-level policy conditioned on skill embedding.
    Produces smoother actions by incorporating skill context.
    """
    
    LOG_STD_MIN, LOG_STD_MAX = -20, 2
    
    def __init__(self, obs_dim, action_dim, num_skills=4, skill_embed_dim=32,
                 hidden_sizes=(512, 512, 256)):
        super().__init__()
        self.skill_embedding = nn.Embedding(num_skills, skill_embed_dim)
        
        input_dim = obs_dim + skill_embed_dim
        self.net = mlp([input_dim] + list(hidden_sizes))
        self.mu_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dim)
        
        # Action smoothing: learnable exponential moving average
        self.action_smoothing = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, obs, skill, prev_action=None, deterministic=False, with_logprob=True):
        """
        Inputs:
        - obs: [B, obs_dim]
        - skill: [B] int64 skill indices
        - prev_action: optional [B, action_dim] for smoothing
        - deterministic: if True, return mean action
        - with_logprob: if True, return log-prob

        Outputs:
        - action: [B, action_dim] in [-1, 1]
        - logprob: [B, 1] or None
        """
        skill_embed = self.skill_embedding(skill)
        x = torch.cat([obs, skill_embed], dim=-1)
        
        net_out = self.net(x)
        mu = self.mu_layer(net_out)
        log_std = torch.clamp(self.log_std_layer(net_out), self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        
        dist = Normal(mu, std)
        raw_action = mu if deterministic else dist.rsample()
        
        # Apply action smoothing for less shaky movements
        if prev_action is not None:
            # Learnable EMA smoothing to reduce high-frequency action jitter.
            alpha = torch.sigmoid(self.action_smoothing)
            raw_action = alpha * raw_action + (1 - alpha) * prev_action
        
        logprob = None
        if with_logprob:
            logprob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
            logprob -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(dim=-1, keepdim=True)
        
        return torch.tanh(raw_action), logprob


# ============================================================================
# Hierarchical SAC Agent
# ============================================================================

class HierarchicalSACAgent:
    """
    Hierarchical Soft Actor-Critic with:
    - High-level skill selector (discrete)
    - Low-level skill-conditioned actor (continuous)
    - Shared critic
    - Action smoothing for less shaky movements
    """
    
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_sizes=(512, 512, 256),
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        device='cuda',
        use_amp=True,
        use_images=False,
    ):
        """
        Inputs:
        - obs_dim: size of state vector
        - action_dim: size of action vector
        - hidden_sizes: MLP sizes for actor/critic
        - lr, gamma, tau: SAC hyperparameters
        - device: torch device
        - use_amp: use mixed precision on CUDA
        - use_images: whether critics accept image inputs
        """
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.action_dim = action_dim
        self.use_amp = use_amp and device != 'cpu'
        self.use_images = use_images
        
        # Determine best dtype
        if self.use_amp:
            self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # High-level skill selector
        self.skill_selector = SkillSelector(obs_dim).to(device)
        
        # Low-level skill-conditioned actor
        self.actor = SkillConditionedActor(
            obs_dim, action_dim,
            num_skills=SkillSelector.NUM_SKILLS,
            hidden_sizes=hidden_sizes
        ).to(device)
        
        # Critic (shared for both levels)
        self.critic = DoubleCritic(obs_dim, action_dim, hidden_sizes, use_images).to(device)
        self.critic_target = DoubleCritic(obs_dim, action_dim, hidden_sizes, use_images).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Compile networks for speed (PyTorch 2.x).
        # If compile introduces instability on a setup, disable it first when debugging.
        self.actor = torch.compile(self.actor, mode='default')
        self.skill_selector = torch.compile(self.skill_selector, mode='default')
        
        # Optimizers
        self.skill_optimizer = optim.AdamW(self.skill_selector.parameters(), lr=lr, fused=(device != 'cpu'))
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr, fused=(device != 'cpu'))
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr, fused=(device != 'cpu'))
        
        # Gradient scaler
        self.scaler = GradScaler(enabled=(self.use_amp and self.amp_dtype == torch.float16))
        
        # Entropy tuning for both levels:
        # - skill alpha: exploration over discrete skills
        # - action alpha: exploration in continuous action space
        # Higher entropy target = more skill exploration (closer to 0 = more uniform)
        self.target_entropy_skill = -0.3 * np.log(SkillSelector.NUM_SKILLS)  # Was -0.5, now more exploration
        self.log_alpha_skill = torch.log(torch.tensor([0.5], device=device))  # Start with higher alpha
        self.log_alpha_skill.requires_grad = True
        self.alpha_skill_optimizer = optim.AdamW([self.log_alpha_skill], lr=lr * 0.5)  # Slower alpha decay
        
        self.target_entropy_action = -0.5 * action_dim
        self.log_alpha_action = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_action_optimizer = optim.AdamW([self.log_alpha_action], lr=lr)
        
        # Previous action for smoothing
        self.prev_action = None
    
    @property
    def alpha_skill(self):
        return self.log_alpha_skill.exp().item()
    
    @property
    def alpha_action(self):
        return self.log_alpha_action.exp().item()
    
    def reset(self):
        """
        Reset per-episode state.

        Output:
        - None. Clears previous action used for smoothing.
        """
        self.prev_action = None
    
    def get_action(self, obs, deterministic=False):
        """
        Get action for a single observation.

        Inputs:
        - obs: [obs_dim] numpy array
        - deterministic: if True, use mean action and argmax skill

        Outputs:
        - action: [action_dim] numpy array in [-1, 1]
        - skill: int skill index
        """
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            # Get skill
            skill, _, _ = self.skill_selector(obs_t, deterministic=deterministic)
            
            # Get action conditioned on skill
            prev = self.prev_action
            if prev is not None:
                prev = torch.as_tensor(prev, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            action, _ = self.actor(obs_t, skill, prev, deterministic=deterministic, with_logprob=False)
            action = action.cpu().numpy()[0]
        
        self.prev_action = action
        return action, skill.cpu().numpy()[0]
    
    def update(self, replay_buffer, batch_size=1024):
        """
        Update actor, critics, skill selector, and entropy terms.

        Inputs:
        - replay_buffer: GPUReplayBuffer
        - batch_size: number of samples per update

        Outputs:
        - metrics dict with losses and alphas (floats)
        """
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        
        batch = replay_buffer.sample(batch_size)
        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']
        images = batch.get('images')
        next_images = batch.get('next_images')
        
        # ---- Critic update ----
        # Standard SAC target:
        #   y = r + gamma*(min(Q1',Q2') - alpha_a*logpi(a'|s') - alpha_s*logpi(skill'|s'))
        with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            with torch.no_grad():
                # Get next skill and action
                next_skill, next_skill_logprob, _ = self.skill_selector(next_obs, deterministic=False)
                next_action, next_action_logprob = self.actor(next_obs, next_skill, deterministic=False)
                # Clone avoids subtle graph reuse/cudagraph edge cases in some torch versions.
                next_action = next_action.clone()
                
                # Compute target Q
                q1_target, q2_target = self.critic_target(next_obs, next_action, next_images)
                q_target = torch.min(q1_target, q2_target)
                q_target = q_target - self.alpha_action * next_action_logprob
                if next_skill_logprob is not None:
                    q_target = q_target - self.alpha_skill * next_skill_logprob
                td_target = rewards + self.gamma * (1 - dones) * q_target
            
            q1, q2 = self.critic(obs, actions, images)
            critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.scaler.step(self.critic_optimizer)
        
        # ---- Actor update ----
        # Maximize Q while retaining entropy (implemented as minimizing alpha*logpi - Q).
        with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            
            skill, skill_logprob, _ = self.skill_selector(obs, deterministic=False)
            new_action, action_logprob = self.actor(obs, skill, deterministic=False)
            
            q1_new, q2_new = self.critic(obs, new_action, images)
            q_new = torch.min(q1_new, q2_new)
            
            actor_loss = (self.alpha_action * action_logprob - q_new).mean()
        
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.scaler.step(self.actor_optimizer)
        
        # ---- Skill selector update ----
        # Train discrete high-level policy against critic value of skill-conditioned action.
        with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            skill, skill_logprob, logits = self.skill_selector(obs, deterministic=False)
            new_action, _ = self.actor(obs, skill, deterministic=False)
            
            q1_skill, q2_skill = self.critic(obs, new_action, images)
            q_skill = torch.min(q1_skill, q2_skill)
            
            skill_loss = (self.alpha_skill * skill_logprob - q_skill).mean()
        
        self.skill_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(skill_loss).backward()
        self.scaler.unscale_(self.skill_optimizer)
        torch.nn.utils.clip_grad_norm_(self.skill_selector.parameters(), 1.0)
        self.scaler.step(self.skill_optimizer)
        
        self.scaler.update()
        
        # ---- Alpha updates ----
        # Automatic entropy temperature tuning for each policy head.
        alpha_skill_loss = -(self.log_alpha_skill * (skill_logprob + self.target_entropy_skill).detach()).mean()
        self.alpha_skill_optimizer.zero_grad(set_to_none=True)
        alpha_skill_loss.backward()
        self.alpha_skill_optimizer.step()
        with torch.no_grad():
            self.log_alpha_skill.clamp_(min=np.log(0.1), max=np.log(2.0))  # Wider range, higher minimum
        
        alpha_action_loss = -(self.log_alpha_action * (action_logprob + self.target_entropy_action).detach()).mean()
        self.alpha_action_optimizer.zero_grad(set_to_none=True)
        alpha_action_loss.backward()
        self.alpha_action_optimizer.step()
        with torch.no_grad():
            self.log_alpha_action.clamp_(min=np.log(0.05), max=np.log(1.0))
        
        # ---- Soft update target ----
        # Polyak averaging for target critics.
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.data.lerp_(p.data, self.tau)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'skill_loss': skill_loss.item(),
            'alpha_skill': self.alpha_skill,
            'alpha_action': self.alpha_action,
        }
    
    def save(self, filepath):
        """
        Save model and optimizer state.

        Input:
        - filepath: destination .pt path
        """
        torch.save({
            'skill_selector': self.skill_selector.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'skill_optimizer': self.skill_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha_skill': self.log_alpha_skill,
            'log_alpha_action': self.log_alpha_action,
            'scaler': self.scaler.state_dict(),
        }, filepath)
    
    def load(self, filepath, resume_training=False):
        """
        Load model weights (and optionally optimizer state).

        Inputs:
        - filepath: checkpoint path
        - resume_training: if True, restores optimizers and scalers
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.skill_selector.load_state_dict(checkpoint['skill_selector'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        
        if resume_training:
            self.skill_optimizer.load_state_dict(checkpoint['skill_optimizer'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.log_alpha_skill.data.copy_(checkpoint['log_alpha_skill'].data)
            self.log_alpha_action.data.copy_(checkpoint['log_alpha_action'].data)
            if checkpoint.get('scaler'):
                self.scaler.load_state_dict(checkpoint['scaler'])


# ============================================================================
# Environment with Domain Randomization
# ============================================================================

class RoboSuiteEnvV2:
    """
    Enhanced RoboSuite environment with:
    - Domain randomization (cube position, lighting)
    - Optional camera observations
    - Improved reward shaping
    """
    
    # Use the gripper-mounted camera you added to Rover2026
    CAMERA_NAME = "robot0_eye_in_hand"  # Gripper camera for egocentric view
    
    def __init__(self, render=False, use_camera=False, domain_randomization=True):
        """
        Inputs:
        - render: enable onscreen renderer
        - use_camera: include gripper camera in observations
        - domain_randomization: randomize cube position each reset
        """
        arm_controller_config = suite.load_part_controller_config(default_controller="JOINT_VELOCITY")
        controller_config = refactor_composite_controller_config(arm_controller_config, "Rover2026", ["right"])
        
        self.use_camera = use_camera
        self.domain_randomization = domain_randomization
        self.image_size = 84
        
        self.env = suite.make(
            env_name="Lift",
            robots=["Rover2026"],
            controller_configs=controller_config,
            has_renderer=render,
            has_offscreen_renderer=use_camera,
            render_camera=self.CAMERA_NAME,
            camera_names=[self.CAMERA_NAME] if use_camera else None,
            camera_heights=self.image_size if use_camera else None,
            camera_widths=self.image_size if use_camera else None,
            ignore_done=False,
            use_camera_obs=use_camera,
            control_freq=20,
            horizon=200,
            reward_shaping=True,
        )
        # This env still adds custom shaping on top of RoboSuite reward below.
        
        self.render_enabled = render
        self._camera_initialized = False
        self._setup_spaces()
        
        # Domain randomization bounds
        self.cube_x_range = (-0.1, 0.1)
        self.cube_y_range = (-0.1, 0.1)
    
    def _setup_spaces(self):
        """
        Derive observation and action dimensions from a reset.

        Output:
        - sets self.obs_dim, self.action_dim, self.image_shape
        """
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
        
        # +4 for phase one-hot encoding (reach, grasp, lift, hold)
        # Phase prior helps hierarchical policy disambiguate task stage.
        self.obs_dim = sum(len(np.array(obs[key]).flatten()) for key in self.obs_keys if key in obs) + 4
        self.action_dim = self.env.action_dim
        self.image_shape = (3, self.image_size, self.image_size) if self.use_camera else None
    
    def _compute_phase(self, obs):
        """
        Compute task phase from observation dict.

        Input:
        - obs: robosuite observation dict

        Output:
        - phase: one-hot [4] for reach/grasp/lift/hold
        """
        cube_pos = obs.get('cube_pos', [0, 0, 0])
        gripper_to_cube = obs.get('gripper_to_cube_pos', [0, 0, 0])
        gripper_qpos = obs.get('robot0_gripper_qpos', [0, 0])
        
        distance = np.linalg.norm(gripper_to_cube)
        gripper_closed = np.mean(gripper_qpos) < 0.02
        height_above_table = max(0, cube_pos[2] - 0.82)
        
        # Hand-coded phase classifier used as compact task-progress feature.
        phase = np.zeros(4, dtype=np.float32)
        if height_above_table > 0.08:
            phase[3] = 1.0  # Hold
        elif height_above_table > 0.01 or (gripper_closed and distance < 0.1):
            phase[2] = 1.0  # Lift
        elif distance < 0.1:
            phase[1] = 1.0  # Grasp
        else:
            phase[0] = 1.0  # Reach
        return phase
    
    def _process_obs(self, obs):
        """
        Flatten selected observation keys and append phase one-hot.

        Input:
        - obs: robosuite observation dict

        Output:
        - state vector float32 [obs_dim]
        """
        obs_list = []
        for key in self.obs_keys:
            if key in obs:
                obs_list.append(np.array(obs[key]).flatten())
        base_obs = np.concatenate(obs_list).astype(np.float32)
        
        # Append phase one-hot as structured curriculum signal.
        phase = self._compute_phase(obs)
        return np.concatenate([base_obs, phase]).astype(np.float32)
    
    def _process_image(self, obs):
        """
        Extract gripper camera image in CHW uint8 format.

        Input:
        - obs: robosuite observation dict

        Output:
        - image: uint8 [C, H, W] or None
        """
        if not self.use_camera:
            return None
        # RoboSuite camera key format: "{camera_name}_image"
        img_key = f"{self.CAMERA_NAME}_image"
        img = obs.get(img_key)
        if img is None:
            return None
        # Convert HWC to CHW and ensure uint8
        img = np.transpose(img, (2, 0, 1)).astype(np.uint8)
        return img
    
    def _randomize_cube_position(self):
        """
        Randomize cube position within bounds (if enabled).

        Output:
        - None. Updates MuJoCo body position in-place.
        """
        if not self.domain_randomization:
            return
        
        # Domain randomization in state space: small XY perturbation per reset.
        try:
            cube_body_id = self.env.sim.model.body_name2id('cube_main')
            base_pos = self.env.sim.model.body_pos[cube_body_id].copy()
            
            # Add random offset
            base_pos[0] += np.random.uniform(*self.cube_x_range)
            base_pos[1] += np.random.uniform(*self.cube_y_range)
            
            self.env.sim.model.body_pos[cube_body_id] = base_pos
            self.env.sim.forward()
        except Exception:
            pass  # Cube randomization not supported
    
    def reset(self):
        """
        Reset environment and return processed observation.

        Output:
        - state: [obs_dim] float32
        - image: optional uint8 [C, H, W] if use_camera
        """
        obs = self.env.reset()
        self._randomize_cube_position()
        obs = self.env._get_observations()  # Re-get observations after randomization
        
        state = self._process_obs(obs)
        image = self._process_image(obs)
        
        if self.use_camera:
            return state, image
        return state
    
    def step(self, action, skill=None):
        """
        Step the environment with optional skill for reward shaping.

        Inputs:
        - action: [action_dim] numpy array in [-1, 1]
        - skill: optional int skill index (for small bonus)

        Outputs:
        - (state, image) if use_camera else state
        - shaped_reward: float
        - done: bool
        - info: dict with phase diagnostics
        """
        obs, reward, done, info = self.env.step(action)
        
        # Get positions
        cube_pos = obs.get('cube_pos', [0, 0, 0])
        gripper_to_cube = obs.get('gripper_to_cube_pos', [0, 0, 0])
        gripper_qpos = obs.get('robot0_gripper_qpos', [0, 0])
        
        distance = np.linalg.norm(gripper_to_cube)
        gripper_closed = np.mean(gripper_qpos) < 0.02
        cube_height = cube_pos[2]
        table_height = 0.82
        height_above_table = max(0, cube_height - table_height)
        
        # =================================================================
        # Progressive shaping:
        # - reach/grasp rewards guide early behavior
        # - lift/success rewards dominate late behavior
        # =================================================================
        
        # Phase 1: REACH (max ~5/step, ~1000/episode if just reaching)
        # Exponential bonus as distance decreases
        reach_reward = 5.0 * np.exp(-5.0 * distance)  # Peaks at ~5 when distance=0
        
        # Phase 2: GRASP (adds ~10-30/step when close)
        grasp_reward = 0.0
        if distance < 0.15:
            proximity_bonus = 10.0 * (1.0 - distance / 0.15)  # 0-10 based on closeness
            grasp_reward = proximity_bonus
            if distance < 0.05:
                grasp_reward += 15.0  # Very close bonus
            if gripper_closed and distance < 0.08:
                grasp_reward += 30.0  # Grasping bonus (big!)
        
        # Phase 3: LIFT (adds 50-500/step - THIS IS THE BIG REWARD)
        lift_reward = 0.0
        if gripper_closed and distance < 0.1:  # Must be grasping
            if height_above_table > 0:
                # Exponential reward for height - this should dominate!
                lift_reward = 100.0 * height_above_table  # Linear base
                lift_reward += 500.0 * (height_above_table ** 2)  # Quadratic boost
                lift_reward += 50.0  # Constant bonus for any lift
        
        # Phase 4: SUCCESS (massive one-time bonus)
        success_reward = 0.0
        if height_above_table > 0.05:
            success_reward = 200.0  # Partial success
        if height_above_table > 0.1:
            success_reward = 500.0  # Full success
        if height_above_table > 0.15:
            success_reward = 1000.0  # Excellent lift
        
        # Skill bonus: light auxiliary shaping to encourage semantic skill alignment.
        # The main learning signal comes from task rewards, not skill matching
        # Phase info is now in observations, so the actor learns phase-specific behavior directly
        skill_bonus = 0.0
        
        # Determine actual phase for logging
        if height_above_table > 0.08:
            actual_phase = 3  # Hold
        elif height_above_table > 0.01 or (gripper_closed and distance < 0.1):
            actual_phase = 2  # Lift
        elif distance < 0.1:
            actual_phase = 1  # Grasp
        else:
            actual_phase = 0  # Reach
        
        # Simple skill consistency: small bonus for matching, no penalty
        if skill is not None and skill == actual_phase:
            skill_bonus = 3.0  # Small bonus, won't dominate task rewards
        
        # Action smoothness penalty (reduce shakiness)
        action_penalty = -0.05 * np.sum(action ** 2)
        
        shaped_reward = reward + reach_reward + grasp_reward + lift_reward + success_reward + skill_bonus + action_penalty
        
        # Emit diagnostics for training logs / offline analysis.
        phase_names = ['reach', 'grasp', 'lift', 'hold']
        info['phase'] = phase_names[actual_phase]
        info['actual_phase'] = actual_phase
        info['skill_bonus'] = skill_bonus
        info['height'] = height_above_table
        info['distance'] = distance
        info['gripper_closed'] = gripper_closed
        
        state = self._process_obs(obs)
        image = self._process_image(obs)
        
        if self.use_camera:
            return (state, image), shaped_reward, done, info
        return state, shaped_reward, done, info
    
    def render(self):
        """Render onscreen if enabled."""
        if self.render_enabled:
            self.env.render()
    
    def close(self):
        """Close the underlying robosuite env."""
        self.env.close()


# ============================================================================
# Parallel Environment Wrapper
# ============================================================================

def worker_v2(remote, parent_remote, env_fn):
    """
    Subprocess worker for a single environment.

    Inputs:
    - remote/parent_remote: multiprocessing Pipes
    - env_fn: callable that builds a RoboSuiteEnvV2
    """
    parent_remote.close()
    # Each process owns an independent env instance.
    env = env_fn()
    use_camera = env.use_camera
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                action, skill = data  # Unpack action and skill
                result = env.step(action, skill=skill)
                if use_camera:
                    (state, image), reward, done, info = result
                    if done:
                        # Auto-reset inside worker: vector interface keeps stepping seamlessly.
                        state, image = env.reset()
                    remote.send((state, image, reward, done, info))
                else:
                    state, reward, done, info = result
                    if done:
                        state = env.reset()
                    remote.send((state, reward, done, info))
            elif cmd == 'reset':
                result = env.reset()
                remote.send(result)
            elif cmd == 'get_spaces':
                remote.send((env.obs_dim, env.action_dim, env.use_camera, env.image_shape))
            elif cmd == 'close':
                env.close()
                remote.close()
                break
    except EOFError:
        pass


class SubprocVecEnvV2:
    """
    Parallel environments with camera support.

    Function:
    - Launches N subprocess envs and provides vectorized step/reset.
    """
    
    def __init__(self, env_fns):
        """
        Inputs:
        - env_fns: list of callables returning RoboSuiteEnvV2

        Outputs:
        - sets obs_dim, action_dim, use_camera, image_shape
        """
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.ps = [mp.Process(target=worker_v2, args=(wr, r, fn))
                   for wr, r, fn in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for wr in self.work_remotes:
            wr.close()
        
        self.remotes[0].send(('get_spaces', None))
        # Infer shared spaces from first worker (all workers are homogeneous).
        self.obs_dim, self.action_dim, self.use_camera, self.image_shape = self.remotes[0].recv()
        self.closed = False
    
    def step(self, actions, skills=None):
        """
        Step all environments.

        Inputs:
        - actions: [N, action_dim] numpy array
        - skills: optional list/array of length N

        Outputs:
        - states (and images if use_camera), rewards, dones, infos
        """
        if skills is None:
            skills = [None] * len(actions)
        for remote, action, skill in zip(self.remotes, actions, skills):
            remote.send(('step', (action, skill)))
        results = [remote.recv() for remote in self.remotes]
        
        if self.use_camera:
            states, images, rewards, dones, infos = zip(*results)
            return np.stack(states), np.stack(images), np.array(rewards), np.array(dones), infos
        else:
            states, rewards, dones, infos = zip(*results)
            return np.stack(states), np.array(rewards), np.array(dones), infos
    
    def reset(self):
        """
        Reset all environments.

        Output:
        - states or (states, images) depending on use_camera
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        
        if self.use_camera:
            states, images = zip(*results)
            return np.stack(states), np.stack(images)
        else:
            return np.stack(results)
    
    def close(self):
        """Close all subprocess environments."""
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


# ============================================================================
# Training Loop
# ============================================================================

def train(args):
    """
    Train hierarchical SAC on the RoboSuite Lift task.

    Inputs:
    - args: argparse Namespace (see __main__ for flags)

    Outputs:
    - Saves checkpoints to checkpoints_v2/<timestamp>
    - Prints training logs to stdout
    """
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    
    if args.cuda and device.type == 'cpu':
        print("ERROR: CUDA requested but not available!")
        sys.exit(1)
    
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    torch.set_num_threads(max(1, min(8, mp.cpu_count() // args.num_envs)))
    
    # Create N parallel env workers for higher sample throughput.
    def make_env():
        return RoboSuiteEnvV2(
            render=False,
            use_camera=args.use_camera,
            domain_randomization=args.domain_rand
        )
    
    print(f"Launching {args.num_envs} parallel environments...")
    print(f"  Camera observations: {args.use_camera}")
    print(f"  Domain randomization: {args.domain_rand}")
    
    env = SubprocVecEnvV2([make_env for _ in range(args.num_envs)])
    
    # Create agent
    agent = HierarchicalSACAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_sizes=(512, 512, 256),
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        device=device,
        use_amp=args.cuda,
        use_images=args.use_camera,
    )
    
    # Resume if specified
    start_episode = 1
    if args.resume and os.path.exists(args.resume):
        agent.load(args.resume, resume_training=True)
        basename = os.path.basename(args.resume)
        if 'ep' in basename:
            try:
                start_episode = int(basename.split('ep')[1].split('.')[0]) + 1
            except:
                pass
        print(f"Resumed from {args.resume}")
    
    # Create replay buffer
    replay_buffer = GPUReplayBuffer(
        capacity=args.buffer_size,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=device,
        use_images=args.use_camera,
        image_shape=env.image_shape if args.use_camera else (3, 84, 84),
    )
    
    # Metrics:
    # - episode_rewards: per-worker running episodic return
    # - all_episode_rewards: rolling window for summary stats
    episode_rewards = np.zeros(args.num_envs)
    all_episode_rewards = deque(maxlen=100)
    best_avg_reward = -float('inf')
    skill_counts = np.zeros(SkillSelector.NUM_SKILLS)
    
    save_dir = os.path.join(ROOT, "checkpoints_v2", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("  HIERARCHICAL SAC TRAINING - Rover2026 Lift Task V2")
    print("="*70)
    print(f"  Skills: {SkillSelector.SKILL_NAMES}")
    print(f"  Episodes: {args.episodes} | Batch: {args.batch_size}")
    print(f"  Save: {save_dir}")
    print("="*70 + "\n")
    
    total_steps = 0
    update_times = deque(maxlen=100)
    train_start = time.time()
    
    # Reset environments
    if args.use_camera:
        obs, images = env.reset()
    else:
        obs = env.reset()
        images = None
    
    for episode in range(start_episode, args.episodes + 1):
        agent.reset()
        
        while True:
            # Get actions
            if total_steps < args.warmup_steps:
                # Pure random exploration before first gradient updates.
                actions = np.random.uniform(-1, 1, (args.num_envs, env.action_dim))
                skills = np.zeros(args.num_envs, dtype=np.int64)
            else:
                with torch.no_grad():
                    if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                        torch.compiler.cudagraph_mark_step_begin()
                    
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    skills_t, _, _ = agent.skill_selector(obs_t, deterministic=False)
                    actions_t, _ = agent.actor(obs_t, skills_t, deterministic=False)
                    
                    actions = actions_t.cpu().numpy()
                    skills = skills_t.cpu().numpy()
            
            # Step environments (pass skills for skill-specific rewards)
            if args.use_camera:
                next_obs, next_images, rewards, dones, infos = env.step(actions, skills=skills)
            else:
                next_obs, rewards, dones, infos = env.step(actions, skills=skills)
                next_images = None
            
            # Store transitions with skill diversity bonus
            for i in range(args.num_envs):
                img = images[i] if images is not None else None
                next_img = next_images[i] if next_images is not None else None
                
                # Auxiliary exploration pressure toward underused skills.
                total_skill_usage = skill_counts.sum() + 1e-8
                skill_usage_pct = skill_counts[skills[i]] / total_skill_usage
                diversity_bonus = 1.0 if skill_usage_pct < 0.25 else 0.0  # Bonus for underused skills
                
                reward_with_bonus = rewards[i] + diversity_bonus
                replay_buffer.add(obs[i], actions[i], reward_with_bonus, next_obs[i], float(dones[i]), img, next_img)
                
                episode_rewards[i] += rewards[i]  # Track original reward for logging
                skill_counts[skills[i]] += 1
                
                if dones[i]:
                    all_episode_rewards.append(episode_rewards[i])
                    episode_rewards[i] = 0
            
            obs = next_obs
            images = next_images
            total_steps += args.num_envs
            
            # Update
            if total_steps >= args.warmup_steps and replay_buffer.size >= args.batch_size:
                update_start = time.perf_counter()
                # Run multiple SGD updates per environment step batch.
                for _ in range(args.updates_per_step):
                    metrics = agent.update(replay_buffer, args.batch_size)
                torch.cuda.synchronize()
                update_times.append(time.perf_counter() - update_start)
            
            if any(dones):
                break
        
        avg_reward = np.mean(all_episode_rewards) if all_episode_rewards else 0
        
        # Logging
        if episode % args.log_freq == 0:
            elapsed = time.time() - train_start
            sps = total_steps / elapsed if elapsed > 0 else 0
            avg_update = np.mean(update_times) * 1000 if update_times else 0
            
            # Skill distribution snapshot since last log.
            skill_pct = skill_counts / (skill_counts.sum() + 1e-8) * 100
            skill_str = " ".join([f"{n[0]}:{p:.0f}%" for n, p in zip(SkillSelector.SKILL_NAMES, skill_pct)])
            
            print(f"  Ep {episode:5d} | Steps {total_steps:8,} | Reward {avg_reward:7.1f} | "
                  f"α_s {agent.alpha_skill:.3f} α_a {agent.alpha_action:.3f} | "
                  f"Skills: {skill_str}")
            
            skill_counts[:] = 0  # Reset for next interval
            
            if avg_reward > best_avg_reward and episode > start_episode + 50:
                # Delay "best model" selection until replay is reasonably populated.
                best_avg_reward = avg_reward
                agent.save(os.path.join(save_dir, "best_model.pt"))
        
        if episode % args.save_freq == 0:
            agent.save(os.path.join(save_dir, f"checkpoint_ep{episode}.pt"))
    
    agent.save(os.path.join(save_dir, "final_model.pt"))
    env.close()
    
    print("\n" + "="*70)
    print("  TRAINING COMPLETE")
    print(f"  Best Reward: {best_avg_reward:.1f}")
    print(f"  Saved to: {save_dir}")
    print("="*70 + "\n")


def evaluate(args):
    """
    Evaluate trained model with visualization.
    - Main window: Birdview camera (default RoboSuite renderer)
    - Optional second window: Gripper camera view (OpenCV)

    Inputs:
    - args.eval: checkpoint path
    - args.use_camera: show gripper camera window
    - args.domain_rand: apply cube randomization

    Outputs:
    - Renders episodes, prints reward/skill usage per episode
    """
    import cv2
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    
    # Evaluation env:
    # - birdview shown in robosuite viewer
    # - optional gripper cam shown in OpenCV window
    arm_controller_config = suite.load_part_controller_config(default_controller="JOINT_VELOCITY")
    controller_config = refactor_composite_controller_config(arm_controller_config, "Rover2026", ["right"])
    
    # Setup camera for separate window display
    gripper_camera = "robot0_eye_in_hand"
    show_gripper_cam = args.use_camera
    
    env = suite.make(
        env_name="Lift",
        robots=["Rover2026"],
        controller_configs=controller_config,
        has_renderer=True,  # Main birdview renderer
        has_offscreen_renderer=show_gripper_cam,  # For gripper camera capture
        render_camera="frontview",  # Birdview for main render
        camera_names=[gripper_camera] if show_gripper_cam else None,
        camera_heights=256 if show_gripper_cam else None,  # Larger for display
        camera_widths=256 if show_gripper_cam else None,
        ignore_done=False,
        use_camera_obs=show_gripper_cam,
        control_freq=20,
        horizon=200,
        reward_shaping=True,
    )
    
    # Observation keys (same as RoboSuiteEnvV2)
    obs_keys = [
        'robot0_joint_pos',
        'robot0_joint_vel',
        'robot0_eef_pos',
        'robot0_eef_quat',
        'robot0_gripper_qpos',
        'cube_pos',
        'gripper_to_cube_pos',
    ]
    
    def process_obs(obs_dict):
        obs_list = []
        for key in obs_keys:
            if key in obs_dict:
                obs_list.append(np.array(obs_dict[key]).flatten())
        base_obs = np.concatenate(obs_list).astype(np.float32)
        
        # Rebuild the same phase feature used during training.
        cube_pos = obs_dict.get('cube_pos', [0, 0, 0])
        gripper_to_cube = obs_dict.get('gripper_to_cube_pos', [0, 0, 0])
        gripper_qpos = obs_dict.get('robot0_gripper_qpos', [0, 0])
        
        distance = np.linalg.norm(gripper_to_cube)
        gripper_closed = np.mean(gripper_qpos) < 0.02
        height = max(0, cube_pos[2] - 0.82)
        
        phase = np.zeros(4, dtype=np.float32)
        if height > 0.08:
            phase[3] = 1.0
        elif height > 0.01 or (gripper_closed and distance < 0.1):
            phase[2] = 1.0
        elif distance < 0.1:
            phase[1] = 1.0
        else:
            phase[0] = 1.0
        
        return np.concatenate([base_obs, phase]).astype(np.float32), distance, height
    
    def randomize_cube():
        if not args.domain_rand:
            return
        try:
            cube_id = env.sim.model.body_name2id('cube_main')
            pos = env.sim.model.body_pos[cube_id].copy()
            pos[0] += np.random.uniform(-0.1, 0.1)
            pos[1] += np.random.uniform(-0.1, 0.1)
            env.sim.model.body_pos[cube_id] = pos
            env.sim.forward()
        except:
            pass
    
    # Infer observation dimensionality from one reset pass.
    test_obs = env.reset()
    obs, _, _ = process_obs(test_obs)
    obs_dim = len(obs)
    
    # Create agent and load weights
    agent = HierarchicalSACAgent(
        obs_dim=obs_dim,
        action_dim=env.action_dim,
        device=device,
        use_amp=False,
        use_images=False,  # We're not using camera obs for policy, just visualization
    )
    agent.load(args.eval)
    
    print(f"\n{'='*60}")
    print(f"  EVALUATING: {args.eval}")
    print(f"{'='*60}")
    print(f"  Domain Randomization: {args.domain_rand}")
    print(f"  Gripper Camera View: {show_gripper_cam}")
    print(f"  Skills: {SkillSelector.SKILL_NAMES}")
    print(f"{'='*60}\n")
    
    if show_gripper_cam:
        cv2.namedWindow("Gripper Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gripper Camera", 400, 400)
    
    for episode in range(args.eval_episodes):
        raw_obs = env.reset()
        randomize_cube()
        raw_obs = env._get_observations()
        
        obs, distance, height = process_obs(raw_obs)
        agent.reset()
        episode_reward = 0
        done = False
        skill_seq = []
        step_count = 0
        
        while not done:
            action, skill = agent.get_action(obs, deterministic=True)
            skill_seq.append(skill)
            
            raw_obs, reward, done, info = env.step(action)
            obs, distance, height = process_obs(raw_obs)
            
            episode_reward += reward
            step_count += 1
            
            # Main renderer is already tied to env; calling render advances the viewer.
            env.render()
            
            # Show gripper camera in separate window
            if show_gripper_cam:
                img_key = f"{gripper_camera}_image"
                if img_key in raw_obs:
                    gripper_img = raw_obs[img_key]
                    # Convert RGB to BGR for OpenCV
                    gripper_img = cv2.cvtColor(gripper_img, cv2.COLOR_RGB2BGR)
                    # Add overlay text
                    cv2.putText(gripper_img, f"Skill: {SkillSelector.SKILL_NAMES[skill]}", 
                               (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(gripper_img, f"Dist: {distance:.3f}m", 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.putText(gripper_img, f"Height: {height:.3f}m", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.imshow("Gripper Camera", gripper_img)
                    cv2.waitKey(1)
            
            time.sleep(0.02)
        
        # Count skill usage
        from collections import Counter
        skill_counts = Counter([SkillSelector.SKILL_NAMES[s] for s in skill_seq])
        skill_str = ", ".join([f"{k}: {v}" for k, v in skill_counts.most_common()])
        
        print(f"  Episode {episode+1}: Reward={episode_reward:.1f} | Height={height:.3f}m | Skills: {skill_str}")
    
    if show_gripper_cam:
        cv2.destroyAllWindows()
    
    env.close()
    print(f"\n  Evaluation complete!")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hierarchical SAC Training V2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode controls: exactly one of --train / --eval expected.
    mode = parser.add_argument_group("Mode")
    mode.add_argument('--train', action='store_true', help='Train')
    mode.add_argument('--eval', type=str, default=None, help='Evaluate model')
    mode.add_argument('--resume', type=str, default=None, help='Resume training')
    
    # Environment
    env_args = parser.add_argument_group("Environment")
    env_args.add_argument('--use_camera', action='store_true', help='Use camera observations')
    env_args.add_argument('--domain_rand', action='store_true', default=True, help='Domain randomization')
    
    # Training
    train_args = parser.add_argument_group("Training")
    train_args.add_argument('--episodes', type=int, default=2000, help='Training episodes')
    train_args.add_argument('--buffer_size', type=int, default=1_000_000, help='Buffer size')
    train_args.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    train_args.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    train_args.add_argument('--gamma', type=float, default=0.99, help='Discount')
    train_args.add_argument('--tau', type=float, default=0.005, help='Soft update')
    train_args.add_argument('--warmup_steps', type=int, default=5000, help='Warmup')
    train_args.add_argument('--updates_per_step', type=int, default=8, help='Updates per step')
    
    # Logging
    log_args = parser.add_argument_group("Logging")
    log_args.add_argument('--log_freq', type=int, default=10, help='Log frequency')
    log_args.add_argument('--save_freq', type=int, default=100, help='Save frequency')
    log_args.add_argument('--eval_episodes', type=int, default=10, help='Eval episodes')
    
    # Hardware
    hw_args = parser.add_argument_group("Hardware")
    hw_args.add_argument('--cuda', action='store_true', help='Use CUDA')
    hw_args.add_argument('--num_envs', type=int, default=16, help='Parallel envs')
    
    args = parser.parse_args()
    
    if args.eval:
        evaluate(args)
    elif args.train:
        train(args)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python train_lift_v2.py --train --cuda")
        print("  python train_lift_v2.py --train --cuda --use_camera --domain_rand")
        print("  python train_lift_v2.py --eval checkpoints_v2/best_model.pt --cuda")
