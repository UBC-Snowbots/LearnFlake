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
import glob
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

try:
    import h5py
except ImportError:
    h5py = None

# Path setup
ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite")
sys.path.insert(0, ROBO_PATH)

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config


def _system_ram_gb() -> float:
    """Best-effort system RAM size in GiB without extra deps."""
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return (page * pages) / (1024 ** 3)
    except Exception:
        return 0.0


def _available_ram_gb() -> float:
    """Best-effort available RAM in GiB (Linux MemAvailable)."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = float(line.split()[1])
                    return kb / (1024 ** 2)
    except Exception:
        pass
    return 0.0


def _apply_low_memory_profile(args):
    """Clamp high-cost defaults for 8GB VRAM / 16GB RAM class laptops."""
    if not args.low_mem:
        return

    # Only overwrite if user left a default-style value.
    if args.num_envs >= 4:
        args.num_envs = 2
    if args.batch_size >= 256:
        args.batch_size = 128
    if args.updates_per_step >= 2:
        args.updates_per_step = 1
    if args.buffer_size >= 250_000:
        args.buffer_size = 120_000
    if args.warmup_steps >= 1000:
        args.warmup_steps = 500
    args.no_compile = True
    args.no_amp = True
    args.sync_replay = True

    print("[low_mem] Applied low-memory profile:")
    print(f"  num_envs={args.num_envs} batch_size={args.batch_size} "
          f"updates_per_step={args.updates_per_step}")
    print(f"  buffer_size={args.buffer_size} warmup_steps={args.warmup_steps} "
          f"no_compile={args.no_compile} no_amp={args.no_amp} sync_replay={args.sync_replay}")


def _expand_hdf5_patterns(patterns):
    """Expand glob patterns to concrete HDF5 file paths."""
    paths = []
    for pat in patterns:
        expanded = sorted(glob.glob(pat))
        if expanded:
            paths.extend(expanded)
        elif os.path.exists(pat):
            paths.append(pat)
    # Deduplicate while preserving order
    out = []
    seen = set()
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def load_demo_transitions(hdf5_paths, obs_dim, action_dim):
    """Load demo transitions from demo_recorder-style HDF5 files."""
    if h5py is None:
        raise RuntimeError("h5py is required for --demo_data but is not installed.")

    obs_all, act_all, next_obs_all, rew_all, done_all = [], [], [], [], []
    num_files = 0
    num_demos = 0
    skipped = 0

    for path in hdf5_paths:
        if not os.path.exists(path):
            continue
        num_files += 1
        with h5py.File(path, "r") as f:
            if "data" not in f:
                continue
            for demo_key in sorted(f["data"].keys()):
                demo = f["data"][demo_key]
                if "obs" not in demo or "actions" not in demo:
                    continue
                obs = np.array(demo["obs"], dtype=np.float32)
                actions = np.array(demo["actions"], dtype=np.float32)
                T = min(len(obs), len(actions))
                if T <= 0:
                    continue
                obs = obs[:T]
                actions = actions[:T]
                if obs.ndim != 2 or actions.ndim != 2:
                    skipped += 1
                    continue
                if obs.shape[1] != obs_dim or actions.shape[1] != action_dim:
                    skipped += 1
                    continue

                if "rewards" in demo:
                    rewards = np.array(demo["rewards"], dtype=np.float32)[:T]
                else:
                    rewards = np.zeros(T, dtype=np.float32)

                if "dones" in demo:
                    dones = np.array(demo["dones"], dtype=np.float32)[:T]
                else:
                    dones = np.zeros(T, dtype=np.float32)
                    dones[-1] = 1.0

                next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0).astype(np.float32)

                obs_all.append(obs)
                act_all.append(actions)
                next_obs_all.append(next_obs)
                rew_all.append(rewards.reshape(-1, 1))
                done_all.append(dones.reshape(-1, 1))
                num_demos += 1

    if not obs_all:
        return None

    data = {
        "obs": np.concatenate(obs_all, axis=0),
        "actions": np.concatenate(act_all, axis=0),
        "next_obs": np.concatenate(next_obs_all, axis=0),
        "rewards": np.concatenate(rew_all, axis=0),
        "dones": np.concatenate(done_all, axis=0),
    }
    data["meta"] = {
        "num_files": num_files,
        "num_demos": num_demos,
        "num_steps": int(data["obs"].shape[0]),
        "skipped_demos": skipped,
    }
    return data


class DemoBatchBuffer:
    """Static demo mini-batch sampler for BC regularization."""

    def __init__(self, obs, actions, device="cuda"):
        self.device = device
        self.obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        self.actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
        self.size = int(self.obs.shape[0])

    def sample(self, batch_size):
        batch_size = min(int(batch_size), self.size)
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
        }


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
                 async_copy=True):
        """
        Args:
            capacity: max number of transitions for state/action buffers.
            obs_dim: length of state observation vector.
            action_dim: length of action vector.
            device: torch device for GPU-resident tensors.
            use_images: whether to store and sample image observations.
            image_shape: CHW shape for images (uint8 on CPU).
            async_copy: use async CUDA stream for H2D flushes.
        """
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.use_images = use_images
        self.image_shape = image_shape
        self.async_copy = bool(async_copy and device != 'cpu')
        
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
        
        self._stream = torch.cuda.Stream(device=device) if self.async_copy else None
    
    def add(self, obs, action, reward, next_obs, done, image=None, next_image=None):
        """
        Stage a transition in the CPU staging buffer.

        Inputs:
        - obs, action, next_obs: np.ndarray or torch.Tensor
        - reward, done: scalars
        - image, next_image: optional uint8 CHW arrays (if use_images)
        """
        idx = self._buffer_ptr
        self._obs_buf[idx] = torch.from_numpy(obs) if isinstance(obs, np.ndarray) else obs
        self._act_buf[idx] = torch.from_numpy(action) if isinstance(action, np.ndarray) else action
        self._rew_buf[idx, 0] = reward
        self._next_buf[idx] = torch.from_numpy(next_obs) if isinstance(next_obs, np.ndarray) else next_obs
        self._done_buf[idx, 0] = done
        
        if self.use_images and image is not None:
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
        
        def _copy_into_device(non_blocking_flag):
            if self.ptr + n <= self.capacity:
                self.obs[self.ptr:self.ptr + n] = self._obs_buf[:n].to(self.device, non_blocking=non_blocking_flag)
                self.actions[self.ptr:self.ptr + n] = self._act_buf[:n].to(self.device, non_blocking=non_blocking_flag)
                self.rewards[self.ptr:self.ptr + n] = self._rew_buf[:n].to(self.device, non_blocking=non_blocking_flag)
                self.next_obs[self.ptr:self.ptr + n] = self._next_buf[:n].to(self.device, non_blocking=non_blocking_flag)
                self.dones[self.ptr:self.ptr + n] = self._done_buf[:n].to(self.device, non_blocking=non_blocking_flag)
            else:
                # Handle wrap-around
                first = self.capacity - self.ptr
                self.obs[self.ptr:] = self._obs_buf[:first].to(self.device, non_blocking=non_blocking_flag)
                self.obs[:n-first] = self._obs_buf[first:n].to(self.device, non_blocking=non_blocking_flag)
                self.actions[self.ptr:] = self._act_buf[:first].to(self.device, non_blocking=non_blocking_flag)
                self.actions[:n-first] = self._act_buf[first:n].to(self.device, non_blocking=non_blocking_flag)
                self.rewards[self.ptr:] = self._rew_buf[:first].to(self.device, non_blocking=non_blocking_flag)
                self.rewards[:n-first] = self._rew_buf[first:n].to(self.device, non_blocking=non_blocking_flag)
                self.next_obs[self.ptr:] = self._next_buf[:first].to(self.device, non_blocking=non_blocking_flag)
                self.next_obs[:n-first] = self._next_buf[first:n].to(self.device, non_blocking=non_blocking_flag)
                self.dones[self.ptr:] = self._done_buf[:first].to(self.device, non_blocking=non_blocking_flag)
                self.dones[:n-first] = self._done_buf[first:n].to(self.device, non_blocking=non_blocking_flag)

        if self.async_copy:
            with torch.cuda.stream(self._stream):
                _copy_into_device(True)
        else:
            _copy_into_device(False)
        
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
            self._flush()
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        batch = {
            'obs': self.obs[idxs],
            'actions': self.actions[idxs],
            'rewards': self.rewards[idxs],
            'next_obs': self.next_obs[idxs],
            'dones': self.dones[idxs],
        }
        
        if self.use_images:
            # Sample from image buffer (may have different indices due to smaller capacity)
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
            skill = logits.argmax(dim=-1)
            log_prob = None
        else:
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
        compile_models=True,
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
        - compile_models: compile actor / skill selector with torch.compile
        """
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.action_dim = action_dim
        self.use_amp = use_amp and device != 'cpu'
        self.use_images = use_images
        self.amp_dtype = torch.float16
        
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
        
        # Compile networks for speed (can consume extra RAM / VRAM on smaller systems)
        self.compile_models = bool(compile_models)
        if self.compile_models:
            self.actor = torch.compile(self.actor, mode='default')
            self.skill_selector = torch.compile(self.skill_selector, mode='default')
        
        # Optimizers
        self.skill_optimizer = optim.AdamW(self.skill_selector.parameters(), lr=lr, fused=(device != 'cpu'))
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr, fused=(device != 'cpu'))
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr, fused=(device != 'cpu'))
        
        # Gradient scaler
        self.scaler = GradScaler(enabled=(self.use_amp and self.amp_dtype == torch.float16))
        
        # Entropy tuning for both levels
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
    
    def update(
        self,
        replay_buffer,
        batch_size=1024,
        demo_batch=None,
        bc_weight=0.0,
        phase_weight=0.0,
    ):
        """
        Update actor, critics, skill selector, and entropy terms.

        Inputs:
        - replay_buffer: GPUReplayBuffer
        - batch_size: number of samples per update
        - demo_batch: optional dict {'obs','actions'} for BC regularization
        - bc_weight: scalar weight for BC regularization term
        - phase_weight: scalar weight for skill-vs-phase cross-entropy

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
        with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            with torch.no_grad():
                # Get next skill and action
                next_skill, next_skill_logprob, _ = self.skill_selector(next_obs, deterministic=False)
                next_action, next_action_logprob = self.actor(next_obs, next_skill, deterministic=False)
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
        bc_loss = torch.tensor(0.0, device=self.device)
        with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            
            skill, skill_logprob, _ = self.skill_selector(obs, deterministic=False)
            new_action, action_logprob = self.actor(obs, skill, deterministic=False)
            
            q1_new, q2_new = self.critic(obs, new_action, images)
            q_new = torch.min(q1_new, q2_new)
            
            actor_loss = (self.alpha_action * action_logprob - q_new).mean()

            if demo_batch is not None and bc_weight > 0.0:
                demo_obs = demo_batch['obs']
                demo_actions = demo_batch['actions']
                # Prefer phase-conditioned BC if phase one-hot exists in obs tail.
                # This keeps each skill semantically grounded.
                if demo_obs.shape[1] >= SkillSelector.NUM_SKILLS:
                    demo_skill = demo_obs[:, -SkillSelector.NUM_SKILLS:].argmax(dim=-1)
                    demo_pred, _ = self.actor(
                        demo_obs, demo_skill, deterministic=True, with_logprob=False
                    )
                    bc_loss = F.mse_loss(demo_pred, demo_actions)
                else:
                    # Fallback for legacy demos without phase in observations.
                    b = demo_obs.shape[0]
                    k = SkillSelector.NUM_SKILLS
                    demo_obs_rep = demo_obs.unsqueeze(1).expand(b, k, demo_obs.shape[1]).reshape(b * k, demo_obs.shape[1])
                    demo_skill_rep = torch.arange(k, device=self.device).unsqueeze(0).expand(b, k).reshape(b * k)
                    demo_pred_rep, _ = self.actor(
                        demo_obs_rep, demo_skill_rep, deterministic=True, with_logprob=False
                    )
                    demo_pred = demo_pred_rep.view(b, k, -1)
                    mse_per_skill = ((demo_pred - demo_actions.unsqueeze(1)) ** 2).mean(dim=-1)
                    bc_loss = mse_per_skill.min(dim=1).values.mean()
                actor_loss = actor_loss + float(bc_weight) * bc_loss
        
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.scaler.step(self.actor_optimizer)
        
        # ---- Skill selector update ----
        phase_ce = torch.tensor(0.0, device=self.device)
        with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            skill, skill_logprob, logits = self.skill_selector(obs, deterministic=False)
            new_action, _ = self.actor(obs, skill, deterministic=False)
            
            q1_skill, q2_skill = self.critic(obs, new_action, images)
            q_skill = torch.min(q1_skill, q2_skill)
            
            skill_loss = (self.alpha_skill * skill_logprob - q_skill).mean()
            if phase_weight > 0.0 and obs.shape[1] >= SkillSelector.NUM_SKILLS:
                phase_targets = obs[:, -SkillSelector.NUM_SKILLS:].argmax(dim=-1)
                phase_ce = F.cross_entropy(logits, phase_targets)
                skill_loss = skill_loss + float(phase_weight) * phase_ce
        
        self.skill_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(skill_loss).backward()
        self.scaler.unscale_(self.skill_optimizer)
        torch.nn.utils.clip_grad_norm_(self.skill_selector.parameters(), 1.0)
        self.scaler.step(self.skill_optimizer)
        
        self.scaler.update()
        
        # ---- Alpha updates ----
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
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.data.lerp_(p.data, self.tau)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'skill_loss': skill_loss.item(),
            'bc_loss': float(bc_loss.item()),
            'phase_ce': float(phase_ce.item()),
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
        def _normalize_state_dict_keys(model, src_state_dict):
            """Handle compiled (<_orig_mod.>) and non-compiled checkpoints."""
            src_keys = list(src_state_dict.keys())
            tgt_keys = list(model.state_dict().keys())
            if not src_keys or not tgt_keys:
                return src_state_dict

            src_prefixed = all(k.startswith("_orig_mod.") for k in src_keys)
            tgt_prefixed = all(k.startswith("_orig_mod.") for k in tgt_keys)
            if src_prefixed == tgt_prefixed:
                return src_state_dict

            if src_prefixed and not tgt_prefixed:
                return {k[len("_orig_mod."):]: v for k, v in src_state_dict.items()}
            return {f"_orig_mod.{k}": v for k, v in src_state_dict.items()}

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.skill_selector.load_state_dict(
            _normalize_state_dict_keys(self.skill_selector, checkpoint['skill_selector'])
        )
        self.actor.load_state_dict(
            _normalize_state_dict_keys(self.actor, checkpoint['actor'])
        )
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
        
        self.render_enabled = render
        self._camera_initialized = False
        self._setup_spaces()
        
        # Domain randomization bounds
        self.cube_x_range = (-0.1, 0.1)
        self.cube_y_range = (-0.1, 0.1)
        
        # Episode-local trackers for progress-based reward shaping.
        self.cube_start_z = None
        self.prev_distance = None
        self.prev_height = None
        self.prev_eef_z = None
        self.prev_action = None
        self.lift_milestones_hit = set()
    
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
        gripper_to_cube = obs.get('gripper_to_cube_pos', [0, 0, 0])
        
        distance = np.linalg.norm(gripper_to_cube)
        height_above_table = self._cube_lift_height(obs)
        
        # One-hot phase encoding
        phase = np.zeros(4, dtype=np.float32)
        if height_above_table > 0.08:
            phase[3] = 1.0  # Hold
        elif height_above_table > 0.015:
            phase[2] = 1.0  # Lift
        elif distance < 0.09:
            phase[1] = 1.0  # Grasp
        else:
            phase[0] = 1.0  # Reach
        return phase

    def _cube_lift_height(self, obs):
        """
        Cube height relative to the episode start height.

        Using relative height is more robust than a fixed table-z constant and
        avoids reward drift across scene variations.
        """
        cube_pos = obs.get('cube_pos', [0, 0, 0])
        cube_z = float(cube_pos[2])
        base_z = cube_z if self.cube_start_z is None else float(self.cube_start_z)
        return max(0.0, cube_z - base_z)
    
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
        
        # Append phase information
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
        # Camera observation key format: {camera_name}_image
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
        
        # Get cube body and randomize position
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

        # Initialize per-episode reward-shaping trackers.
        self.cube_start_z = float(obs.get('cube_pos', [0.0, 0.0, 0.0])[2])
        self.prev_distance = float(np.linalg.norm(obs.get('gripper_to_cube_pos', [0.0, 0.0, 0.0])))
        self.prev_height = 0.0
        eef_pos = obs.get('robot0_eef_pos', [0.0, 0.0, 0.0])
        self.prev_eef_z = float(eef_pos[2]) if len(eef_pos) >= 3 else None
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.lift_milestones_hit = set()
        
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
        info = dict(info or {})
        
        # Get positions
        gripper_to_cube = obs.get('gripper_to_cube_pos', [0, 0, 0])
        gripper_qpos = obs.get('robot0_gripper_qpos', [0, 0])
        eef_pos = obs.get('robot0_eef_pos', [0, 0, 0])
        
        distance = float(np.linalg.norm(gripper_to_cube))
        gripper_closed = np.mean(gripper_qpos) < 0.02
        height_above_table = self._cube_lift_height(obs)
        eef_z = float(eef_pos[2]) if len(eef_pos) >= 3 else None

        prev_distance = distance if self.prev_distance is None else float(self.prev_distance)
        prev_height = height_above_table if self.prev_height is None else float(self.prev_height)
        prev_eef_z = eef_z if self.prev_eef_z is None else self.prev_eef_z

        # -----------------------------------------------------------------
        # Cohesive, progress-based shaping:
        # - reward progress (not just state occupancy) to reduce reward farming
        # - explicit penalties for "descend blindly" behavior
        # -----------------------------------------------------------------
        reach_phi = np.exp(-8.0 * distance)
        reach_phi_prev = np.exp(-8.0 * prev_distance)
        reach_reward = 5.0 * (reach_phi - reach_phi_prev) + 0.15 * reach_phi

        gripper_cmd = float(action[-1]) if len(action) > 0 else 0.0
        near_for_grasp = distance < 0.07
        touch_zone = distance < 0.045
        grasp_reward = 0.0
        if near_for_grasp and gripper_cmd > 0.2:
            grasp_reward += 0.35
        if touch_zone and gripper_closed:
            grasp_reward += 0.75
        if gripper_cmd > 0.25 and distance > 0.12:
            grasp_reward -= 0.45
        if gripper_cmd < -0.25 and (distance < 0.08 or height_above_table > 0.01):
            grasp_reward -= 0.5

        height_delta = height_above_table - prev_height
        lift_reward = 0.0
        if gripper_closed and (distance < 0.09 or height_above_table > 0.003):
            lift_reward += 30.0 * float(np.clip(height_delta, -0.02, 0.02))
            lift_reward += 4.0 * float(np.clip(height_above_table / 0.12, 0.0, 1.5))

        success_reward = 0.0
        for thr, bonus in ((0.015, 2.0), (0.04, 6.0), (0.08, 14.0)):
            if height_above_table >= thr and thr not in self.lift_milestones_hit:
                self.lift_milestones_hit.add(thr)
                success_reward += bonus

        downward_penalty = 0.0
        if eef_z is not None and prev_eef_z is not None:
            eef_dz = eef_z - prev_eef_z
            if eef_dz < -0.002 and distance > 0.12 and height_above_table < 0.005:
                downward_penalty -= 6.0 * abs(eef_dz)
        if len(gripper_to_cube) >= 3 and gripper_to_cube[2] < -0.03 and distance > 0.08 and height_above_table < 0.005:
            downward_penalty -= 0.25
        if prev_height - height_above_table > 0.004:
            downward_penalty -= 15.0 * (prev_height - height_above_table)
        
        # Skill bonus: simplified - just a small consistency bonus
        # The main learning signal comes from task rewards, not skill matching
        # Phase info is now in observations, so the actor learns phase-specific behavior directly
        skill_bonus = 0.0
        
        # Determine actual phase from the same helper used for observation phase
        # so "detected phase" is consistent everywhere.
        actual_phase = int(np.argmax(self._compute_phase(obs)))
        
        # Simple skill consistency: small bonus for matching, no penalty
        if skill is not None and skill == actual_phase:
            skill_bonus = 0.25
        
        # Action smoothness penalty (reduce shakiness)
        prev_action = action if self.prev_action is None else self.prev_action
        action_penalty = -0.01 * float(np.sum(action ** 2))
        action_penalty -= 0.005 * float(np.sum((action - prev_action) ** 2))
        
        shaped_reward = (
            float(reward)
            + reach_reward
            + grasp_reward
            + lift_reward
            + success_reward
            + skill_bonus
            + action_penalty
            + downward_penalty
        )
        shaped_reward = float(np.clip(shaped_reward, -10.0, 30.0))

        # Update trackers for next step.
        self.prev_distance = distance
        self.prev_height = height_above_table
        self.prev_eef_z = eef_z
        self.prev_action = np.array(action, dtype=np.float32, copy=True)
        
        # Store phase info for debugging
        phase_names = ['reach', 'grasp', 'lift', 'hold']
        info['phase'] = phase_names[actual_phase]
        info['actual_phase'] = actual_phase
        info['reward_env'] = float(reward)
        info['reward_reach'] = float(reach_reward)
        info['reward_grasp'] = float(grasp_reward)
        info['reward_lift'] = float(lift_reward)
        info['reward_success'] = float(success_reward)
        info['reward_down'] = float(downward_penalty)
        info['reward_action'] = float(action_penalty)
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

    args.demo_ratio = float(np.clip(args.demo_ratio, 0.0, 1.0))
    args.demo_bc_weight = float(max(0.0, args.demo_bc_weight))
    args.demo_batch_size = int(max(1, args.demo_batch_size))
    
    print(f"Using device: {device}")
    ram_gb = _system_ram_gb()
    if ram_gb > 0:
        print(f"System RAM: {ram_gb:.1f} GiB")
    
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU VRAM: {vram_gb:.1f} GiB")
        if (vram_gb <= 9.0 and args.num_envs >= 8 and args.batch_size >= 512 and not args.low_mem):
            print("WARNING: configuration is aggressive for <=9 GiB VRAM laptops.")
            print("         Consider: --low_mem (or reduce --num_envs/--batch_size/--updates_per_step).")
    
    torch.set_num_threads(max(1, min(8, mp.cpu_count() // args.num_envs)))
    
    # Create environments
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
    hidden_sizes = tuple(args.hidden)
    agent = HierarchicalSACAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_sizes=hidden_sizes,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        device=device,
        use_amp=(args.cuda and (not args.no_amp)),
        use_images=args.use_camera,
        compile_models=(not args.no_compile),
    )
    
    # Resume if specified
    start_episode = 1
    resumed = False
    if args.resume and os.path.exists(args.resume):
        agent.load(args.resume, resume_training=True)
        basename = os.path.basename(args.resume)
        if 'ep' in basename:
            try:
                start_episode = int(basename.split('ep')[1].split('.')[0]) + 1
            except:
                pass
        resumed = True
        print(f"Resumed from {args.resume}")
    
    # Create replay buffer
    replay_buffer = GPUReplayBuffer(
        capacity=args.buffer_size,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=device,
        use_images=args.use_camera,
        image_shape=env.image_shape if args.use_camera else (3, 84, 84),
        async_copy=(not args.sync_replay),
    )
    # Approximate replay memory (state tensors only).
    # columns = obs + action + reward + next_obs + done
    replay_cols = env.obs_dim + env.action_dim + 1 + env.obs_dim + 1
    replay_gib = (args.buffer_size * replay_cols * 4) / (1024 ** 3)
    print(f"  Replay (state tensors) ~ {replay_gib:.2f} GiB on {device}")
    print(f"  AMP: {not args.no_amp} | compile: {not args.no_compile} | sync_replay: {args.sync_replay}")

    # Optional demo integration: replay prefill + demo mini-batch regularization.
    demo_batch_buffer = None
    demo_prefilled_steps = 0
    demo_skip_warmup = False
    if args.demo_data:
        demo_paths = _expand_hdf5_patterns(args.demo_data)
        if not demo_paths:
            raise ValueError(f"No demo files matched --demo_data patterns: {args.demo_data}")
        print(f"Loading demos from {len(demo_paths)} files...")
        demo_data = load_demo_transitions(demo_paths, env.obs_dim, env.action_dim)
        if demo_data is None:
            raise ValueError("Demo loading failed: no compatible transitions were found.")

        meta = demo_data["meta"]
        print(f"  Demo transitions: {meta['num_steps']} from {meta['num_demos']} demos "
              f"({meta['num_files']} files, skipped={meta['skipped_demos']})")

        demo_batch_buffer = DemoBatchBuffer(
            demo_data["obs"], demo_data["actions"], device=device
        )

        if args.demo_prefill_steps < 0:
            prefill_target = meta["num_steps"]
        else:
            prefill_target = min(args.demo_prefill_steps, meta["num_steps"])

        if prefill_target > 0:
            select_idx = np.random.permutation(meta["num_steps"])[:prefill_target]
            for idx in select_idx:
                replay_buffer.add(
                    demo_data["obs"][idx],
                    demo_data["actions"][idx],
                    float(demo_data["rewards"][idx, 0]),
                    demo_data["next_obs"][idx],
                    float(demo_data["dones"][idx, 0]),
                )
            if replay_buffer._buffer_ptr > 0:
                replay_buffer._flush()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            demo_prefilled_steps = int(prefill_target)
            print(f"  Prefilled replay with {demo_prefilled_steps} demo transitions")

        demo_skip_warmup = bool(args.demo_skip_warmup)
        del demo_data
    
    # Metrics
    episode_rewards = np.zeros(args.num_envs)
    episode_max_heights = np.zeros(args.num_envs, dtype=np.float32)
    episode_success = np.zeros(args.num_envs, dtype=bool)
    all_episode_rewards = deque(maxlen=100)
    all_episode_max_heights = deque(maxlen=100)
    all_episode_success = deque(maxlen=100)
    best_avg_reward = -float('inf')
    skill_counts = np.zeros(SkillSelector.NUM_SKILLS)
    actual_phase_counts = np.zeros(SkillSelector.NUM_SKILLS, dtype=np.int64)
    skill_match_count = 0
    skill_match_total = 0
    
    save_dir = os.path.join(ROOT, "checkpoints_v2", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("  HIERARCHICAL SAC TRAINING - Rover2026 Lift Task V2")
    print("="*70)
    print(f"  Skills: {SkillSelector.SKILL_NAMES}")
    print(f"  Episodes: {args.episodes} | Batch: {args.batch_size}")
    print(f"  Hidden: {hidden_sizes}")
    print(f"  Phase supervision weight: {args.phase_supervision_weight}")
    print(f"  Skill diversity bonus: {args.skill_div_bonus}")
    print(f"  Save: {save_dir}")
    print("="*70 + "\n")
    
    total_steps = 0
    update_times = deque(maxlen=100)
    train_start = time.time()
    stop_requested = False
    avail_ram_gb = _available_ram_gb()
    gpu_free_gb = None
    last_metrics = {"bc_loss": 0.0}

    if demo_prefilled_steps > 0 and demo_skip_warmup:
        total_steps = max(total_steps, args.warmup_steps)
        print("  Skipping random warmup because replay was prefixed with demos.")
    
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
            in_random_warmup = (
                total_steps < args.warmup_steps
                and (not resumed or args.warmup_on_resume)
                and not (demo_prefilled_steps > 0 and demo_skip_warmup)
            )
            if in_random_warmup:
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
                
                # Optional diversity bonus (off by default to avoid distorting task rewards).
                diversity_bonus = 0.0
                if args.skill_div_bonus > 0.0:
                    total_skill_usage = skill_counts.sum() + 1e-8
                    skill_usage_pct = skill_counts[skills[i]] / total_skill_usage
                    if skill_usage_pct < 0.25:
                        diversity_bonus = float(args.skill_div_bonus)
                
                reward_with_bonus = rewards[i] + diversity_bonus
                replay_buffer.add(obs[i], actions[i], reward_with_bonus, next_obs[i], float(dones[i]), img, next_img)
                
                episode_rewards[i] += rewards[i]  # Track original reward for logging
                skill_counts[skills[i]] += 1
                info_i = infos[i] if infos is not None else {}
                h_i = float(info_i.get('height', 0.0))
                episode_max_heights[i] = max(episode_max_heights[i], h_i)
                if h_i >= args.success_height:
                    episode_success[i] = True
                phase_i = int(info_i.get('actual_phase', 0))
                if 0 <= phase_i < len(actual_phase_counts):
                    actual_phase_counts[phase_i] += 1
                skill_match_total += 1
                if int(skills[i]) == phase_i:
                    skill_match_count += 1
                
                if dones[i]:
                    all_episode_rewards.append(episode_rewards[i])
                    all_episode_max_heights.append(float(episode_max_heights[i]))
                    all_episode_success.append(float(episode_success[i]))
                    episode_rewards[i] = 0
                    episode_max_heights[i] = 0.0
                    episode_success[i] = False
            
            obs = next_obs
            images = next_images
            total_steps += args.num_envs
            
            # Update
            if total_steps >= args.warmup_steps and replay_buffer.size >= args.batch_size:
                update_start = time.perf_counter()
                for _ in range(args.updates_per_step):
                    demo_batch = None
                    if (demo_batch_buffer is not None and args.demo_bc_weight > 0.0
                            and args.demo_ratio > 0.0):
                        demo_bs = max(1, int(args.batch_size * args.demo_ratio))
                        if args.demo_batch_size > 0:
                            demo_bs = min(demo_bs, args.demo_batch_size)
                        demo_batch = demo_batch_buffer.sample(demo_bs)

                    last_metrics = agent.update(
                        replay_buffer,
                        args.batch_size,
                        demo_batch=demo_batch,
                        bc_weight=args.demo_bc_weight,
                        phase_weight=args.phase_supervision_weight,
                    )
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
            avail_ram_gb = _available_ram_gb()
            gpu_free_gb = None
            if device.type == 'cuda':
                try:
                    free_b, total_b = torch.cuda.mem_get_info()
                    gpu_free_gb = free_b / (1024 ** 3)
                except Exception:
                    gpu_free_gb = None
            
            # Skill distribution
            skill_pct = skill_counts / (skill_counts.sum() + 1e-8) * 100
            skill_str = " ".join([f"{n[0]}:{p:.0f}%" for n, p in zip(SkillSelector.SKILL_NAMES, skill_pct)])
            actual_phase_pct = actual_phase_counts / (actual_phase_counts.sum() + 1e-8) * 100
            phase_str = " ".join([f"{n[0]}:{p:.0f}%" for n, p in zip(SkillSelector.SKILL_NAMES, actual_phase_pct)])
            match_pct = 100.0 * skill_match_count / max(skill_match_total, 1)
            avg_max_h = np.mean(all_episode_max_heights) if all_episode_max_heights else 0.0
            succ_rate = (np.mean(all_episode_success) * 100.0) if all_episode_success else 0.0
            bc_str = ""
            if demo_batch_buffer is not None and args.demo_bc_weight > 0.0:
                bc_str = f" bc={last_metrics.get('bc_loss', 0.0):.4f}"
            phase_str_extra = ""
            if args.phase_supervision_weight > 0.0:
                phase_str_extra = f" ce={last_metrics.get('phase_ce', 0.0):.4f}"
            
            print(f"  Ep {episode:5d} | Steps {total_steps:8,} | Reward {avg_reward:7.1f} | "
                  f"α_s {agent.alpha_skill:.3f} α_a {agent.alpha_action:.3f} | "
                  f"Skills(pred): {skill_str} | Phase(actual): {phase_str} | "
                  f"match={match_pct:.1f}% | "
                  f"max_h(avg100)={avg_max_h:.3f} succ@{args.success_height:.2f}m={succ_rate:.1f}%{bc_str}{phase_str_extra} | "
                  f"RAM_avail={avail_ram_gb:.1f}GiB"
                  + (f" GPU_free={gpu_free_gb:.1f}GiB" if gpu_free_gb is not None else ""))

            # Safety exit before host lock-up.
            if avail_ram_gb > 0 and avail_ram_gb < args.min_avail_ram_gb:
                print(f"Stopping early: available RAM {avail_ram_gb:.2f} GiB "
                      f"< threshold {args.min_avail_ram_gb:.2f} GiB")
                stop_requested = True
            if gpu_free_gb is not None and gpu_free_gb < args.min_free_vram_gb:
                print(f"Stopping early: free VRAM {gpu_free_gb:.2f} GiB "
                      f"< threshold {args.min_free_vram_gb:.2f} GiB")
                stop_requested = True
            
            skill_counts[:] = 0  # Reset for next interval
            actual_phase_counts[:] = 0  # Reset for next interval
            skill_match_count = 0
            skill_match_total = 0
            
            if avg_reward > best_avg_reward and episode > start_episode + 50:
                best_avg_reward = avg_reward
                agent.save(os.path.join(save_dir, "best_model.pt"))
        
        if episode % args.save_freq == 0:
            agent.save(os.path.join(save_dir, f"checkpoint_ep{episode}.pt"))

        if stop_requested:
            break
    
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
    
    # Create environment with birdview render + optional gripper camera for observation
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
    
    def process_obs(obs_dict, cube_base_z):
        obs_list = []
        for key in obs_keys:
            if key in obs_dict:
                obs_list.append(np.array(obs_dict[key]).flatten())
        base_obs = np.concatenate(obs_list).astype(np.float32)
        
        # Compute phase
        cube_pos = obs_dict.get('cube_pos', [0, 0, 0])
        gripper_to_cube = obs_dict.get('gripper_to_cube_pos', [0, 0, 0])
        
        distance = np.linalg.norm(gripper_to_cube)
        height = max(0, float(cube_pos[2]) - float(cube_base_z))
        
        phase = np.zeros(4, dtype=np.float32)
        if height > 0.08:
            phase[3] = 1.0
        elif height > 0.015:
            phase[2] = 1.0
        elif distance < 0.09:
            phase[1] = 1.0
        else:
            phase[0] = 1.0
        
        return np.concatenate([base_obs, phase]).astype(np.float32), distance, height, int(np.argmax(phase))
    
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
    
    # Calculate obs_dim
    test_obs = env.reset()
    test_base_z = float(test_obs.get('cube_pos', [0.0, 0.0, 0.0])[2])
    obs, _, _, _ = process_obs(test_obs, test_base_z)
    obs_dim = len(obs)
    
    # Create agent and load weights
    hidden_sizes = tuple(args.hidden)
    agent = HierarchicalSACAgent(
        obs_dim=obs_dim,
        action_dim=env.action_dim,
        hidden_sizes=hidden_sizes,
        device=device,
        use_amp=False,
        use_images=False,  # We're not using camera obs for policy, just visualization
        compile_models=(not args.no_compile),
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
        episode_base_z = float(raw_obs.get('cube_pos', [0.0, 0.0, 0.0])[2])
        
        obs, distance, height, actual_phase = process_obs(raw_obs, episode_base_z)
        max_height = height
        agent.reset()
        episode_reward = 0
        done = False
        skill_seq = []
        actual_phase_seq = []
        skill_match = 0
        step_count = 0
        
        while not done:
            action, skill = agent.get_action(obs, deterministic=True)
            skill_seq.append(skill)
            actual_phase_seq.append(actual_phase)
            if int(skill) == int(actual_phase):
                skill_match += 1
            
            raw_obs, reward, done, info = env.step(action)
            obs, distance, height, actual_phase = process_obs(raw_obs, episode_base_z)
            max_height = max(max_height, height)
            
            episode_reward += reward
            step_count += 1
            
            # Render main birdview
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
        phase_counts = Counter([SkillSelector.SKILL_NAMES[p] for p in actual_phase_seq])
        phase_str = ", ".join([f"{k}: {v}" for k, v in phase_counts.most_common()])
        success = max_height >= args.success_height
        status = "SUCCESS" if success else "FAIL"
        match_pct = 100.0 * skill_match / max(len(skill_seq), 1)
        
        print(f"  Episode {episode+1}: {status} | Reward={episode_reward:.1f} | "
              f"Height(final={height:.3f}m, max={max_height:.3f}m) | "
              f"Skills(pred): {skill_str} | Phase(actual): {phase_str} | match={match_pct:.1f}%")
    
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
    
    # Mode
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
    train_args.add_argument('--hidden', type=int, nargs='+', default=[512, 512, 256],
                            help='Actor / critic hidden sizes')
    train_args.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    train_args.add_argument('--gamma', type=float, default=0.99, help='Discount')
    train_args.add_argument('--tau', type=float, default=0.005, help='Soft update')
    train_args.add_argument('--warmup_steps', type=int, default=5000, help='Warmup')
    train_args.add_argument('--warmup_on_resume', action='store_true',
                            help='Keep random warmup even when --resume is provided')
    train_args.add_argument('--updates_per_step', type=int, default=8, help='Updates per step')
    train_args.add_argument('--demo_data', type=str, nargs='*', default=[],
                            help='Demo HDF5 files / globs for replay prefill + BC regularization')
    train_args.add_argument('--demo_prefill_steps', type=int, default=-1,
                            help='How many demo transitions to prefill into replay (-1 = all)')
    train_args.add_argument('--demo_ratio', type=float, default=0.25,
                            help='Fraction of batch size to use for demo BC minibatch')
    train_args.add_argument('--demo_batch_size', type=int, default=256,
                            help='Cap for demo BC minibatch size per update')
    train_args.add_argument('--demo_bc_weight', type=float, default=0.1,
                            help='Weight for demo behavior-cloning regularization term')
    train_args.add_argument('--phase_supervision_weight', type=float, default=0.2,
                            help='Weight for cross-entropy alignment of selected skill vs detected phase')
    train_args.add_argument('--skill_div_bonus', type=float, default=0.0,
                            help='Optional reward bonus for underused skills (0 disables)')
    train_args.add_argument('--demo_skip_warmup', action='store_true',
                            help='Skip random warmup when replay was prefilled with demos')
    train_args.add_argument('--low_mem', action='store_true',
                            help='Safer settings for low-memory GPUs/laptops')
    train_args.add_argument('--no_compile', action='store_true',
                            help='Disable torch.compile for lower peak memory / startup cost')
    train_args.add_argument('--no_amp', action='store_true',
                            help='Disable mixed precision for maximum stability')
    train_args.add_argument('--sync_replay', action='store_true',
                            help='Disable async replay H2D copies (safer on unstable laptops)')
    train_args.add_argument('--min_avail_ram_gb', type=float, default=1.0,
                            help='Early-stop guard when available system RAM drops below this')
    train_args.add_argument('--min_free_vram_gb', type=float, default=0.4,
                            help='Early-stop guard when free VRAM drops below this')
    train_args.add_argument('--success_height', type=float, default=0.10,
                            help='Height above table (m) considered a successful lift')
    
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
    _apply_low_memory_profile(args)
    
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
