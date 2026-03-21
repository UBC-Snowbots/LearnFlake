#!/usr/bin/env python3
"""
Behavior Cloning (BC) Trainer for Imitation Learning

Loads demonstration HDF5 files produced by demo_recorder.py and trains a
policy network π(s) → a using supervised learning (MSE loss).

The trained policy uses the same observation/action spaces as train_lift_v2.py,
so its weights can later be loaded into HierarchicalSACAgent for RL fine-tuning.

Usage:
    # Train BC from recorded demos
    python3 bc_trainer.py --data demos/demos_20260224_153000.hdf5 --cuda

    # Train with multiple demo files
    python3 bc_trainer.py --data demos/demos_1.hdf5 demos/demos_2.hdf5 --cuda

    # Evaluate trained BC policy in MuJoCo viewer
    python3 bc_trainer.py --eval bc_checkpoints/best_bc.pt --cuda

Architecture:
    The BC policy is a GaussianActor (same class as train_lift_v2.py) that maps
    obs → action deterministically.  This means you can directly transfer its
    weights into the HierarchicalSACAgent.actor for RL warm-starting.

    For simplicity, BC does NOT use the skill selector — it learns a flat
    obs → action mapping.  The skill hierarchy is learned during RL fine-tuning.
"""

import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    import h5py
except ImportError:
    print("ERROR: h5py is required.  Install with:  pip install h5py")
    sys.exit(1)

# Path setup for RoboSuite (needed for eval mode)
ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite")
if os.path.exists(ROBO_PATH) and ROBO_PATH not in sys.path:
    sys.path.insert(0, ROBO_PATH)


# ============================================================================
# BC Policy Network
# ============================================================================

def mlp(sizes, activation=nn.SiLU, output_activation=nn.Identity):
    """MLP with LayerNorm (same architecture as train_lift_v2)."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(activation())
        else:
            layers.append(output_activation())
    return nn.Sequential(*layers)


class BCPolicy(nn.Module):
    """
    Simple deterministic policy for Behavior Cloning.

    Architecture mirrors GaussianActor from train_lift_v2 but outputs
    a deterministic action (no stochasticity needed for BC).
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_sizes: tuple = (512, 512, 256)):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes))
        self.action_head = nn.Linear(hidden_sizes[-1], action_dim)

        # Same init as train_lift_v2 GaussianActor
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.constant_(self.action_head.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns actions in [-1, 1] (tanh squashed)."""
        h = self.net(obs)
        return torch.tanh(self.action_head(h))


# ============================================================================
# Demo Dataset
# ============================================================================

class DemoDataset(Dataset):
    """Loads (obs, action) pairs from one or more HDF5 demo files."""

    def __init__(self, hdf5_paths: list[str]):
        self.obs_all = []
        self.actions_all = []

        for path in hdf5_paths:
            print(f"  Loading {path} ...")
            with h5py.File(path, 'r') as f:
                data_grp = f['data']
                n_demos = len(data_grp)
                for i in range(n_demos):
                    demo = data_grp[f'demo_{i}']
                    self.obs_all.append(demo['obs'][:])
                    self.actions_all.append(demo['actions'][:])
                meta = f['metadata']
                print(f"    {n_demos} demos, obs_dim={meta.attrs['obs_dim']}, "
                      f"action_dim={meta.attrs['action_dim']}")

        self.obs = np.concatenate(self.obs_all, axis=0)
        self.actions = np.concatenate(self.actions_all, axis=0)
        print(f"  Total: {len(self)} (obs, action) pairs")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.actions[idx]),
        )


# ============================================================================
# Training
# ============================================================================

def train_bc(args):
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    dataset = DemoDataset(args.data)
    obs_dim = dataset.obs.shape[1]
    action_dim = dataset.actions.shape[1]

    # Train / validation split (90/10)
    n = len(dataset)
    n_val = max(1, int(0.1 * n))
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=(device.type == 'cuda'))

    # Policy
    policy = BCPolicy(obs_dim, action_dim, hidden_sizes=(512, 512, 256)).to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Checkpointing
    save_dir = os.path.join(ROOT, "bc_checkpoints", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')

    print(f"\n{'='*60}")
    print(f"  BEHAVIOR CLONING TRAINING")
    print(f"{'='*60}")
    print(f"  obs_dim={obs_dim}  action_dim={action_dim}")
    print(f"  train={n_train}  val={n_val}  epochs={args.epochs}")
    print(f"  batch_size={args.batch_size}  lr={args.lr}")
    print(f"  save_dir={save_dir}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        policy.train()
        train_losses = []
        for obs_batch, action_batch in train_loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)

            pred_actions = policy(obs_batch)
            loss = F.mse_loss(pred_actions, action_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # ---- Validate ----
        policy.eval()
        val_losses = []
        with torch.no_grad():
            for obs_batch, action_batch in val_loader:
                obs_batch = obs_batch.to(device)
                action_batch = action_batch.to(device)
                pred_actions = policy(obs_batch)
                loss = F.mse_loss(pred_actions, action_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        # ---- Logging ----
        if epoch % args.log_freq == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:4d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"LR: {lr:.2e}")

        # ---- Save best ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'policy': policy.state_dict(),
                'obs_dim': obs_dim,
                'action_dim': action_dim,
                'hidden_sizes': (512, 512, 256),
                'epoch': epoch,
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_bc.pt'))

        # ---- Periodic save ----
        if epoch % args.save_freq == 0:
            torch.save({
                'policy': policy.state_dict(),
                'obs_dim': obs_dim,
                'action_dim': action_dim,
                'hidden_sizes': (512, 512, 256),
                'epoch': epoch,
                'val_loss': val_loss,
            }, os.path.join(save_dir, f'bc_epoch{epoch}.pt'))

    # Final save
    torch.save({
        'policy': policy.state_dict(),
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'hidden_sizes': (512, 512, 256),
        'epoch': args.epochs,
        'val_loss': val_loss,
    }, os.path.join(save_dir, 'final_bc.pt'))

    print(f"\n{'='*60}")
    print(f"  BC TRAINING COMPLETE")
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    print(f"  Saved to: {save_dir}")
    print(f"{'='*60}\n")


# ============================================================================
# Evaluation — run BC policy in MuJoCo to see if the arm moves sensibly
# ============================================================================

def evaluate_bc(args):
    import robosuite as suite
    from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    ckpt = torch.load(args.eval, map_location=device, weights_only=False)
    obs_dim = ckpt['obs_dim']
    action_dim = ckpt['action_dim']
    hidden_sizes = ckpt.get('hidden_sizes', (512, 512, 256))

    policy = BCPolicy(obs_dim, action_dim, hidden_sizes).to(device)
    policy.load_state_dict(ckpt['policy'])
    policy.eval()

    print(f"Loaded BC policy from {args.eval}  (epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', '?'):.6f})")

    # Create env (same as train_lift_v2 eval)
    arm_ctrl = suite.load_part_controller_config(default_controller="JOINT_VELOCITY")
    ctrl_cfg = refactor_composite_controller_config(arm_ctrl, "Rover2026", ["right"])

    env = suite.make(
        env_name="Lift",
        robots=["Rover2026"],
        controller_configs=ctrl_cfg,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=200,
        reward_shaping=True,
    )

    # Obs processing (must match training exactly)
    obs_keys = [
        'robot0_joint_pos', 'robot0_joint_vel', 'robot0_eef_pos',
        'robot0_eef_quat', 'robot0_gripper_qpos', 'cube_pos',
        'gripper_to_cube_pos',
    ]

    def proc_obs(raw_obs):
        parts = [np.array(raw_obs[k]).flatten() for k in obs_keys if k in raw_obs]
        base = np.concatenate(parts).astype(np.float32)
        # Phase (4-dim)
        cube = raw_obs.get('cube_pos', [0, 0, 0])
        g2c = raw_obs.get('gripper_to_cube_pos', [0, 0, 0])
        gq = raw_obs.get('robot0_gripper_qpos', [0, 0])
        dist = np.linalg.norm(g2c)
        gc = np.mean(gq) < 0.02
        h = max(0, cube[2] - 0.82)
        phase = np.zeros(4, dtype=np.float32)
        if h > 0.08: phase[3] = 1.0
        elif h > 0.01 or (gc and dist < 0.1): phase[2] = 1.0
        elif dist < 0.1: phase[1] = 1.0
        else: phase[0] = 1.0
        return np.concatenate([base, phase]).astype(np.float32)

    print(f"\nRunning BC policy for {args.eval_episodes} episodes...\n")

    for ep in range(args.eval_episodes):
        raw_obs = env.reset()
        obs = proc_obs(raw_obs)
        ep_reward = 0.0
        done = False
        max_h = 0.0

        while not done:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                action = policy(obs_t).cpu().numpy().flatten()

            raw_obs, reward, done, info = env.step(action)
            obs = proc_obs(raw_obs)
            ep_reward += reward

            h = max(0, raw_obs.get('cube_pos', [0, 0, 0])[2] - 0.82)
            max_h = max(max_h, h)

            env.render()
            time.sleep(0.02)

        print(f"  Episode {ep+1}: reward={ep_reward:.1f}, max_height={max_h:.3f}m")

    env.close()
    print("\nEvaluation complete!")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Behavior Cloning Trainer / Evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    parser.add_argument('--data', nargs='+', type=str, default=None,
                        help='HDF5 demo file(s) for training')
    parser.add_argument('--eval', type=str, default=None,
                        help='BC checkpoint to evaluate')

    # Training hyperparams
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

    # Logging
    parser.add_argument('--log-freq', type=int, default=5, help='Print every N epochs')
    parser.add_argument('--save-freq', type=int, default=50, help='Save every N epochs')
    parser.add_argument('--eval-episodes', type=int, default=10, help='Episodes for eval')

    # Hardware
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')

    args = parser.parse_args()

    if args.eval:
        evaluate_bc(args)
    elif args.data:
        train_bc(args)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python bc_trainer.py --data demos/demos_20260224.hdf5 --cuda")
        print("  python bc_trainer.py --eval bc_checkpoints/20260224/best_bc.pt --cuda")
