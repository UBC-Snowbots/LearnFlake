#!/usr/bin/env python3
"""
Behavioral Cloning (BC) Trainer for DAgger Pipeline

Loads demonstration HDF5 files (from demo_recorder.py or dagger_collect.py),
trains an MLP policy via supervised learning, and saves a checkpoint.

Supports DAgger iteration: pass multiple HDF5 files (or globs) and they are
automatically aggregated into one training dataset.

Usage:
    # Train on a single demo file:
    python3 bc_train.py demos/demos_20250101_120000.hdf5

    # DAgger: aggregate ALL hdf5 files in demos/ and retrain:
    python3 bc_train.py demos/*.hdf5

    # Custom hyperparameters:
    python3 bc_train.py demos/*.hdf5 --epochs 200 --lr 1e-3 --batch-size 256

    # Resume from a checkpoint:
    python3 bc_train.py demos/*.hdf5 --resume models/bc_latest.pt

    # Evaluate a trained model (quick sanity check):
    python3 bc_train.py --eval models/bc_latest.pt
"""

import os
import sys
import glob
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    print("ERROR: h5py required — pip install h5py")
    sys.exit(1)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Observation keys must match cartesian_control_ros.py / demo recorder schema.
OBS_KEYS = [
    'robot0_joint_pos',
    'robot0_joint_vel',
    'robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
    'cube_pos',
    'gripper_to_cube_pos',
]
NUM_PHASE_DIMS = 4


def compute_phase(obs: dict) -> np.ndarray:
    """Compute one-hot phase encoding (reach / grasp / lift / hold)."""
    cube_pos = obs.get('cube_pos', [0, 0, 0])
    gripper_to_cube = obs.get('gripper_to_cube_pos', [0, 0, 0])
    gripper_qpos = obs.get('robot0_gripper_qpos', [0, 0])

    distance = np.linalg.norm(gripper_to_cube)
    gripper_closed = np.mean(gripper_qpos) < 0.02
    height_above_table = max(0, cube_pos[2] - 0.82)

    phase = np.zeros(NUM_PHASE_DIMS, dtype=np.float32)
    if height_above_table > 0.08:
        phase[3] = 1.0
    elif height_above_table > 0.01 or (gripper_closed and distance < 0.1):
        phase[2] = 1.0
    elif distance < 0.1:
        phase[1] = 1.0
    else:
        phase[0] = 1.0
    return phase


def process_obs(obs: dict) -> np.ndarray:
    """Convert raw RoboSuite observation dict to the flat BC input vector."""
    obs_list = []
    for key in OBS_KEYS:
        if key in obs:
            obs_list.append(np.array(obs[key]).flatten())
    base_obs = np.concatenate(obs_list).astype(np.float32)
    phase = compute_phase(obs)
    return np.concatenate([base_obs, phase]).astype(np.float32)


# ============================================================================
# Dataset
# ============================================================================

class DemoDataset(Dataset):
    """Load (obs, action) pairs from one or more HDF5 files.

    Schema expected (from demo_recorder.py):
        data/demo_N/obs       (T, obs_dim)
        data/demo_N/actions   (T, action_dim)
    """

    def __init__(self, hdf5_paths: list[str], subsample: int = 1,
                 min_arm_action_norm: float = 0.0,
                 idle_keep_ratio: float = 1.0,
                 rng_seed: int = 0):
        """
        Args:
            hdf5_paths: List of HDF5 file paths to load.
            subsample:  Keep every N-th timestep (1 = all, 2 = half, etc.).
        """
        self.obs_all = []
        self.act_all = []
        self.obs_dim = None
        self.action_dim = None
        self.arm_norm_samples = []
        self.total_steps_before_filter = 0
        self.total_steps_after_filter = 0
        self.arm_norm_mean = 0.0
        self.arm_norm_median = 0.0
        self.arm_norm_p90 = 0.0
        self.arm_near_zero_frac_005 = 0.0
        self.arm_near_zero_frac_010 = 0.0
        self.arm_near_zero_frac_020 = 0.0
        rng = np.random.default_rng(rng_seed)

        total_demos = 0
        for path in hdf5_paths:
            if not os.path.exists(path):
                print(f"  WARNING: {path} not found, skipping")
                continue
            with h5py.File(path, 'r') as f:
                data = f['data']
                for demo_key in sorted(data.keys()):
                    demo = data[demo_key]
                    obs = np.array(demo['obs'], dtype=np.float32)
                    actions = np.array(demo['actions'], dtype=np.float32)
                    self.total_steps_before_filter += len(actions)

                    if subsample > 1:
                        obs = obs[::subsample]
                        actions = actions[::subsample]

                    # Optional: drop most idle timesteps so BC doesn't collapse to
                    # "do nothing" on early iterations with sparse demos.
                    if min_arm_action_norm > 0.0 and actions.shape[1] >= 6:
                        arm_norm = np.linalg.norm(actions[:, :6], axis=1)
                        moving_mask = arm_norm >= min_arm_action_norm
                        idle_mask = ~moving_mask

                        if idle_keep_ratio >= 1.0:
                            keep_mask = np.ones_like(moving_mask, dtype=bool)
                        elif idle_keep_ratio <= 0.0:
                            keep_mask = moving_mask
                        else:
                            keep_idle = rng.random(np.count_nonzero(idle_mask)) < idle_keep_ratio
                            keep_mask = moving_mask.copy()
                            keep_mask[idle_mask] = keep_idle

                        obs = obs[keep_mask]
                        actions = actions[keep_mask]
                        arm_norm = arm_norm[keep_mask]
                    elif actions.shape[1] >= 6:
                        arm_norm = np.linalg.norm(actions[:, :6], axis=1)
                    else:
                        arm_norm = np.linalg.norm(actions, axis=1)

                    self.arm_norm_samples.append(arm_norm)
                    self.total_steps_after_filter += len(actions)

                    self.obs_all.append(obs)
                    self.act_all.append(actions)
                    total_demos += 1

                    if self.obs_dim is None:
                        self.obs_dim = obs.shape[1]
                        self.action_dim = actions.shape[1]

        if not self.obs_all:
            raise ValueError("No demonstrations loaded! Check file paths.")

        self.obs_all = np.concatenate(self.obs_all, axis=0)
        self.act_all = np.concatenate(self.act_all, axis=0)
        arm_norm_all = np.concatenate(self.arm_norm_samples, axis=0)
        self.arm_norm_mean = float(arm_norm_all.mean())
        self.arm_norm_median = float(np.median(arm_norm_all))
        self.arm_norm_p90 = float(np.percentile(arm_norm_all, 90))
        self.arm_near_zero_frac_005 = float((arm_norm_all < 0.005).mean())
        self.arm_near_zero_frac_010 = float((arm_norm_all < 0.010).mean())
        self.arm_near_zero_frac_020 = float((arm_norm_all < 0.020).mean())

        print(f"  Loaded {total_demos} demos from {len(hdf5_paths)} files")
        print(f"  Total timesteps: {len(self.obs_all)}")
        if min_arm_action_norm > 0.0:
            kept_frac = self.total_steps_after_filter / max(self.total_steps_before_filter, 1)
            print(f"  Idle filter: min_arm_action_norm={min_arm_action_norm:.4f} "
                  f"idle_keep_ratio={idle_keep_ratio:.2f}  kept={kept_frac:.3f}")
        print(f"  obs_dim={self.obs_dim}  action_dim={self.action_dim}")
        print("  Arm action norm stats: "
              f"mean={self.arm_norm_mean:.4f} "
              f"median={self.arm_norm_median:.4f} "
              f"p90={self.arm_norm_p90:.4f}")
        print("  Arm near-zero fractions: "
              f"<0.005={self.arm_near_zero_frac_005:.3f} "
              f"<0.010={self.arm_near_zero_frac_010:.3f} "
              f"<0.020={self.arm_near_zero_frac_020:.3f}")

    def __len__(self):
        return len(self.obs_all)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.obs_all[idx], dtype=torch.float32),
            torch.tensor(self.act_all[idx], dtype=torch.float32),
        )


# ============================================================================
# Policy Network
# ============================================================================

class BCPolicy(nn.Module):
    """MLP policy for Behavioral Cloning.

    Architecture matches train_lift_v2.py's style: LayerNorm + SiLU activations.
    Outputs raw (un-squashed) action — the action space is already [-1, 1] from
    the JOINT_VELOCITY normalisation in cartesian_control_ros.py.
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_sizes: tuple = (256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.SiLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())  # Bound output to [-1, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ============================================================================
# Training Loop
# ============================================================================

def train(args):
    """Main training function."""
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Expand glob patterns ----
    hdf5_paths = []
    for pattern in args.data:
        expanded = sorted(glob.glob(pattern))
        if not expanded:
            hdf5_paths.append(pattern)  # pass as-is, Dataset will warn
        else:
            hdf5_paths.extend(expanded)

    print("=" * 60)
    print("  Behavioral Cloning Trainer")
    print("=" * 60)
    print(f"  Data files: {hdf5_paths}")

    # ---- Load data ----
    dataset = DemoDataset(
        hdf5_paths,
        subsample=args.subsample,
        min_arm_action_norm=args.min_arm_action_norm,
        idle_keep_ratio=args.idle_keep_ratio,
        rng_seed=args.seed,
    )
    if args.min_arm_action_norm <= 0.0 and dataset.arm_near_zero_frac_010 > 0.6:
        print("  WARNING: >60% of samples are near-zero arm actions.")
        print("           This often leads to weak first-iteration BC behavior.")
        print("           Try: --min-arm-action-norm 0.01 --idle-keep-ratio 0.3")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=False,
    )

    # ---- Create or load model ----
    obs_dim = dataset.obs_dim
    action_dim = dataset.action_dim
    policy = BCPolicy(obs_dim, action_dim, hidden_sizes=tuple(args.hidden)).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        policy.load_state_dict(ckpt['model'])
        print(f"  Resumed from {args.resume} (epoch {ckpt.get('epoch', '?')})")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    print(f"  Model params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"  Epochs: {args.epochs}  Batch: {args.batch_size}  LR: {args.lr}")
    print(f"  Device: {device}")
    print(f"  Batches / epoch: {len(loader)}")
    print("-" * 60)

    # ---- Train ----
    best_loss = float('inf')
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        policy.train()
        epoch_loss = 0.0
        n_batches = 0

        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device, non_blocking=True)
            act_batch = act_batch.to(device, non_blocking=True)

            pred = policy(obs_batch)
            loss = loss_fn(pred, act_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % args.log_every == 0 or epoch == 1:
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.6f}  lr={lr:.2e}")

        # Save best + periodic checkpoints
        if avg_loss < best_loss:
            best_loss = avg_loss
            _save(policy, obs_dim, action_dim, epoch, avg_loss,
                  os.path.join(args.output_dir, "bc_best.pt"))

        if epoch % args.save_every == 0:
            _save(policy, obs_dim, action_dim, epoch, avg_loss,
                  os.path.join(args.output_dir, f"bc_epoch{epoch}.pt"))

    # Final save
    path = os.path.join(args.output_dir, "bc_latest.pt")
    _save(policy, obs_dim, action_dim, args.epochs, avg_loss, path)
    print("-" * 60)
    print(f"  Training complete!  Best loss: {best_loss:.6f}")
    print(f"  Model saved to: {path}")


def _save(policy, obs_dim, action_dim, epoch, loss, path):
    torch.save({
        'model': policy.state_dict(),
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'epoch': epoch,
        'loss': loss,
        'saved_at': datetime.now().isoformat(),
    }, path)


# ============================================================================
# Evaluation — run trained policy in MuJoCo
# ============================================================================

def evaluate(args):
    """Load a trained BC model and run it in MuJoCo viewer."""
    # RoboSuite / MuJoCo path bootstrap
    ROOT = os.path.dirname(os.path.abspath(__file__))
    ROBO_PATH = os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite")
    if os.path.exists(ROBO_PATH) and ROBO_PATH not in sys.path:
        sys.path.insert(0, ROBO_PATH)

    import robosuite as suite
    from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
    from robosuite.wrappers import VisualizationWrapper

    device = torch.device(args.device)
    ckpt = torch.load(args.eval, map_location=device, weights_only=True)
    obs_dim = ckpt['obs_dim']
    action_dim = ckpt['action_dim']

    policy = BCPolicy(obs_dim, action_dim, hidden_sizes=tuple(args.hidden)).to(device)
    policy.load_state_dict(ckpt['model'])
    policy.eval()

    loss_val = ckpt.get('loss', None)
    loss_str = f"{float(loss_val):.6f}" if loss_val is not None else "n/a"
    print(f"Loaded BC model from {args.eval}")
    print(f"  obs_dim={obs_dim}  action_dim={action_dim}  "
          f"epoch={ckpt.get('epoch', '?')}  loss={loss_str}")

    # Same env config as cartesian_control_ros.py
    arm_ctrl = suite.load_part_controller_config(default_controller="JOINT_VELOCITY")
    arm_ctrl["output_max"] = [1.0, 1.0, 0.8, 0.8, 0.8, 1.0]
    arm_ctrl["output_min"] = [-1.0, -1.0, -0.8, -0.8, -0.8, -1.0]
    ctrl_cfg = refactor_composite_controller_config(arm_ctrl, "Rover2026", ["right"])

    env = suite.make(
        env_name="Lift",
        robots=["Rover2026"],
        controller_configs=ctrl_cfg,
        has_renderer=not args.no_render,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=400,
        reward_shaping=True,
    )
    if not args.no_render:
        env = VisualizationWrapper(env, indicator_configs=None)

    num_episodes = args.eval_episodes
    print(f"\nRunning {num_episodes} episodes in MuJoCo viewer...\n")

    for ep in range(num_episodes):
        raw_obs = env.reset()
        obs_flat = process_obs(raw_obs)
        ep_reward = 0.0
        done = False
        step = 0
        action_abs_sum = 0.0
        arm_norm_sum = 0.0

        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs_flat, dtype=torch.float32,
                                     device=device).unsqueeze(0)
                action = policy(obs_t).cpu().numpy().flatten()
                action = np.clip(action * args.eval_action_scale, -1.0, 1.0)
            action_abs_sum += float(np.mean(np.abs(action)))
            arm_norm_sum += float(np.linalg.norm(action[:6]))

            # Pad to env action dim if needed (model may output 7, env expects 7)
            env_action = np.zeros(env.action_dim, dtype=np.float64)
            env_action[:len(action)] = action

            raw_obs, reward, done, info = env.step(env_action)
            obs_flat = process_obs(raw_obs)
            ep_reward += reward
            step += 1
            if not args.no_render:
                env.render()

        cube_h = raw_obs.get('cube_pos', [0, 0, 0])[2] - 0.82
        success = cube_h > 0.04
        status = "SUCCESS" if success else "FAIL"
        mean_abs_action = action_abs_sum / max(step, 1)
        mean_arm_norm = arm_norm_sum / max(step, 1)
        print(f"  Episode {ep + 1}/{num_episodes}: {status}  "
              f"reward={ep_reward:.2f}  steps={step}  cube_h={cube_h:.3f}  "
              f"|a|={mean_abs_action:.3f} arm_norm={mean_arm_norm:.3f}")

    if num_episodes > 0 and args.eval_action_scale <= 1.0:
        print("Tip: if arm_norm is near zero, re-run eval with --eval-action-scale 2.0 "
              "or train with idle filtering flags below.")

    env.close()
    print("\nDone!")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Behavioral Cloning trainer for DAgger pipeline")
    parser.add_argument('data', nargs='*', default=[],
                        help='HDF5 demo files or glob patterns (e.g. demos/*.hdf5)')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, nargs='+', default=[256, 256],
                        help='Hidden layer sizes (default: 256 256)')
    parser.add_argument('--subsample', type=int, default=1,
                        help='Keep every N-th timestep (default: 1 = all)')
    parser.add_argument('--min-arm-action-norm', type=float, default=0.0,
                        help='Drop mostly-idle timesteps below this arm action norm (default: 0.0 = off)')
    parser.add_argument('--idle-keep-ratio', type=float, default=1.0,
                        help='If filtering idle timesteps, keep this fraction of idle samples (default: 1.0)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed used for idle filtering')

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory for model checkpoints (default: models/)')
    parser.add_argument('--save-every', type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log-every', type=int, default=10,
                        help='Print loss every N epochs')

    # Resume / eval
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from a .pt checkpoint')
    parser.add_argument('--eval', type=str, default=None,
                        help='Evaluate a trained model in MuJoCo viewer')
    parser.add_argument('--eval-episodes', type=int, default=5,
                        help='Number of evaluation episodes (default: 5)')
    parser.add_argument('--eval-action-scale', type=float, default=1.0,
                        help='Multiply policy action during eval before clipping [-1,1]')
    parser.add_argument('--no-render', action='store_true',
                        help='Run eval headless (no MuJoCo viewer window)')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

    if args.eval:
        evaluate(args)
    elif args.data:
        train(args)
    else:
        parser.print_help()
        print("\nError: provide HDF5 data files for training, or --eval for evaluation")
        sys.exit(1)


if __name__ == '__main__':
    main()
