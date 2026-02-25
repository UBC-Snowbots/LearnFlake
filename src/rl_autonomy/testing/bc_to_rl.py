#!/usr/bin/env python3
"""
BC → RL Weight Transfer

Loads a trained Behavior Cloning (BC) checkpoint and transfers its weights
into a HierarchicalSACAgent from train_lift_v2.py so RL can start from
a warm policy instead of random exploration.

The BC policy (BCPolicy) and the RL actor (SkillConditionedActor) have
different architectures:
    BC:   obs → net(512,512,256) → action_head → tanh
    RL:   [obs, skill_embed(32)] → net(512+32 → 512,512,256) → mu_layer → tanh

Strategy:
    1. Copy the shared hidden layers (net) weights where dimensions match.
    2. Copy BC's action_head → RL actor's mu_layer (same shape).
    3. Initialize RL actor's log_std_layer to small values (low initial stochasticity).
    4. Leave the skill_embedding randomly initialized (RL will learn to use it).
    5. The first linear layer of the RL net has extra columns for the skill embedding
       input — we copy the BC weights into the obs columns and zero-init the rest.

Usage:
    # Create a warm-started RL checkpoint from BC weights
    python3 bc_to_rl.py \\
        --bc-checkpoint bc_checkpoints/20260224/best_bc.pt \\
        --output warm_start_rl.pt

    # Then train RL with warm start:
    python3 train_lift_v2.py --train --cuda --resume warm_start_rl.pt
"""

import os
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

# Reuse definitions from train_lift_v2 and bc_trainer
ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite")
if os.path.exists(ROBO_PATH) and ROBO_PATH not in sys.path:
    sys.path.insert(0, ROBO_PATH)


# ============================================================================
# Rebuild the RL architecture (copied from train_lift_v2.py to be self-contained)
# ============================================================================

def mlp(sizes, activation=nn.SiLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(activation())
        else:
            layers.append(output_activation())
    return nn.Sequential(*layers)


class SkillSelector(nn.Module):
    NUM_SKILLS = 4
    SKILL_NAMES = ['Reach', 'Grasp', 'Lift', 'Hold']
    def __init__(self, obs_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [self.NUM_SKILLS])
    def forward(self, obs, deterministic=False):
        logits = self.net(obs)
        if deterministic:
            return logits.argmax(dim=-1), None, logits
        dist = Categorical(logits=logits)
        skill = dist.sample()
        return skill, dist.log_prob(skill).unsqueeze(-1), logits


class SkillConditionedActor(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -20, 2
    def __init__(self, obs_dim, action_dim, num_skills=4, skill_embed_dim=32,
                 hidden_sizes=(512, 512, 256)):
        super().__init__()
        self.skill_embedding = nn.Embedding(num_skills, skill_embed_dim)
        input_dim = obs_dim + skill_embed_dim
        self.net = mlp([input_dim] + list(hidden_sizes))
        self.mu_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.action_smoothing = nn.Parameter(torch.tensor(0.3))
    def forward(self, obs, skill, prev_action=None, deterministic=False, with_logprob=True):
        x = torch.cat([obs, self.skill_embedding(skill)], dim=-1)
        net_out = self.net(x)
        mu = self.mu_layer(net_out)
        log_std = torch.clamp(self.log_std_layer(net_out), self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        raw_action = mu if deterministic else dist.rsample()
        if prev_action is not None:
            alpha = torch.sigmoid(self.action_smoothing)
            raw_action = alpha * raw_action + (1 - alpha) * prev_action
        logprob = None
        if with_logprob:
            logprob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
            logprob -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(dim=-1, keepdim=True)
        return torch.tanh(raw_action), logprob


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(512, 512, 256)):
        super().__init__()
        self.q = mlp([obs_dim + action_dim] + list(hidden_sizes) + [1])
    def forward(self, obs, action, images=None):
        return self.q(torch.cat([obs, action], dim=-1))


class DoubleCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(512, 512, 256),
                 use_images=False, image_feature_dim=256):
        super().__init__()
        self.q1 = Critic(obs_dim, action_dim, hidden_sizes)
        self.q2 = Critic(obs_dim, action_dim, hidden_sizes)
    def forward(self, obs, action, images=None):
        return self.q1(obs, action), self.q2(obs, action)


# BC policy (from bc_trainer.py)
class BCPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(512, 512, 256)):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes))
        self.action_head = nn.Linear(hidden_sizes[-1], action_dim)
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.constant_(self.action_head.bias, 0.0)
    def forward(self, obs):
        return torch.tanh(self.action_head(self.net(obs)))


# ============================================================================
# Transfer logic
# ============================================================================

def transfer_bc_to_rl(bc_path: str, output_path: str, device: str = 'cpu'):
    """
    Load BC checkpoint, build fresh RL agent, transfer weights, save.
    """
    # ---- Load BC ----
    bc_ckpt = torch.load(bc_path, map_location=device, weights_only=False)
    obs_dim = bc_ckpt['obs_dim']
    action_dim = bc_ckpt['action_dim']
    bc_hidden = bc_ckpt.get('hidden_sizes', (512, 512, 256))

    bc_policy = BCPolicy(obs_dim, action_dim, bc_hidden)
    bc_policy.load_state_dict(bc_ckpt['policy'])
    bc_policy.eval()

    print(f"Loaded BC policy: obs_dim={obs_dim}, action_dim={action_dim}, "
          f"hidden_sizes={bc_hidden}")
    print(f"  BC val_loss={bc_ckpt.get('val_loss', '?')}, epoch={bc_ckpt.get('epoch', '?')}")

    # ---- Build RL components ----
    rl_hidden = (512, 512, 256)
    skill_embed_dim = 32
    rl_actor = SkillConditionedActor(obs_dim, action_dim, num_skills=4,
                                      skill_embed_dim=skill_embed_dim,
                                      hidden_sizes=rl_hidden)
    rl_skill_selector = SkillSelector(obs_dim)
    rl_critic = DoubleCritic(obs_dim, action_dim, rl_hidden)
    rl_critic_target = DoubleCritic(obs_dim, action_dim, rl_hidden)
    rl_critic_target.load_state_dict(rl_critic.state_dict())

    # ---- Transfer weights ----
    bc_sd = bc_policy.state_dict()
    rl_sd = rl_actor.state_dict()

    transferred = 0
    skipped = 0

    # 1. Transfer action_head → mu_layer (exact match)
    rl_sd['mu_layer.weight'] = bc_sd['action_head.weight'].clone()
    rl_sd['mu_layer.bias'] = bc_sd['action_head.bias'].clone()
    transferred += 2
    print("  Transferred: action_head → mu_layer")

    # 2. Initialize log_std to small values (low initial stochasticity)
    nn.init.constant_(rl_sd['log_std_layer.weight'], 0.0)
    nn.init.constant_(rl_sd['log_std_layer.bias'], -2.0)  # std ≈ 0.14
    print("  Initialized: log_std_layer (low stochasticity)")

    # 3. Transfer hidden layers
    # BC net: net.0 (Linear obs→512), net.1 (LN), net.2 (SiLU),
    #         net.3 (Linear 512→512), net.4 (LN), net.5 (SiLU),
    #         net.6 (Linear 512→256), net.7 (Identity)
    # RL net: net.0 (Linear obs+32→512), net.1 (LN), net.2 (SiLU),
    #         net.3 (Linear 512→512), ...same after layer 0
    for bc_key, bc_param in bc_sd.items():
        if not bc_key.startswith('net.'):
            continue

        rl_key = bc_key  # Same key names in both
        if rl_key not in rl_sd:
            skipped += 1
            continue

        rl_param = rl_sd[rl_key]

        if bc_param.shape == rl_param.shape:
            # Shapes match exactly — direct copy
            rl_sd[rl_key] = bc_param.clone()
            transferred += 1
        elif bc_key == 'net.0.weight':
            # First linear layer: BC has (512, obs_dim), RL has (512, obs_dim + 32)
            # Copy BC weights into the obs columns, zero-init the skill columns
            assert rl_param.shape[0] == bc_param.shape[0], \
                f"Output dim mismatch: BC={bc_param.shape[0]}, RL={rl_param.shape[0]}"
            assert rl_param.shape[1] == bc_param.shape[1] + skill_embed_dim, \
                f"Input dim mismatch: RL={rl_param.shape[1]}, expected={bc_param.shape[1] + skill_embed_dim}"

            new_weight = torch.zeros_like(rl_param)
            new_weight[:, :bc_param.shape[1]] = bc_param  # obs columns
            # skill_embed columns stay zero → initially ignores skill input
            rl_sd[rl_key] = new_weight
            transferred += 1
            print(f"  Transferred: {bc_key} (padded {skill_embed_dim} skill columns with zeros)")
        elif bc_key == 'net.0.bias':
            # Bias has same size for first layer
            rl_sd[rl_key] = bc_param.clone()
            transferred += 1
        else:
            print(f"  SKIPPED: {bc_key} shape mismatch BC={bc_param.shape} RL={rl_param.shape}")
            skipped += 1

    rl_actor.load_state_dict(rl_sd)
    print(f"\n  {transferred} params transferred, {skipped} skipped")

    # ---- Compile (matching train_lift_v2 expectations) ----
    # Note: We DON'T compile here — train_lift_v2 does it on load.

    # ---- Save as HierarchicalSACAgent-compatible checkpoint ----
    # Build fresh optimizers (they'll be re-created on resume anyway)
    import torch.optim as optim

    rl_checkpoint = {
        'skill_selector': rl_skill_selector.state_dict(),
        'actor': rl_actor.state_dict(),
        'critic': rl_critic.state_dict(),
        'critic_target': rl_critic_target.state_dict(),
        'skill_optimizer': optim.AdamW(rl_skill_selector.parameters(), lr=3e-4).state_dict(),
        'actor_optimizer': optim.AdamW(rl_actor.parameters(), lr=3e-4).state_dict(),
        'critic_optimizer': optim.AdamW(rl_critic.parameters(), lr=3e-4).state_dict(),
        'log_alpha_skill': torch.log(torch.tensor([0.5])),
        'log_alpha_action': torch.zeros(1),
        'scaler': torch.amp.GradScaler('cuda', enabled=False).state_dict(),
        # Extra metadata
        'bc_source': bc_path,
        'bc_val_loss': bc_ckpt.get('val_loss'),
        'bc_epoch': bc_ckpt.get('epoch'),
    }

    torch.save(rl_checkpoint, output_path)
    print(f"\n  Saved RL-compatible checkpoint to: {output_path}")
    print(f"  Use with:  python train_lift_v2.py --train --cuda --resume {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Transfer BC policy weights into RL agent checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--bc-checkpoint', required=True, type=str,
                        help='Path to bc_trainer.py checkpoint (best_bc.pt)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for RL checkpoint (default: bc_warm_start.pt)')
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(ROOT, 'bc_warm_start.pt')

    transfer_bc_to_rl(args.bc_checkpoint, args.output)


if __name__ == '__main__':
    main()
