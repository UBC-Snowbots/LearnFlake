#!/usr/bin/env python3
"""
BC -> HRL-SAC warm-start checkpoint converter.

Supports BC checkpoints from:
- bc_trainer.py  (key: 'policy')
- bc_train.py    (key: 'model')

Output checkpoint is compatible with train_lift_v2.py --resume.
"""

import argparse
import os
import re
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


ROOT = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# Shared model builders (aligned with train_lift_v2.py)
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
        self.net = mlp([obs_dim + skill_embed_dim] + list(hidden_sizes))
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


# ============================================================================
# BC checkpoint models
# ============================================================================

class BCTrainerPolicy(nn.Module):
    """Matches bc_trainer.py format: net + action_head."""

    def __init__(self, obs_dim, action_dim, hidden_sizes=(512, 512, 256)):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes))
        self.action_head = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, obs):
        return torch.tanh(self.action_head(self.net(obs)))


class BCTrainPolicy(nn.Module):
    """Matches bc_train.py format: sequential net with final tanh."""

    def __init__(self, obs_dim, action_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.SiLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


def _infer_bc_train_hidden(state_dict: dict[str, torch.Tensor]) -> tuple[int, ...]:
    linear_layers = []
    for key, value in state_dict.items():
        m = re.match(r"net\.(\d+)\.weight", key)
        if m and value.ndim == 2:
            linear_layers.append((int(m.group(1)), tuple(value.shape)))
    linear_layers.sort(key=lambda x: x[0])
    # All linears except final output linear
    hidden = [shape[0] for _, shape in linear_layers[:-1]]
    return tuple(hidden)


def _get_final_bc_train_linear_index(state_dict: dict[str, torch.Tensor]) -> int:
    idxs = []
    for key in state_dict.keys():
        m = re.match(r"net\.(\d+)\.weight", key)
        if m:
            idxs.append(int(m.group(1)))
    if not idxs:
        raise ValueError("Could not find any linear layer in bc_train checkpoint")
    return max(idxs)


def load_bc_checkpoint(path: str, device: str = "cpu") -> dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)

    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]

    if "policy" in ckpt:
        fmt = "bc_trainer"
        hidden = tuple(ckpt.get("hidden_sizes", (512, 512, 256)))
        policy = BCTrainerPolicy(obs_dim, action_dim, hidden)
        policy.load_state_dict(ckpt["policy"])
        sd = policy.state_dict()
        mu_w_key = "action_head.weight"
        mu_b_key = "action_head.bias"
        loss = ckpt.get("val_loss")
    elif "model" in ckpt:
        fmt = "bc_train"
        hidden = tuple(ckpt.get("hidden_sizes", ()))
        if not hidden:
            hidden = _infer_bc_train_hidden(ckpt["model"])
        policy = BCTrainPolicy(obs_dim, action_dim, hidden)
        policy.load_state_dict(ckpt["model"])
        sd = policy.state_dict()
        out_idx = _get_final_bc_train_linear_index(sd)
        mu_w_key = f"net.{out_idx}.weight"
        mu_b_key = f"net.{out_idx}.bias"
        loss = ckpt.get("loss")
    else:
        raise ValueError("Unsupported BC checkpoint format: missing 'policy' or 'model' key")

    return {
        "format": fmt,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_sizes": hidden,
        "state_dict": sd,
        "mu_w_key": mu_w_key,
        "mu_b_key": mu_b_key,
        "loss": loss,
        "epoch": ckpt.get("epoch"),
    }


# ============================================================================
# Transfer
# ============================================================================

def transfer_bc_to_rl(
    bc_path: str,
    output_path: str,
    device: str = "cpu",
    rl_hidden: tuple[int, ...] | None = None,
    match_bc_hidden: bool = False,
):
    bc = load_bc_checkpoint(bc_path, device=device)

    obs_dim = bc["obs_dim"]
    action_dim = bc["action_dim"]
    bc_hidden = tuple(bc["hidden_sizes"])

    if match_bc_hidden:
        rl_hidden = bc_hidden
    elif rl_hidden is None:
        rl_hidden = (512, 512, 256)

    if len(rl_hidden) == 0:
        raise ValueError("rl_hidden must contain at least one layer size")

    print(
        f"Loaded BC checkpoint ({bc['format']}): obs_dim={obs_dim}, action_dim={action_dim}, "
        f"bc_hidden={bc_hidden}, epoch={bc['epoch']}, loss={bc['loss']}"
    )
    print(f"Target RL hidden sizes: {rl_hidden}")

    skill_embed_dim = 32
    rl_actor = SkillConditionedActor(
        obs_dim,
        action_dim,
        num_skills=SkillSelector.NUM_SKILLS,
        skill_embed_dim=skill_embed_dim,
        hidden_sizes=rl_hidden,
    )
    rl_skill_selector = SkillSelector(obs_dim)
    rl_critic = DoubleCritic(obs_dim, action_dim, rl_hidden)
    rl_critic_target = DoubleCritic(obs_dim, action_dim, rl_hidden)
    rl_critic_target.load_state_dict(rl_critic.state_dict())

    bc_sd = bc["state_dict"]
    rl_sd = rl_actor.state_dict()
    transferred = 0
    skipped = 0

    # 1) Transfer action output -> mu_layer
    mu_w_key = bc["mu_w_key"]
    mu_b_key = bc["mu_b_key"]
    bc_mu_w = bc_sd[mu_w_key]
    bc_mu_b = bc_sd[mu_b_key]
    if rl_sd["mu_layer.weight"].shape == bc_mu_w.shape:
        rl_sd["mu_layer.weight"] = bc_mu_w.clone()
        rl_sd["mu_layer.bias"] = bc_mu_b.clone()
        transferred += 2
        print(f"  Transferred: {mu_w_key}/{mu_b_key} -> mu_layer")
    else:
        skipped += 2
        print(
            f"  SKIPPED mu_layer: BC={tuple(bc_mu_w.shape)} RL={tuple(rl_sd['mu_layer.weight'].shape)}"
        )

    # 2) Initialize log_std to low stochasticity
    nn.init.constant_(rl_sd["log_std_layer.weight"], 0.0)
    nn.init.constant_(rl_sd["log_std_layer.bias"], -2.0)

    # 3) Transfer hidden layers where compatible
    skip_keys = {mu_w_key, mu_b_key}
    for bc_key, bc_param in bc_sd.items():
        if not bc_key.startswith("net.") or bc_key in skip_keys:
            continue
        if bc_key not in rl_sd:
            skipped += 1
            continue

        rl_param = rl_sd[bc_key]
        if bc_param.shape == rl_param.shape:
            rl_sd[bc_key] = bc_param.clone()
            transferred += 1
            continue

        # Special handling for first layer: RL has extra skill embedding columns.
        if bc_key == "net.0.weight":
            out_ok = rl_param.shape[0] == bc_param.shape[0]
            in_ok = rl_param.shape[1] == bc_param.shape[1] + skill_embed_dim
            if out_ok and in_ok:
                padded = torch.zeros_like(rl_param)
                padded[:, :bc_param.shape[1]] = bc_param
                rl_sd[bc_key] = padded
                transferred += 1
                print("  Transferred: net.0.weight with zero-padded skill columns")
                continue

        skipped += 1

    rl_actor.load_state_dict(rl_sd)
    print(f"Transfer summary: {transferred} tensors transferred, {skipped} skipped")

    import torch.optim as optim

    rl_checkpoint = {
        "skill_selector": rl_skill_selector.state_dict(),
        "actor": rl_actor.state_dict(),
        "critic": rl_critic.state_dict(),
        "critic_target": rl_critic_target.state_dict(),
        "skill_optimizer": optim.AdamW(rl_skill_selector.parameters(), lr=3e-4).state_dict(),
        "actor_optimizer": optim.AdamW(rl_actor.parameters(), lr=3e-4).state_dict(),
        "critic_optimizer": optim.AdamW(rl_critic.parameters(), lr=3e-4).state_dict(),
        "log_alpha_skill": torch.log(torch.tensor([0.5])),
        "log_alpha_action": torch.zeros(1),
        "scaler": torch.amp.GradScaler("cuda", enabled=False).state_dict(),
        "bc_source": bc_path,
        "bc_format": bc["format"],
        "bc_loss": bc["loss"],
        "bc_epoch": bc["epoch"],
        "bc_hidden_sizes": bc_hidden,
        "rl_hidden_sizes": tuple(rl_hidden),
    }

    torch.save(rl_checkpoint, output_path)
    print(f"Saved RL-compatible checkpoint to: {output_path}")
    print(
        "Run: python3 train_lift_v2.py --train --cuda "
        f"--resume {output_path} --hidden {' '.join(str(x) for x in rl_hidden)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Transfer BC policy weights into an HRL-SAC checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bc-checkpoint", required=True, type=str,
                        help="Path to BC checkpoint from bc_train.py or bc_trainer.py")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path for RL checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--rl-hidden", type=int, nargs="+", default=[512, 512, 256],
                        help="RL hidden sizes for actor/critic")
    parser.add_argument("--match-bc-hidden", action="store_true",
                        help="Use BC hidden sizes for RL actor/critic")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(ROOT, "bc_warm_start.pt")

    transfer_bc_to_rl(
        bc_path=args.bc_checkpoint,
        output_path=args.output,
        device=args.device,
        rl_hidden=tuple(args.rl_hidden),
        match_bc_hidden=args.match_bc_hidden,
    )


if __name__ == "__main__":
    main()
