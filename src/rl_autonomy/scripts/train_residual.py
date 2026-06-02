"""Train a tube-clipped residual on the M1 DLS-IK expert (TRACKER §38).

The ceiling-raiser past DAgger (§37.1): DAgger matches the IK's ~62% per-attempt
quality but cannot beat it. Here the agent learns a small residual added to the
live IK action; RL can push past 62% on the keys the IK gets *almost* right
(precision / workspace-edge), while the tube clip keeps exploration inside the
4 mm success basin (the thing that sank the v3–v7 online-SAC runs).

  a_final[:6] = clip(a_ik[:6] + tube * residual[:6], -1, 1);  a_final[6] = -1

The actor head is zero-initialised so training starts at residual ≈ 0 ⇒ the pure
IK baseline (already ~62%), then RL only improves. Standard RLPD-SAC otherwise
(env reward = xy_focus dense + success bonus). random_key=True trains across all
87 keys at the tuned keyboard position (-0.10,-0.10, §36.6).

Eval the result with: eval_orchestrator --residual --residual-tube <T> --keyboard-offset=-0.10,-0.10

Run inside rover_gpu:
    python3 -m rl_autonomy.scripts.train_residual \
        --steps 200000 --tube 0.15 \
        --save-dir checkpoints/approach_v16_residual --log-dir logs/approach_v16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl_autonomy.algos import RLPDSAC, RLPDConfig
from rl_autonomy.envs import make_residual_env
from rl_autonomy.envs._wrapper_utils import find_inner
from rl_autonomy.envs.obs_adapter import ObsAdapter


def _share_normalizer(train_env, eval_env) -> None:
    train_oa = find_inner(train_env, ObsAdapter)
    eval_oa = find_inner(eval_env, ObsAdapter)
    if train_oa is None or eval_oa is None:
        print("[residual] warning: ObsAdapter not found; skipping RMS share")
        return
    eval_oa.rms = train_oa.rms
    eval_oa.training = False
    print("[residual] eval env shares train RMS (eval frozen)")


def _zero_init_actor_head(agent: RLPDSAC) -> None:
    """Zero the actor head so the initial residual is ~0 ⇒ start at pure IK."""
    with torch.no_grad():
        agent.actor.head.weight.zero_()
        agent.actor.head.bias.zero_()
    print("[residual] zero-init actor head ⇒ training starts at the IK baseline")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--tube", type=float, default=0.15,
                   help="residual magnitude cap. Small keeps exploration inside "
                        "the 4mm basin. 0.10-0.20 is the sweet spot.")
    p.add_argument("--reward-mode", choices=["dense", "pbrs_only", "xy_focus"],
                   default="xy_focus")
    p.add_argument("--keyboard-offset", type=str, default="-0.10,-0.10",
                   help="keyboard placement 'x,y' (use the =form for the leading minus)")
    p.add_argument("--utd", type=int, default=2)
    p.add_argument("--warmstart", type=int, default=5_000)
    p.add_argument("--frame-stack", type=int, default=3)
    p.add_argument("--no-zero-init", action="store_true",
                   help="skip zero-init of the actor head (start with random residual)")
    p.add_argument("--save-dir", type=Path, default="checkpoints/approach_residual")
    p.add_argument("--log-dir", type=Path, default="logs/approach_residual")
    p.add_argument("--save-every", type=int, default=25_000)
    p.add_argument("--eval-every", type=int, default=20_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--log-every", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    save_dir = Path(args.save_dir).resolve(); save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir).resolve(); log_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    kb_off = tuple(float(v) for v in args.keyboard_offset.split(","))
    print(f"[residual] tube={args.tube} reward={args.reward_mode} "
          f"keyboard_offset={kb_off} steps={args.steps:_}")

    train_env = make_residual_env(tube=args.tube, reward_mode=args.reward_mode,
                                  keyboard_offset=kb_off, random_key=True,
                                  frame_stack=args.frame_stack, seed=args.seed)
    eval_env = make_residual_env(tube=args.tube, reward_mode=args.reward_mode,
                                 keyboard_offset=kb_off, random_key=True,
                                 frame_stack=args.frame_stack, seed=args.seed + 1)
    _share_normalizer(train_env, eval_env)

    cfg = RLPDConfig(update_to_data=args.utd, warmstart_steps=args.warmstart, seed=args.seed)
    agent = RLPDSAC(env=train_env, config=cfg, device=args.device, eval_env=eval_env)
    if not args.no_zero_init:
        _zero_init_actor_head(agent)
    print(f"[residual] obs(actor)={agent.actor_dim} action={agent.action_dim} "
          f"device={agent.device}")

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=str(log_dir))
    except Exception:
        tb = None

    last_save = 0

    def progress(env_step: int, info: dict[str, Any]) -> None:
        nonlocal last_save
        if tb is not None and env_step % 100 == 0:
            for k in ("critic_loss", "actor_loss", "alpha", "eval_return"):
                v = info.get(k)
                if v is not None and np.isfinite(v):
                    tb.add_scalar(f"residual/{k}", float(v), env_step)
        if env_step - last_save >= args.save_every:
            agent.save(str(save_dir / f"residual_step_{env_step:09d}.pt"))
            agent.save(str(save_dir / "residual_latest.pt"))
            last_save = env_step

    agent.learn(total_timesteps=args.steps, log_every=args.log_every,
                eval_every=args.eval_every, eval_episodes=args.eval_episodes,
                progress=progress)
    agent.save(str(save_dir / "residual_final.pt"))
    if tb is not None:
        tb.close()
    print(f"[residual] done. final: {save_dir / 'residual_final.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
