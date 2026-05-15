"""Train the Strike skill with RLPD-SAC.

Strike: from a hover pose above any key, extend the solenoid to make
contact and hold for ≥3 ticks. Action: only the solenoid bit; arm joints
are zeroed by the env's ActionAdapter. Episode horizon: 50 steps.

Strike has no key-position component — random key per episode covers all
spatial cases naturally. No phased curriculum. Keep it simple.

Run inside rover_gpu:
    python3 -m rl_autonomy.scripts.train_strike \
        --steps 100_000 \
        --save-dir checkpoints/strike_v1 \
        --log-dir logs/strike_v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl_autonomy.algos import RLPDSAC, RLPDConfig
from rl_autonomy.envs import make_env
from rl_autonomy.scripts.train_approach import _share_normalizer


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--save-dir", type=Path, default="checkpoints/strike_v1")
    p.add_argument("--log-dir", type=Path, default="logs/strike_v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--frame-stack", type=int, default=3)
    p.add_argument("--utd", type=int, default=4)            # smaller MDP → less UTD needed
    p.add_argument("--warmstart", type=int, default=2_000)
    p.add_argument("--log-every", type=int, default=1_000)
    p.add_argument("--save-every", type=int, default=20_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    args = p.parse_args()

    save_dir = Path(args.save_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[strike] save_dir={save_dir}  log_dir={log_dir}")
    print(f"[strike] steps={args.steps:_}  UTD={args.utd}  warmstart={args.warmstart}")

    train_env = make_env(mode="strike", frame_stack=args.frame_stack,
                         domain_rand=False, seed=args.seed)
    eval_env = make_env(mode="strike", frame_stack=args.frame_stack,
                        domain_rand=False, seed=args.seed + 1)

    # Share train env's RunningMeanStd with eval env so the eval policy sees
    # the same normalization (same fix as train_approach, see TRACKER §23.4).
    _share_normalizer(train_env, eval_env)

    cfg = RLPDConfig(
        update_to_data=args.utd,
        warmstart_steps=args.warmstart,
        seed=args.seed,
    )
    agent = RLPDSAC(env=train_env, config=cfg, device=args.device, eval_env=eval_env)
    print(f"[strike] obs(actor)={agent.actor_dim}  obs(critic)={agent.critic_dim}  "
          f"action={agent.action_dim}  device={agent.device}")

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(log_dir))
    except ImportError:
        writer = None

    last_save = 0

    def progress(env_step: int, info: dict[str, Any]) -> None:
        nonlocal last_save
        if writer and env_step % 100 == 0:
            for k in ("critic_loss", "actor_loss", "alpha", "target_q_mean"):
                v = info.get(k)
                if v is not None and np.isfinite(v):
                    writer.add_scalar(f"train/{k}", float(v), env_step)
        if env_step - last_save >= args.save_every:
            agent.save(str(save_dir / f"strike_step_{env_step:09d}.pt"))
            agent.save(str(save_dir / "strike_latest.pt"))
            last_save = env_step

    history = agent.learn(
        total_timesteps=args.steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        progress=progress,
    )
    if writer:
        for s, r in zip(history["env_step"], history["episode_return"]):
            writer.add_scalar("episode/return", r, s)
        writer.close()

    final_path = save_dir / "strike_final.pt"
    agent.save(str(final_path))
    print(f"[strike] done. final checkpoint: {final_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
