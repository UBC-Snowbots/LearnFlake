"""Train the Approach skill with RLPD-SAC + key-phase curriculum.

Approach: move EEF to within 4 mm XY / 5 mm Z / 5° tilt of any keyboard key.
Action space: 7-D JOINT_POSITION + masked solenoid (Approach forces solenoid
retracted via the env's ActionAdapter).

TRACKER §12 Phase 3. Wires:
  - rl_autonomy.envs.make_env(mode='approach')
  - rl_autonomy.curricula.KeyPhaseCurriculum
  - rl_autonomy.algos.RLPDSAC
  - TensorBoard scalar logging

Run inside rover_gpu:
    python3 -m rl_autonomy.scripts.train_approach \
        --steps 1_000_000 \
        --save-dir checkpoints/approach_v1 \
        --log-dir logs/approach_v1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

import gymnasium as gym

from rl_autonomy.algos import RLPDSAC, RLPDConfig
from rl_autonomy.curricula import KeyPhaseCurriculum
from rl_autonomy.envs import make_env


# ---------------------------------------------------------------------------
# Curriculum-coupled env
# ---------------------------------------------------------------------------

class CurriculumEnv(gym.Wrapper):
    """Re-samples the target key from the curriculum at every reset, and
    pipes per-episode success back to the curriculum for advance decisions."""

    def __init__(self, env: gym.Env, curriculum: KeyPhaseCurriculum):
        super().__init__(env)
        self.curriculum = curriculum
        self._current_key: str = "g"

    def reset(self, **kw):
        self._current_key = self.curriculum.sample_key()
        # Walk down to the underlying KeyboardEnv to set the target.
        underlying = self._find_underlying()
        if underlying is not None:
            underlying.set_target_key(self._current_key)
        obs, info = self.env.reset(**kw)
        info = dict(info)
        info["curriculum_phase"] = self.curriculum.current_phase
        info["target_key"] = self._current_key
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        if terminated or truncated:
            success = bool(info.get("is_success", False))
            self.curriculum.report_outcome(self._current_key, success)
            info["curriculum_phase"] = self.curriculum.current_phase
            info["curriculum_rolling_success"] = self.curriculum.rolling_success_rate()
        return obs, reward, terminated, truncated, info

    def _find_underlying(self):
        env = self.env
        while hasattr(env, "env"):
            if hasattr(env, "underlying"):
                return env.underlying
            env = env.env
        return getattr(env, "underlying", env)


def _find_obs_adapter(env):
    """Walk the wrapper stack to find the ObsAdapter (which holds the RMS)."""
    from rl_autonomy.envs.obs_adapter import ObsAdapter
    cur = env
    while True:
        if isinstance(cur, ObsAdapter):
            return cur
        if hasattr(cur, "env"):
            cur = cur.env
            continue
        return None


def _share_normalizer(train_env, eval_env) -> None:
    """Point the eval env's ObsAdapter at the train env's RMS, and freeze it."""
    train_oa = _find_obs_adapter(train_env)
    eval_oa = _find_obs_adapter(eval_env)
    if train_oa is None or eval_oa is None:
        print("[approach] warning: ObsAdapter not found in one of the envs; skipping RMS share")
        return
    eval_oa.rms = train_oa.rms        # share the underlying object — train updates flow to eval
    eval_oa.training = False          # belt and suspenders: don't let eval update stats
    print(f"[approach] eval env shares train env's RunningMeanStd accumulator (eval frozen)")


# ---------------------------------------------------------------------------
# TensorBoard logger
# ---------------------------------------------------------------------------

class _TBLogger:
    """Optional TensorBoard scalar logger. Falls back to no-op if torch.utils.tensorboard
    isn't importable (which it should always be — pinned in pyproject)."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
            self.enabled = True
        except ImportError:
            print(f"[warn] tensorboard not importable; logging disabled")
            self.writer = None
            self.enabled = False

    def add_scalar(self, name: str, value: float, step: int) -> None:
        if self.enabled and value is not None and np.isfinite(value):
            self.writer.add_scalar(name, value, step)

    def close(self) -> None:
        if self.enabled:
            self.writer.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    p.add_argument("--steps", type=int, default=1_000_000, help="total env steps")
    p.add_argument("--save-dir", type=Path, default="checkpoints/approach_v1")
    p.add_argument("--log-dir", type=Path, default="logs/approach_v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda or cpu (auto if unset)")
    p.add_argument("--frame-stack", type=int, default=3)
    p.add_argument("--domain-rand", action="store_true",
                   help="enable domain randomization (default off; turn on in Phase 4)")
    p.add_argument("--utd", type=int, default=10,
                   help="update-to-data ratio. 10 = RLPD default; 1 = vanilla SAC")
    p.add_argument("--warmstart", type=int, default=5_000)
    p.add_argument("--save-every", type=int, default=50_000,
                   help="save checkpoint every N env steps")
    p.add_argument("--eval-every", type=int, default=20_000,
                   help="run eval every N env steps")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--log-every", type=int, default=1_000,
                   help="terminal log every N env steps")
    args = p.parse_args()

    save_dir = Path(args.save_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # RNG
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[approach] save_dir={save_dir}")
    print(f"[approach] log_dir ={log_dir}")
    print(f"[approach] device  ={args.device or 'auto'}")
    print(f"[approach] steps   ={args.steps:_}")
    print(f"[approach] UTD     ={args.utd}")
    print(f"[approach] DR      ={'on' if args.domain_rand else 'off'}")

    # Envs
    train_env = make_env(
        mode="approach", frame_stack=args.frame_stack,
        domain_rand=args.domain_rand, render=False, seed=args.seed,
    )
    eval_env = make_env(
        mode="approach", frame_stack=args.frame_stack,
        domain_rand=False, render=False, seed=args.seed + 1,
    )

    curriculum = KeyPhaseCurriculum(advance_threshold=0.85, window=200, seed=args.seed)
    train_env = CurriculumEnv(train_env, curriculum)
    eval_env = CurriculumEnv(eval_env, KeyPhaseCurriculum(seed=args.seed + 2))

    # Share the train env's RunningMeanStd with the eval env so the eval policy
    # sees the same normalization it was trained with. Without this, the eval
    # env's RMS is barely populated and observations are mis-scaled by the time
    # they reach the policy. Also freeze the eval env's RMS so it doesn't drift.
    # (See TRACKER §23 — original v1 had independent RMSs and a 2.5× train/eval
    # return gap that traced back to this.)
    _share_normalizer(train_env, eval_env)

    cfg = RLPDConfig(
        update_to_data=args.utd,
        warmstart_steps=args.warmstart,
        seed=args.seed,
    )
    agent = RLPDSAC(env=train_env, config=cfg, device=args.device, eval_env=eval_env)
    print(f"[approach] obs(actor)={agent.actor_dim}  obs(critic)={agent.critic_dim}  "
          f"action={agent.action_dim}  device={agent.device}")

    tb = _TBLogger(str(log_dir))

    # Forward train info to TB and intermittently save the agent.
    last_save = 0
    last_log = 0
    last_curriculum_phase = -1

    def progress(env_step: int, info: dict[str, Any]) -> None:
        nonlocal last_save, last_log, last_curriculum_phase
        if env_step % 100 == 0:
            for k in ("critic_loss", "actor_loss", "alpha", "actor_entropy",
                      "target_q_mean", "demo_fraction"):
                v = info.get(k)
                if v is not None:
                    tb.add_scalar(f"train/{k}", float(v), env_step)
        if env_step - last_save >= args.save_every:
            agent.save(str(save_dir / f"approach_step_{env_step:09d}.pt"))
            agent.save(str(save_dir / "approach_latest.pt"))
            last_save = env_step
        # Curriculum phase transitions
        ph = curriculum.current_phase
        if ph != last_curriculum_phase:
            print(f"[approach] curriculum advanced to phase {ph} "
                  f"(rolling success {curriculum.rolling_success_rate():.2f})", flush=True)
            tb.add_scalar("curriculum/phase", ph, env_step)
            last_curriculum_phase = ph
        tb.add_scalar(
            "curriculum/rolling_success",
            float(curriculum.rolling_success_rate()), env_step,
        )

    history = agent.learn(
        total_timesteps=args.steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        progress=progress,
    )

    # Episode-level metrics: dump to TB at the end too.
    for i, (s, r) in enumerate(zip(history["env_step"], history["episode_return"])):
        tb.add_scalar("episode/return", r, s)

    # Final save
    final_path = save_dir / "approach_final.pt"
    agent.save(str(final_path))
    tb.close()

    print(f"[approach] done. final checkpoint: {final_path}")
    print(f"[approach] curriculum reached phase {curriculum.current_phase} "
          f"({curriculum.rolling_success_rate():.2f} rolling success)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
