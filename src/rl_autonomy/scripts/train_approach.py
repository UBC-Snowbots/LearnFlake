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
        from rl_autonomy.envs import KeyboardEnv
        from rl_autonomy.envs._wrapper_utils import find_inner
        return find_inner(self.env, KeyboardEnv)


def _share_normalizer(train_env, eval_env) -> None:
    """Point the eval env's ObsAdapter at the train env's RMS, and freeze it."""
    from rl_autonomy.envs._wrapper_utils import find_inner
    from rl_autonomy.envs.obs_adapter import ObsAdapter
    train_oa = find_inner(train_env, ObsAdapter)
    eval_oa = find_inner(eval_env, ObsAdapter)
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
    p.add_argument("--utd", type=int, default=2,
                   help=("update-to-data ratio. 2 = v1 default after §26 (collapse "
                         "from α-decay despite §25 reward fix). 5 = §25 default. "
                         "10 = RLPD paper default but unstable on this task without "
                         "demos. 1 = vanilla SAC."))
    p.add_argument("--warmstart", type=int, default=5_000)
    p.add_argument("--save-every", type=int, default=50_000,
                   help="save checkpoint every N env steps")
    p.add_argument("--eval-every", type=int, default=20_000,
                   help="run eval every N env steps")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--log-every", type=int, default=1_000,
                   help="terminal log every N env steps")
    p.add_argument("--demos", type=Path, default=None,
                   help=("path to an HDF5 file of demo transitions (from "
                         "tools.gen_demos). When given, RLPD's demo buffer "
                         "is populated before learning starts and the "
                         "demo_fraction schedule kicks in. See TRACKER §29.4."))
    p.add_argument("--bc-warmup-epochs", type=int, default=0,
                   help=("if >0, run BCPretrain.fit on agent.actor against the "
                         "demo (obs, action) pairs for N epochs before SAC "
                         "starts. Puts the actor INSIDE demo state distribution "
                         "at SAC step 0 — Option E from TRACKER §32.4. "
                         "Resets actor_opt momentum after to avoid stale buffers."))
    p.add_argument("--demo-fraction-final", type=float, default=None,
                   help=("override RLPDConfig.demo_fraction_final. Default = 0.25 "
                         "(v1/v1.1/v1.2/v1.3). TRACKER §32.4 Option D sets this "
                         "to 0.5 to keep demos at 50%% of every batch — keeps the "
                         "success-rich demo signal strong for the whole run."))
    p.add_argument("--reward-mode", choices=["dense", "pbrs_only", "xy_focus"], default="dense",
                   help=("'dense' = original v1 reward (tolerance shaping + PBRS + "
                         "sparse). 'pbrs_only' = TRACKER §30 Option A — drop the "
                         "Gaussian tolerance dense terms (kept the hover attractor "
                         "alive in v1/v1.1); keep PBRS, success, collision, time."))
    p.add_argument("--actor-hidden", type=str, default=None,
                   help=("comma-separated actor MLP hidden widths, e.g. '512,512,512'. "
                         "Default = RLPDConfig.actor_hidden = (256, 256, 256). "
                         "TRACKER §34.5 Option A — bigger actor for diverse-demo BC. "
                         "Critic hidden is left at the default (512, 512, 512)."))
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
    print(f"[approach] reward  ={args.reward_mode}")
    print(f"[approach] actor_h ={cfg.actor_hidden}  critic_h={cfg.critic_hidden}")

    # Envs
    train_env = make_env(
        mode="approach", frame_stack=args.frame_stack,
        domain_rand=args.domain_rand, render=False, seed=args.seed,
        reward_mode=args.reward_mode,
    )
    eval_env = make_env(
        mode="approach", frame_stack=args.frame_stack,
        domain_rand=False, render=False, seed=args.seed + 1,
        reward_mode=args.reward_mode,
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

    cfg_kwargs = dict(
        update_to_data=args.utd,
        warmstart_steps=args.warmstart,
        seed=args.seed,
    )
    if args.demo_fraction_final is not None:
        cfg_kwargs["demo_fraction_final"] = args.demo_fraction_final
        # Keep f_init >= f_final so the schedule is monotone-decreasing-or-flat.
        if args.demo_fraction_final > 0.5:
            cfg_kwargs["demo_fraction_init"] = args.demo_fraction_final
    if args.actor_hidden is not None:
        hidden = tuple(int(w) for w in args.actor_hidden.split(",") if w.strip())
        if len(hidden) < 2 or any(w < 8 for w in hidden):
            raise ValueError(
                f"--actor-hidden must be 2+ ints each ≥8; got {hidden}"
            )
        cfg_kwargs["actor_hidden"] = hidden
    cfg = RLPDConfig(**cfg_kwargs)
    print(f"[approach] demo_f  =init {cfg.demo_fraction_init} → final "
          f"{cfg.demo_fraction_final} over {cfg.demo_fraction_decay_steps} steps")
    agent = RLPDSAC(env=train_env, config=cfg, device=args.device, eval_env=eval_env)
    print(f"[approach] obs(actor)={agent.actor_dim}  obs(critic)={agent.critic_dim}  "
          f"action={agent.action_dim}  device={agent.device}")

    if args.demos is not None:
        from rl_autonomy.data import load_demos_into_buffer
        from rl_autonomy.envs._wrapper_utils import find_inner
        from rl_autonomy.envs.obs_adapter import ObsAdapter
        oa = find_inner(train_env, ObsAdapter)
        # No re-normalize: at this point train RMS is uninitialized (count≈ε).
        # The agent's RMS will warm up from online experience naturally; demo
        # obs are clipped to ±10 from gen time, which is the same clip the
        # agent uses post-normalize, so they live in roughly the right range.
        n_loaded = load_demos_into_buffer(
            args.demos, agent.replay.demos, obs_adapter=oa, re_normalize=False,
        )
        print(f"[approach] loaded {n_loaded} demo transitions from {args.demos}")
        if n_loaded == 0:
            print("[approach] WARNING: demo file empty; training will behave like vanilla SAC")

        if args.bc_warmup_epochs > 0 and n_loaded > 0:
            from rl_autonomy.algos.bc_pretrain import BCPretrain, BCConfig
            print(f"[approach] BC warm-start: {args.bc_warmup_epochs} epochs on "
                  f"{n_loaded} demo transitions")
            d = agent.replay.demos
            obs_np = d.actor_obs[:d.size].detach().cpu().numpy()
            act_np = d.action[:d.size].detach().cpu().numpy()
            bc = BCPretrain(
                agent.actor,
                BCConfig(epochs=args.bc_warmup_epochs, batch_size=256,
                         lr=1e-3, weight_decay=1e-4, device=str(agent.device)),
            )
            bc_history = bc.fit(obs_np, act_np)
            print(f"[approach] BC done. epoch 0 loss={bc_history['loss'][0]:.4f}, "
                  f"final loss={bc_history['loss'][-1]:.4f}")
            # Reset actor_opt momentum so the post-BC SAC update doesn't apply
            # stale AdamW state to the new weights. Recreate with same config.
            agent.actor_opt = torch.optim.AdamW(
                agent.actor.parameters(),
                lr=cfg.actor_lr, weight_decay=cfg.weight_decay,
            )

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
