"""Numerical inspection of a trained policy's rollouts.

Loads a checkpoint, runs deterministic eval episodes on a fixed key, and
prints per-step trajectory data (eef_pos, target offset, success flag,
contact flag). Useful for diagnosing what a policy is doing when it's
'failing' — flailing? hovering? hitting the keyboard?

Run:
    python3 -m rl_autonomy.tools.inspect_policy \\
        --checkpoint checkpoints/approach_v1_attempt2/approach_latest.pt \\
        --key g --episodes 3 --max-steps 200
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from rl_autonomy.algos import RLPDSAC, RLPDConfig
from rl_autonomy.envs import KeyboardEnv, make_env
from rl_autonomy.envs._wrapper_utils import find_inner


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--key", default="g")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    env = make_env(mode="approach", frame_stack=3, domain_rand=False, seed=0)
    keyboard = find_inner(env, KeyboardEnv)
    assert keyboard is not None

    agent = RLPDSAC(env=env, config=RLPDConfig(), device=args.device)
    agent.load(str(args.checkpoint))
    print(f"[inspect] loaded {args.checkpoint}")
    print(f"[inspect] agent env_steps={agent._env_steps}  gradient_steps={agent._gradient_steps}")
    # Orphan-checkpoint workaround per TRACKER §28: checkpoints saved before
    # the RMS-persistence fix produce a frozen policy at eval time. Warm up.
    if not agent.has_rms():
        print(f"[inspect] no RMS in checkpoint → warming up via 5000 random-action steps")
        agent.warm_up_env_rms(n_steps=5000, action_source="random")
    print()

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        keyboard.set_target_key(args.key)
        # Burn the env to apply the target key
        obs, _ = env.reset(seed=ep)
        keyboard.set_target_key(args.key)

        xy_log = []
        z_log = []
        tilt_log = []
        success_step = None
        contact_step = None

        for step in range(args.max_steps):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            xy, z, tilt = keyboard._compute_approach_errors()
            in_contact = keyboard._in_contact()
            xy_log.append(xy)
            z_log.append(z)
            tilt_log.append(tilt)
            if info.get("is_success") and success_step is None:
                success_step = step
            if in_contact and contact_step is None:
                contact_step = step
            if term or trunc:
                break

        xy_arr = np.array(xy_log) * 1000  # mm
        z_arr = np.array(z_log) * 1000    # mm
        tilt_arr = np.rad2deg(np.array(tilt_log))

        print(f"=== episode {ep+1}/{args.episodes}  key={args.key}  steps={len(xy_log)} ===")
        print(f"  trajectory:")
        # Sample every ~10% of trajectory
        n = len(xy_log)
        sample_idx = list(range(0, n, max(1, n // 10)))
        if n - 1 not in sample_idx:
            sample_idx.append(n - 1)
        print(f"  {'step':>4}  {'xy(mm)':>7}  {'z(mm)':>7}  {'tilt(°)':>8}")
        for i in sample_idx:
            print(f"  {i:>4}  {xy_arr[i]:>7.1f}  {z_arr[i]:>7.1f}  {tilt_arr[i]:>8.2f}")
        print(f"  summary: xy min={xy_arr.min():.1f}mm  mean={xy_arr.mean():.1f}mm  final={xy_arr[-1]:.1f}mm")
        print(f"           z  min={z_arr.min():.1f}mm  mean={z_arr.mean():.1f}mm  final={z_arr[-1]:.1f}mm")
        print(f"           tilt min={tilt_arr.min():.1f}°  mean={tilt_arr.mean():.1f}°  final={tilt_arr[-1]:.1f}°")
        print(f"           success step: {success_step}  in_contact first @ step: {contact_step}")
        print()
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
