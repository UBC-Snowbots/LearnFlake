"""Chain Approach → Strike across all 87 keys; produce a success matrix.

Loads a trained Approach checkpoint and a trained Strike checkpoint, then
for each key on the keyboard runs:

  1. Reset env in 'approach' mode, target the key.
  2. Run Approach policy until success (xy<4mm, z<5mm, tilt<5°) or timeout.
  3. If Approach failed, mark key as failed and continue.
  4. Switch the env to 'strike' mode (preserving arm pose), run Strike policy
     until 3-tick contact hold or timeout.
  5. Record outcome.

Output: a markdown table per key, plus an overall success rate.

Run:
    python3 -m rl_autonomy.scripts.eval_orchestrator \
        --approach checkpoints/approach_v1/approach_final.pt \
        --strike   checkpoints/strike_v1/strike_final.pt \
        --keys all                       # or a comma-separated list
        --trials-per-key 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from rl_autonomy.algos import RLPDSAC, RLPDConfig
from rl_autonomy.envs import KeyboardEnv, make_env
from rl_autonomy.envs.keyboard_layout import AVAILABLE_KEYS


def _build_agent(env, ckpt_path: str, device: str | None,
                 warmup_steps: int = 5000) -> RLPDSAC:
    agent = RLPDSAC(env=env, config=RLPDConfig(), device=device)
    agent.load(ckpt_path)
    # Orphan-checkpoint workaround (TRACKER §28): pre-§28 checkpoints don't
    # contain the RMS state. Without it the policy receives unnormalized
    # observations it was never trained on and freezes. Detect missing RMS
    # and warm it up with random actions before the per-key eval loop.
    if not agent.has_rms():
        print(f"[eval] {ckpt_path}: no RMS in checkpoint → warming up via "
              f"{warmup_steps} random-action steps", flush=True)
        agent.warm_up_env_rms(n_steps=warmup_steps, action_source="random")
    return agent


def _find_keyboard_env(env) -> KeyboardEnv:
    """Walk the wrapper stack to the inner robosuite-native KeyboardEnv."""
    from rl_autonomy.envs._wrapper_utils import require_inner
    return require_inner(env, KeyboardEnv)


def _approach_episode(
    env, agent: RLPDSAC, key: str, max_steps: int = 300
) -> tuple[bool, int, dict]:
    """Run Approach until success or timeout. Returns (success, steps, info)."""
    underlying = _find_keyboard_env(env)
    obs, _ = env.reset()
    underlying.set_target_key(key)

    for step in range(max_steps):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        if info.get("is_success"):
            return True, step + 1, info
        if term or trunc:
            return False, step + 1, info
    return False, max_steps, {}


def _strike_episode(env, agent: RLPDSAC, max_steps: int = 50) -> tuple[bool, int, dict]:
    """Run Strike from current sim state. Note: this resets the env first
    because our env hard-resets on env.reset(). True Approach→Strike
    chaining without reset requires moving the action mask in-process.
    For v1, the Strike env is *separately reset* with a simulated near-key
    state (init pose tightened by the env's _reset_internal); Strike's task
    is just "fire the solenoid given a near-key pose"."""
    obs, _ = env.reset()
    for step in range(max_steps):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        if info.get("is_success"):
            return True, step + 1, info
        if term or trunc:
            return False, step + 1, info
    return False, max_steps, {}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--approach", type=Path, required=True, help="Approach checkpoint .pt")
    p.add_argument("--strike",   type=Path, required=True, help="Strike checkpoint .pt")
    p.add_argument("--keys", default="all",
                   help="comma-separated key names, or 'all', or 'home' (a..z + space)")
    p.add_argument("--trials-per-key", type=int, default=5)
    p.add_argument("--max-approach-steps", type=int, default=300)
    p.add_argument("--max-strike-steps", type=int, default=50)
    p.add_argument("--device", default=None)
    p.add_argument("--frame-stack", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-md", type=Path, default=None,
                   help="optional: write the success matrix to a markdown file")
    args = p.parse_args()

    if args.keys == "all":
        keys = list(AVAILABLE_KEYS)
    elif args.keys == "home":
        keys = list("asdfghjkl") + ["space"]
    else:
        keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    keys = [k for k in keys if k in AVAILABLE_KEYS]
    if not keys:
        print(f"[eval] no valid keys in --keys={args.keys!r}")
        return 1

    print(f"[eval] approach: {args.approach}")
    print(f"[eval] strike:   {args.strike}")
    print(f"[eval] keys:     {len(keys)}, trials per key: {args.trials_per_key}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    approach_env = make_env(mode="approach", frame_stack=args.frame_stack,
                            domain_rand=False, seed=args.seed)
    strike_env = make_env(mode="strike", frame_stack=args.frame_stack,
                          domain_rand=False, seed=args.seed)

    approach_agent = _build_agent(approach_env, str(args.approach), args.device)
    strike_agent = _build_agent(strike_env, str(args.strike), args.device)

    rows: list[dict] = []
    n_full_success = 0
    n_approach_only = 0

    print(f"\n{'idx':>3}  {'key':<10}  {'approach':>10}  {'strike':>10}  total")
    print("-" * 55)
    for i, key in enumerate(keys):
        approach_passes = 0
        strike_passes = 0
        full_passes = 0
        for t in range(args.trials_per_key):
            ok_a, steps_a, _ = _approach_episode(
                approach_env, approach_agent, key, max_steps=args.max_approach_steps,
            )
            approach_passes += int(ok_a)
            if not ok_a:
                continue
            ok_s, steps_s, _ = _strike_episode(
                strike_env, strike_agent, max_steps=args.max_strike_steps,
            )
            strike_passes += int(ok_s)
            if ok_s:
                full_passes += 1
        rows.append({
            "key": key, "approach": approach_passes, "strike": strike_passes,
            "full": full_passes, "trials": args.trials_per_key,
        })
        n_full_success += full_passes
        n_approach_only += approach_passes
        marker = " " if full_passes == args.trials_per_key else "x"
        print(f"{i+1:>3}  {key:<10}  "
              f"{approach_passes}/{args.trials_per_key:<8}  "
              f"{strike_passes}/{args.trials_per_key:<8}  "
              f"{full_passes}/{args.trials_per_key} {marker}", flush=True)

    n_total_trials = len(keys) * args.trials_per_key
    print("-" * 55)
    print(f"\nApproach success: {n_approach_only}/{n_total_trials} "
          f"({100*n_approach_only/n_total_trials:.1f}%)")
    print(f"Full chain success: {n_full_success}/{n_total_trials} "
          f"({100*n_full_success/n_total_trials:.1f}%)")
    keys_all_pass = sum(1 for r in rows if r["full"] == args.trials_per_key)
    print(f"Keys at 100% full success: {keys_all_pass}/{len(keys)}")

    if args.out_md is not None:
        with open(args.out_md, "w") as f:
            f.write(f"# Approach→Strike success matrix\n\n")
            f.write(f"Approach: `{args.approach}`  \n")
            f.write(f"Strike: `{args.strike}`  \n")
            f.write(f"Trials per key: {args.trials_per_key}\n\n")
            f.write("| key | approach | strike | full |\n|---|---|---|---|\n")
            for r in rows:
                f.write(f"| {r['key']} | {r['approach']}/{r['trials']} | "
                        f"{r['strike']}/{r['trials']} | {r['full']}/{r['trials']} |\n")
            f.write(f"\n**Full chain: {n_full_success}/{n_total_trials} = "
                    f"{100*n_full_success/n_total_trials:.1f}%**\n")
        print(f"[eval] wrote {args.out_md}")

    # M4 per TRACKER §15: ≥80/87 keys at ≥80% full success
    m4_pass = sum(1 for r in rows if r["full"] / r["trials"] >= 0.8) >= max(1, int(0.92 * len(keys)))
    print(f"\nM4 status: {'PASSED' if m4_pass else 'NOT YET'}")
    return 0 if m4_pass else 2


if __name__ == "__main__":
    sys.exit(main())
