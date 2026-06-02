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
    # Pin the key BEFORE reset (env built random_key=False) so the observation
    # reflects the target from step 0 and the optional key-aware base
    # pre-rotation (TRACKER §36) is computed for the eval key, not a random one.
    underlying.set_target_key(key)
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


def _chain_strike_phase(env, kb, aa, res_wrap, max_steps: int = 60) -> bool:
    """TRUE chaining (TRACKER §39): the Approach env is already positioned over
    the key. Switch to strike IN-PROCESS (no reset) and extend the solenoid
    open-loop until contact-hold or timeout. Restores approach mode after."""
    aa.set_mode("strike")
    if res_wrap is not None:
        res_wrap.bypass = True
    kb.mode = "strike"
    kb._contact_steps = 0
    kb.done = False
    extend = np.array([0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32)
    success = False
    for _ in range(max_steps):
        try:
            env.step(extend)
        except Exception:
            break
        if kb._check_success():
            success = True
            break
    aa.set_mode("approach")
    if res_wrap is not None:
        res_wrap.bypass = False
    kb.mode = "approach"
    return success


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
    p.add_argument("--strike",   type=Path, default=None,
                   help="Strike checkpoint .pt (not needed with --chain, which uses "
                        "open-loop solenoid extend)")
    p.add_argument("--chain", action="store_true",
                   help="TRUE Approach→Strike chaining (TRACKER §39): after Approach "
                        "succeeds, switch to strike IN-PROCESS (no reset) and extend "
                        "the solenoid open-loop. Requires --residual. The real "
                        "end-to-end M4 number.")
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
    p.add_argument("--key-aware-init", action="store_true",
                   help="pre-rotate the arm base toward each key's column at reset "
                        "(TRACKER §36). MUST match how the Approach checkpoint was "
                        "trained (train_dagger --key-aware-init).")
    p.add_argument("--keyboard-offset", type=str, default="-0.15,0.0",
                   help="keyboard placement 'x,y' (m). MUST match the offset the "
                        "Approach checkpoint was trained with (TRACKER §36.6).")
    p.add_argument("--residual", action="store_true",
                   help="Approach checkpoint is a residual-on-IK policy "
                        "(TRACKER §38): build the residual env so the IK base is "
                        "added to the policy's action.")
    p.add_argument("--residual-tube", type=float, default=0.15,
                   help="residual tube cap; MUST match training (--residual only).")
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

    kb_off = tuple(float(v) for v in args.keyboard_offset.split(","))
    if args.residual:
        from rl_autonomy.envs import make_residual_env
        print(f"[eval] residual-on-IK mode (tube={args.residual_tube})")
        approach_env = make_residual_env(tube=args.residual_tube, reward_mode="xy_focus",
                                         keyboard_offset=kb_off, random_key=False,
                                         frame_stack=args.frame_stack, seed=args.seed)
    else:
        approach_env = make_env(mode="approach", frame_stack=args.frame_stack,
                                domain_rand=False, random_key=False, seed=args.seed,
                                key_aware_init=args.key_aware_init, keyboard_offset=kb_off)
    approach_agent = _build_agent(approach_env, str(args.approach), args.device)

    # Chain mode needs in-process handles to the approach env's inner pieces.
    chain_kb = chain_aa = chain_res = None
    strike_env = strike_agent = None
    if args.chain:
        if not args.residual:
            print("[eval] --chain requires --residual"); return 1
        from rl_autonomy.envs._wrapper_utils import find_inner
        from rl_autonomy.envs.action_adapter import ActionAdapter
        from rl_autonomy.envs import ResidualIKWrapper
        chain_kb = find_inner(approach_env, KeyboardEnv)
        chain_aa = find_inner(approach_env, ActionAdapter)
        chain_res = find_inner(approach_env, ResidualIKWrapper)
        print("[eval] TRUE Approach→Strike chaining (in-process, open-loop solenoid)")
    else:
        if args.strike is None:
            print("[eval] --strike is required without --chain"); return 1
        strike_env = make_env(mode="strike", frame_stack=args.frame_stack,
                              domain_rand=False, seed=args.seed, keyboard_offset=kb_off)
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
            if args.chain:
                ok_s = _chain_strike_phase(
                    approach_env, chain_kb, chain_aa, chain_res,
                    max_steps=max(args.max_strike_steps, 60),
                )
            else:
                ok_s, _, _ = _strike_episode(
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
