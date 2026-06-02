"""Train the Approach skill with DAgger (Dataset Aggregation).

TRACKER §35. Motivation: across v1..v1.8 (TRACKER §29–§34) the strongest result
was *pure behavior cloning* (v8: 10/435). Online SAC/RLPD always *degraded* the
BC policy because its exploration noise is wider than the 4 mm success basin.
The residual failure of pure BC is the textbook one: **covariate shift /
compounding error** — the cloned policy drifts into states the i.i.d. demo set
never covered near the key (it reaches ~48 mm then wanders back out).

DAgger is the canonical cure, and it is *uniquely* applicable here because we
own a **deterministic expert queryable at any state**: the M1 DLS-Jacobian IK
controller (``rl_autonomy.algos.IKExpert``). Most modern interactive-IL papers
have to ration expert queries with human/uncertainty gating; we don't, so we use
**vanilla automated DAgger** (Ross, Gordon & Bagnell, AISTATS 2011) and skip all
the gating variants.

The loop:
  round 0: roll out the expert (beta=1) → label every state with the expert
           action → BC-fit the actor.  (= BC on on-expert-distribution data.)
  round i: roll out the *current policy* (beta per schedule) → label every
           visited state with the expert → AGGREGATE with all prior data →
           BC-fit on the full aggregate.

Every state the learner actually visits gets a corrective expert label, which
directly populates the near-key fine-correction region BC undersamples. That
turns BC's O(T^2 eps) drift into DAgger's O(T eps).

Normalization is kept self-consistent by warming a RunningMeanStd up front and
then **freezing** it: all aggregated rounds and the saved checkpoint share one
normalizer, so the eval-time obs scaling matches training exactly.

Checkpoints are written through ``RLPDSAC.save`` so ``eval_orchestrator`` (and
the architecture-aware ``RLPDSAC.load``) consume them with zero changes.

Run inside rover_gpu:
    python3 -m rl_autonomy.scripts.train_dagger \
        --keys central --rounds 6 --rollouts-per-round 60 \
        --save-dir checkpoints/approach_v12_dagger \
        --log-dir logs/approach_v12
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl_autonomy.algos import RLPDSAC, RLPDConfig, IKExpert
from rl_autonomy.algos.bc_pretrain import BCPretrain, BCConfig
from rl_autonomy.envs import make_env, KeyboardEnv
from rl_autonomy.envs._wrapper_utils import find_inner
from rl_autonomy.envs.obs_adapter import ObsAdapter
from rl_autonomy.envs.keyboard_layout import PHASE_A_KEYS, PHASE_B_KEYS, AVAILABLE_KEYS


KEY_GROUPS = {
    "phase_a": list(PHASE_A_KEYS),
    "phase_b": list(PHASE_B_KEYS),
    "all": list(AVAILABLE_KEYS),
    # M1's strongest keys (home row + neighbours) — where the IK expert is most
    # reliable, so the cleanest test of "does DAgger close the covariate gap".
    "central": ["g", "h", "f", "j", "d", "k", "s", "l", "t", "y", "r", "u"],
    # A ~24-key sample spread across rows/regions (corners, edges, f-row,
    # numbers, home row, modifiers, arrows). Use as --eval-keys when training on
    # 'all' so the per-round model-selection eval reflects the all-87 objective
    # instead of just the easy central cluster (TRACKER §35.7 — fixes the v13
    # model-selection bug where dagger_best was picked by central keys only).
    "stratified": [
        "esc", "f6", "f12", "grave", "5", "0", "backspace",
        "tab", "t", "p", "backslash", "caps", "a", "f", "j",
        "semicolon", "enter", "lshift", "b", "m", "space", "left", "up", "right",
    ],
}


# ---------------------------------------------------------------------------
# Rollout / collection
# ---------------------------------------------------------------------------

def _pin_key_reset(env, kb: KeyboardEnv, key: str):
    """Reset the env with ``key`` pinned as the target (env built random_key=False)."""
    kb.set_target_key(key)
    obs, info = env.reset()
    kb.set_target_key(key)          # belt-and-suspenders against any re-randomize
    return obs, info


def collect_episode(
    env, kb: KeyboardEnv, agent: RLPDSAC, expert: IKExpert,
    *, key: str, beta: float, max_steps: int, deterministic_policy: bool,
    rng: np.random.Generator, label: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray], bool, int]:
    """Run one episode on ``key``.

    At each step we record (normalized actor obs, **expert** action) — the
    expert labels whatever state the rollout is in. The *driving* action is the
    expert action with probability ``beta``, else the policy action (the DAgger
    beta-mixture that shapes the visited-state distribution).

    Returns (obs_list, expert_action_list, success, n_steps).
    """
    obs, _ = _pin_key_reset(env, kb, key)
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    success = False

    for _ in range(max_steps):
        a_expert = expert.action(kb)                      # label for current state
        a_policy, _ = agent.predict(obs, deterministic=deterministic_policy)
        if label:
            obs_list.append(np.asarray(obs["actor"], dtype=np.float32).copy())
            act_list.append(np.asarray(a_expert, dtype=np.float32).copy())

        drive = a_expert if rng.random() < beta else a_policy
        obs, _, terminated, truncated, info = env.step(drive)
        if info.get("is_success"):
            success = True
        if terminated or truncated:
            break

    return obs_list, act_list, success, len(act_list)


def eval_success(
    env, kb: KeyboardEnv, agent: RLPDSAC, *, keys: list[str], trials: int,
    max_steps: int, rng: np.random.Generator,
) -> tuple[float, dict[str, int]]:
    """Deterministic-policy success rate over keys x trials (no data kept)."""
    per_key: dict[str, int] = {}
    n_success = 0
    n_total = 0
    for key in keys:
        k_succ = 0
        for _ in range(trials):
            _, _, succ, _ = collect_episode(
                env, kb, agent, IKExpert(), key=key, beta=0.0,
                max_steps=max_steps, deterministic_policy=True, rng=rng, label=False,
            )
            k_succ += int(succ)
            n_total += 1
        per_key[key] = k_succ
        n_success += k_succ
    return (n_success / max(1, n_total)), per_key


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    p.add_argument("--keys", default="central",
                   help="key group (central|phase_a|phase_b|all) or comma-separated keys")
    p.add_argument("--rounds", type=int, default=6,
                   help="DAgger rounds AFTER round 0 (round 0 = expert-driven BC)")
    p.add_argument("--rollouts-per-round", type=int, default=60,
                   help="episodes collected per round (spread across the key set)")
    p.add_argument("--rms-warmup-episodes", type=int, default=24,
                   help="expert episodes used to warm the RunningMeanStd before it "
                        "is frozen (data discarded). Keeps all rounds + eval on one "
                        "consistent normalizer.")
    p.add_argument("--bc-epochs", type=int, default=150,
                   help="BC epochs per DAgger round (rounds >= 1)")
    p.add_argument("--bc-epochs-round0", type=int, default=400,
                   help="BC epochs for round 0 (the initial fit)")
    p.add_argument("--beta-decay", type=float, default=0.0,
                   help="beta_i = beta_decay**i for i>=1 (prob the expert drives a "
                        "step). 0.0 = policy drives entirely from round 1 (standard "
                        "practical DAgger). 0.5 = geometric mixing.")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--keep-only-success", action="store_true",
                   help="aggregate only states from episodes where the expert/policy "
                        "reached success. Default off (vanilla DAgger keeps all "
                        "visited states — the recovery labels are the point).")
    p.add_argument("--eval-trials", type=int, default=5,
                   help="trials per key for the per-round success eval")
    p.add_argument("--eval-keys", default=None,
                   help="key group/list for eval (default = --keys)")
    p.add_argument("--reward-mode", choices=["dense", "pbrs_only", "xy_focus"],
                   default="xy_focus",
                   help="env reward mode. Irrelevant to BC/DAgger (no reward used) "
                        "but kept consistent with the eval env.")
    p.add_argument("--actor-hidden", type=str, default=None,
                   help="comma-separated actor hidden widths, e.g. '256,256,256'")
    p.add_argument("--frame-stack", type=int, default=3)
    p.add_argument("--keyboard-offset", type=str, default="-0.15,0.0",
                   help="keyboard placement 'x,y' (m). TRACKER §36.6: the default "
                        "(-0.15,0.0) leaves the left half outside the arm's "
                        "dexterous window; (-0.10,-0.10) reaches 71/87 vs 64.")
    p.add_argument("--key-aware-init", action="store_true",
                   help="pre-rotate the arm base toward the target key's column at "
                        "reset (TRACKER §36). Lets the IK expert reach the left-side "
                        "keys it otherwise misses by 5-22mm. Eval must use the same "
                        "flag (eval_orchestrator --key-aware-init).")
    p.add_argument("--save-dir", type=Path, default="checkpoints/approach_dagger")
    p.add_argument("--log-dir", type=Path, default="logs/approach_dagger")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    save_dir = Path(args.save_dir).resolve(); save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir).resolve(); log_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    keys = KEY_GROUPS.get(args.keys) or [
        k.strip() for k in args.keys.split(",") if k.strip() in AVAILABLE_KEYS
    ]
    if not keys:
        print(f"[dagger] no valid keys in --keys={args.keys!r}")
        return 1
    eval_keys = KEY_GROUPS.get(args.eval_keys or args.keys) or [
        k.strip() for k in (args.eval_keys or args.keys).split(",")
        if k.strip() in AVAILABLE_KEYS
    ]

    print(f"[dagger] keys ({len(keys)}): {keys}")
    print(f"[dagger] rounds={args.rounds} rollouts/round={args.rollouts_per_round} "
          f"max_steps={args.max_steps} beta_decay={args.beta_decay}")
    print(f"[dagger] save_dir={save_dir}")

    kb_off = tuple(float(v) for v in args.keyboard_offset.split(","))
    if len(kb_off) != 2:
        raise ValueError(f"--keyboard-offset must be 'x,y'; got {args.keyboard_offset!r}")
    print(f"[dagger] keyboard_offset={kb_off}")

    # ---- envs (random_key=False so we pin the target per episode) ----
    train_env = make_env(mode="approach", frame_stack=args.frame_stack,
                         domain_rand=False, random_key=False, seed=args.seed,
                         reward_mode=args.reward_mode, key_aware_init=args.key_aware_init,
                         keyboard_offset=kb_off)
    eval_env = make_env(mode="approach", frame_stack=args.frame_stack,
                        domain_rand=False, random_key=False, seed=args.seed + 1,
                        reward_mode=args.reward_mode, key_aware_init=args.key_aware_init,
                        keyboard_offset=kb_off)
    kb = find_inner(train_env, KeyboardEnv)
    eval_kb = find_inner(eval_env, KeyboardEnv)
    assert kb is not None and eval_kb is not None

    # ---- agent (for actor + RMS + save/load compat). Critic is unused. ----
    cfg_kwargs: dict[str, Any] = dict(seed=args.seed)
    if args.actor_hidden is not None:
        hidden = tuple(int(w) for w in args.actor_hidden.split(",") if w.strip())
        if len(hidden) < 2 or any(w < 8 for w in hidden):
            raise ValueError(f"--actor-hidden must be 2+ ints each >=8; got {hidden}")
        cfg_kwargs["actor_hidden"] = hidden
    cfg = RLPDConfig(**cfg_kwargs)
    agent = RLPDSAC(env=train_env, config=cfg, device=args.device, eval_env=eval_env)
    device = str(agent.device)
    expert = IKExpert()
    print(f"[dagger] actor={cfg.actor_hidden} obs={agent.actor_dim} act={agent.action_dim} "
          f"device={device}")

    # ---- normalizer: warm up on expert rollouts, then FREEZE + share with eval ----
    train_oa = find_inner(train_env, ObsAdapter)
    eval_oa = find_inner(eval_env, ObsAdapter)
    assert train_oa is not None and eval_oa is not None
    train_oa.training = True
    print(f"[dagger] warming RMS on {args.rms_warmup_episodes} expert episodes...")
    for i in range(args.rms_warmup_episodes):
        collect_episode(train_env, kb, agent, expert, key=keys[i % len(keys)],
                        beta=1.0, max_steps=args.max_steps,
                        deterministic_policy=True, rng=rng, label=False)
    train_oa.training = False                 # FREEZE — one normalizer everywhere
    eval_oa.rms = train_oa.rms                 # share the frozen accumulator
    eval_oa.training = False
    print(f"[dagger] RMS frozen (count={train_oa.rms.count:.0f}); shared train<->eval")

    # ---- TB ----
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=str(log_dir))
    except Exception:
        tb = None

    # ---- DAgger loop ----
    agg_obs: list[np.ndarray] = []
    agg_act: list[np.ndarray] = []
    best_success = -1.0

    for rnd in range(args.rounds + 1):
        beta = 1.0 if rnd == 0 else (args.beta_decay ** rnd)
        deterministic = (rnd == 0)            # round 0 = expert; later = policy rollouts
        t0 = time.time()

        n_succ = 0
        round_obs: list[np.ndarray] = []
        round_act: list[np.ndarray] = []
        for ep in range(args.rollouts_per_round):
            key = keys[ep % len(keys)]
            o, a, succ, _ = collect_episode(
                train_env, kb, agent, expert, key=key, beta=beta,
                max_steps=args.max_steps, deterministic_policy=deterministic, rng=rng,
            )
            n_succ += int(succ)
            if args.keep_only_success and not succ:
                continue
            round_obs.extend(o)
            round_act.extend(a)

        agg_obs.extend(round_obs)
        agg_act.extend(round_act)
        roll_succ = n_succ / max(1, args.rollouts_per_round)

        # BC-fit on the FULL aggregate (DAgger aggregates).
        X = np.asarray(agg_obs, dtype=np.float32)
        Y = np.asarray(agg_act, dtype=np.float32)
        epochs = args.bc_epochs_round0 if rnd == 0 else args.bc_epochs
        bc = BCPretrain(agent.actor, BCConfig(
            epochs=epochs, batch_size=256, lr=1e-3, weight_decay=1e-4, device=device))
        hist = bc.fit(X, Y)
        # Refresh actor optimizer reference (BCPretrain made its own).
        agent.actor_opt = torch.optim.AdamW(
            agent.actor.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay)

        # Per-round eval (deterministic policy success rate).
        ev_succ, per_key = eval_success(
            eval_env, eval_kb, agent, keys=eval_keys, trials=args.eval_trials,
            max_steps=300, rng=rng)

        dt = time.time() - t0
        print(f"[dagger] round {rnd}: beta={beta:.2f} rollout_succ={roll_succ:.2f} "
              f"agg={len(agg_obs)} bc_loss {hist['loss'][0]:.2f}->{hist['loss'][-1]:.2f} "
              f"EVAL_succ={ev_succ:.3f} ({dt:.0f}s)", flush=True)
        good = sorted([k for k, v in per_key.items() if v > 0])
        print(f"[dagger]   keys with success: {good}", flush=True)
        if tb is not None:
            tb.add_scalar("dagger/eval_success", ev_succ, rnd)
            tb.add_scalar("dagger/rollout_success", roll_succ, rnd)
            tb.add_scalar("dagger/bc_loss_final", hist["loss"][-1], rnd)
            tb.add_scalar("dagger/dataset_size", len(agg_obs), rnd)

        agent.save(str(save_dir / f"dagger_round_{rnd:02d}.pt"))
        agent.save(str(save_dir / "dagger_latest.pt"))
        if ev_succ > best_success:
            best_success = ev_succ
            agent.save(str(save_dir / "dagger_best.pt"))
            print(f"[dagger]   new best (eval_succ={ev_succ:.3f}) -> dagger_best.pt", flush=True)

    if tb is not None:
        tb.close()
    print(f"[dagger] done. best eval success={best_success:.3f}. "
          f"checkpoints in {save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
