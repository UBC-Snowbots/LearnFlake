# DAgger for the Approach skill

**Status:** active (TRACKER §35, v1.9). Supersedes the online-RL approach
(RLPD-SAC, TRACKER §30–§34) for the Approach skill.

## TL;DR

Across v1–v1.8, **pure behaviour cloning was the best result** (v8: 10/435) and
every attempt to refine it with online SAC/RLPD made it *worse*. The residual
failure of BC here is **covariate shift / compounding error**: the cloned
policy drifts into near-key fine-correction states the i.i.d. demo set never
covered, reaches ~48 mm, and wanders back out. The textbook cure is **DAgger**,
and it applies *unusually cleanly* here because we own a **deterministic expert
queryable at any state** — the M1 DLS-Jacobian IK controller.

## Why DAgger and not the obvious alternatives

| Why people reach for it | Why it doesn't fit (here) |
|---|---|
| Online SAC/RLPD | Exploration noise (cm-scale) ≫ success basin (4 mm). Every run (v3–v7) degraded BC. |
| Bigger actor (v10) | BC NLL hit the same floor with 3.6× params → not a capacity wall. |
| More / cleaner demos (v9, v11) | Same 3/435 ceiling → not a data-quantity or label-noise problem. |
| Diffusion / ACT / flow-matching | Their wins are multimodality + long-horizon open-loop. Our expert is deterministic, unimodal, low-dim, closed-loop. Overkill, and chunking *hurts* tight 4 mm feedback. |
| **DAgger** | **We have a free, deterministic expert. Covariate shift is exactly what DAgger fixes. ✅** |

Because the expert is free to query, we use **vanilla automated DAgger** and
skip the query-rationing variants (HG-/Ensemble-/Safe-/Thrifty-/RND-/Tube-DAgger)
— those only exist to budget expensive *human* labels.

## The key insight that reframed everything

The actor observation **already contains the exact goal vector**
(`target_offset_eef`, `obs_adapter.py:49`). The task was never an
observability/exploration problem — the §30–§32 reward-shaping detour was
chasing the wrong cause. With the goal in-hand and a queryable expert, the only
gap left is the distribution-mismatch one, which is precisely DAgger's domain.

## The algorithm (`scripts/train_dagger.py`)

```
warm a RunningMeanStd on N_warm expert episodes, then FREEZE it   # one normalizer everywhere
D = {}                                                            # aggregate dataset
for round in 0..R:
    beta = 1.0 if round == 0 else beta_decay**round              # prob the expert *drives* a step
    for ep in rollouts_per_round:
        reset env with a pinned target key
        for each step:
            a_expert = IKExpert.action(kb)        # label for the CURRENT state, always
            a_policy = actor(obs)
            record (obs, a_expert) into D         # <-- the relabelled corrective data
            step env with (a_expert if rand()<beta else a_policy)   # policy shapes the state distribution
    BC-fit actor on ALL of D (DAgger aggregates)
    eval deterministic-policy success on the key set; checkpoint
```

- **Round 0** (β=1) is expert-driven BC — the regenerated BC baseline.
- **Rounds ≥1** (β=0 by default) let the *policy* drive, so the expert labels
  the policy's own visited states — including the recovery states BC never saw.
- The expert labels **every** visited state by default. The corrective labels on
  bad states *are* the recovery signal; `--keep-only-success` can restrict to
  successful episodes if the imperfect (~44%) expert injects too much noise.
- One **frozen** RMS across all rounds + eval keeps obs normalization identical
  at train and deploy (the old pure-BC path had a gen-time-vs-eval RMS mismatch).

## Normalization (why a frozen RMS)

DAgger aggregates data across rounds. If the normalizer kept updating, stored
obs from earlier rounds would be normalized against a now-stale RMS. So we warm
the RMS on a batch of expert rollouts, **freeze** it, and reuse it for every
round and for eval (and save it in the checkpoint, which `eval_orchestrator`
applies). Self-consistent end to end.

## Usage

```bash
# inside rover_gpu
python3 -m rl_autonomy.scripts.train_dagger \
    --keys central \           # g h f j d k s l t y r u  (M1's strongest)
    --rounds 6 \
    --rollouts-per-round 60 \
    --beta-decay 0.0 \         # policy drives from round 1 (standard practical DAgger)
    --reward-mode xy_focus \
    --save-dir checkpoints/approach_v12_dagger \
    --log-dir logs/approach_v12

# all-87-key M4 success matrix on the best checkpoint
python3 -m rl_autonomy.scripts.eval_orchestrator \
    --approach checkpoints/approach_v12_dagger/dagger_best.pt
```

Key flags: `--keys {central|phase_a|phase_b|all|<comma-list>}`,
`--rollouts-per-round`, `--rounds`, `--bc-epochs[-round0]`, `--beta-decay`,
`--keep-only-success`, `--actor-hidden`, `--rms-warmup-episodes`.

## Files

- `algos/expert_ik.py` — `IKExpert` / `ik_step`: the DLS-Jacobian expert,
  queryable at any state. Single source of truth (also used by `gen_demos`).
- `scripts/train_dagger.py` — the DAgger trainer.
- `tests/test_expert_ik.py`, `tests/test_dagger.py` — unit coverage.

## What DAgger can and cannot do here

- **Can:** cure the covariate-shift drift → robustly match the expert on the
  keys it covers. This is the expected win (10/435 → 200/435, TRACKER §35–§35.8).
- **Cannot:** exceed the expert's ~44% per-attempt quality. If DAgger plateaus
  at expert level, the ceiling-raiser is **residual RL on the IK base with a
  tube-clipped delta** (keeps exploration inside the 4 mm basin) — TRACKER §35.2.
- **Cannot:** beat a **kinematic limit**. TRACKER §36: the ~23 left-side keys are
  not position-unreachable but **tilt-dead** — the arm can put the EEF over them
  (sub-mm XY) but physically cannot keep the actuator within 5° of vertical there
  (min achievable tilt 25–50°). No Approach policy (DAgger *or* residual RL) fixes
  that; it needs a **physical/spec change** (reposition the keyboard, or relax the
  tilt tolerance if the real solenoid strikes at an angle). So M4 (80/87) is a
  setup decision, not an algorithm one. The reachable-in-full-pose set is the
  centre/right ~60 keys.

## References

See TRACKER §35.5. Primary: Ross, Gordon & Bagnell, *DAgger*, AISTATS 2011;
Spencer et al., *Three Regimes of Covariate Shift*, RSS 2021.
