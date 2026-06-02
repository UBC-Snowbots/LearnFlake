# Residual RL on the IK base (the ceiling-raiser)

**Status:** active (TRACKER §38, v16). The follow-on to DAgger (`dagger.md`).

## Why

DAgger (`dagger.md`) took Approach to 268/435 (61.6%) but is **capped at the IK
expert's per-attempt quality (~62%)** — it can only *match* the expert, never beat
it. 84/87 keys are reachable but only 36 are at ≥80%; the gap is keys the IK gets
*almost* right (precision / workspace-edge) where it succeeds <80% of tries.

The only lever that can exceed the expert is **RL** — but naive online RL failed
for v1 (TRACKER §3x, runs v3–v7) because its **cm-scale exploration noise is wider
than the 4 mm success basin**, so any exploration step left the basin and the
policy never improved. The fix is **residual RL with a tube clip**.

## The idea

The agent learns a small **residual** added to the live IK action:

```
a_final[:6] = clip( a_ik[:6] + tube · residual[:6], -1, 1 )    # 6 joints
a_final[6]  = -1                                               # solenoid retracted (Approach)
```

- The **IK does the gross motion** (it already drives to within mm on most keys);
  the residual only has to learn the **local correction** the IK gets wrong.
- The **tube** (small, default 0.15) caps the residual magnitude, so the effective
  action — and therefore exploration — stays in a neighbourhood of the good IK
  trajectory: **RL refines inside the 4 mm basin instead of wandering out.** This
  is the structural fix for what killed v3–v7.
- The actor head is **zero-initialised**, so training starts at residual ≈ 0 ⇒ the
  pure IK baseline (~62%), and RL only improves from there. (Smoke run: episode
  return is +110 from step 1k, vs the deeply-negative starts of from-scratch SAC.)
- **Deployable:** the IK controller runs on the real arm (joint state + target are
  available), so this ships as "IK + learned correction" — a strong sim-to-real
  story; the tube also band-limits the learned part (ActionAdapter smooths it).

## Implementation

- `envs/residual_ik.py`
  - `ResidualIKWrapper(gym.Wrapper)` — wraps `KeyboardGymEnv`; the wrapped action
    is the residual; it adds the crisp live `IKExpert` action and clips.
  - `make_residual_env(...)` — builds
    `ObsAdapter → FrameStack → ActionAdapter(smooth+mask residual) → ResidualIKWrapper(add IK) → KeyboardGymEnv → KeyboardEnv`.
    ActionAdapter is **outside** the residual wrapper so the *residual* is
    band-limited while the IK stays crisp/closed-loop.
- `scripts/train_residual.py` — RLPDSAC on the residual env, zero-init head,
  `random_key=True` (trains all 87 keys) at `keyboard_offset=(-0.10,-0.10)`.
- `scripts/eval_orchestrator.py --residual --residual-tube T` — evaluates a
  residual checkpoint (rebuilds the residual env so the IK base is added).
- Tests: `tests/test_residual_ik.py` (4 — spaces, zero-residual==IK, tube-clip
  math + solenoid mask, tube>0).

## Usage

```bash
# train
python3 -m rl_autonomy.scripts.train_residual \
    --steps 200000 --tube 0.15 --keyboard-offset=-0.10,-0.10 \
    --save-dir checkpoints/approach_v16_residual --log-dir logs/approach_v16

# eval all 87 keys (tube + offset MUST match training; use the =form for the minus)
python3 -m rl_autonomy.scripts.eval_orchestrator \
    --approach checkpoints/approach_v16_residual/residual_final.pt \
    --strike   checkpoints/approach_v16_residual/residual_final.pt \
    --residual --residual-tube 0.15 --keyboard-offset=-0.10,-0.10 \
    --keys all --out-md results/m4_residual_v16.md
```

## Tuning notes

- **tube** is the key knob. Too small → can't correct the larger misses; too large
  → exploration leaks out of the 4 mm basin (the v3–v7 failure). 0.10–0.20 is the
  expected sweet spot; sweep if v16 underwhelms.
- It can only help **reachable** keys. The 3 hard corners (`scrlk del right`) and
  any residual M4 shortfall are a spec question (tilt tolerance / 2-pose), not an
  RL one (TRACKER §36.6, §37.1).

## References

TRACKER §35.2, §37.1, §38. Silver et al., *Residual Policy Learning* (2018);
Johannink et al., *Residual RL for Robot Control*; Xu et al., *Compliant Residual
DAgger* (NeurIPS 2025).
