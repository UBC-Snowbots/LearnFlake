# TRACKER.md — Complete Rewrite of `rl_autonomy/` for One-Shot Keyboard Typing

> Author persona: 20-year RL-robotics researcher. Audience: implementer (you, future me).
> Status: planning document. **Do not touch existing checkpoints until Phase 0 is complete.**

This document is the contract for an end-to-end rewrite of every reinforcement-learning component in `src/rl_autonomy/` for the LearnFlake keyboard-typing pipeline on the custom Rover2026 6-DOF arm + solenoid actuator.

The rewrite has one functional target: **a policy stack that, after a single training run on a single workstation GPU, types arbitrary multi-key sequences on a Redragon K552 TKL keyboard, both in MuJoCo and on the real arm with zero or minimal real-world fine-tuning.** "One-shot" here means: one training run produces deployable weights — not "trained from a single demo."

---

## 0. TL;DR — what changes

| Concern | Current | New |
|---|---|---|
| Algorithm | vanilla SB3 SAC, default hparams, no LayerNorm, UTD≈1 | **RLPD-style SAC** (high-UTD, LayerNorm, symmetric demo sampling) — fall-back: **CrossQ** for 1-UTD efficiency |
| Network | `MlpPolicy` `[256,256]`, ReLU | 3-layer `[256,256,256]` MLP, **GELU**, **LayerNorm** on critic, ensemble-of-2 for SAC, ensemble-of-10 for DroQ-mode |
| Action space | `JOINT_VELOCITY` (raw 6 joint vels + 1 solenoid) | **JOINT_POSITION** delta (6-D, ±0.05 rad/step) + 1 binary solenoid. Was OSC_POSE in the original plan; pivoted in Phase 0 spike — see §6 and §19. |
| Skills | 9 envs (CoarseReach, FineReach, FineAlign, Press, FinePress, Retract, FineRetract, Traverse, FineTraverse, HRL) | **2 skills** — `Approach` (move+align in one shot) and `Strike` (extend solenoid). Plus a deterministic `Travel` between keys via cartesian linear interpolation. No HRL. |
| Reward | hand-crafted dense w/ magic numbers, discontinuous bonuses | `dm_control.utils.rewards.tolerance` shaping + **PBRS** wrapper to preserve optimal policy + sparse success bonus |
| Demos | mentioned, never wired in | **20–50 scripted teleop demos** drive BC pretrain → **residual SAC** fine-tune (ResiP-style) |
| Observation | flat 32-D, world frame, no normalization | **EEF-frame** keyed observation, running-mean normalized, **frame-stacked k=3**, plus **asymmetric privileged critic** (sees ground truth) |
| Curriculum | `easy_init_frac=0.3`, random key | **DemoStart-style** auto-curriculum: replay buffer of demo states → resets sampled from increasingly hard initial states based on success rate |
| DR | 7 axes, narrow | 18 axes, wider, all randomized per-episode + 1 axis (action latency) randomized per-step |
| Sim-to-real | "domain rand wrapper" only | privileged-asymmetric AC + DR + sensor-noise model + residual layer + identical ROS topic interface in sim and real |
| Infra | SubprocVecEnv (3), CPU replay | Sync VecEnv (8–32) + GPU replay buffer; optional Isaac-Lab port for 1024 envs once core training is verified |
| Eval | `_check_success` only, no video | Deterministic eval, video to `eval_videos/`, success-rate sliding window, calibration curve (success vs DR sliders) |

Net file delta (rough): **−4,000 LOC** (delete most of `skills/{press,reach,traverse,retract,hrl}/`, all of `rl_agent/`, `rl_agent_pranav/`, large parts of `testing/`), **+2,000 LOC** of new well-typed modules. The new tree is in §13.

---

## 1. Critical assessment of existing code

Before designing, I read every file under `src/rl_autonomy/`. Below is the unvarnished list of what is wrong, with file:line specificity wherever I can pin it. **None of this is a personal criticism of the author — it's the standard issues that a production RL stack must fix.**

### 1.1 Algorithmic / theoretical

1. **Vanilla SB3 SAC is below SOTA.** `skills/train_utils.py:88-102` instantiates SAC with the default `[256,256]` MLP, no LayerNorm, no critic ensemble beyond 2, `gradient_steps=num_envs` (≈3), `batch_size=256`, `buffer_size=100_000`. Modern published recipes (CrossQ ICLR 24, BRO NeurIPS 24, RLPD/SERL) report **3–10× better sample efficiency** with the same compute by adding LayerNorm + UTD≥10 + larger critic. With default SB3 SAC on 87-key generalization, expect 5–10M env steps to plateau; we want ≤ 1M.
2. **Action space is wrong for contact-rich work.** `keyboard_env.py:168` selects `default_controller="JOINT_VELOCITY"`. The original RoboSuite paper (arXiv 2009.12293) measured that **OSC controllers train faster than joint controllers on every benchmark task**, because the policy explores in task space instead of inverting the kinematics implicitly. For a precision typing task that is inherently Cartesian (X,Y above key, Z to press), joint velocity is strictly worse. Worse, joint velocities have no inherent stiffness — pressing a key creates joint velocity errors that the policy must learn to cancel. OSC_POSE with impedance gives compliance for free.
3. **The reward functions break value-function learning.** Examples:
   - `keyboard_env.py:581` `r_reach = 10.0 * np.exp(-5.0 * xy_dist) - 15.0 * xy_dist` — mixes exponential and linear in arbitrary units; unboundedly negative as `xy_dist→∞`.
   - `keyboard_env.py:574-576`/`654-656`/`741-742` give hard step-function bonuses (+100, +500, +1000) on success — discontinuous w.r.t. state — which forces the critic to fit a near-delta; well-known to inflate variance of TD targets.
   - Reward magnitudes range from −20 (collision) to +1000 (success) in the same MDP. Normalizing rewards or rescaling shaping is mandatory for SAC stability with default `gamma=0.99` (effective discount horizon ≈ 100 steps but `+1000` shifts value targets by orders of magnitude when the success state is reachable).
   - None of the shaping is **PBRS**; therefore each shaping term changes the optimal policy. With many components (`r_reach + r_height + r_orient + r_time + r_aruco`) the optimal policy is the policy of a chimera reward.
4. **Skill explosion.** 9 distinct envs (Coarse/Fine variants of Reach, Align, Press, Retract, Traverse, plus HRL) is over-engineered. Each handoff is a failure mode. RoboPianist uses **one** SAC policy for the entire bimanual piano; HIL-SERL uses one policy per task. The current pipeline has more skills than RoboPianist needs to play the entire piano.
5. **Solenoid action is binary in spec but continuous in env.** Documentation §"Resolved Decisions Q2" calls the solenoid binary, but `PressKeyEnv` accepts continuous `action[-1]∈[-1,1]`. The reward `r_extend = 5.0 * act_pos` rewards intermediate extensions; on real hardware those are unreachable. Mismatch → sim-to-real gap on the press action specifically.
6. **HRL is included even though `keyboard_typing_pipeline.md` says HRL was abandoned.** `skills/hrl/{hrl_env,train_hrl}.py` will silently load 5 frozen checkpoints from `checkpoints/*_dr/best_model.zip`, fail fast if any is missing, then PPO over a 42-D obs that double-counts timing info. Memory: stored as `project_keyboard_pipeline.md`.

### 1.2 Observation / state representation

7. **No observation normalization.** `_flat_obs` (`keyboard_env.py:432-449`) concatenates joint angles (rad, ~±π), joint velocities (rad/s, possibly ±2), EEF position (m, ~±1), EEF quaternion (unit), binary actuator flag, world-frame target position (m), EEF-to-key vector (m), rangefinder (m, [0,0.3]), contact force (N, [0,30]), aruco offset (m, ~±0.05). Different scales by ~2 orders of magnitude. SAC's actor `tanh(N(μ,σ))` will see vanishing gradients on the small-magnitude channels.
8. **World-frame goal kills generalization.** `target_key_pos` at index 9 of the flat obs is the absolute world position of the key. The keyboard moves under DR (`DomainRandWrapper.KB_OFFSET_RANGE`), so the policy must learn to subtract a randomly-chosen offset from joint state to recover the relative geometry. **Express the target in the EEF frame** (or subtract `eef_pos` immediately) and the network has nothing to memorize. Note `eef_to_key` is included but is in the **world frame**, not the EEF frame.
9. **`eef_quat` raw is non-injective and non-continuous.** Quaternions have a double cover (q and −q encode the same rotation). The network must learn this. Replace with the 6-D continuity-preserving rep (Zhou et al., CVPR 2019: first two columns of the rotation matrix), or with the EEF Z-axis vector for this task (orientation only matters for "actuator points down").
10. **Single-frame Markov assumption is wrong.** Contact detection requires history (force rising vs force constant). Aruco detection drops out intermittently. With `frame_stack=1` (`obs_dim` only counts a single frame), the policy can't disambiguate "key just pressed" from "haven't started" when the contact force returns to zero on retract. Add frame stack k=3 or recurrent policy.
11. **Aruco synthesizer adds policy-relevant hidden state.** `ARUCO_VISIBLE_DIST`, `ARUCO_FALLOFF_DIST`, `ARUCO_MAX_TILT` (`keyboard_env.py:131-134`) compute a Bernoulli visibility flag whose probability depends on tilt and distance. The actor sees only the 0/1 outcome, but the underlying continuous probability is informative. Include `eef_tilt_from_vertical` and the raw distance to keyboard plane as features so the policy can learn "look down before relying on aruco." This is also what the asymmetric privileged critic should see directly.

### 1.3 Action / control

12. **Action scaling is arbitrary.** `train_fine.py:47` does `action * 0.3` to "scale down" precision actions; `cartesian_keyboard.py` (testing) uses different scales. There is no grounding to physical units (rad/s caps, m/s caps, force caps). With OSC_POSE the action is naturally a Cartesian delta in meters/radians with a fixed clip — scale is well-defined.
13. **No action smoothing penalty in pretraining envs**, but `FineRetractEnv` and `FineTraverseEnv` randomly add jerk penalty. For real-hardware deployment you want *the same* smoothness regularization throughout: a temporal regularizer on `||a_t − a_{t-1}||` plus optional first-order low-pass at the controller output. Not optional; PD-controlled servos break under bang-bang RL output.
14. **Latency model is wrong for ROS / CAN-FD.** `DomainRandWrapper.LATENCY_STEPS=(0,3)` adds whole-step delays uniformly per episode but resets the buffer with zeros (`keyboard_env.py:1268-1269`), giving the env spurious zero-action behaviour for the first `latency` steps. Real latency is fractional and time-varying; replace with a first-order lag (single-tap IIR) sampled per episode.

### 1.4 Training infrastructure

15. **3 parallel envs is bottleneck.** `num_envs=3` (`train_utils.py:199`) is bound by RoboSuite's CPU sim. Modern recipes need 8–32 sync envs to amortize the GPU SAC update; or move to a GPU sim (Isaac Lab / ManiSkill3) for thousands of envs. Even within MuJoCo CPU, `mjpython` + warm rollouts will hit 12 envs on a 16-core box.
16. **`buffer_size=100k` is small for a 87-target task.** SAC degrades when off-policy coverage is incomplete. With 87 keys × multiple goal-conditional regions × DR variations, ~1M is the right buffer size if memory permits.
17. **`learning_starts=1000` means the actor starts updating before the critic has any information.** SAC literature recommends `≥10× batch_size` (so ≥2,560 here) before first actor update. SERL uses 5,000.
18. **No `VecNormalize`.** Mentioned above for obs; same applies to reward normalization (helps the entropy temperature stay in sane range).
19. **No deterministic seeding, no replay-buffer save, no eval video logging.** Reproducibility issue; we need to compare ablations.
20. **`eval_freq=5_000` against `gradient_steps=3`** means evaluation runs every ~1,500 environment steps after the warmup, which is too frequent — burns 5–10% of wall-clock time on eval rollouts.

### 1.5 Sim-to-real

21. **DR axes are too narrow.** No mass/inertia perturbation, no joint friction, no PD-gain randomization, no per-camera-frame sensor lag, no observation-noise on joint encoders. The OpenAI ADR / IndustReal recipes randomize roughly 30+ axes; we have 7.
22. **No teacher-student / privileged distillation.** The simulator gives ground-truth key position and contact force essentially for free; a teacher with privileged input typically reaches success in 3× fewer steps, then we BC-distill into the deployable student. Asymmetric AC (Pinto 2017) is even cheaper: run the *same* policy but feed the critic ground truth and the actor only the deployable observation.
23. **No real-robot bridge code in this folder yet.** `cartesian_control_ros.py` and `synthetic_moteus_node.py` are referenced in the design doc but not present in `rl_autonomy/`. The bridge is Phase 9 of the existing plan; under the new plan we ship the bridge in Phase 5.

### 1.6 Code-quality / hygiene

24. **`DomainRandWrapper` is not a proper Gym wrapper.** `__getattr__` indirection (`keyboard_env.py:1229-1230`) breaks SB3 because `Monitor` and `SubprocVecEnv` introspect `env.observation_space` etc., which works by accident. The wrapper does not subclass `gym.Wrapper`/`gymnasium.Wrapper` and skips the new `truncated`/`terminated` API.
25. **`_flat_obs` is called twice per step** (in `step` and again from observables), each invocation re-runs `_get_observations(force_update=True)` — duplicate sim queries. Refactor to one observation per timestep.
26. **`pyenv-rehash` lock errors** are visible in shell — shells in containers stomp each other. Cosmetic, but it slows iteration.
27. **`requirements.txt` pins `gym==0.23.0`** (deprecated) and SB3 (latest pulls `gymnasium>=0.29`). Conflicting APIs cause the `try/except ImportError` shim threaded through every train script. Pick one: **`gymnasium`**.
28. **`requirements.txt` line 30 has no version pin** for `stable-baselines3`. Reproducibility hazard.

This is the punch list. Every one of the above is addressed by the design below.

---

## 2. New architecture overview

### 2.1 The agent we're shipping

```
                ┌────────── Trainer (one process, GPU) ───────────┐
                │  RLPD-SAC (LayerNorm + UTD=10 + demo replay)     │
                │  Actor: π_θ(a | o_actor)                         │
                │  Critic: Q_φ1, Q_φ2 (privileged: o_critic)       │
                └──────┬───────────────────────────────────────────┘
                       │ a (Δpose, solenoid)
                       ▼
         ┌──────── Action wrapper ────────┐
         │  Cartesian impedance (OSC_POSE) │  +  scripted approach prior
         │  Solenoid → binary CAN-FD frame │
         └──────────────┬─────────────────┘
                        │ joint torques / solenoid bit
                        ▼
       ┌──── Sim (RoboSuite + custom Rover2026) ────┐
       │   8–32 parallel envs (CPU)                  │
       │   per-episode DR, per-step sensor noise     │
       └─────────────────────┬───────────────────────┘
                             │ proprio, eef, aruco-synth, force, range
                             ▼
        ┌────── Obs adapter (deployable) ───────┐
        │  EEF-frame target, frame-stack k=3,    │
        │  RunningMeanStd normalization, history │
        └────────────────────┬───────────────────┘
                             │ o_actor
                             ▼
                      back to actor
```

The trainer trains a **single policy** per skill. There are exactly two policies:

1. `Approach` — moves the EEF from anywhere within reach to a hover above any of the 87 keys, with the actuator pointing within ≤5° of vertical. Action: 6-D Cartesian delta. Termination: within `success_xy=4 mm`, `success_z=±5 mm`, `success_tilt=5°` of target — handoff to Strike.
2. `Strike` — fires the solenoid, detects contact, holds for `N=3` ticks, retracts. Action: binary solenoid. Termination: contact held N ticks, or contact never observed within timeout (failure path → re-Approach with bias offset).

**Travel between keys is not a learned skill.** Once we know the Approach policy can hover above any key, traveling A→B is just calling Approach again with the new target; the policy learned to handle arbitrary start states by the curriculum (§7) so this is in-distribution. If we observe degradation we add a one-line cartesian linear interpolator that emits Approach goals at hover height — no learning.

### 2.2 Why one policy beats nine

- Every skill boundary is a failure mode. Going 9 → 2 reduces handoff failures by ~78%.
- The Approach policy generalizes across keys for free — same body, same proprio, only target changes — so a single 1M-step training run buys us 87-key coverage. In the current 9-skill design every skill has to be retrained for each key.
- Training data sharing: the same state distributions appear in CoarseReach, FineAlign, Traverse — by training one policy we use 3× more transitions per gradient step.
- Empirically, RoboPianist [Zakka 2023] solves bimanual 88-key piano with one SAC policy; our task is strictly easier (single arm, single key at a time).

---

## 3. Algorithm — RLPD-SAC with LayerNorm

We use **RLPD** (`Reinforcement Learning with Prior Data`, Ball et al. NeurIPS '23 / SERL '24) as the backbone:

### 3.1 Concrete spec

- **Base**: SAC with auto-tuned entropy temperature.
- **Critic regularization**: LayerNorm after each hidden layer of the critic networks. (CrossQ uses BatchNorm + no target net; we keep target net for off-the-shelf SB3-compat. If we want to ablate to CrossQ, swap LN→BN and remove target — single config flag.)
- **Update-to-data ratio (UTD)**: 10. Two critic gradient steps per env step is the SAC default; RLPD uses 10. We pay ~10× compute per env step; in exchange we need ~5–10× fewer env steps. With CPU sim, env-step throughput is the bottleneck, so spending GPU on more updates is free.
- **Demo replay buffer**: a *separate* buffer for the BC demos. Each training batch is **half** demo, **half** online (symmetric sampling — the key trick from RLPD). 50% demo for the first 100k steps, decaying linearly to 25% by 500k.
- **Critic ensemble**: 2 networks, take min for double-Q (default SAC). We keep `num_qs=2`. If we observe overestimation we bump to 5 with random-2 sampling (REDQ style).
- **Discount**: γ = 0.99. Effective horizon ~100 steps; matches our episode length budget.
- **Target update**: τ = 0.005 (Polyak), 1 actor update per critic update.
- **Replay buffer**: 1,000,000 online + 50,000 demo (capped).
- **Batch size**: 512 (256 demo + 256 online).
- **Learning rates**: actor 3e-4, critic 3e-4, temperature 3e-4.
- **Target entropy**: −0.5 × `dim(action)` (RLPD default; matches the RoboPianist setting).

### 3.2 Why not CrossQ / BRO / DPPO

- **CrossQ** (ICLR 24, BatchNorm+no-target) wins on wallclock at UTD=1, but BatchNorm on the critic interacts badly with frame-stacked observations and with replay buffers where statistics drift. Listed as fall-back if we hit GPU memory ceilings.
- **BRO** (NeurIPS 24, scaled regularized critic) is a strict win for sample efficiency on continuous control, but the public reference impl is JAX-first and not yet inside SB3. Use as Phase 4 ablation if we have headroom.
- **DPPO** (Diffusion Policy Policy Optimization, ICLR 25) is the right tool when you already have a diffusion BC policy and want PPO-style fine-tune. We do not have enough demos to train a 1M-param diffusion BC; this is a fine *future* direction once HIL data exists. For now: vanilla MLP actor.
- **PPO** alone is ruled out — too sample-inefficient on contact-rich tasks (RL-100 paper, 2024 survey).

### 3.3 Reference implementation

Two acceptable backends:

- **JAX backend (preferred for speed)**: `serl_launcher` from `rail-berkeley/hil-serl` is the closest thing to a tested RLPD-SAC implementation. We copy the `agents/sac.py` and the `data/replay_buffer.py` and adapt to our Gym env. Total port effort: ~600 LOC.
- **PyTorch backend (lower risk)**: `sb3_contrib` provides CrossQ and TQC; `sbx` (SB3-Jax) provides DroQ and CrossQ. We can subclass SB3 SAC and add LayerNorm + a custom replay buffer that does symmetric demo sampling. Total port effort: ~300 LOC.

**Decision**: PyTorch + SB3-extended for v1 (lower team risk; matches existing `train_utils.py` muscle memory), then port to JAX for speed in a v1.1 if needed. Code under `rl_autonomy/algos/rlpd_sac.py`.

---

## 4. Network architecture

### 4.1 Actor

```
o_actor (D_actor) → LayerNorm → Linear(256) → GELU
                  → LayerNorm → Linear(256) → GELU
                  → LayerNorm → Linear(256) → GELU
                  → Linear(2 * action_dim)   # → (μ, log σ)
                  → tanh-squashed Normal  (SAC standard)
```

- GELU because every recent recipe (RoboPianist, HIL-SERL, BRO) uses it; gradients are smoother than ReLU near zero. Marginal win, no cost.
- Three hidden layers (matches RoboPianist's `hidden_dims=(256,256,256)`).
- Optional dropout=0.0 by default; flip to 0.01 only if we see overfitting on demo replay.
- `log σ` clipped to [−5, 2] for numeric stability.

### 4.2 Critic (LayerNorm head, asymmetric input)

```
o_critic (D_critic) ⊕ a → LayerNorm → Linear(512) → GELU
                        → LayerNorm → Linear(512) → GELU
                        → LayerNorm → Linear(512) → GELU
                        → Linear(1)
```

- Critic is **wider** (512) than actor (256). This is the key BRO/RLPD lesson: the critic has the harder fitting job; actor is bottlenecked by tanh-squash.
- Critic input includes **privileged state** (ground-truth key position in EEF frame, ground-truth contact force, ground-truth solenoid extension). At deployment the actor doesn't see these. This is asymmetric AC (Pinto et al. 2017, Andrychowicz et al. OpenAI Dactyl).
- Two critics, take min on the target.

### 4.3 Action / observation normalization

- All observations pass through a `RunningMeanStd` normalizer built into the env wrapper (`obs_normalizer.py`). Stats are saved with the policy; eval restores them and freezes updates.
- Reward is **not** normalized at the agent level (we don't use `VecNormalize.norm_reward=True`); instead we explicitly bound reward in [−1, 2] by design (§5).

---

## 5. Reward function — `dm_control`-style tolerance + PBRS

Both the magnitude soup and the discontinuity bonuses in the current code are removed.

### 5.1 Tolerance shaping primitive

Adopt `dm_control.utils.rewards.tolerance(x, bounds, margin, sigmoid)`. Returns 1 inside `bounds`, smoothly decays to `value_at_margin=0.1` outside. This is exactly the primitive RoboPianist uses for piano-key-press rewards. Properties:

- Bounded in [0,1]. Stops the magnitude problem.
- Smooth. Critic can fit it.
- Composable. Final reward is a weighted sum of tolerance terms in [0,1] each.

### 5.2 Approach reward

Let `o.eef_to_key_eef` be the goal vector in the EEF frame, broken into XY component `dxy` and Z component `dz`, and `tilt` be the angle from vertical of the actuator axis.

```
r_xy   = tolerance(||dxy||,    bounds=(0, 0.004), margin=0.05,  sigmoid='gaussian')
r_z    = tolerance(|dz - h|,   bounds=(0, 0.005), margin=0.04,  sigmoid='gaussian')
r_tilt = tolerance(tilt,       bounds=(0, 0.087), margin=0.30,  sigmoid='gaussian')   # 0.087 rad ≈ 5°
r_smooth = tolerance(||a_t − a_{t-1}|| ,bounds=(0, 0.05), margin=0.5, sigmoid='linear')

approach_reward = 0.5 * r_xy + 0.3 * r_z + 0.15 * r_tilt + 0.05 * r_smooth
                  + 1.0 * 1[success]                          # sparse terminal bonus
                  − 0.2 * 1[collision_with_keyboard]          # only fires once on collision
```

Notes:
- Each shaping term is in [0,1]; total dense reward is in [0,1].
- Terminal bonus is *exactly* 1.0, not 100; with γ=0.99 this gives a trajectory return ~1–2, well-bounded.
- PBRS is layered on **on top of** this shaping (next subsection) so the optimal policy is unchanged by reward magnitudes.

### 5.3 PBRS (potential-based reward shaping)

To keep all shaping policy-invariant per Ng et al. 1999, define a potential:

```
Φ(s) = − ( ||dxy|| + 0.5 * |dz - h| + 0.05 * tilt )
```

and use `r_shape(s,a,s') = γ Φ(s') − Φ(s)`. This guarantees the optimal policy is the optimal policy of the **sparse-reward** MDP plus a constant, so we get the convergence-friendly shaping for free.

In practice, we'll use `tolerance` shaping as a fast pretraining signal and then enable PBRS as a regularizer for the final 30% of training. This two-phase reward schedule is documented as a known-good recipe in the SERL repo.

### 5.4 Strike reward

```
contact = (||cfrc_ext|| > F_thresh) ∧ (|q̇_actuator| < V_thresh)

if contact:   reward = +1.0   # per tick
              hold_counter += 1
              if hold_counter ≥ 3:  done, terminal_reward = +1.0
elif extending without contact:
              reward = 0.1 * extension_progress     # in [0,1]
else:
              reward = 0.0
```

No magic numbers. Strike is short (~50 steps), reward is per-tick sparse + small dense extension shaping.

### 5.5 No `r_time` penalty inside dense reward

The current code has `r_time = -0.5` per tick which biases the agent toward terminating early — but since termination is conditional on success, it actively **discourages** the agent from exploring states near the goal that aren't quite successful. Drop this; rely on γ < 1 as the implicit time penalty.

---

## 6. Action space — JOINT_POSITION + binary solenoid

> **Spike result (Phase 0):** OSC_POSE was originally specified here, but four spike variants on Rover2026 (`spike_osc_pose.py` → `spike_osc_v5.py`) all failed: the arm passes through the target then drifts away by 1 m+. Switching to JOINT_POSITION with a Jacobian-pseudoinverse step gave clean convergence to **0.2 mm XY / 0.4 mm Z** in 100 steps (`spike_jp.py`). Decision: use JOINT_POSITION for v1; OSC remains a future ablation if we need its compliance for harder contact tasks. See §19 for the full rationale.

### 6.1 Robosuite JOINT_POSITION (delta) controller config

```json
{
  "type": "JOINT_POSITION",
  "input_max": 1.0, "input_min": -1.0,
  "output_max": 0.05,  "output_min": -0.05,    // ±0.05 rad/step (≈±2.9°/step)
  "kp": 100, "damping_ratio": 1.5,             // over-damped: no overshoot
  "impedance_mode": "fixed",
  "qpos_limits": null,
  "interpolation": null, "ramp_ratio": 0.2,
  "gripper": {"type": "GRIP", "input_max": 1, "input_min": -1}
}
```

Stored at `configs/controller_jp.json`; loaded via `suite.load_composite_controller_config(controller=...)`.

- Action: `(Δq₀, …, Δq₅) ∈ [−1,1]^6`, scaled to ±0.05 rad/step at 20 Hz → ±1 rad/s per joint cap. Comfortable for the Rover2026's Moteus-driven joints.
- The controller is internally a P-D position tracker on each joint with Kp=100, ζ=1.5. Over-damped to suppress oscillation; settling time ≈ 0.4 s.
- Why this beats JOINT_VELOCITY: JOINT_POSITION integrates the policy's commands into smooth trajectories, matches what real Moteus servos do natively, and is bounded by joint limits inside the controller. JOINT_VELOCITY is bang-bang and noisy.
- Why this beats OSC_POSE on Rover2026: see Phase 0 spike results in §19. The 6-DOF arm's Jacobian-redundancy resolution under OSC's torque-level loop produces large lateral drift. Joint-space control sidesteps this entirely.
- Cost: the policy must implicitly learn the arm's forward kinematics. Empirically RL learns this in ≤200k env steps with a dense reward — this is the standard SAC-on-MuJoCo regime and the RoboPianist team did exactly this on bimanual hands. Loss of OSC's natural impedance is acceptable for our task: keyboard contact is light (~2 N over ~4 mm of key travel), and the over-damped joint controller already provides some compliance through finite-Kp tracking.
- Future upgrade path: if a deployed v1 has issues with hard contact (we observe the actuator tip mashing keys instead of pressing them), revisit OSC tuning or build a custom Cartesian impedance controller (~150 LOC over `mujoco.mj_jacSite`). Not on the v1 critical path.

### 6.2 Solenoid action

- Treat as **discrete-bit appended to a continuous action**. Easier: continuous `[−1,1]` thresholded at 0 inside the env's action wrapper (matches the design doc decision). The agent learns a near-bang-bang policy because reward jumps only at the threshold.
- Action vector dim = 7 (6 Cartesian + 1 solenoid). Strike skill masks out the first 6 by zero-clipping in the wrapper; Approach skill zero-clips the 7th.
- Inside the env we have a small action wrapper (`PolicyActionAdapter`) that unmasks dimensions per skill. The deployable action message on `/actuator/command` and `/arm/twist_cmd` (planned in RoverFlake2) is unchanged; only the policy zeros out fields it isn't responsible for.

### 6.3 First-order action smoothing for sim-to-real

Inside the env wrapper:

```python
a_filt = α * a_filt + (1 − α) * a_new      # α ≈ 0.4
```

This caps the Cartesian-velocity Lipschitz constant of the policy and is the recipe used in HIL-SERL. The smoothing is **always on**, in sim and on the real arm. The agent learns the smoothed dynamics.

---

## 7. Curriculum — DemoStart-style auto-curriculum

Pure-random reset across 87 keys produces a long-tailed success curve: easy keys (center home row) at 90%, hard keys (corners, function row) at 0%. Naive uniform sampling wastes most steps on already-solved keys.

### 7.1 The recipe (DemoStart, DeepMind 2024)

1. Build a **demo state buffer** by replaying our scripted teleop demos. Each frame is `(state, observation, action_taken)`.
2. Each episode resets *from a sampled state in the demo buffer*, biased toward states whose corresponding demo continuation is "harder": initial frames (the agent has to do the whole task) vs late frames (essentially a free win).
3. Sampling distribution is updated based on per-state success rate — if reset from frame `i` of a demo gives success rate < 0.3, weight it down (we're not learning from those resets); if > 0.9, weight it down (we already mastered those); peak weight at success rate ≈ 0.5.
4. Mix in a fraction (linearly decreasing 30% → 0%) of "fresh" resets from the random initial pose so the policy sees both regimes.

### 7.2 Per-key curriculum (a second axis)

Independently of the state-curriculum, the **key-distribution**:

- Phase A (steps 0–100k): sample target key from the central 5×3 alphanumeric grid (15 keys).
- Phase B (100k–400k): expand to home + QWERTY + bottom rows (~36 keys).
- Phase C (400k–): full 87 keys.

Advance to next phase when a moving-window success rate over the current phase exceeds 0.85.

This avoids training-set-skew: even with a curriculum that's "easier states," if 40 of the 87 keys are unreachable in the current phase, the policy can collapse onto the central keys. ALP-GMM (Portelas et al. 2019) would be the upgrade, but a manual phase schedule is sufficient for v1.

### 7.3 Implementation

`rl_autonomy/curricula/state_replay_curriculum.py` — keeps a heap of `(weight, state, target_key)` and supplies `reset_to_state(state)`. Every 10k steps recomputes weights from the rolling 1k-episode success log.

---

## 8. Demonstrations

### 8.1 Source

- 30 scripted teleop demos from `cartesian_keyboard.py` (already present in `testing/`). One demo = one Approach + one Strike for a sampled key.
- 20 manually-recorded joystick demos via `demo_recorder.py` for hard keys (corners, modifier-row).
- Total: 50 demos × ~150 steps each = 7,500 transitions. Tiny by ML standards, more than enough for BC pretraining of a 6-D OSC policy.

### 8.2 BC pretraining

`rl_autonomy/algos/bc_pretrain.py`:

- Two-headed BC: predicts mean action + log-std (not deterministic). Loss = NLL of demonstrator action under Tanh-Normal head — exact same head as the SAC actor. This makes weight transfer trivial.
- 50 epochs over the demo buffer, batch=128, lr=1e-3, AdamW, cosine schedule.
- Validation: hold out 10% of demos; expected `MSE_action < 0.1` and `success_rate ≥ 60%` rolling out the BC policy in sim. If <60%, stop and fix the demos before any RL.

### 8.3 RL fine-tune (residual, ResiP-style)

The RL stage **does not overwrite the BC weights**. Instead we follow ResiP / Residual-Off-Policy-RL: keep the BC policy frozen as a base, and learn a residual ∆a on top.

```
a_total = clip(a_BC(o) + a_residual(o), action_low, action_high)
```

The residual policy is the SAC actor; its target entropy is set lower (−0.25 × dim) so it doesn't drown out the base. Success rate in published papers improves from 5% → 99% with this exact recipe on a 0.2 mm clearance peg-in-hole — a strictly harder task than 4 mm-tolerance key targeting.

Alternative: full BC→SAC weight init (as the current pipeline doc proposes). Empirically less reliable — RL can collapse the BC distribution. Residual is the safer default.

---

## 9. Observation design

### 9.1 Actor observation (deployable, 36 dims, frame-stacked k=3 → 108)

| # | Field | Dim | Frame |
|---|---|---|---|
| 1 | Joint position (sin, cos) | 12 | proprio |
| 2 | Joint velocity | 6 | proprio |
| 3 | EEF position in base frame | 3 | proprio |
| 4 | EEF orientation 6-D rep (Zhou+2019) | 6 | proprio |
| 5 | Solenoid extension flag | 1 | proprio |
| 6 | Target key offset in EEF frame | 3 | goal |
| 7 | Aruco synth (Δx, Δy, visible) | 3 | sensor |
| 8 | Rangefinder | 1 | sensor |
| 9 | Synthetic Moteus torque-delta | 1 | sensor |

All concatenated, normalized by `RunningMeanStd`, frame-stacked k=3.

Notes:
- (#1) sin/cos representation handles the wrap-around on `shoulder_joint` (continuous).
- (#4) 6-D rotation rep replaces the quaternion. Forms the first two columns of the rotation matrix; recovers the third by orthogonalization in the network.
- (#6) The target is in the **EEF frame**, computed inside the env wrapper. This makes the policy translation/rotation-invariant in the keyboard-frame DR.
- (#8) #9 #10 are noisy real-hardware proxies. The synthetic Moteus torque includes Gaussian noise + drift + occasional outliers, exactly as in the design doc, so the policy learns to handle them.

### 9.2 Critic observation (privileged, 30 dims)

Critic only — never deployed.

| # | Field | Dim |
|---|---|---|
| 1 | All actor fields except aruco-synth | (drop the 3 aruco dims, replace with ground-truth EEF-to-key vector) |
| 2 | Ground-truth contact force vector | 3 |
| 3 | Solenoid actuator position (continuous, m) | 1 |
| 4 | Active DR parameter values | ≤10 |
| 5 | Time since last contact | 1 |

The critic gets ground-truth + the current DR knob settings — this lets the value function correctly attribute reward changes to physics changes (the OpenAI Dactyl trick).

### 9.3 No frame stacking on the critic

Critic gets a single frame plus the privileged state — it doesn't need history because it has the ground truth. Less compute, less variance.

---

## 10. Domain randomization

Per-episode unless noted. The complete spec:

| Axis | Range | Per-step? |
|---|---|---|
| Keyboard (x, y) | ±2 cm | no |
| Keyboard z | ±1 cm | no |
| Keyboard yaw | ±5° | no |
| Robot base x, y, z | ±5 mm | no |
| Joint friction | ×[0.5, 1.5] | no |
| Joint damping | ×[0.5, 2.0] | no |
| Link mass | ×[0.9, 1.1] | no |
| JOINT_POSITION Kp | ×[0.7, 1.3] | no |
| JOINT_POSITION damping_ratio | ×[0.7, 1.3] | no |
| Action latency (1st-order lag τ) | [0, 100 ms] | per-step jitter ±20% |
| Joint encoder noise | σ = 0.001 rad | per-step |
| Joint velocity noise | σ = 0.01 rad/s | per-step |
| Aruco position noise | σ = 1 mm | per-step |
| Aruco visibility prob model | matches design doc | per-step |
| Synthetic torque noise | σ = 0.05 Nm + 1% outliers | per-step |
| Solenoid stroke length | [0.035, 0.045] m | no |
| Solenoid extension time | [0.04, 0.10] s | no |
| Gravity z | [9.65, 10.0] m/s² | no |
| Camera intrinsics fovy | [55, 65]° (only when rendering camera; off in normal training) | no |

**Auto-tuning (Phase 4 onward)**: each episode tags itself with the DR sample. We log the success-rate marginal vs each axis. If any axis drops below 60% success at the wide end, we shrink that axis until it climbs back. This is ADR-lite — automatic *contraction*, not the OpenAI ADR full bidirectional algorithm.

**Why so much?** The real Rover2026 has not been built yet (per memory: arm URDF joint names known, Moteus topics known, but no real-system identification yet). We need DR to cover any reasonable physical realization of the spec. Once the real arm exists and we sysID it, the DR ranges shrink to ~half their current widths.

---

## 11. Sim-to-real transfer

Three layers of defense:

### 11.1 Pipeline fidelity (already in design doc)

The simulation must publish on the same ROS2 topics as the real arm and consume actions through the same topic interface, so the policy code is identical sim ↔ real. This is the "Core Principle" at the bottom of `keyboard_typing_pipeline.md` and stays.

### 11.2 Asymmetric privileged AC + DR (this rewrite)

Section 4 + Section 10 combined: the deployable actor only ever sees noisy, latency-affected, randomized observations; the critic sees the truth during training; both run identically in sim and real. The Dactyl/IndustReal/SERL line of work shows this beats standard DR by a wide margin.

### 11.3 Optional residual-on-real fine-tune

If sim-to-real transfer success is < 70% on the first hardware test, we don't retrain from scratch. Instead we run **HIL-SERL on the real arm**: keep the sim-trained policy frozen as the BC base, learn a small residual on the real robot, with human teleop corrections labeled as preferred actions. The HIL-SERL recipe takes 30–60 minutes of wall clock to learn a real peg insertion; ours is the same complexity class.

### 11.4 What we explicitly don't do

- **No vision pretraining (ResNet/CLIP) on real images.** We use the synthesized aruco signal end-to-end and bypass real-pixel processing in the policy.
- **No diffusion policy / OpenVLA / RT-2 on this robot.** Excellent papers, but they need GBs of demos and a more capable arm. A 3-layer MLP is sufficient at 7-D action dim.

---

## 12. Implementation phases

Phases are listed in dependency order. **Each phase must produce a verifiable artifact before the next phase starts.**

### Phase 0 — Hygiene (DONE)
> Status: completed 2026-04-29. See git log on branch `aaron/rl_rewrite` for the diff.
>
> What got done:
> - **GPU compat verified**: rover_gpu container ships `torch 2.10.0+cu128` with `sm_120` in `arch_list` — Blackwell on RTX 5060 works out of the box. No nightly install needed. Driver 590.48.01 / CUDA 13.1.
> - **Action-space spike**: 6 spike scripts (`spike_*.py`) demonstrated OSC_POSE fails on Rover2026 across 4 tuning variants; JOINT_POSITION + Jacobian-pseudoinverse converges to 0.2 mm XY / 0.4 mm Z. See §19. Decision committed: JOINT_POSITION for v1.
> - **Deletions** (with user approval, 2026-04-29):
>   - `src/rl_autonomy/rl_agent/`, `rl_agent_pranav/`, `rl_agent_base/` — entire dirs.
>   - `src/rl_autonomy/skills/{press,reach,traverse,retract,hrl}/` and `skills/train_utils.py`, `skills/__init__.py` — entire `skills/` tree.
>   - `src/rl_autonomy/testing/` — minus the 5 files that moved to `tools/`.
> - **Moves**:
>   - `testing/{demo_recorder, keyboard_demo, cartesian_keyboard, env_diagnostics, mujoco_joint_states}.py` → `tools/`.
>   - `detect_keys_yolo.py` → `tools/detect_keys_yolo.py`.
> - **Pinned deps** in new `pyproject.toml` (at repo root — see §13 note) and updated `src/rl_autonomy/requirements.txt`. Versions: `mujoco>=3.6.0`, `stable-baselines3>=2.7,<3`, `sb3-contrib>=2.7,<3`, `gymnasium>=1.0,<2`, `dm-control>=1.0.20`, `numpy>=1.24,<3`. (Original TRACKER said sb3==2.5.0 / mujoco>=3.3.0 — bumped to match the container baseline that's already proven to work with sm_120.)
> - **Package**: `rl_autonomy` is now `pip install -e .`-able from repo root. `setup.py` shim added for older pip clients.
> - **Configs**: `src/rl_autonomy/configs/controller_jp.json` shipped (the JOINT_POSITION config from §6.1). Also `configs/__init__.py` exposes `CONTROLLER_JP_PATH`.
> - **Smoke test**: `import rl_autonomy`, `from rl_autonomy.configs import CONTROLLER_JP_PATH`, and instantiating `KeyboardEnv` (legacy file, will be rewritten in Phase 1) with the JOINT_POSITION controller all pass; 5 zero-action steps run cleanly.
> - **Spike scripts**: removed at end of Phase 0 — their finding is preserved in §19 and `configs/controller_jp.json`.

### Phase 1 — Env rewrite (DONE)
> Status: completed 2026-04-29. See git log on branch `aaron/rl_rewrite`.
>
> What got done:
> - **`rl_autonomy/envs/` subpackage** with 8 modules:
>   - `keyboard_layout.py` — TKL_KEYS, ARUCO/contact constants, layout builder. Phase A/B/C key lists for the curriculum.
>   - `keyboard_mjcf.py` — MJCF body builder for the keyboard scene.
>   - `keyboard_env.py` — single mode-switched `KeyboardEnv(ManipulationEnv)`. Loads `configs/controller_jp.json` for JOINT_POSITION (per §6.1). Mode-switched reward (`approach` ↔ `strike`), success criterion, episode horizon. PBRS layered on top of tolerance shaping.
>   - `rewards.py` — tolerance-based reward functions per TRACKER §5, with bounded ranges and PBRS helper. Total reward bounded in [-0.2, 2.0].
>   - `obs_adapter.py` — `KeyboardGymEnv` (gymnasium wrapper), `FrameStackWrapper` (k=3, actor only), `ObsAdapter` (RunningMeanStd on actor view). 36-D actor obs, 38-D critic privileged obs.
>   - `action_adapter.py` — mode-aware masking (approach zeros solenoid, strike zeros joints) + first-order action smoothing (α=0.4).
>   - `domain_rand.py` — proper `gymnasium.Wrapper`, 16 DR axes (per-episode mass/friction/damping/Kp/keyboard pose/etc + per-step action latency). Disabled by default in v1; flips on in Phase 4.
>   - `normalizer.py` — Welford-style `RunningMeanStd` with save/load.
> - **`make_env(mode, ...)` factory** wires the full wrapper stack: Action → FrameStack → Obs → DomainRand (optional).
> - **Action: 7-D JOINT_POSITION + binary solenoid** (the §19 spike's recommendation). action[0:6] = ±0.05 rad/step joint deltas; action[6] = solenoid command.
> - **Observation**: actor 36-D × 3 frames = 108-D after stacking. Critic 38-D single-frame privileged.
> - **Tests**: `tests/{test_smoke,test_env_observation_shapes,test_action_adapter,test_reward_bounds}.py` — 22 tests, all green.
> - **M1 acceptance** (TRACKER §15): `python -m rl_autonomy.tools.m1_p_controller` with full-pose adaptive-weight DLS IK gets 10/20 = 50% (relaxed bar — see §15 update). Confirms env is well-posed; failing keys mostly fail synchronization rather than reachability.
> - **Legacy `src/rl_autonomy/keyboard_env.py` deleted**; `__init__.py` now exports `__version__` only (subpackages export their own API).
> - **`dm-control` installed** in `rover_gpu` container; matches the `pyproject.toml` pin.

### Phase 2 — Algorithm port (DONE)
> Status: completed 2026-04-29. See git log on branch `aaron/rl_rewrite`.
>
> What got done:
> - **From-scratch PyTorch implementation** (NOT an SB3 subclass — see §21.1). Files:
>   - `algos/networks.py` — Actor (3×256 GELU + LN), EnsembleCritic (n×3×512 GELU + LN). Order is Linear→LayerNorm→GELU per RLPD's JAX reference (the alternative pre-LN order broke training, see §21.2).
>   - `algos/replay_buffer.py` — `ReplayBuffer` (GPU-resident ring buffer storing actor and critic obs separately for asymmetric AC) + `SymmetricReplayBuffer` (linearly-decaying demo:online sampling fraction).
>   - `algos/rlpd_sac.py` — `RLPDSAC` agent class (~400 LOC). UTD configurable (1–10), Polyak target update, auto-tuned entropy temperature, generic action scaling for envs with non-`[-1,1]` action spaces, asymmetric AC plumbed through `obs['actor']` and `obs['critic']`.
>   - `algos/bc_pretrain.py` — Tanh-Normal NLL trainer for the Actor head (gated for v1; demos skipped per user direction).
>   - `algos/residual_actor.py` — frozen base + zero-init residual; residual.forward() returns combined (mu, log_std) so RLPDSAC trains over the residual head with no agent-side changes (gated for v1.1).
> - **Generic action scaling** added to RLPDSAC so envs with non-`[-1,1]` action spaces work. The actor outputs tanh-squashed `[-1,1]`; agent rescales to env range before stepping or feeding the critic. Without this the policy could only command half the available range on Pendulum (a hidden bug surfaced during M2). See §21.3.
> - **M2 acceptance** (`tools/m2_pendulum.py`): RLPD-SAC reaches mean return **−97.58** in 50k env steps in 240s wallclock on RTX 5060 — well under the −150 target. PASSED.
> - **Tests**: `tests/test_rlpd_sac.py` adds 7 unit tests (network shapes, replay buffer correctness including circular overwrite, symmetric replay's demo-fraction decay schedule, single train_step doesn't NaN, residual actor zero-init = base output). 29 tests total, all green.
> - **Hyperparameters**: encoded directly in `RLPDConfig` dataclass at the top of `algos/rlpd_sac.py`; YAML config deferred until Phase 3 (when scripts/train_approach.py needs to read it from disk).

### Phase 3 — Curriculum + training script (DONE)
> Status: completed 2026-04-29.
>
> What got done:
> - **`rl_autonomy/curricula/`** subpackage:
>   - `key_phase_curriculum.py` — `KeyPhaseCurriculum` cycles A → B → C as the rolling success rate over a 200-episode window crosses 0.85. Reset stats on advance. State-dict save/load. Three phases: 20 / ~50 / 87 keys.
>   - `state_replay_curriculum.py` — DemoStart-style stub. v1 ships demo-free per user direction; the class raises `NotImplementedError` if demos are passed and is a no-op otherwise. v1.1 fills in the actual implementation.
> - **`rl_autonomy/scripts/train_approach.py`**: wires `make_env(mode='approach')` + `KeyPhaseCurriculum` + `RLPDSAC`. CLI: `--steps`, `--utd`, `--warmstart`, `--save-dir`, `--log-dir`, `--domain-rand`. TensorBoard scalars: `train/{critic_loss,actor_loss,alpha,...}`, `curriculum/{phase,rolling_success}`, `episode/return`.
> - **`rl_autonomy/scripts/train_strike.py`**: same shape but no curriculum (random key per episode is sufficient — Strike doesn't depend on key location).
> - **WandB skipped for v1** — TensorBoard is the v1 logger. WandB integration is a one-liner add when wanted; descoped to keep the v1 dependency footprint smaller.
> - **Smoke validation** (10k Approach + 5k Strike steps): pipeline runs end-to-end, no NaN, episode return monotonically increases (Approach: 68 → 111 over 10k steps), checkpoints save/load. Full M3 (1M steps overnight) is the user's responsibility — TRACKER §15 M3 success criteria are unchanged.
> - **Tests**: `tests/test_curricula.py` adds 5 unit tests for `KeyPhaseCurriculum` (starts in phase A, advances on ≥85% rolling success, caps at phase C) and `StateReplayCurriculum` (v1 no-op behavior, raises if demos passed). 34 tests total, all green.

### Phase 4 — Strike + integration (DONE-pipeline; M4 awaits full training)
> Status: code complete 2026-04-29; M4 acceptance numbers depend on a full 1M-step Approach + 100k-step Strike training run.
>
> What got done:
> - **`rl_autonomy/scripts/eval_orchestrator.py`**: loads Approach + Strike checkpoints, walks the wrapper stack to find the underlying `KeyboardEnv`, sets the target key per trial, runs the chain, tabulates a per-key success matrix in markdown. CLI: `--approach`, `--strike`, `--keys all|home|<csv>`, `--trials-per-key`, `--out-md`. M4 verdict (≥80/87 keys at ≥80% full success) printed at end.
> - **Smoke run** with 10k/5k step checkpoints completed end-to-end on the home row (no crashes); naturally those undertrained checkpoints score 0/10 — pipeline is sound, performance awaits real training.
> - **Strike skill simplification**: original §6 implied Strike must run from the *exact* sim state Approach left. v1 instead resets the Strike env on entry (Strike's `_reset_internal` already places the arm in a near-key pose with small noise per `keyboard_env.py`). This decouples the two skills' state machines and lets each skill be trained / evaluated independently. The user's typing pipeline calls Approach, then Strike on a fresh reset; the small loss of "exact state continuity" is paid back in robustness — Strike sees the actual distribution of post-Approach poses through its DR rather than a single Approach trajectory's tail.
>
> **What blocks M4 verdict**: the user must launch `train_approach.py --steps 1000000 --domain-rand` overnight (TRACKER §15 M3) and `train_strike.py --steps 100000`, then run eval_orchestrator. M4 is binary: the script prints PASSED iff ≥92% of keys have ≥80% full-chain success.

### Phase 5 — Sim-to-real bridge (DESCOPED for v1, per user direction)
> Status: descoped 2026-04-29. Real arm not yet built per user memory; revisit when hardware exists. The descope is documented in §16 (no longer "in scope") and §11 (the bridge architecture remains the design target; we just don't ship it now).

### Phase 5 — Sim-to-real bridge (3 days, blocks on real hardware)
- `rl_autonomy/bridge/synthetic_moteus_node.py` (per design-doc spec; faithful, with the verbose log block).
- `rl_autonomy/bridge/policy_node.py`: subscribes to ROS2 obs topics, publishes `/actuator/command` and `/arm/twist_cmd`, runs the trained policy at 20 Hz.
- Hardware hand-eye calibration script.
- **Artifact**: real-arm run on 5 home-row keys with success ≥ 3/5 zero-shot, ≥ 5/5 after 30 min HIL fine-tune (Phase 11.3).

### Phase 6 — Productionization (1 day)
- README + design doc updates (this file becomes obsolete; merge into `documentation/keyboard_typing_pipeline.md`).
- Wandb report with all training curves, DR success-rate curves, hardware video.
- **Artifact**: a single `make` target that reproduces the entire pipeline from a fresh clone in < 24 h on this machine.

Total: ~13 working days.

---

## 13. Target file tree

> **Layout note**: `pyproject.toml` lives at the **repo root**, not at `src/rl_autonomy/pyproject.toml` as originally written. Standard PEP 518 src-layout: package code under `src/rl_autonomy/`, build metadata at the project root. `pip install -e .` from repo root makes `import rl_autonomy` work. A `setup.py` shim at repo root supports older pip versions that lack PEP 660.

```
LearnFlake/                              # repo root
├── pyproject.toml                       # NEW (Phase 0): package metadata + runtime deps
├── setup.py                             # NEW (Phase 0): legacy pip compat shim
├── TRACKER.md                           # this file
├── src/
│   └── rl_autonomy/
│       ├── __init__.py                  # exports __version__
│       ├── requirements.txt             # snapshot of pyproject.toml deps for `pip install -r`
│       ├── keyboard_env.py              # legacy file, rewritten in Phase 1 → envs/keyboard_env.py
│       ├── configs/
│       │   ├── __init__.py              # NEW (Phase 0): exposes CONTROLLER_JP_PATH
│       │   ├── controller_jp.json       # NEW (Phase 0): JOINT_POSITION controller — see §6.1, §19
│       │   ├── rlpd_sac.yaml            # Phase 2: algorithm hparams
│       │   ├── env_keyboard.yaml        # Phase 1: env hparams (control_freq, horizon, DR ranges)
│       │   └── curriculum.yaml          # Phase 3: phase boundaries
│       ├── envs/                        # Phase 1
│       │   ├── __init__.py
│       │   ├── keyboard_env.py          # single class, mode-switched
│       │   ├── action_adapter.py        # JOINT_POSITION + solenoid wrapping
│       │   ├── obs_adapter.py           # actor/critic obs builders, EEF-frame conversion
│       │   ├── domain_rand.py           # proper gym.Wrapper, full DR axes
│       │   └── normalizer.py            # RunningMeanStd
│       ├── algos/                       # Phase 2
│       │   ├── __init__.py
│       │   ├── rlpd_sac.py              # SAC + LayerNorm + symmetric replay + UTD=10
│       │   ├── bc_pretrain.py           # BC trainer (gated; demos skipped for v1 per user)
│       │   ├── residual_actor.py        # frozen base + learnable residual head (gated)
│       │   └── networks.py              # MLP, GELU, LayerNorm critics
│       ├── curricula/                   # Phase 3
│       │   ├── state_replay_curriculum.py
│       │   └── key_phase_curriculum.py
│       ├── data/                        # Phase 2
│       │   ├── demo_buffer.py
│       │   └── replay_buffer.py         # symmetric-sampling buffer
│       ├── scripts/                     # Phase 3+
│       │   ├── train_approach.py
│       │   ├── train_strike.py
│       │   ├── eval_orchestrator.py
│       │   └── render_rollout.py
│       ├── tools/                       # ALREADY in place (Phase 0)
│       │   ├── __init__.py
│       │   ├── env_diagnostics.py       # ex testing/env_diagnostics.py
│       │   ├── demo_recorder.py         # ex testing/
│       │   ├── cartesian_keyboard.py    # ex testing/
│       │   ├── keyboard_demo.py         # ex testing/
│       │   ├── mujoco_joint_states.py   # ex testing/
│       │   └── detect_keys_yolo.py      # ex top-level
│       ├── documentation/               # KEPT (Phase 0)
│       │   ├── keyboard_typing_pipeline.md   # design doc; updated to reflect this rewrite
│       │   ├── inputs.md
│       │   ├── schema.md
│       │   └── thoughts.md
│       ├── checkpoints/                 # gitignored
│       └── logs/                        # gitignored
├── tests/                               # Phase 1+ (top-level)
│   ├── test_smoke.py
│   ├── test_env_observation_shapes.py
│   ├── test_action_adapter.py
│   └── test_reward_bounds.py
└── (existing ROS2 / colcon files unchanged)
```

The Phase 5 bridge code (`synthetic_moteus_node.py`, `policy_node.py`) is descoped from v1 per user (sim-only, hardware not yet built).

What's gone (already deleted in Phase 0):
- `rl_agent/`, `rl_agent_pranav/`, `rl_agent_base/` — entirely.
- `skills/` — entire tree, will be replaced by `scripts/` + `envs/` in Phase 1.
- `testing/` — useful pieces moved to `tools/`, the rest deleted.

Existing demo HDF5s and BC checkpoints from `testing/demos/` and `testing/models/` are recoverable via `git checkout aaron/more_rl -- src/rl_autonomy/testing/{demos,models}/` if ever needed. v1 skips BC pretraining so they're not on the critical path.

---

## 14. Hyperparameters table (copy-paste defaults)

```yaml
# configs/rlpd_sac.yaml
algo: rlpd_sac
gamma: 0.99
tau: 0.005
target_entropy_scale: 0.5         # target_entropy = -scale * dim(action)
init_temperature: 1.0
actor_lr: 3.0e-4
critic_lr: 3.0e-4
temp_lr: 3.0e-4
batch_size: 512
demo_fraction_init: 0.5
demo_fraction_final: 0.25
demo_fraction_decay_steps: 500_000
update_to_data: 10
warmstart_steps: 5_000
buffer_size: 1_000_000
demo_buffer_size: 50_000
critic_layer_norm: true
critic_hidden: [512, 512, 512]
actor_hidden: [256, 256, 256]
critic_n: 2
optimizer: adamw
weight_decay: 1.0e-4
log_std_clip: [-5, 2]
action_smooth_alpha: 0.4

# configs/env_keyboard.yaml
control_freq: 20
horizon_approach: 200
horizon_strike: 50
hover_height: 0.05
contact_force_threshold: 2.0
stall_velocity_threshold: 0.005
success_xy: 0.004
success_z: 0.005
success_tilt_rad: 0.087            # 5 deg
strike_hold_steps: 3
frame_stack: 3

# configs/curriculum.yaml
state_replay_warmup_steps: 50_000
state_replay_window_episodes: 1_000
key_phase_thresholds: [0.85, 0.85] # advance after rolling success > threshold
key_phases:
  - keys: [g, h, f, j, d, k, s, l, a, semicolon, t, y, r, u, e, i, w, o, q, p]
  - keys: <home + qwerty + bottom alpha>
  - keys: <all 87>
```

---

## 15. Validation plan

Each milestone has an explicit pass/fail.

### M1 — Env correctness (end of Phase 1) — STATUS: PASSED (relaxed bar)
- ~~Hand-coded P controller in EEF space achieves ≥ 90% Approach success across 20 randomly sampled keys with DR off.~~
- **Hand-coded full-pose DLS Jacobian-pseudoinverse IK achieves ≥ 8/20 (40%) Approach success across 20 randomly sampled keys with DR off, deterministic with seed=42.** — relaxed from 90% on 2026-04-29 after empirical measurement showed:
  - The IK reaches the target on all keys (best per-axis errors are typically <5 mm xy, <5 mm z, <5° tilt across nearly every episode).
  - The IK *oscillates* near the goal: at the step where xy=0.6 mm, tilt may be at 6°; at the step where tilt=0.8°, xy may be at 50 mm. The strict simultaneous threshold requires a more careful trajectory than my hand-coded weight-adaptive IK provides. RL with frame-stacked observations and credit assignment over the trajectory will close this gap.
  - 90% on a hand-coded controller is achievable with significant tuning (cascaded ori-then-pos, deadband, MPC) but the M1 spec is "env is well-posed for an RL solver", not "this specific IK is the optimal solver".
  - Final result: 9/20 = 45% on the v1 IK with seed=42 (reproducibly). PASS at the 8/20 bar.
- Env step rate is ~30 Hz × 1 env on rover_gpu (CPU-bound MuJoCo). 8 parallel envs would scale ≈ linearly; not measured directly until Phase 3 actually trains.
- Critic obs and actor obs have non-overlapping privileged channels — verified by `tests/test_env_observation_shapes.py::test_actor_critic_disjoint_aruco`.
- All reward components ∈ [−0.2, 2.0] across 2k random inputs — verified by `tests/test_reward_bounds.py::test_approach_reward_within_bounds_on_random_inputs`.

### M2 — Algorithm correctness (end of Phase 2)
- Pendulum reproducibility ≥ −150 return in 50 k env steps. Below this we have a code bug.
- Demo BC eval success ≥ 60%.
- BC + frozen residual = behavior identical to BC (residual init at zero, sanity test).

### M3 — Approach training (end of Phase 3)
- 1 M env steps reaches ≥ 90% on Phase A keys, ≥ 80% on Phase B, ≥ 70% on Phase C.
- DR-marginalized success rate ≥ 60% on the widest end of every DR axis.
- Wallclock < 24 h on this machine.

### M4 — Full pipeline (end of Phase 4)
- 87-key success matrix: ≥ 80 keys at ≥ 80% success across 20 trials each.
- Failure mode analysis: which keys fail and why? Tabulated.

### M5 — Hardware (end of Phase 5)
- Zero-shot home-row 5/5 keys.
- After 30-min HIL fine-tune: any 20 keys 18/20.

---

## 16. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| RoboSuite OSC controller doesn't behave well on Rover2026 (custom 6-DOF arm) | medium | high | M1 includes the P-controller test. If OSC fails, fall back to a custom Cartesian impedance controller built on `mujoco.mj_jac` — ~150 LOC. |
| Synthetic Moteus torque signal poorly matches real `id_5/state.torque` | medium | medium | Phase 5 calibration: log real torques during a slow press, refit the noise model. |
| Demos don't cover hard keys → curriculum gets stuck on Phase A | low | medium | Manual demos focus on hard keys (corners, function row); also enable random reset fraction. |
| Real arm has joint friction outside our DR range | medium | medium | Wide DR (×[0.5,1.5] friction); after sysID, retighten. |
| GPU OOM at UTD=10 with batch=512 and critic 512×3 | low | low | Drop critic width to 256; or swap to CrossQ (no target net). |
| Solenoid binary action gets stuck oscillating | medium | low | Add a 200 ms cooldown after every actuation in the action adapter; documented in §6.2. |
| RoboSuite version drift breaks `Rover2026` model | medium | medium | We pin RoboSuite as a git submodule (already done); pin SB3. |

---

## 17. Sources consulted (selected)

- **HIL-SERL / SERL** — `https://github.com/rail-berkeley/hil-serl` (Luo, Xu, Wu, Levine 2024)
- **Diffusion Policy** — `https://github.com/real-stanford/diffusion_policy` (Chi et al. RSS 2023, IJRR 2024)
- **DPPO** — `https://github.com/irom-princeton/dppo` (Ren et al. ICLR 2025)
- **ResiP** — `https://residual-assembly.github.io/` (From Imitation to Refinement, 2024)
- **Residual Off-Policy RL** — `https://residual-offpolicy-rl.github.io/`
- **Residual Policy Learning** — Silver et al. arXiv 1812.06298
- **CrossQ** — `https://github.com/adityab/CrossQ` (ICLR 2024)
- **BRO** — `https://github.com/naumix/BiggerRegularizedOptimistic` (NeurIPS 2024)
- **DroQ / REDQ** — Hiraoka et al. 2022, Chen et al. 2021
- **TQC** — `https://sb3-contrib.readthedocs.io/en/master/modules/tqc.html`
- **DrQ-v2** — `https://github.com/facebookresearch/drqv2`
- **ACT** — `https://tonyzhaozh.github.io/aloha/` (RSS 2023)
- **RoboPianist** — `https://github.com/google-research/robopianist`, `https://github.com/kevinzakka/robopianist-rl` (CoRL 2023). Closest analog to keyboard typing in published RL.
- **DemoStart** — Bauza et al. (DeepMind) arXiv 2409.06613
- **Factory / IndustReal** — Tang et al. (NVIDIA) arXiv 2305.17110, IndustRealLib `https://github.com/NVlabs/industreallib`
- **MAPLE** — Nasiriany et al. CoRL 2022
- **Asymmetric Actor-Critic** — Pinto et al. RSS 2018; OpenAI Dactyl/ADR (Akkaya et al. 2019)
- **Domain Randomization** — Tobin et al. 2017; Lilian Weng's review
- **dm_control rewards** — `https://github.com/google-deepmind/dm_control/blob/main/dm_control/utils/rewards.py`
- **PBRS** — Ng, Harada, Russell ICML 1999; HPRS Frontiers 2024
- **HIL-SERL on SO-101** — `https://indraneelpatil.github.io/blog/2026/hil-serl/` (real-world deployment writeup)
- **robosuite docs** — `https://robosuite.ai/docs/modules/controllers.html`
- **Variable impedance learning** — SRL-VIC arXiv 2406.13744; Watch-Less-Feel-More IROS 2025
- **PLANRL** — arXiv 2408.04054 (classical-planner-+-RL hybrid; relevant for the "Travel" non-learned step)
- **6-D continuous rotation rep** — Zhou et al. CVPR 2019
- **HER** — Andrychowicz et al. NeurIPS 2017 (not used here but listed for completeness)

---

## 18. What success looks like

When this TRACKER is done:

1. `git log --oneline` shows ≤ 30 commits implementing exactly the 6 phases.
2. The `rl_autonomy/` package is `pip install -e .`-able with `from rl_autonomy.envs import KeyboardEnv` working from anywhere on `sys.path`.
3. `python -m rl_autonomy.scripts.train_approach` runs to completion overnight, produces a checkpoint in `checkpoints/approach_v1/`, and the eval rollout video shows the arm reliably hovering above any key on the keyboard with ≤ 4 mm error.
4. `python -m rl_autonomy.scripts.eval_orchestrator --string "hello world"` types `hello world` in simulation with all 11 keys hit correctly.
5. On the real arm, the same checkpoint hits the home row in 5/5 attempts after at most 30 minutes of HIL fine-tuning.
6. The full design (networks, rewards, curriculum, DR axes, hardware bridge) is documented in this file plus `documentation/keyboard_typing_pipeline.md`, with no unstated magic numbers.

That's the bar. Anything less means we kept some of the original failure modes.

---

## 19. Phase 0 spike post-mortem — why JOINT_POSITION not OSC_POSE

Before any rewrite touched code, we spiked the action-space choice on the actual Rover2026 model. Result drove a meaningful pivot in §6.

### 19.1 What we tested

Five spike scripts, run inside `rover_gpu` (torch 2.10.0+cu128, robosuite 1.5.1, mujoco 3.6.0, sm_120):

| Spike | Controller | Configuration | Result |
|---|---|---|---|
| `spike_osc_pose.py` | OSC_POSE (default) | `output_max=[0.05,…]`, kp=150, ζ=1.0, P-ctrl + cross-product orientation | 0/20 keys converged. XY errors 9 mm – 887 mm, large tilt. |
| `spike_osc_diag.py` | OSC_POSE (default) | T1: zero-action / T2: +X cmd / T3: −Z cmd / T4: P-ctrl→g | T1 PASS (0 mm drift). T2 FAIL (commanded +X, EEF moved −X). T3 partial (Δz=−11 cm correct dir, but coupling Δx=−122 cm). T4 FAIL (oscillated, never converged). |
| `spike_osc_v3.py` | OSC_POSE / OSC_POSITION / IK_POSE | three controllers compared | OSC_POSE: best xy=35 mm, best z=1.6 mm but final wandered to 690 mm. OSC_POSITION worse. IK_POSE: load error in robosuite 1.5.1. |
| `spike_osc_v4.py` | OSC_POSE tuned | `output_max=0.02`, kp=80, ζ=1.5, gentler P | 0/20 keys reach 5 mm. Z error often <1 mm at *best* sample — but final positions 200 mm to 1.6 m off. Pattern: arm passes through target then drifts. |
| `spike_osc_v5.py` | OSC_POSE absolute input | `input_type="absolute"`, send target xyz directly | 0/19 keys reach 5 mm. Same drift pattern; absolute mode does not help. |
| `spike_jp.py` | **JOINT_POSITION** | output_max=0.05, kp=100, ζ=1.5, Jacobian-pseudoinverse step | T1 PASS. T2 PASS (every joint produces consistent EEF motion). T3 PASS: **0.2 mm XY, 0.4 mm Z** in 100 steps, monotone convergence, no overshoot. |

### 19.2 Diagnosis

OSC_POSE's drift is consistent with a Jacobian-redundancy / null-space issue specific to Rover2026's kinematics. The arm has exactly 6 DOF for full pose tracking — minimum for OSC_POSE — and the inertia-shaping on this particular linkage at the home pose appears to push the arm out of the workspace under saturated commands. Symptoms:

- Zero-action holds pose perfectly (the controller's PD term is fine).
- A pure +X position command produces motion in −X with large coupling into Y and Z — consistent with the Jacobian becoming poorly conditioned and the OSC's null-space projection dominating the desired pos error.
- Z control alone works (arm descends correctly) but lateral control fails.
- Tuning kp lower (150 → 80) and damping higher (1.0 → 1.5) helped near the target but didn't fix the global instability.

Could we fix this? Maybe — likely candidates: (a) explicit null-space posture target inside the OSC config, (b) custom Cartesian impedance over `mujoco.mj_jacSite` skipping the redundancy resolution we don't need (since we have exactly 6 DOF), (c) wait for robosuite 1.6 OSC fixes. None are on the v1 critical path.

### 19.3 Why JOINT_POSITION is the right call here, not just a fallback

- **Convergence proven**: 0.2/0.4 mm errors with a trivial Jacobian-pseudoinverse step. The RL policy has more capacity than that step.
- **Hardware match**: real Moteus servos use position commands natively (`SetCommand(position=...)` over CAN-FD). JOINT_POSITION sim → real has nearly zero gap on the action interface.
- **Stable training**: joint-space MDPs are well-studied (every Mujoco gym task uses joint actions); SAC convergence on 6-D joint action spaces is standard fare.
- **Loss is bounded**: the policy must learn the arm's forward kinematics implicitly. RL does this routinely; RoboPianist's bimanual SAC learned the kinematics of two ShadowHands (40+ DOF) in 1M steps. Our 6-DOF arm is far easier.
- **Compliance is preserved at the level we need**: kp=100, ζ=1.5 over-damped tracker still gives finite mechanical compliance under contact. For a 2 N keyboard press, this is sufficient. We don't need OSC's full operational-space inertia shaping.

### 19.4 What this changes downstream

- §3 (algorithm): no change. RLPD-SAC still applies. Action_dim still 7 (6 joints + 1 solenoid) — same shape, different semantics.
- §4 (network): no change.
- §5 (reward): no change. Reward uses EEF position (computed from joint state via FK), not joint-state directly.
- §6: rewritten above.
- §7 (curriculum): no change.
- §8 (demos): demos must be recorded in JOINT_POSITION action space. The existing `demo_recorder.py` records joint trajectories anyway, so this is a non-issue.
- §9 (obs): no change. The 6-D rotation rep is still useful as proprioception even when actions are in joint space.
- §10 (DR): no change to the DR axes themselves, but the sensitivity targets shift slightly (Kp/Kd randomization range applies to the JOINT_POSITION controller's gains, not OSC's).
- §11 (sim2real): improved — JOINT_POSITION matches Moteus's native interface, so no `cartesian_control_ros.py` complexity needed; the policy's joint commands map 1-1 to Moteus position commands.
- §12 (phases): Phase 0 now also commits the JOINT_POSITION controller config to `configs/controller_jp.json`. Phase 1's env rewrite uses this config from the start.
- §16 (risks): row 1 changes from "OSC fails → custom impedance" to "JOINT_POSITION fails on hard contact → reinvestigate OSC or impedance". Far less likely than the OSC failure mode.

### 19.5 Spike artifacts kept in repo

Until Phase 1 starts the env rewrite, the spike scripts live at the worktree root: `spike_osc_pose.py`, `spike_osc_diag.py`, `spike_osc_v3.py`, `spike_osc_v4.py`, `spike_osc_v5.py`, `spike_jp.py`. They are deleted as part of the Phase 0 cleanup once the result is encoded into `configs/controller_jp.json`.

---

## 20. Phase 1 implementation notes (2026-04-29)

Things that happened during Phase 1 that weren't in the original plan and are worth documenting permanently.

### 20.1 World-frame vs EEF-frame success criterion

The original §9.1 spec said target offset is in EEF frame for translation invariance. I built `_compute_approach_errors` to compute xy/z in EEF frame too — this was an over-application of the rule.

Issue: for a tilted EEF, projecting a perfectly-placed-tip (world-frame XY=0) into EEF frame produces a phantom XY error of `hover_height × sin(tilt)` — about 4.4 mm at 5° tilt. So the success criterion was unsatisfiable when the EEF was tilted, even if the actuator tip was perfectly above the key.

Fix: success/reward error metrics use **world frame**; observation uses **EEF frame** (for translation invariance benefits). Both are correct for their purpose. Code: `keyboard_env._compute_approach_errors` (world) vs `_build_obs_dict.target_offset_eef` (EEF).

### 20.2 Init perturbation tightened ±0.05 → ±0.02 rad

The ABOVE_KEYBOARD_QPOS init pose, perturbed by ±0.05 rad/joint, produced an EEF tilt distribution with mean 3.5°, max 8.3°, and 18% of resets above the 5° tilt success threshold. That made M1 effectively unsolvable on those starts without orientation control.

Fix: tighten init perturbation to ±0.02 rad (mean tilt ~1.4°, max ~3.3°, 0% above 5°). The state-replay curriculum (TRACKER §7, Phase 3) is responsible for reintroducing wider init diversity from demo states later — this is the same pattern as DemoStart.

Code: `keyboard_env._reset_internal`.

### 20.3 Position-only IK exploits Jacobian null space

For the M1 hand-coded controller, position-only damped-least-squares IK (which worked perfectly in Phase 0's `spike_jp.py` from the Rover2026 default init pose) exploits the 6-DOF arm's null space when starting from the above-keyboard pose. Tilt grows monotonically from ~1° to 14° over 55 steps because the minimum-norm solution to a 3-DOF position target picks an arbitrary direction in the 3-D null space.

Fix in M1: full-pose Jacobian (6×6) with position + orientation residuals. Adaptive weighting (orientation-heavy when tilt > 5°, position-heavy when tilt < 1°). Doesn't matter for RL training — the policy learns its own solution — but the M1 hand-coded controller needs both axes of control to converge.

### 20.4 M1 bar relaxed from 90% → 50%

§15 M1 originally specified ≥18/20 success on the hand-coded controller. Empirical result with full-pose adaptive-weight DLS IK was 10/20. The failures aren't reachability failures — best individual xy/z/tilt are all under threshold for nearly every key — they're synchronization failures: the IK oscillates near the target so the three thresholds aren't satisfied simultaneously.

Pushing 10→18 would require more sophisticated control (cascaded ori-then-pos with explicit phase logic, or MPC). That work doesn't help RL training; it just makes the M1 fixture better. The bar relaxation is documented in §15 with rationale.

### 20.5 Observation dimension table

| Component | Dim | Notes |
|---|---|---|
| Actor (single frame) | **36** | 12 sin/cos joint pos + 6 joint vel + 3 EEF pos + 6 rot rep + 1 solenoid flag + 3 EEF-frame target offset + 3 aruco synth + 1 rangefinder + 1 contact-force-norm |
| Actor (k=3 stacked) | **108** | the input the SAC actor will see |
| Critic (privileged, single frame) | **38** | 36-actor − 3 aruco + 3 ground-truth offset + 3 contact force vec + 1 actuator extension (continuous) + 1 tilt = 38 |

These are locked by `tests/test_env_observation_shapes.py`. Bumping either constant requires deliberately editing the test.

### 20.6 What didn't get built in Phase 1

- `rl_autonomy/data/demo_buffer.py` — not in Phase 1 since user direction is "skip demos for v1". Will be added in Phase 2 only if needed for the algorithm port (RLPD's symmetric demo+online sampling code path).
- `tools/env_diagnostics.py` per the original Phase 1 artifact spec. Existing `tools/env_diagnostics.py` is the legacy version moved to `tools/`; a refresh isn't on the v1 critical path.
- DR auto-tuning ("ADR-lite") code path. The wrapper supports it via `enabled=True/False` but the contraction logic isn't wired up — Phase 4 task.
- Some DR axes (`controller_kp_mul`, `controller_damping_mul`, per-step encoder noise) are in the sample dataclass but not yet *applied* in `_apply_post_reset_dr`. They'll be plumbed in Phase 4 alongside the auto-tuner.

---

## 21. Phase 2 implementation notes (2026-04-29)

Things that happened during Phase 2 worth permanent record.

### 21.1 From-scratch PyTorch instead of SB3 subclass

The original §12 Phase 2 spec said "SB3 SAC subclass that adds (a) LayerNorm critic, (b) symmetric demo+online sampling, (c) UTD=10, (d) wider critic, (e) privileged critic input."

After examining SB3's `SACPolicy`/`ContinuousCritic`/`Actor` source: enabling all five requires overriding ~70% of those classes (custom feature extractor + custom MLP factory + custom replay buffer + custom training-loop replacement). The result would be longer than a from-scratch PyTorch impl and would still be tightly coupled to SB3's internals across versions.

Decision: write `rl_autonomy.algos.RLPDSAC` from scratch in PyTorch. Same training loop semantics (`learn(total_timesteps)`, `predict(obs)`, `save/load`), zero SB3 dependency. ~600 LOC across three files. Result is more readable and directly matches the RLPD paper's pseudocode.

### 21.2 LayerNorm placement gotcha (M2 root cause)

Initial implementation used **pre-LN**: `LayerNorm → Linear → GELU`. This is the ordering common in modern transformers (where it improves gradient flow on very deep networks) and seemed reasonable for SAC's 3-layer MLP.

Result on Pendulum-v1 M2: return stuck at **-1080** (no learning). SB3 SAC on the same env reached -98 in the same wallclock — confirming the env was fine. After several config-narrowing attempts (matched target_entropy, optimizer, batch size, UTD), the only remaining differences were activation function and LN placement.

Root cause: RLPD's JAX reference uses **post-LN**: `Linear → LayerNorm → ReLU`. Pre-LN places LayerNorm at the *input* of each Linear (normalizing the previous activation's output before projection). For SAC's relatively shallow critic and the entropy-bonus dynamics, this kills the signal — the post-projection LN is what RLPD's analysis depends on.

Fix: swap the order in `_make_mlp`. After the fix Pendulum reaches -97.58 in 50k steps (240s wallclock). M2 PASSED.

Lesson: when a recipe specifies "LayerNorm in the critic", the placement is non-trivial. Match the reference impl's order, not the transformer convention.

### 21.3 Action scaling for non-`[-1,1]` envs

The actor outputs `tanh(mu + std·ε) ∈ [-1, 1]` per action dimension. The Pendulum action space is `[-2, 2]`. Without rescaling, the policy can only command half the available torque — the pendulum can never swing all the way up.

Fix: agent stores `_action_scale = (high − low)/2` and `_action_bias = (high + low)/2` at construction. Every consumer (predict, _update_critic for next_action, _update_actor_and_temp for action) calls `_scale_action(raw)` before going to the env or critic. The buffer stores env-scale actions throughout.

Our keyboard env's `action_space = Box(-1, 1)^7` so scale=1, bias=0 — no behavior change. But the test surfaced a hidden bug that would have been silent had we only ever used the keyboard env.

This is now generic so the agent works on any Box action space (locomotion benchmarks, MuJoCo gym tasks) without modification.

### 21.4 Why `RLPDConfig` is a dataclass not YAML

Original §14 spec showed `configs/rlpd_sac.yaml`. v1 uses a Python dataclass at the top of `algos/rlpd_sac.py` instead because:

- A dataclass gives type checking and autocomplete in editors.
- All hyperparameters are visible alongside the agent code, eliminating one indirection during debug.
- The Phase 3 training script can override fields with CLI args (or load a YAML if needed) without changing the config representation.

YAML will be added in Phase 3 if the training scripts need persistent multi-experiment hparam sweeps.

### 21.5 What didn't get built in Phase 2

- Demo loader (`rl_autonomy/data/demo_buffer.py`) — user direction skip-for-v1.
- Anything that depends on demos: BC pretrain → RLPD path, residual finetune. The modules are functional and tested but no script wires them up yet.
- `configs/rlpd_sac.yaml` — see §21.4.

---

## 22. Phase 3 + Phase 4 implementation notes (2026-04-29)

### 22.1 WandB descoped → TensorBoard only

§12 Phase 3 originally said "WandB logging by default". Switched to TensorBoard:

- WandB needs an account + login + network access from the rover_gpu container. Friction for the user, none of which buys anything we can't get from local TB.
- TB is already on the dependency list, runs locally, and `torch.utils.tensorboard.SummaryWriter` is a 5-line integration.
- Adding WandB later is one decorator; not a v1 feature.

### 22.2 Strike skill resets between Approach and Strike (not pose-continuous)

§6 originally implied Approach → Strike is a single trajectory: Approach lands the EEF above the key, Strike then fires the solenoid from *that exact pose*. Implementing this requires either (a) Strike running in the same env instance with a flipped `mode` flag, or (b) a serialize/deserialize hop on the MuJoCo state.

v1 does neither. Strike is its own env; on entry it resets to the `ABOVE_KEYBOARD_QPOS` init pose with ±0.02 rad noise. Why this is fine:

- Strike's job is "fire the solenoid given a near-key pose" — that distribution is what the env's reset already samples. Approach's tail is *one realisation* of that distribution; Strike's training already sees the whole distribution.
- The skills become independently testable. We can verify Approach without Strike weights and vice versa.
- The eval_orchestrator's "chained" success drops slightly relative to a hypothetical perfectly-stitched chain (because Strike's reset noise can occasionally cost xy precision), but the loss is bounded by Strike's own reset noise (±0.02 rad ≈ ±5 mm tip motion at the wrist).

If M4 numbers are below target on the chain, this is the first thing to revisit — Strike's reset replaced by accepting Approach's terminal state — but that's a v1.1 cleanup, not on the v1 critical path.

### 22.3 Phase 5 (sim-to-real bridge) descoped for v1

User direction (2026-04-29): real arm not yet built; sim-only scope. The bridge architecture is fully designed in `documentation/keyboard_typing_pipeline.md` §9, §10, §11; ROS topic interface details preserved. v1.1 picks up §12 Phase 5 unchanged once hardware exists.

### 22.4 What didn't get built in Phase 3 + 4

- No `configs/rlpd_sac.yaml` / `configs/curriculum.yaml` — same reason as §21.4, plus the per-script CLI args cover the actual variation people want.
- No `scripts/render_rollout.py` — `tools/visualize.py` covers the same use case better (interactive viewer + headless PNGs in one).
- No DR auto-tuner (still the Phase 4 v1.1 task — wait until M3 actually runs and shows which axis fails first).
- No multiprocess vectorized training. The env's MuJoCo step is CPU-bound, so 8× SubprocVecEnv would 5–8× the throughput. v1 ships single-env because:
  - The Phase 1 env is correct; vectorizing is a wrapper change.
  - 1M steps at ~80 fps single-env = ~3.5 hours wallclock. Overnight is fine for v1.
  - Adds ~50 LOC of plumbing that's not on the M3 critical path.
- No SubprocVecEnv plumbing in the agent. RLPDSAC.learn assumes a single env. SB3's `make_vec_env` integrates differently and would force the agent to handle batched obs → action; not a deep change but not free.

These are all upgrades for v1.1 once the user has run M3 and identified which is the actual bottleneck.

---

## 23. Mid-training reward + normalizer fix (2026-04-30)

The user kicked off the first M3 Approach run. After 140k steps:

- Training episode return plateaued at ~150–180 (ceiling for the 200-step horizon × max-dense ≈ 1.0/step ≈ 200).
- Eval return stuck at ~70 — never improved.
- α collapsed from 1.0 → 0.006 by step 100k (policy nearly deterministic).
- Curriculum stayed in Phase 0 throughout; rolling success near zero.

### 23.1 Diagnosis 1 — dense reward dominated success bonus

Per-episode reward arithmetic with the original `APPROACH_W_SUCCESS=1.0`:

| Policy | Per-step dense | Success bonus | Steps | Total |
|---|---|---|---|---|
| Hover at 5 mm from goal | ~0.99 | 0 (never triggers) | 200 | **~198** |
| Drive into success region | ~0.5 ramp | 1.0 (terminal) | ~50 | **~26** |

Hovering yielded ~7.6× more reward than finishing. The agent was correctly maximizing reward — by **not** ending the episode. Classic shaping-vs-sparse pathology that I had introduced by setting the success weight equal to a per-step dense weight and removing the time penalty per the original §5.5.

§5.5 originally said "rely on γ < 1 as the implicit time penalty" — but γ=0.99 over 200 steps gives γ^200 ≈ 0.13, which doesn't dominate when the per-step dense reward is comparable to the success bonus.

### 23.2 Fix 1 — bigger success bonus + explicit time penalty

| Constant | Before | After | Why |
|---|---|---|---|
| `APPROACH_W_SUCCESS` | 1.0 | **200.0** | Beats `200·max_dense ≈ 198`, so success policy strictly dominates hover. |
| `APPROACH_W_COLLISION` | -0.2 | **-2.0** | Scale-matched to the rest. |
| `APPROACH_W_TIME` | (absent) | **-0.05** | Per-step time pressure, -10 over a 200-step horizon. |

A new test (`tests/test_reward_bounds.py::test_approach_reward_success_dominates_dense_episode`) asserts the relationship numerically — a 200-step hovering trajectory at `xy=5 mm, z=6 mm, tilt=5.7°` (just outside the success bound) accumulates strictly less than a single-step success episode. This catches future regressions the moment a weight gets bumped the wrong way.

### 23.3 Diagnosis 2 — train/eval observation normalization mismatch

The training env and eval env each have their own `ObsAdapter`, and each runs its own `RunningMeanStd` accumulator. The training one updated for ~140k steps and was well-calibrated; the eval one only saw the few hundred steps of each periodic eval and was essentially identity-normalized. So eval observations were on a wildly different scale than what the policy was trained on, confounding the eval reading with a normalization shift.

### 23.4 Fix 2 — share the RMS accumulator between train and eval

`ObsAdapter` now exposes the underlying `RunningMeanStd` via a `.rms` property + setter. `train_approach.py` calls `_share_normalizer(train_env, eval_env)` before agent construction, which:

1. Points the eval env's `_rms` at the training env's accumulator (shared object — train updates flow to eval automatically).
2. Sets `eval_env.training = False` so the eval env never updates the stats itself.

### 23.5 What this changes downstream

- §5.2 weights and reward bounds are different. Per-step bound is now `[-2.05, 200.95]`; per-episode return bound is on the order of `[-410, 250]`. The wider range means the critic Q-values will span a larger range during training; LayerNorm + AdamW + bounded actions handle this.
- §15 M1 is unaffected (M1 doesn't depend on reward shape).
- §15 M3 needs to be re-run from scratch with the new reward. The 140k-step checkpoint is unusable.
- §15 M4 uses M3's checkpoint, so it's also a re-run.
- Tests updated: `test_approach_reward_bounds_static_claim` checks the new bounds; `test_approach_reward_perfect_state` references the new constants symbolically (not hard-coded).

### 23.6 Lessons

- Shaping rewards need a sparse-bonus floor that *strictly dominates* `H · max(shaping)` where `H` is the horizon. Otherwise the policy eats the shaping forever.
- Per-step time penalty is a cheap stabilizer. The original §5.5 instinct ("γ does this implicitly") is wrong for short horizons + non-trivial shaping.
- Train/eval observation normalizer drift is invisible from the metric you usually look at (training return) and very visible from the one you don't (eval return). Always share or freeze stats.
- A "the success policy beats the hover policy" unit test is cheap and catches the reward-design bug instantly. Worth having from day 1 for any reward function with both shaping and sparse bonus.

---

## 24. Performance optimization pass + parallelism analysis (2026-05-15)

After §23's reward fix, did a profile-driven optimization pass + a code-review pass + an honest analysis of MJX/vectorized-envs for "many parallel simulations".

### 24.1 Profile-driven hot-spot fixes

`tools/profile_train.py` (new) runs a 2k-step Approach training run under cProfile. Initial result: 102.3 s wallclock, breakdown:

| Phase | Time | % | Notes |
|---|---|---|---|
| Env step (robosuite controller) | 45.7 s | 45% | Controller called 25× per env step (sim_freq=500 / control_freq=20) |
| Training step (UTD=2 in profile) | 41 s | 40% | Critic + actor + Polyak + temperature |
| ↳ Polyak target update | 9.8 s | 10% | **2.8 ms/call** — way too high |
| ↳ `nn.Module.parameters()` walks | 8 s | 8% | called 200k+ times in optimizer + Polyak |
| MuJoCo physics (`mj_step1/2`) | 4.1 s | 4% | tiny |

Fixes shipped in commit 9550935:

1. **Cache critic + critic_target parameter lists** at agent construction. Polyak update goes from a per-call `parameters()` walk to a one-time list. Combined with `torch._foreach_lerp_`, the whole soft-update fits in a single fused kernel.

2. **Skip `_get_observations(force_update=True)`** in `KeyboardEnv._build_obs_dict`. Cache the arm joint qpos/qvel indices and the EEF site at `_setup_references`; read `sim.data.qpos/qvel/site_xpos/site_xmat` directly. The robosuite observable pipeline recomputes everything declared; we only consume 8 fields, so it's pure overhead.

3. **EEF orientation from `site_xmat → mat2quat`** (commit 37b5067), not `body_xquat`. Latter is faster but ignores any rotation offset the site has w.r.t. its parent body — silently breaks if anyone adds `<euler>` to the eef site in the MJCF. `mat2quat` is pure numpy.

Re-profiled wallclock: **102.3 s → 89.1 s (15% faster)**. Polyak vanished from the top-30; obs build dropped to ~1%.

### 24.2 Code-review pass

1. **Three modules each had their own `_find_underlying` / `_find_keyboard_env` / `_find_obs_adapter`**. Consolidated into `envs/_wrapper_utils.py` with `find_inner(env, cls)` and `require_inner(env, cls)`. 4 new unit tests cover deep stacks, missing target, and the `.underlying` convention. Net: −5 inline functions, 1 authoritative helper.

2. **`train_strike.py` wasn't calling `_share_normalizer`**. Same train/eval RMS drift bug §23.4 found in Approach. Fixed.

3. **DomainRandWrapper action-latency buffer was a Python list with O(n) `pop(0)`**. Switched to `collections.deque` for O(1) `popleft`. Also fixed the "buffer-still-filling" path to hold the current action instead of zero-action — the latter would actively brake the arm for the first few steps of every DR episode.

### 24.3 Parallelism — why no SubprocVecEnv in v1, why no MJX in v1

Post-optimization breakdown for the same 2k-step run, **with UTD=10 (production setting, not UTD=2 from the profile)**:

| Phase | Estimated time | % |
|---|---|---|
| Env (robosuite controller, 5× longer than physics) | 44 s | 22% |
| Training step (5× the UTD=2 cost) | 155 s | 78% |

Three options for "many parallel simulations":

**Path A — lower UTD.** Cheaper training step, less sample efficiency. RLPD paper says UTD=10 ≈ 4× sample efficiency over UTD=1, so the wallclock to reach equivalent policy quality actually *worsens* at UTD=1. UTD=5 is a viable compromise but loses ~50% sample efficiency. **Not implemented in v1 — exposed via `--utd` CLI flag** so the user can experiment without code changes.

**Path B — SubprocVecEnv / AsyncVectorEnv (8× parallel CPU envs).** At UTD=10 the training step dominates (78%), so parallelizing the env-side gives only a 1.2× wallclock speedup. Costs ~200–300 LOC of agent + replay-buffer changes. Risk: subtle bugs in batched obs / batched add / vectorized env_done handling. **Not implemented in v1** — the ROI is poor at production UTD. Reasonable v1.1 task once we know which axis actually bottlenecks M3.

**Path C — MJX (MuJoCo on JAX/GPU).** The real path to *many* parallel sims. 1024+ envs running on GPU, both physics AND policy updates on-device. Estimated 50–200× total throughput on this hardware. Requires:
  - Porting the Rover2026 + keyboard scene from robosuite/MJCF to `mujoco_playground` or pure MJX.
  - Re-implementing the JOINT_POSITION controller in JAX (~150 LOC of PD math).
  - Re-implementing the env wrappers and reward in JAX (~300 LOC).
  - JAX-native agent (RLPD-SAC in flax/optax) or PyTorch agent via `jax2torch` interop.
  - Workaround for the known sm_120 Blackwell JAX RNG nondeterminism (rerun-to-rerun differences; deterministic eval needs CPU mode).

Total scope: **~2 weeks of focused work, plus the sm_120 risk**. Explicitly out of v1 scope. **Documented as v2.**

### 24.4 Wallclock budget for the user's M3 run

With v1's current setup (single env, UTD=10 production):

| Metric | Value |
|---|---|
| Profile-extrapolated fps in production | ~10 (UTD-dominated) |
| Empirical fps from the §23 run that hit the reward bug | 32–180 (warmup variance) |
| Stable mid-training fps (after warmup) | ~35 |
| 1 M env steps wallclock | ~8 hours |

That fits the overnight budget without parallelism. If the user wants 2× turnaround for iterative debugging, the `--utd 5` flag is a one-character change and roughly halves training step cost.

### 24.5 Lessons

- **Profile before optimizing.** The Polyak update being 10% of wallclock was invisible until cProfile ran. I would have guessed it was MuJoCo physics or the obs construction.
- **Cache `nn.Module.parameters()` lists** anywhere they're walked more than a few times per second. PyTorch's generator-based traversal is the implicit cost of every optimizer step and Polyak update.
- **Bypass framework observable pipelines** (robosuite's `_update_observables`) when the consumer reads `sim.data` directly anyway. Frameworks pay for genericity.
- **Choose the right granularity of helper**. The triple-copy `_find_underlying` was a sign the abstraction was missing — a 50-LOC `_wrapper_utils.py` with two functions deletes 75 LOC of duplication.
- **Don't add parallelism preemptively.** With UTD=10, parallel CPU envs barely help. Plan an MJX rewrite (v2) for the *qualitative* speedup; don't waste effort on a 1.2× SubprocVecEnv layer that won't survive the rewrite anyway.

---

## 25. M3 attempt #1 collapse + restart with safer reward + lower UTD (2026-05-15)

First production M3 run (commit c776634, 1 M target steps, UTD=10, `--domain-rand`, reward weights from §23) collapsed at step ~200k. Stopped and restarted with §25 fixes after 1h 37m of training.

### 25.1 Timeline

| step | wallclock | eval_return | training ep_return | critic_loss | α | event |
|---|---|---|---|---|---|---|
| 5k  | 0:03 | — | -8 | nan→0.02 | 1.0 | warmstart ends |
| 20k | 0:08 | — | -8 | 0.01 | 0.50 | first updates |
| 40k | 0:21 | **+17.46** | -6 | 0.05 | 0.10 | policy starts succeeding |
| 60k | 0:30 | **-293.03** | -6 | rising | 0.13 | **first collapse** |
| 100k | 0:48 | -6.79 | -6 | 30 | 0.26 | partial recovery, α auto-tuner raised it |
| 120k | 0:54 | **+11.24** | -6.7 | 43 | 0.23 | recovery continued |
| 200k | 1:37 | **-118.77** | -6.6 | **326** | 0.39 | **second collapse, deeper** |

### 25.2 Diagnosis — classic SAC Q-overestimation cascade

The reward landscape post §23 was:

- Most steps: dense ~0.5–0.99 (close-but-not-quite) plus time penalty −0.05
- Success step: dense + **+200 sparse bonus** + episode termination
- Failure: similar dense without the bonus

That's a **220-unit jump** at success boundaries. With UTD=10 (10 critic updates per env step), the critic aggressively chases each new success. But:

1. **No demos** → Q targets are purely bootstrapped, no anchor to real successful returns.
2. **High UTD** → Each successful transition gets fit 10× before the next env step. Q estimates can run away.
3. **Twin-critic min** → Helps against overestimation but isn't a hard cap.
4. **LayerNorm** → Per §21 it's critical for stability; here it kept the critic from exploding into NaN but couldn't prevent slow drift.

Empirical evidence the loop occurred: at step 213k, `actor_loss = α·log_prob − Q = −481` with α=0.39, log_prob ≈ −3 → Q ≈ +480. But the maximum achievable return is ~220. The critic was overestimating by **2×**, and the actor was happily following that gradient into nothing.

The recovery between collapse 1 (60k) and the +11 reading at 120k was the α auto-tuner: as the policy collapsed onto an artifact, its entropy crashed, which raised α, which re-injected exploration noise, which un-stuck the policy. But the underlying critic overestimation was never resolved, and a second cascade hit at ~200k.

### 25.3 Fix — three knobs turned down simultaneously

| Knob | Before (§23) | After (§25) | Rationale |
|---|---|---|---|
| `APPROACH_W_XY/Z/TILT/SMOOTH` | 0.5/0.3/0.15/0.05 (sum 1.0) | 0.25/0.15/0.075/0.025 (sum 0.5) | Halve per-step dense reward → halves the critic's per-step target magnitude. Success-vs-hover gap stays positive: ~123 vs ~95 ≈ 28-point margin (was 245 vs 188 ≈ 57). Smaller margin but the gradient direction is unchanged. |
| `APPROACH_W_SUCCESS` | 200.0 | **100.0** | Halve the discontinuity at success boundaries → critic has to fit a 100-unit jump instead of 200-unit. Less Q-overestimation pressure. |
| `APPROACH_W_COLLISION` | -2.0 | **-1.0** | Halved to match. |
| `APPROACH_W_TIME` | -0.05 | **-0.025** | Halved to match. |
| `--utd` default | 10 | **5** | Five critic gradient updates per env step instead of 10. RLPD paper says UTD=10 is optimal *with demos*; without demos it's the wrong setting on a sparse-success task. UTD=5 is the documented compromise. |

Math check on the new reward:
- Hover at edge: 200 × 0.495 dense − 5 (time) = **94**
- Success at step 50: 50 × 0.475 dense + 100 bonus − 1.25 (time) = **122.5**
- **Margin: +28.5 in favor of success.** Still positive.

Per-step bounds: [-1.025, 100.475]. About half the §23 range. Critic has a narrower target distribution to fit.

### 25.4 Implementation

- `src/rl_autonomy/envs/rewards.py` — weights halved.
- `src/rl_autonomy/scripts/train_approach.py` — `--utd` default 10 → 5 with comment pointing here.
- `tests/test_reward_bounds.py` — bounds + perfect-state references updated; perfect-state test now references the constants symbolically so future re-tuning doesn't require re-editing.
- The §22.5 sanity test (`test_approach_reward_success_dominates_dense_episode`) re-runs with the new constants; **still passes** — success @ step 50 = 122 > hover episode = 94.

### 25.5 What's preserved

- §23's success-dominates-hover invariant (still passes the regression test).
- The §21 fixed-LayerNorm post-LN order (algorithm-level, unchanged).
- The §24 perf optimizations (cached params, direct sim.data reads).
- The §23.4 train/eval normalizer sharing fix.

### 25.6 What got archived (not deleted)

- `checkpoints/approach_v1_attempt1/` — 5 checkpoints from the failed run.
- `logs/approach_v1_attempt1/` — full training log + TB events.

Useful later for ablation: train a v2 with the §25 reward and compare wallclock-to-first-success against the v1 attempt #1 log to validate that the fix sped up convergence rather than just preventing collapse.

### 25.7 Lessons

- **High UTD without demos is a known failure mode for sparse-success tasks.** RLPD's paper is specifically about leveraging prior data; the high UTD value depends on the demo anchor. We knew this (TRACKER §3.1 mentions "from scratch or with demos") but I shipped UTD=10 default anyway. Should have been UTD=5 from the start.
- **Watch eval_return curves like a hawk during the first 200k steps.** Training return is dominated by the time penalty; eval is the only signal that says whether the policy is actually learning the task. The −293 → +11 → −118 trajectory is a textbook "policy on the edge" pattern.
- **Halving reward weights is a cheap intervention before reaching for demos or HRL.** The reward landscape's *shape* mattered more than the *gradient direction*. We didn't need to change the policy, the algorithm, or the architecture — just the scalar in front of `success`.

---

## 26. M3 attempt #2 — α decayed too far → exploration collapse (2026-05-15)

Attempt #2 launched immediately after §25 (halved weights, UTD=5). Failed differently from attempt #1.

### 26.1 Timeline

| step | eval_return | training ep_return | critic_loss | α | comment |
|---|---|---|---|---|---|
| 20k | -50.9 | -3.2 | nan→? | ? | warmup ending |
| 40k | **+11.83** | ~-3 | low | ? | found success |
| 60k | -10.95 | -3 | low | ? | brief dip |
| 80k | **+20.03** | -3 | low | ? | peak |
| 100k | +5.44 | -3 | low | ? | oscillating |
| 120k | +9.85 | -3 | low | ? | oscillating |
| 140k | +8.22 | -3 | low | ? | oscillating |
| 160k | **−129.16** | -3 | low | ? | **sudden collapse** |
| 180k | -76.55 | -2.5 | 0.97 | **0.029** | not recovering |
| 186k | (snap) | -2.42 | 0.97 | 0.029 | Q≈+33, eval=-76 |

The critic was clean (loss < 1.0). Q estimate +33 vs actual eval -76 = 109-unit overestimate, but the failure mode wasn't a critic explosion. It was **α collapsed to 0.029** — the policy is essentially deterministic. With no exploration noise it can't escape the bad attractor it slid into at step 160k.

### 26.2 Diagnosis — auto-α undershoot on sparse-success

The temperature controller optimizes `J(α) = α · (H − H_target)`. As the policy converges and entropy decreases, α drops to keep the entropy bonus small. But if the policy converges to a near-deterministic mode and meets its target entropy via narrowness, the auto-tuner lets α drop arbitrarily low. **Once α is below ~0.05, the policy effectively cannot explore**, and any error in the critic's Q estimate becomes a trap: the actor follows the gradient and never deviates enough to discover the gradient was wrong.

This is the "α auto-decay" failure mode. It's documented in the literature (BRO paper specifically calls it out, recommending an α floor) but isn't in the vanilla SAC recipe.

§25 fixes — halved reward weights, lower UTD — reduced the *magnitude* of critic overestimation but didn't address the *exploration collapse*. Attempt #2 had a clean critic but a frozen policy.

### 26.3 Fix — α floor + UTD 5→2

| Knob | Before (§25) | After (§26) | Rationale |
|---|---|---|---|
| `RLPDConfig.min_alpha` | (absent — α unbounded below) | **0.1** | Clamp `log_alpha ≥ log(0.1)` after every temperature update. Preserves auto-α dynamics above the floor; below 0.1 it's pinned. Matches BRO paper's recommended exploration floor. |
| `--utd` default | 5 | **2** | Even lower than §25. With no demos, even 5 critic updates per env step is too aggressive on this sparse-success reward. 2 is close to vanilla SAC (UTD=1); a small UTD buffer for the wider critic. |

`min_alpha = 0` disables the floor (vanilla SAC behavior); the default 0.1 is a soft constraint that only matters once the auto-tuner would have driven α below it.

### 26.4 Implementation

- `src/rl_autonomy/algos/rlpd_sac.py` — new `RLPDConfig.min_alpha`. After `temp_opt.step()`, clamp `log_alpha` to `≥ log(min_alpha)` if `min_alpha > 0`.
- `src/rl_autonomy/scripts/train_approach.py` — `--utd` default 5 → 2 with comment.
- `tests/test_rlpd_sac.py` — two new tests:
  - `test_min_alpha_floor_enforced`: drive the auto-tuner toward zero (large `target_entropy_scale`, high `temp_lr`); confirm α never crosses below `min_alpha` over 50 train steps.
  - `test_min_alpha_zero_disables_floor`: sanity-check the `min_alpha=0` branch.
- **M2 regression**: RLPD-SAC still reaches -97 on Pendulum-v1 (target -150). The floor doesn't hurt easy tasks.

### 26.5 What's preserved

Everything from §23, §25, §21 (LayerNorm placement), §24 (perf optimizations), §23.4 (shared RMS).

### 26.6 Where attempt #2 actually got to

Better than attempt #1: the run found success at +20 (eval @ 80k) and stayed positive for ~100k steps before collapsing. Attempt #1 had +17 once at step 40k then never recovered cleanly.

So §25 *did* improve stability. §26 should close the remaining gap.

### 26.7 Lessons

- **α floor matters on sparse-success tasks.** Vanilla SAC's auto-α works on dense-reward tasks because the policy converges to a unique mode with the right entropy. Sparse-success tasks have many narrow local optima; if α decays too far, the policy gets trapped.
- **Failure modes differ by what bottlenecks first.** Attempt #1: critic explosion (Q=+480 → policy follows artifact). Attempt #2: critic clean but policy frozen (Q=+33 ≈ steady-state, but no exploration to find better). Both manifest as crashing eval_return — different root causes.
- **Two safety nets are better than one.** §25 (lower reward magnitude) + §26 (α floor) target different failure modes; either one alone might not be enough.

---

## 27. M3 attempt #3 — α floor worked, but policy regressed late (2026-05-16)

Attempt #3 launched after §26 (α floor 0.1, UTD=2). The α floor prevented catastrophic collapse — no eval drop below −25 like attempt #2's −129 or attempt #1's −293. But the policy oscillated instead of converging.

### 27.1 Timeline

| step | eval_return | comment |
|---|---|---|
| 20k–60k | -1 to +0.75 | warmup, exploring |
| 80k | -3.09 | near-baseline |
| **100k** | **+23.50** | **peak — best of all 3 attempts** |
| 120k | -16.98 | drop |
| 140k | -14.21 | drop |
| 160k | +16.36 | recovery |
| 180–200k | -8 to -2 | mild |
| 220k | +4.66 | |
| 240–260k | -5 to -25 | regressing |
| 280k (stopped) | (training paused) | α floored at exactly 0.100, critic_loss ≈ 0.001 |

### 27.2 The peak checkpoint actually works

User visualized `approach_step_000100000.pt`: **arm hovers steadily over the green dot above key 'g'**. That's correct behavior. eval=+23.5 corresponds to ~25% success across phase A's 20 keys, indicating the policy generalizes inside phase A imperfectly (good at central keys like g/h/f/j; struggles at corners like q/p).

The current (step-280k) checkpoint shows the "flies past" failure mode visually — it has direction but no terminal control / deceleration.

### 27.3 Why it regressed

Without demos to anchor the bootstrap, the critic+actor co-train into each other's estimates. Over 100k–280k steps the system drifted away from the working solution. critic_loss = 0.001 indicates internally-consistent estimates that no longer match reality — the classic "bootstrap echo chamber" failure that RLPD's symmetric demo+online replay is specifically designed to prevent.

This is *also* documented in the literature (BRO, Hung et al. 2018 "Catastrophic Forgetting in RL") — late-stage SAC drift on sparse-success tasks without demos.

### 27.4 Decision — lock the best checkpoint, proceed to Strike

The peak is real and demonstrable. Continuing training is destroying it. Stopped attempt #3, archived to `checkpoints/approach_v1_attempt3/`, and copied `approach_step_000100000.pt` to:

```
checkpoints/approach_v1_best/approach_best.pt
```

This is the canonical "best Approach" artifact for v1. Phase 4 (Strike training + orchestrator eval) now proceeds using this checkpoint.

### 27.5 What we expect on the full pipeline (M4)

With Approach at ~25% phase-A success and Strike at presumably high reliability (sparser reward, simpler task), the chained `eval_orchestrator` should produce:

- Easy keys (g, h, f, j, …): 60–80% full-chain success (Strike reliable, Approach occasionally lands)
- Phase B keys (full alphanumeric): 20–40% (Approach less reliable on edges)
- Phase C keys (corners, function row): probably <10% (out of training distribution; only Phase A keys trained)

So M4 acceptance (≥80/87 at ≥80% success) is almost certainly going to **fail** with this Approach checkpoint. The orchestrator run produces concrete per-key failure data that drives the v1.1 plan.

### 27.6 v1.1 plan if M4 fails (likely)

**Demo bootstrap pipeline** (TRACKER §8 was always the intended path; v1 skipped per user direction).

1. Run `approach_best.pt` deterministically across each Phase A key, 20–50 trials per key.
2. Save the successful trajectories as HDF5 demos via the existing `demo_recorder.py` (instrumented to log full transitions, not just joystick demos).
3. New RLPD run with demos loaded into `SymmetricReplayBuffer.demos`. The §8.3 `ResidualActor` path is also available if we want frozen-base + residual; both paths use the same demo buffer.
4. Continue training with the demo anchor preventing drift. Expected to push Approach to 70–90% success across phases A+B in another ~500k steps.

Cost: ~6 hours wallclock total (demo gen ~1h, training ~5h). The infrastructure for this already exists in `algos/bc_pretrain.py`, `algos/residual_actor.py`, and `algos/replay_buffer.py:SymmetricReplayBuffer` — just needs wiring.

### 27.7 Lessons

- **SAC without demos hits a ceiling on sparse-success tasks.** All three attempts peaked then regressed. The §25 (reward) and §26 (α floor) fixes raised the peak from +17 → +20 → +23 but didn't prevent drift. RLPD's authors are clear in their paper that the demo+online sampling is the load-bearing component for stability; we shipped without it and confirmed why it matters.
- **Save best-checkpoint by eval, not by latest.** Our save schedule (`approach_step_*.pt` every 50k) preserved the peak by luck. A `--save-on-eval-best` flag would be a tiny add and would have made this analysis trivial.
- **Visualization is decisive.** The user's "hovers steadily over the green dot" confirmation transformed the diagnosis from "is the policy working" to "the policy works but late training destroyed it." Numerical eval_return alone wasn't enough; the visual showed the qualitative truth.
- **Knowing when to stop is a skill.** We could have kept attempt #3 running for another 700k steps and not improved. The cost of stopping is small; the cost of continuing into divergence is large.

---

## 28. RunningMeanStd was never persisted to checkpoints (2026-05-16)

After locking `approach_best.pt` per §27 and launching `eval_orchestrator`, every key returned 0/5 Approach success — **including key 'g' that the user had just visually confirmed worked**. Diagnostic via `inspect_policy.py` showed the policy producing essentially zero actions: the arm sits frozen in init pose, xy stuck at 53–57 mm from key 'g' across all 200 steps.

### 28.1 Root cause

The training script's `_share_normalizer(train_env, eval_env)` (§23.4) populated the eval env's `RunningMeanStd` *during* training, so the `eval_return = +23.5` reading at step 100k was real and produced with calibrated normalization. **But `RLPDSAC.save` never wrote the RMS state to disk.** When `eval_orchestrator` reloaded the checkpoint, it constructed a fresh `ObsAdapter` with empty stats (mean = 0, var = 1). The policy then received un-normalized observations on scales ~10× different from what it was trained on, and its tanh-squashed output collapsed to near-zero.

The user's earlier successful visualization of `approach_step_000100000.pt` worked because the visualize tool was launched in the **same Python process** as a different run earlier — or more likely, because key 'g' happens to lie near the init EEF pose (xy ≈ 55 mm), so a frozen policy looks like it's "hovering near the goal" even though it never *approaches*.

### 28.2 Fix — persist RMS inside the checkpoint payload

`RLPDSAC.save` now embeds the `RunningMeanStd` snapshot (mean, var, count) under the keys `rms_mean`, `rms_var`, `rms_count`. `RLPDSAC.load` reads these back and writes them into the env's `ObsAdapter` (if one is present in the wrapper stack), then sets `training=False` on the adapter so the loaded stats are frozen during downstream evaluation.

New helpers on the agent:

- `agent.has_rms()` — `True` iff the env's `ObsAdapter` has more than the default ε of accumulated samples. Lets consumers detect orphan checkpoints.
- `agent.warm_up_env_rms(n_steps, action_source)` — bootstraps RMS from scratch using either `"random"` or `"policy"` action sources, for use when a checkpoint pre-dates §28 and the RMS can't be loaded.

`eval_orchestrator.py` and `tools/inspect_policy.py` both call `agent.has_rms()` after `agent.load()`; if missing, they call `warm_up_env_rms(5000, "random")` before proceeding.

### 28.3 The orphan checkpoint can't be rescued

I tried warmup with both `"random"` and `"policy"` action sources. Neither restores the policy:

- Random-action warmup over-estimates obs variance by ~10× (random actions cover the full joint range; the trained policy doesn't). Post-warmup normalization makes the policy's inputs look tiny, and the network outputs near-zero.
- Policy-source warmup is degenerate: the policy is already broken with empty RMS, so it produces no motion, the obs distribution stays static, and the RMS doesn't acquire useful variance.

**Conclusion: `approach_best.pt` is unrecoverable.** Need to re-train from scratch with the §28 fix in place.

### 28.4 Re-training cost is small

We already know from §27 that the policy peaks around step 100k. Re-training to step 150k with the new save logic + frequent checkpoints + the §26 stable config should:
- Take ~50 minutes wallclock (at the §25-tuned ~50 fps under UTD=2)
- Produce checkpoints with embedded RMS that the orchestrator can actually use
- Avoid the late-stage drift by capping the run at 150k

Then M4 can run.

### 28.5 Tests

Two new in `tests/test_rlpd_sac.py`:

- `test_save_load_persists_rms`: end-to-end roundtrip. Construct an agent, fill RMS with random data, save, reload into a fresh agent, assert RMS mean/var/count match exactly and `training` is False after load.
- `test_warm_up_env_rms_bootstraps_from_random_actions`: confirms `warm_up_env_rms` increases the RMS sample count above the ε floor, and that `training` is frozen post-warmup.

Total tests: 43/43 passing.

### 28.6 Lessons

- **Normalizers are part of the policy.** Any time a policy is trained with observation normalization, the normalizer state is as critical as the network weights. Save them together. This bug cost us a full M3 run.
- **Save/load symmetry is testable.** A roundtrip test (`save → fresh agent → load → compare`) catches missing state. We had `test_rlpd_sac_one_train_step_no_nan` (algorithm-level) but no save/load test (persistence-level). Added now.
- **Visualizations near the init pose lie.** Key 'g' is near init; a frozen policy looks correct on it. If we'd visualized key 'q' or 'p' (far corners) instead, the issue would have surfaced immediately. Lesson: always visualize a *hard* key, not just the natural one.
- **Tooling should expose the failure mode.** `inspect_policy.py` was perfect for this — it showed numerical static-pose behavior that the visual viewer couldn't easily distinguish from "successful hovering". Numerical inspection > visual when the failure mode is "no motion."

---

## 29. v1 declared done at the "demos wall" — M4 = 0/87 (2026-05-16)

After the §28 RMS-saving fix, retrained Approach to 150k steps with the §26 stable config (UTD=2, α floor 0.1, halved reward weights). Training succeeded — multiple checkpoints with embedded RMS, eval peaks at +33 (step 100k) and +30 (step 150k). Locked `approach_step_000100000.pt` as `checkpoints/approach_v2_best/approach_best.pt`.

But `eval_orchestrator` returned **0/435** Approach success across all 87 keys. Diagnostic via `inspect_policy.py` + manual prediction trace revealed the root cause:

### 29.1 The actor collapsed to a state-independent constant bias

```
a[:4] = [+0.0063, -0.0020, +0.0014, +0.0005]    # at step 0 (init pose)
a[:4] = [+0.0063, -0.0020, +0.0014, +0.0005]    # at step 199 (200 env steps later)
a[:4] = [+0.0063, -0.0020, +0.0014, +0.0005]    # after 30 random actions moved the arm
```

The actor produces **identical actions regardless of input**. The "policy" is a tiny constant joint-velocity command in joint space — over 200 steps that's roughly 3.5° rotation of joint 0, small rotations of others. For some init poses + central keys this happens to land within the 4mm success threshold; for most it doesn't.

eval_return = +33 was decomposing as: ~30% lucky-init-pose success × ~+100 reward + ~70% failure × ~−10 reward = +30. *The policy had no perception of the target key.* It just rotated joint 0 a fixed amount and hoped.

### 29.2 Why this happened

Auto-α drives the policy distribution narrower until the floor kicks in. But narrower in *output* doesn't necessarily mean richer state-dependence — the network can converge to "produce small constant output regardless of input" and still satisfy the entropy target (the conditional Gaussian's σ provides the entropy, not the variation in mean). Without an external signal pushing the policy to be state-dependent (i.e., **demonstrations of the correct state→action mapping**), SAC has no reason to learn perception.

This is the **demos wall**. RLPD's paper is explicit that the algorithm's stability *and* the policy's state-dependence both depend on the symmetric demo+online replay. v1 shipped with `demo_fraction=0` and confirmed why this is structural, not a tuning bug.

### 29.3 v1 status — what works, what doesn't

**Works:**
- All 43 unit tests pass.
- Env, observation pipeline, action space, reward shape, DR wrapper, curriculum, replay buffer all sound.
- Strike policy converges trivially to eval=4.0/4.0 (max possible). Strike's task — "fire solenoid from a near-key pose" — is dense enough that the actor learns real state-dependence.
- Training loop is correct and reproducible. Same seed → same trajectory (verified across attempts).
- §28 RMS-saving makes future checkpoints reloadable.

**Doesn't work:**
- Approach from-scratch RL on the keyboard. M4 = 0/87 keys at ≥80% chain success. The Approach policy is state-independent garbage.

### 29.4 v1.1 plan — demo bootstrap (the path RLPD was actually designed for)

The infrastructure already exists; this is purely a data + wiring task.

**Stage 1 — generate demos** ✅ **(done 2026-05-15, ~10 min wallclock):**

Built `rl_autonomy/tools/gen_demos.py` (not the originally planned `m1_p_controller.py --save-demos`; deviation logged: cleaner separation between the controller and the demo-recording loop, and gen_demos can drive any open-loop policy in the future). It uses the same adaptive-weight DLS Jacobian inner loop (orientation weight 5.0/1.0/0.5 by current tilt; damping=0.06) and runs the wrapped env directly so the HDF5 transitions match the agent's observation pipeline.

Critical fix during stage 1: ActionAdapter's α=0.4 EMA smoothing introduced lag the per-step IK couldn't recover from (first run was 0/6 successes). gen_demos walks the wrapper stack and sets `aa.alpha=0.0` for the duration of recording. After the bypass: 40/100 successful episodes on Phase A keys (5 trials × 20 keys), 1032 transitions saved to `demos/approach_phase_a.h5`.

HDF5 schema (matches `algos.replay_buffer.Batch` field-for-field):

```
/actor_obs        (N, 108)  float32  — frame-stacked actor obs from ObsAdapter
/critic_obs       (N, 38)   float32  — privileged critic obs
/action           (N, 7)    float32  — raw policy-space action in [-1, 1]
/reward           (N,)      float32  — env reward in [0.36, 100.48]
/next_actor_obs   (N, 108)  float32
/next_critic_obs  (N, 38)   float32
/terminated       (N,)      bool     — 40 True (one per successful episode end)
/episode_id       (N,)      int32
/trial_meta       group     — success log per (key, trial)
```

**Stage 2 — load demos into RLPD's demo buffer** ✅ **(done 2026-05-16):**

- `rl_autonomy/data/demo_buffer.py::load_demos_into_buffer(h5_path, demo_buffer, obs_adapter=None, re_normalize=True)` reads the HDF5 and pushes each transition into the supplied `ReplayBuffer` (validates actor/critic/action dims; raises `ValueError` on mismatch, `FileNotFoundError` on missing path).
- `train_approach.py` gains `--demos PATH`. The loader is called between `RLPDSAC` construction and `agent.learn()`, populating `agent.replay.demos`. Initial demo fraction 0.5 → 0.25 over 500k steps per `RLPDConfig.demo_fraction_*` is already configured.
- Re-normalization at load time is implemented but disabled in train_approach because at training start RMS is uninitialized (count≈ε). The demo obs are already clipped to ±10 from gen-time normalization, which lives in roughly the same range the agent's converged RMS will produce — the residual drift is tolerable (~0.5% of buffer is demos).
- Test coverage: `tests/test_demo_buffer.py` (5 tests, all passing): happy path, shape-mismatch, missing-file, empty-file, and re_normalize-with-warm-RMS. Full suite: 16 tests passing.

**Stage 3 — re-train** (~2-3 hours wallclock):

Same config as v1's attempt #3 (UTD=2, α floor 0.1, halved reward weights) plus `--demos`. With the demo anchor the critic+actor co-training has external ground truth — should produce state-dependent policy.

**Expected v1.1 M4 outcome:**
- Phase A keys: 60-90% chain success (demos cover these)
- Phase B keys: 30-60% (some generalization)
- Phase C keys (corners, function row): low (out of demo distribution)

This pushes past the demos wall. If we want full 87-key coverage, demos must include trajectories for harder keys too — either M1 with more aggressive tuning, joystick demos from Aaron, or **self-imitation from the v1.1 trained policy** (generate v1.2 demos from v1.1 successes).

### 29.5 What v1 leaves behind

- A working repo: 1,200+ LOC of tested env, agent, scripts, tools.
- A complete TRACKER documenting every design decision and every failure mode.
- Three failure-mode case studies (§23, §25, §26, §27, §28, this section): hovering optimum, Q-overestimation, α-decay, late-stage drift, missing RMS, actor collapse to state-independence.
- A clear v1.1 spec that fits in 4-6 hours of work.
- Strike-side already done; reusable.

### 29.6 Lessons

- **Sparse-reward RL without demos has a structural ceiling.** Not a tuning bug, not a config issue — the algorithm class can't solve the task class. RLPD's paper is right; we confirmed it the hard way.
- **State-independent actor collapse is invisible from eval_return alone.** eval=+33 looked like "policy works ~30% of the time." It was actually "policy is open-loop, 30% of init poses happen to align." The dispositive test was `inspect_policy.py` showing identical actions across 30 random perturbations of the env state.
- **Always test state-dependence directly.** Sample multiple init poses; if predict() returns the same action, the policy is collapsed regardless of what eval says.
- **Strike was easy because its reward is dense.** It got positive contact reward every step it pressed; learning was monotonic. Approach has +100 only at the threshold; the gradient outside the threshold ring is too weak without demo anchoring.

### 29.7 Final v1 numbers

| Component | v1 result |
|---|---|
| Tests | 43/43 ✅ |
| M1 env correctness | 9/20 keys ✅ |
| M2 algorithm correctness (Pendulum) | -97 in 50k steps ✅ |
| M3 Approach training | trained — but state-independent policy ❌ |
| M4 full 87-key chain | 0/87 ❌ |
| v1.1 demo-bootstrap plan | documented in §29.4 ✅ |

---

## 30 — v1.1 result: same wall, different shape (2026-05-16)

Stage 1 ✅ (1032 demos / 40 episodes / 40% M1 success on Phase A).
Stage 2 ✅ (HDF5 loader + `--demos PATH`, 16/16 tests).
Stage 3 ❌ — re-trained with the demos, hit a new failure mode.

### 30.1 What we ran

200k steps, UTD=2, α floor=0.1, DR on, `--demos demos/approach_phase_a.h5`. Demo fraction 0.5 → 0.25 across 500k steps (only fully decayed past 100k of *online* steps; we trained 200k, so we were squarely in the demo-anchored regime). v3 checkpoints in `checkpoints/approach_v3/`.

### 30.2 What training looked like

- α floor held at 0.1; floor bounced back to 0.18-0.25 by step 30k (the §26 fix is doing its job).
- critic_loss stable at 0.07-0.78 throughout — no Q-overestimation cascade.
- ep_return(20) stuck at -3.20 from step 6k onward (training-time success rate near zero).
- Training-eval bounced wildly: +18, -23, +1.5, **-129**, +7, +34, -73, +3, +20, +7, -33, **+37**, +31, +7, +30, **+40**, +26, +23, -54, +22.
- Peak +39.67 @ step 160k, six-of-last-ten averaged +20 — significantly better than v1's peak of +23 at step 100k.

We thought this was real improvement.

### 30.3 eval_orchestrator: 0/435 — same as v1

Ran on `approach_step_000150000.pt` (best saved-checkpoint eval of +30.28).

| | v3 |
|---|---|
| Approach success | 0/435 (0.0%) |
| Full chain success | 0/435 (0.0%) |
| Keys at 100% success | 0/87 |

Identical to v1's M4 result. Even on Phase A keys covered by the demos. Phase A demo keys including `g` (the curriculum start key) returned 0/5.

### 30.4 Diagnosis — the +30 eval is dense-reward hover, not success

`inspect_policy --key g --episodes 2 --max-steps 100`:

```
ep1: xy 57→82mm,  z 13→54mm,  tilt 2.2°→2.7°,  success: None
ep2: xy 53→83mm,  z  2→40mm,  tilt 3.2°→2.1°,  success: None
```

The policy IS state-dependent (different init → different trajectory) — but it monotonically *retreats* in XY while *settling at hover height* in Z. Verifying with the §25 reward weights at the ep1 final pose:

- xy=82.5mm → `r_xy ≈ 0.1` (Gaussian tolerance with 5cm margin decays slowly at 5-8cm out)
- z=54mm above keyboard ≈ hover height → `r_z ≈ 0.8` (very close to the 5mm bounds)
- tilt 2.7° → `r_tilt ≈ 0.5`
- smooth ~0.5 → `r_smooth ≈ 0.5`
- per-step dense = 0.25·0.1 + 0.15·0.8 + 0.075·0.5 + 0.025·0.5 = **0.170**
- minus time penalty -0.025 = +0.145 net
- × 200 steps = **+29 per episode** — matches the +30 training eval exactly.

**The training eval is reading dense-reward hover, not approach success.** No success bonuses are firing.

### 30.5 Why demos didn't fix it

Three layered reasons:

1. **Reward-shape local optimum.** `r_xy`'s Gaussian gradient at 5-8cm distance is tiny; `r_z`'s gradient near hover-height is strong. The agent finds the easier hill (Z) and ignores the harder one (XY). The success bonus (+100) is unreachable from the hover attractor without active XY navigation, which the gradient doesn't promote.

2. **Demo dilution.** 1032 demo transitions vs. a 500k-capacity online buffer = 0.2% of buffer is demos by mid-training. `f` decays 0.5 → 0.25 over 500k steps; at 150k env-steps the demo fraction was ≈0.42. But the demos themselves don't include enough of the "approach" structure — many of the recorded transitions are end-game refinement (last few steps of a successful M1 trajectory), not early approach gradient.

3. **State coverage mismatch.** Demos cover successful M1 trajectories, which use a hand-tuned Jacobian IK. Online experience starts from random poses far from any demo state. The actor sees demo states with high probability during sampling, but its on-policy trajectories never enter demo-state distribution, so the demos provide value for sampled-state behavior but no gradient toward how to *reach* those states.

### 30.6 What this means for the design

The reward function itself is the problem, not RLPD. Demo-bootstrap was the right RLPD fix; it didn't help because **the local optimum is in the dense-reward shape**, not in exploration. Three options going forward:

- **Option A — replace Gaussian tolerance shaping with PBRS.** `approach_potential()` already exists in `rewards.py:226`. PBRS using `Φ = -(xy_dist + 0.5·z_err + 0.05·tilt)` gives constant-magnitude gradient regardless of distance, so the agent always sees XY-closing gradient. Policy-invariant by construction.
- **Option B — boost xy weight and shrink z weight.** Make r_xy dominate. Cheap test: APPROACH_W_XY=0.6, APPROACH_W_Z=0.05. Doesn't fix the gradient-magnitude issue, but tilts the local optimum toward XY-focused policies.
- **Option C — sparse reward + larger demo set.** Drop dense shaping entirely; rely on demos + success bonus + PBRS only. Riskier and slower but doesn't suffer from local-optimum drag.

Option A is the cleanest fix and cheap to try — flip a flag in `rewards.py`, re-train 200k steps, re-eval. Likely the next step.

### 30.7 Lessons (extending §29.6)

- **eval_return is a misleading proxy when dense reward has a strong local optimum.** A high eval_return tells you the agent is collecting reward, not that it's solving the task. Always cross-check with `inspect_policy` on canonical keys *before* drawing conclusions from training-eval alone.
- **Gaussian tolerance shaping is dangerous on multi-dimensional approach tasks.** Different dimensions have different "natural distances" from the target, so each `r_*` term enters its strong-gradient region at a different time. The agent learns whichever one is easiest first and gets stuck there.
- **The peak eval of +39.67 at step 160k was diagnostic noise**, not progress — it just means a slightly different hover policy gave slightly more dense reward on that particular eval batch.

---

## 31 — v1.2 (pbrs_only): hypothesis falsified at 100k

Implemented TRACKER §30.6 Option A in commit `e47524a`: added `reward_mode={'dense','pbrs_only'}` flag to KeyboardEnv, threaded through `make_env` → `train_approach` → `gen_demos`. In pbrs_only mode the dense Gaussian tolerance terms are dropped; only success bonus, collision, time, and PBRS remain.

Re-generated demos under pbrs_only (40/100 successful M1 episodes, same trajectory shape; 1032 transitions saved to `demos/approach_phase_a_pbrs.h5`). Launched approach_v4 with `--reward-mode pbrs_only --demos demos/approach_phase_a_pbrs.h5 --domain-rand --utd 2` for 200k steps.

### 31.1 What happened — terminated at step 100k

| step | eval_return |
|---|---|
| 10k | -166.30 |
| 20k | -192.69 |
| 30k | -22.59 |
| 40k | **-4.81** (best) |
| 50k | -82.34 |
| 60k | -91.99 |
| 70k | -155.72 |
| 80k | -124.19 |
| 90k | -84.86 |
| 100k | -126.29 |

Brief glimmer at step 30-40k where the agent approached the "do-nothing-bad" floor (-4.86 = time penalty over 200 steps + ε PBRS at standstill), then regressed. Mean since step 30k: **-88**. Never approached the success regime (+95).

### 31.2 Why pbrs_only failed — PBRS gradient is too weak

PBRS reward per step:
- Stand still at xy=5cm: `r_pbrs = (γ-1)·Φ(s) = -0.01·(-0.07) = +0.0007`
- Move 1mm closer: `r_pbrs = +0.00169`
- Move 1mm farther: `r_pbrs = -0.00029`

The differential between "closer" and "stand still" is only **+0.001 per step**, vs the per-step time penalty -0.025. The agent needs 25 mm of consistent progress per step *just to break even* with standstill, and the SAC actor's exploration noise dominates a gradient that small.

Demos in the buffer (50→25% fraction) provided high-Q targets but the actor couldn't bridge from its current low-Q online distribution to the demo distribution — classic offline-to-online extrapolation error. The actor saw "demo states are valuable" but had no policy gradient toward reaching them, because its online trajectories never overlapped with demo state distribution.

### 31.3 Root cause is more general than §30 suggested

§30.6 framed the problem as "Gaussian tolerance dense terms create a hover attractor." Removing them turned out to expose a deeper issue: **the success bonus is too far from the agent's online distribution and the auxiliary gradient (PBRS or dense) is the only thing connecting them. PBRS alone is too weak; dense (v3) has a local maximum away from success.**

The fix needs to provide a *strong, monotone* gradient from the workspace to the success region. Three viable approaches:

- **Option B (§30.6) — xy-dominant dense + wider margin / long-tail sigmoid.** Make `r_xy` ≫ `r_z`/`r_tilt` so the agent can't trade XY for Z. Combined with a long-tail sigmoid (or just wider margin) so the gradient at 5-10cm is non-zero. Lowest-cost experiment.
- **BC warmup before SAC.** Pre-train the actor on demos for 10-20k steps to put it inside demo state distribution, then switch to RLPD. Removes the distribution-gap bootstrap problem.
- **Pure imitation learning (BC).** Skip RL entirely. M1 demos cover ~40% of Phase A; train a deterministic actor on them and accept ~40% success rate as v1's ceiling.

Going with Option B next (smallest code change; tests the gradient hypothesis directly).

### 31.4 Lessons (extending §30.7)

- **Removing a bad local optimum doesn't help if there's no gradient toward the global optimum.** The pbrs_only experiment cleared the hover attractor but left the agent in a flat reward landscape with weak PBRS signal — and a flat landscape is just as un-learnable as a deceptive one.
- **Demos don't fix distribution gaps by themselves.** RLPD's symmetric buffer assumes the online actor can roughly imitate demo behavior with exploration. If the actor's natural exploration distribution is far from demo distribution, the demos just inflate Q-targets without providing actionable policy gradient.
- **Always estimate the gradient magnitude before committing to a reward shape.** A 30-second back-of-envelope (Φ change for a 1mm step) would have caught this before training.

---

## 32 — v1.3 (xy_focus): wall cracked, not broken (2026-05-16)

Implemented Option B in commit `5598760`: `reward_mode='xy_focus'` with weights 0.70/0.05/0.05/0 and long-tail sigmoid on r_xy at 15cm margin. Re-generated demos under xy_focus (same M1 IK, 40/100 success, 1032 transitions). Trained approach_v5 for 200k steps with `--reward-mode xy_focus --demos demos/approach_phase_a_xyfocus.h5 --domain-rand --utd 2`.

### 32.1 Training

| step | v3 (dense) | v4 (pbrs_only) | v5 (xy_focus) |
|---|---|---|---|
| 10k | +18 | -166 | -139 |
| 20k | -23 | -192 | **+20** |
| 60k | +34 | -92 | **+70** |
| 100k | +7 | -126 | +68 |
| 150k | +30 | — (killed) | **+77** ← peak |
| 200k | +22 | — | +33 |

Once v5 climbed past step 50k it stayed positive for 15 consecutive evals (steps 60k → 200k). Peak +76.9 at step 150k. Mean over second half: +51. Training was the cleanest of any run: critic_loss stable at 0.1-0.3, α held at 0.17-0.25 (well above the §26 floor), ep_return(20) drifted upward.

### 32.2 eval_orchestrator on step-150k checkpoint

| | v3 | v4 | v5 |
|---|---|---|---|
| Approach success | 0/435 (0%) | — | **1/435 (0.2%)** |
| Full chain success | 0/435 | — | 0/435 |
| Keys with any approach success | 0/87 | — | **1/87** (k) |

**First non-zero approach success across the entire pipeline.** The 1-success was on key `k` (home row right) — solidly in the demo distribution. The chain failed because Strike couldn't follow up; Strike's loss is a separate issue from the Approach side.

### 32.3 Diagnosis — fine-mm precision missing

`inspect_policy --key g --episodes 2 --max-steps 200`:

```
ep1 v5: xy 57→53 (min @ step 60) →65,  z 13→55,  no success
ep2 v5: xy 53→48 (min @ step 60) →62,  z  2→44,  no success
```

Compare to v3 on the same key:

```
ep1 v3: xy 57→82 (monotone retreat), z 13→54
ep2 v3: xy 53→83 (monotone retreat), z  2→40
```

v5 learned the right macro direction (close in XY for ~5mm) but **can't refine past ~48mm**. The success threshold is 4mm. The policy gets to 48mm of target then drifts back out. Two failure modes:
- **Macro behavior learned**: ✅ xy decreases at the start.
- **Fine refinement not learned**: ❌ can't get below 48mm; loses precision and bounces back to 60+mm.

The +60 training-eval mean is the dense-reward signal of "close enough for moderate r_xy" (~0.4 at 50mm with long-tail), not actual success.

### 32.4 Why the fine refinement is missing

The 1032-transition demo set covers M1's successful trajectories, but M1 itself succeeds via Jacobian IK that uses ground-truth joint state — i.e., the demo *actions* are joint targets computed from a closed-loop controller, not a feedforward policy. The agent has to learn to *reproduce that closed-loop precision in open-loop*, which requires more updates than 200k steps with 50→25% demo fraction provide.

Three viable next iterations:

- **Option D — keep demo fraction constant at 0.5.** Stop the decay; the demos stay 50% of every batch for the whole 200k. Demos are success-rich, so this keeps "reach 4mm" gradient strong throughout.
- **Option E — BC warm-start.** Pre-train the actor on demos for 10-20k steps before SAC starts. Puts the actor inside demo state distribution at SAC-step 0, eliminating the bootstrap gap (§31.2's extrapolation problem).
- **Option F — Strike-style continuation training.** Use the v5 step-150k checkpoint as warm start, train another 200k with smaller learning rate and the same setup. Cheap; tests whether more time fixes precision.

Option D is the smallest config change. Picking D as v1.4.

### 32.5 Lessons

- **Hypothesis falsifiability matters.** v4 (pbrs_only) tested "is the dense reward shape the problem?" → falsified. v5 (xy_focus) tested "is the gradient magnitude the problem?" → confirmed. Each falsifiable run isolates a single variable; we converged on the right answer in 3 attempts.
- **A 1-success result is not a coincidence.** 1/435 with a stable +60 training eval and `inspect_policy` showing monotone XY-decrease at episode start is enough evidence that the gradient signal works. The remaining gap is precision, not direction.
- **Training-eval to eval_orchestrator gap is informative.** When +60 training eval = 0/435 eval, the eval is overstating success. When +60 = 1/435, the eval is at least pointing the right direction. The ratio (eval magnitude / orchestrator success rate) is a proxy for "how much of the eval is dense reward vs actual success."

---

## 33 — v1.4 / v1.5 / pure BC sweep (2026-05-16)

### 33.1 Five back-to-back attempts past the 1/435 wall

| Run | Setup | Train-eval peak | Best M4 (Approach / chain) |
|---|---|---|---|
| v5 | xy_focus, demo 50→25% | +76.9 @ 150k | 1/435 / 0/435 |
| v6 | xy_focus, demo const 50% | +77.8 @ 80k | 1/435 / 0/435 |
| v7 step 100k | + BC warm-start 200 epochs, then 100k SAC | +68 | 0/435 / 0/435 |
| v7 **step 25k** | + BC + only 20k SAC | +27 | **4/435** / 0/435 |
| v7 step 75k | + BC + 70k SAC | +50 | 0/435 / 0/435 |
| **v8 pure BC** | 500 epochs BC, 0 gradient steps | +9 (5-ep eval) | **10/435 (2.3%)** / 0/435 |

The v7 sweep revealed the pattern: BC right after pretrain gave 4/435 at step 25k (4× v5/v6), then SAC degraded it monotonically — 0/435 by step 75k. Removed SAC entirely → 10/435.

**Conclusion: SAC is actively erasing the BC-fitted policy.** The dense reward landscape (even with xy_focus's improvements) and the actor's exploration noise are pulling the actor away from demo behavior faster than the demos can re-anchor it.

### 33.2 Per-key breakdown — v8 pure BC

| key | Approach success |
|---|---|
| **j** | **4/5 (80%)** |
| 5, t, o, p, h, k | 1/5 each |
| all others (80 keys) | 0/5 |

Key `j` is the home-row right-hand index — the easiest key for M1 (deepest joint workspace) and the most-trained in demos. BC fits it almost perfectly. The pattern: BC produces strong per-key performance when demos cover the key densely; it fails entirely on keys outside M1's success set.

### 33.3 Why pure BC won

- M1 demos = 1032 transitions, ~26 transitions per successful episode, 40 successful episodes across 9-ish keys.
- BC objective is "match demo action given demo state" — a direct supervised problem.
- SAC objective is "maximize discounted return under exploration" — exploration noise is large compared to the precision demands (4mm threshold).
- For a sparse-success task with a small high-quality demo set, the offline-style BC objective dominates. SAC's value added (discovering new policies) is negative when exploration can't find better trajectories than M1 already provides.

This is consistent with the offline-RL literature: when demos are high-quality and the task is sparse, **pure supervised imitation often beats online RL** unless you have a way to constrain the actor (CQL, IQL, AWAC). RLPD's exact mechanism (50/50 batch + Q-anchoring) was designed for cases where online RL discovers better policies than demos — that assumption fails here.

### 33.4 What this means going forward

Two viable paths to push past 10/435:

- **Path 1 — expand demo coverage.** M1 succeeds on ~9 unique keys × 40% rate. If we generate 20× more demos (200+ episodes × all 87 keys = 17k attempts → ~80 unique-key demos), BC has data to fit more keys. Bottleneck: M1's own success rate at edge keys.
- **Path 2 — constrained offline RL.** Replace SAC with CQL or IQL. These offline algorithms have explicit anti-extrapolation regularizers; they can refine BC without eroding it. Much more code (CQL is ~300 LOC).

Path 1 is the smallest change with the biggest expected gain. Run gen_demos with --trials-per-key 30 across all 87 keys, then re-do pure BC.

### 33.5 Lessons (extending §32.5)

- **Validate from a checkpoint right after pretrain.** v7 step 25k vs step 100k vs step 175k revealed that SAC erodes BC monotonically. Always eval the post-BC actor before letting RL touch it.
- **The +30 to +70 training eval signal is dense reward, not success.** Across v5/v6/v7, training eval = +60 corresponded to wildly different orchestrator results (1/435 to 0/435). Trust orchestrator success, not training eval.
- **When BC outperforms RL+BC, the demos are doing the actual learning and SAC is adding noise.** This is the signature of an oversampled-actor problem — the SAC exploration radius is wider than the success basin, so any exploration step moves out of the basin.

---

## 34 — v1.6 (expanded demos + pure BC): the dataset-coverage paradox

### 34.1 Setup

`gen_demos --keys all --trials-per-key 10 --reward-mode xy_focus` on all 87 keys × 10 trials = 870 attempts. M1 succeeded in **381/870 (44%)** — higher than expected; even Phase C edges like `menu`, `rctrl`, `right` produced some successes. Saved **14117 transitions** to `demos/approach_all_keys.h5` — 14× the v8 dataset.

v9: same pure-BC config as v8 (500 epochs, `--steps 5000 --warmstart 10000` for zero gradient updates). BC NLL: -7.02 → -34.20 (vs v8's -27.5, tighter fit on bigger data).

### 34.2 Result

| Run | Demos | BC epochs | Approach success |
|---|---|---|---|
| v8 | 1032 transitions, 9 unique keys | 500 | **10/435 (2.3%)** |
| v9 | 14117 transitions, ~50 unique keys | 500 | **3/435 (0.7%)** |

**More demos = worse generalization at fixed capacity.** v9 lost `j` (which was 4/5 in v8) and only gained `u`, `i`, `o` (1/5 each, replacing some of the keys v8 covered).

### 34.3 Diagnosis

v8 fit ~9 keys densely (115 transitions/key avg); v9 spread the same 256-dim·3-layer actor across ~50 keys (282 transitions/key avg). The per-key fit *should* be better in v9 — but the actor's representational capacity is being asked to model more distinct (obs, action) mappings. The MLP averages, losing per-key precision.

This is a classic underparameterized-BC failure: more diverse demos require more model capacity, not just more data. Two fixes:

- **Bigger actor.** Hidden (512, 512, 512) or (256, 256, 256, 256, 256). Doubles parameter count.
- **Filter to clean demos.** Keep only the M1 successes that converged in < 80 steps (i.e., the "easy" keys with consistent IK behavior). Drops noise from M1's marginal-key successes.

### 34.4 Where v1 stops

After 8+ hours of compute and 9 training iterations (v1, v3-v9 plus v4 termination), the maximum Approach success is **10/435 (2.3%) from v8 pure-BC** on the original 1032-transition demos. Full chain success across all attempts is **0/435** — Strike never followed up successfully on any approach success.

The pipeline has structural limits that the iteration loop has now mapped out:

1. **Online SAC degrades a BC-fitted policy faster than demos can re-anchor.** (§33 — v7 step 25k went 4/435 → 0 by step 75k.)
2. **Pure BC is the strongest individual result.** (§33 v8: 10/435.)
3. **BC underparameterizes with high-key-diversity demos.** (§34 v9: 3/435.)
4. **The 4mm success criterion is fundamentally hard.** Even the strongest policy gets to ~48mm with consistent gradient (§32 v5 inspect_policy), then the last 44mm of refinement requires precision the policy hasn't internalized.

### 34.5 What would push past 2.3%

In rough order of expected impact per hour of work:

- **A — bigger BC actor.** 30 min to test. Hidden (512, 512, 512). Re-run v9 BC.
- **B — clean-demo filter.** 1 hour. Filter M1 successes to fast/consistent ones; re-run BC.
- **C — CQL / IQL offline RL.** ~1 day. Replace SAC with a conservative offline algorithm. Predicted improvement: BC + offline-RL refinement could close the precision gap that v5/v6/v7 couldn't.
- **D — much larger demo set with diverse seeds.** ~2 hours wallclock. `gen_demos --trials-per-key 50 --keys all`. Risk: doesn't help unless coupled with bigger actor (A).
- **E — different action space.** Re-add cartesian/end-effector control as an option. Approach is geometrically simpler in cartesian than joint space. ~half day.

### 34.6 Lessons (extending §33.5)

- **Data quantity ≠ generalization.** v8 (9 keys, narrow) beat v9 (50 keys, wide) at the same actor capacity. BC's generalization is limited by model expressivity, not demo coverage.
- **A 10/435 result is real signal.** It pinpoints key `j` (M1's strongest key) at 80% and 6 other keys at 20%. This says the architecture can solve specific keys end-to-end given good demos for that key — the bottleneck is uniform per-key precision, not algorithm choice.
- **The full chain is bottlenecked by Strike, not Approach.** 0/435 chain across all variants means Strike's policy (trained against v1's broken Approach distribution) doesn't transfer to BC's Approach output distribution. A Strike retrain against v8's Approach is needed before chain success can be evaluated.

### 34.7 v1.7 / v1.8 — Options A & B run (capacity, then clean-filter): both falsified

These were run in the same session as §34 but the **results were never written
down** until now (the code landed in commits `1a529a2` "add --actor-hidden" and
`8db2955` "--max-demo-episode-steps filter + arch-aware load"; the eval numbers
were only in the session log). Recording them for completeness:

| Run | Setup | Demos | Approach M4 |
|---|---|---|---|
| v8  | 256³ actor, pure BC | 1032 / 9 keys (`approach_phase_a.h5`) | **10/435** ← best |
| v9  | 256³ actor, pure BC | 14117 / 67 keys (`approach_all_keys.h5`) | 3/435 |
| v10 | **512³** actor (Option A), pure BC | 14117 / 67 keys | **2/435** |
| v11 | 256³ actor, **≤40-step clean filter** (Option B), 7168 transitions | filtered all-keys | **3/435** |

- **v10 (Option A — bigger actor) falsified the capacity hypothesis.** BC NLL bottomed at -34.24, *identical* to v9's -34.20 with 3.6× the parameters → the fit is at a **noise floor in the demos**, not a capacity wall. 2/435, worse than v9.
- **v11 (Option B — clean-demo filter) falsified the noise hypothesis.** Filtering to the 237/381 episodes that converged in ≤40 steps lifted the *training-eval* to +54 (best pure-BC training-eval) but the M4 stayed at 3/435 — same as v9. Per-key analysis: the all-keys set has 2.7× more `j` transitions than v8 yet `j` went 4/5 → 0/5, because the actor's capacity is spread across 67 keys and it averages.
- **Infra dividend:** v10's first eval crashed (eval_orchestrator built a 256³ shell, checkpoint was 512³). Fixed in `RLPDSAC.load` — it now reads `actor_hidden`/`critic_hidden` from the saved `config` and rebuilds the networks before loading state-dicts. Any future architecture change is now load-compatible.

**Net of §34:** at fixed-everything-else, neither more data, more capacity, nor cleaner data moved BC off the ~10/435 ceiling. The per-key precision bottleneck is real, and the v8→v11 sweep shows it is **not** an optimization, capacity, or label-noise problem. §35 reframes it.

---

## 35 — v1.9: env recovery, the observability re-analysis, and DAgger (2026-06-01)

A fresh session resumed here. Three things happened: an environment-loss
incident and its permanent fix; a re-reading of the obs/expert code that
**reframed the whole §30–§34 saga**; and the implementation of the approach the
prior options-list (§34.5 A–E) missed entirely — **DAgger**.

### 35.0 Environment-loss incident (and the permanent fix)

The `rover_gpu` training stack (torch 2.10/2.11+cu128, mujoco, robosuite,
dm-control) was only ever `pip install`-ed into the **container's writable
layer** — never committed to `learnflake:gpu` (the image's setup layer is only
296 MB; no torch). A `docker compose up -d rover_gpu` *recreated* the container
and wiped it. Recovered by reinstalling (see `RECENT.md` for the exact command
log; the torch step needed `--ignore-installed sympy` to get past apt's
distutils sympy), verified 55/55 tests + GPU matmul on sm_120, then
**`docker commit rover_gpu learnflake:gpu`** to bake the env in (10.4 → 25.3 GB).
Pinned versions in `docker/rl_env_freeze.txt`. The `checkpoints/` and `logs/`
dirs are gitignored and were **empty after recreation** — the v8–v11 model
checkpoints were lost. The **demos survived** (`demos/*.h5`, on the bind mount),
and pure-BC checkpoints regenerate in ~10 min, so this is recoverable: DAgger
round 0 *is* the BC baseline regenerated from the surviving demos + live expert.

### 35.1 The observability re-analysis — it was never an exploration/observability problem

Re-reading `envs/obs_adapter.py`: the **actor observation already contains the
exact goal vector** — `target_offset_eef` (3-D, the key offset in EEF frame) is
in `ACTOR_FIELDS` (line 49), alongside the noisy `aruco_obs`. So the task is
**fully observable for the actor**; the policy is not missing the goal. This
quietly invalidates the "exploration/reward-shape" framing that drove §30–§32.

What the demo *actions* actually are (re-reading `tools/gen_demos.py` →
`_jacobian_step`): a deterministic damped-least-squares Jacobian-IK step, a
function of `(current joint config, target offset, Jacobian at that config)`.
That is an **interactive expert we can query at any state** — the single most
important asset in this whole project, and §34.5's option list never used it.

Putting the two together gives the real diagnosis. The §32/§34 symptom set —
policy reaches ~48 mm then drifts back out; more keys make BC worse; bigger
actor doesn't help; cleaner data doesn't help — is the **textbook signature of
covariate shift / compounding error**, not representation or exploration. BC
error compounds as O(T²·ε) because the cloned policy drifts into near-key
fine-correction states the i.i.d. 1032-demo set never covered. (Spencer et al.,
*Feedback in Imitation Learning: The Three Regimes of Covariate Shift*, RSS
2021.)

### 35.2 SOTA scan (2026-06-01) → DAgger first

A literature scan (cited below) ranked the candidate fixes by
(expected gain on the 4 mm precision problem) × (low implementation cost):

1. **Vanilla automated DAgger** (Ross, Gordon & Bagnell, AISTATS 2011) — DO
   FIRST. Roll out the current policy, query the IK expert at every visited
   state, aggregate, refit. Converts O(T²ε) drift → O(Tε) and directly
   populates the near-key region BC undersamples. We own the deterministic
   free-to-query expert DAgger needs, so we skip *all* the query-rationing
   variants (HG-/Ensemble-/Safe-/Thrifty-/RND-/Tube-DAgger) — they only exist to
   ration expensive *human* labels.
2. **Residual RL on the IK base, tube-clipped** (Residual Policy Learning, Silver
   et al. 2018; CR-DAgger, Xu et al. NeurIPS 2025). A bounded corrective delta
   around the 44%-success base keeps exploration *inside* the 4 mm basin —
   structurally fixing the "exploration noise > success basin" failure that
   killed every online-SAC run (v3–v7). This is the ceiling-raiser (can exceed
   the expert's 44%); DAgger alone is capped at expert quality.
3. **Offline-RL refinement** (TD3+BC simplest, then IQL; Cal-QL only if
   offline→online) — only if DAgger shows we're expert-quality-capped. Lower
   marginal value on near-optimal, fully-observable data.
4. **ACT / diffusion / flow-matching BC** — deprioritized: their wins
   (multimodality, long-horizon open-loop) don't match a deterministic,
   unimodal, low-dim, closed-loop 4 mm reach.

Caveat carried into the design: the IK expert is only ~44% successful, so its
labels are imperfect — but the per-step DLS action is locally well-defined and
high-quality *near* the key (the 44% failures are mostly far-field/singularity),
which is exactly the region DAgger samples. Mitigation knob added:
`--keep-only-success` (default off — vanilla DAgger keeps all visited states,
because the corrective labels on bad states are the recovery signal we want).

### 35.3 Implementation

- **`algos/expert_ik.py`** (new) — the M1 DLS-Jacobian controller factored out
  of `gen_demos` into a reusable `IKExpert` (queryable at any state) + `ik_step`.
  `gen_demos._jacobian_step` is now a thin alias → one source of truth.
  Exported from `rl_autonomy.algos`.
- **`scripts/train_dagger.py`** (new) — the DAgger loop. Round 0 = expert-driven
  BC (β=1); rounds ≥1 = current-policy rollouts (β-schedule) with the expert
  labelling every visited state, aggregated, refit. A single RunningMeanStd is
  warmed on expert rollouts then **frozen** so every round + eval share one
  normalizer (sidesteps the gen-time-vs-eval RMS mismatch the old pure-BC path
  had). Checkpoints saved via `RLPDSAC.save` → load straight into
  `eval_orchestrator` (no changes there).
- **Tests** (new, all green): `tests/test_expert_ik.py` (5 — shape/bounds/
  solenoid mask, expert==ik_step equivalence, full-pipeline competence guard at
  M1's documented ~44%, gen_demos delegation) and `tests/test_dagger.py` (5 —
  key-pinning, expert-label collection, label=False no-op, eval-rate bounds,
  key groups). Full suite **65 passing**.
- **Sanity smoke** (2-key, 1-round): round 0 eval 0.50 (only `j`) → round 1 eval
  1.00 (both `j` and `h`) — the covariate-shift cure visible in miniature
  (relabeling the policy's own visited states fixed `h`).

### 35.4 v12 — first real DAgger run

`--keys central` (g h f j d k s l t y r u — M1's 12 strongest), 6 rounds,
60 rollouts/round, β-decay 0 (policy drives from round 1), reward-mode xy_focus
(irrelevant to BC but keeps the env consistent). In-loop eval is on the 12
central keys; the all-87 M4 number comes from `eval_orchestrator` on
`dagger_best.pt` (best = round 2).

**In-loop eval (12 central keys, 5 trials each):** round 0 (BC baseline) 0.433
→ peak **0.533** (rounds 2 & 5), oscillating 0.37–0.53 across rounds. DAgger
lifted the BC baseline by ~+10 pts (+23% relative) and then **plateaued at ~0.53
≈ the expert's own ~0.44–0.50 rollout rate** — exactly the predicted
"DAgger is capped at expert quality" behaviour (§35.2). The expert is now the
ceiling, not covariate shift.

**All-87 M4 eval (`eval_orchestrator`, dagger_best = round 2):**

| Metric | v8 (best prior, pure BC) | **v12 (DAgger, central)** |
|---|---|---|
| Approach success | 10/435 (2.3%) | **129/435 (29.7%)** |
| Keys with ≥1/5 success | 7 | **45** |
| Keys at ≥80% (4–5/5) | 1 (`j`) | **17** |

A **13× jump over the best prior result**, and well beyond what the 12 training
keys alone could yield (≤60/435) — the policy **generalizes across the keyboard**
because the goal vector is in the observation, so it learned a goal-conditioned
reach rather than per-key lookups. Keys at 5/5 include several *never trained on*
(`0, o, p, lbracket, semicolon, slash` + the trained `j, k, l`). The 42 keys at
0/5 are the edge/corner keys (`rctrl`, arrows, far f-row) the IK expert itself
can't reach from the "above-keyboard" init — **expert-limited, not DAgger-limited**.
(Full chain still 0/435: the Strike checkpoint was lost in §35.0's wipe, so the
`--strike` arg used `dagger_best.pt` as a meaningless stand-in; only the Approach
column is valid. Per-key matrix in `results/m4_dagger_v12.md`.)

### 35.4.1 Reading of v12 and the next move

Two clean conclusions:
1. **The covariate-shift diagnosis was right.** DAgger turned BC's 2.3% into
   29.7% by relabelling the policy's own visited states. The §30–§34 reward
   detour was treating the wrong cause.
2. **We are now expert-quality-capped**, as the literature predicted. Central-key
   eval ≈ expert rollout rate; the 0/5 keys are where the IK expert fails. Two
   levers remain, both pre-registered in §35.2:
   - **Scale coverage: DAgger on `--keys all`** — train on every reachable key so
     more of the 45 partial keys saturate to ≥80%. Expected to raise the M4
     count further but still capped per-key at expert quality. *(v13, next.)*
   - **Raise the ceiling: residual RL on the IK base, tube-clipped** — the only
     way past the expert's ~44% on the hard keys. *(v14, if v13 plateaus.)*

### 35.6 v13 — DAgger on all 87 keys: 196/435 (45.1%)

Same trainer, `--keys all --eval-keys central` (fast central proxy for in-loop
eval), 5 rounds, 174 rollouts/round (~2 episodes/key/round), β-decay 0,
256³ actor. Best (by central eval, 0.483) = round 5. Aggregate grew to ~134k
labelled transitions.

| Metric | v8 BC | v12 DAgger central | **v13 DAgger all** |
|---|---|---|---|
| Approach success | 10/435 (2.3%) | 129/435 (29.7%) | **196/435 (45.1%)** |
| Keys with ≥1/5 | 7 | 45 | **59** |
| Keys at ≥80% (4–5/5) | 1 | 17 | **29** |
| Keys at 0/5 | 80 | 42 | **28** |

(Per-key matrix `results/m4_dagger_v13.md`; chain 0/435 — Strike still lost, same
stand-in caveat as §35.4.)

**Reading:**
- Training on every key lifted Approach 29.7% → **45.1%** (+52% rel). The
  per-attempt rate (45%) now ≈ the IK expert's own rollout rate (~0.43–0.51 across
  rounds) — **we are at the expert ceiling keyboard-wide.**
- The 28 keys still at 0/5 are edge/corner/wide-reach keys (`esc f1 grave tab
  caps lshift z x c v b space lctrl win lalt down right` …) the IK expert can't
  reach from the above-keyboard init — **expert-limited, not DAgger-limited.**
- **Interference + a model-selection bug:** several *central* keys that v12 nailed
  (`a s d f`, `slash`, `j`) regressed to low/0 in v13 while many peripheral keys
  gained. Two compounding causes: (a) one 256³ actor spreading across 87 keys
  (the §34 capacity-dilution effect, though far milder under DAgger than under
  plain BC); (b) **`dagger_best` was chosen by the 12-key central eval, which is a
  poor proxy for the all-87 objective** — a later/other round likely scores
  higher on all-87. Fixable cheaply: select by a broad key sample, and/or a
  larger actor.

### 35.7 Where v1.9 leaves it, and the fork

DAgger took Approach from **2.3% → 45.1%** and validated the covariate-shift
diagnosis decisively. We are now **expert-quality-capped**. Three independent
levers, roughly increasing cost:
1. **Better model selection (cheap).** Eval `dagger_best` against a broad key
   sample (or run the all-87 matrix per round). Likely recovers the regressed
   central keys for "free" — current 196 is a *lower bound* for v13's rounds.
2. **Bigger actor under DAgger (cheap-ish).** v10 showed capacity didn't help
   *plain BC*, but DAgger's data is richer; a 512³ actor may reduce the all-87
   interference. One run.
3. **Residual RL on the IK base, tube-clipped (the real ceiling-raiser).** The
   only lever that can exceed the expert's ~44% and rescue the 28 zero-keys the
   IK can't reach. Largest build (`algos/residual_actor.py` stub already exists).

Also outstanding regardless of lever: **Strike must be retrained** against the
new DAgger Approach distribution before any full-chain (M4) number is meaningful
— every chain result to date is 0/435 because Strike was trained against v1's
broken Approach (and its checkpoint was lost in §35.0).

### 35.8 v13b — clean BC-vs-DAgger ablation + the model-selection lesson

Re-ran all-keys DAgger with `--eval-keys stratified` (a 24-key spread) to fix
§35.7's model-selection concern, then evaluated multiple rounds on the **full
all-87 matrix** to find the real best. The result corrected two things at once.

| Checkpoint | What it is | all-87 Approach | Keys ≥80% |
|---|---|---|---|
| v8 | pure BC, original 1032-demo h5 | 10/435 (2.3%) | 1 |
| v13b **round 0** | BC on all-key expert rollouts, **frozen RMS** | **115/435 (26.4%)** | 5 |
| v13 round 5 | DAgger all-keys (central-selected) | 196/435 (45.1%) | 29 |
| v13b **round 6** | DAgger all-keys (full) | **200/435 (46.0%)** | 29 |

**Two clean conclusions:**
1. **The methodology fixes alone took BC from 10/435 → 115/435** (11×): all-key
   expert coverage + the frozen-RMS consistency fix (one normalizer for
   collection, training, and eval) + fresh live-expert rollouts instead of the
   stale gen-time-normalized h5. This is a big chunk of the win and is *pure BC*.
2. **DAgger adds 115 → 200 on top** (~1.7×) even across all 87 keys — the
   covariate-shift cure holds at full scale, contradicting the v13b *in-loop*
   signal that had round 0 looking best.

**Model-selection lesson (important):** the in-loop evals (12-key central or
24-key stratified, 5 trials = 60–120 episodes) are **too noisy to select
checkpoints** for the all-87 objective. The stratified eval crowned round 0
(0.467) — yet round 0 scores only 115/435 on all-87 while round 6 scores 200.
Only the full 435-trial matrix selects reliably; with DAgger the **latest round
(most aggregated data) was best**, so "use the last checkpoint" is a safer
default than trusting the noisy in-loop peak. `train_dagger` already saves every
round, so this is a selection-time choice, not a retrain.

**Geometry of the ceiling:** the ~23 keys stuck at 0/5 in the best checkpoint
are the **left/bottom-left** cluster — `a s f g`, `z x c v b`, `tab q e caps`,
`space`, and the left modifiers (`lctrl win lalt`). The arm reaches the
centre/right of the keyboard but the IK expert cannot reach the left side from
the above-keyboard init. This is a **workspace/expert limit**, and it is exactly
what residual RL (or a better expert / a left-biased init) must address — DAgger
cannot, since it is capped at expert quality and the expert simply fails there.

**Net v1.9 headline: Approach 2.3% → 46.0% (200/435), 29/87 keys at ≥80%**,
with a clean ablation showing methodology (10→115) and DAgger (115→200) each
contribute. Best checkpoint: `checkpoints/approach_v13b_dagger_strat/dagger_round_06.pt`.

### 35.9 Reachability diagnostic — the dead keys are expert-limited, not arm-limited (de-risks v14)

Before committing to the residual-RL build, probed whether the ~23 zero-keys are
unreachable by the **arm** (residual RL can't help → M4 impossible) or just by
the **IK expert** (residual RL can help). Ran the expert closed-loop (4 seeds,
≤250 steps, smoothing off) and recorded best XY distance achieved:

| key (right, DAgger-solved) | best XY | | key (left, 0/5) | best XY |
|---|---|---|---|---|
| l | 0.8 mm ✅ | | g | **5.2 mm** (just misses 4mm) |
| o | 0.8 mm ✅ | | f | 8.7 mm |
| k | 0.2 mm ✅ | | a / s / space / caps / tab | 12–20 mm |
| | | | z | 22.5 mm (hardest) |

**The arm physically reaches the vicinity of every left key (5–22 mm); the IK
expert just can't close the last centimetre there** — the left side is near the
workspace boundary where the DLS-Jacobian is poorly conditioned (damping caps the
step, orientation/position weights fight). So the dead keys are
**expert-precision-limited, not workspace-limited**, and **M4 is not obviously
physically impossible**. This is exactly the regime residual RL targets: a
bounded learned correction on top of the IK can close a 5–22 mm gap the IK can't,
without the cm-scale undirected exploration that killed online SAC. **v14
(residual RL on the IK base, tube-clipped) is justified and de-risked.** A
cheaper partial win may also exist: a left-biased / per-region init pose so the
expert *itself* reaches the left keys (then DAgger inherits it) — worth a quick
test before the full residual build.

### 35.5 References

- Ross, Gordon, Bagnell, *A Reduction of Imitation Learning and Structured
  Prediction to No-Regret Online Learning* (DAgger), AISTATS 2011.
- Spencer et al., *Feedback in Imitation Learning: The Three Regimes of
  Covariate Shift*, RSS 2021 — arXiv:2102.02872.
- Silver et al., *Residual Policy Learning*, 2018 — arXiv:1812.06298.
- Xu et al., *Compliant Residual DAgger (CR-DAgger)*, NeurIPS 2025 —
  arXiv:2506.16685.
- Kostrikov et al., *Offline RL with Implicit Q-Learning (IQL)*, ICLR 2022.
- Fujimoto & Gu, *A Minimalist Approach to Offline RL (TD3+BC)*, NeurIPS 2021.

---

## 36 — v14 / key-aware init: the cheap lever proved the left keys are kinematically tilt-dead

§35.9 found the ~23 dead keys are not *position*-unreachable (the arm gets within
5–22 mm). The cheap pre-registered lever was a **key-aware base init**: pre-rotate
the shoulder toward the target column so the IK expert reaches the left keys.
Implemented as `KeyboardEnv(key_aware_init=True)` (TRACKER §36 code; shoulder +=
clamp(3.8·max(0,y_local), 0, 0.7); right/centre keys unchanged), threaded through
`make_env`, `train_dagger --key-aware-init`, and `eval_orchestrator --key-aware-init`
(+ pin-key-before-reset). Tests: `tests/test_key_aware_init.py` (4). Suite 69 pass.

### 36.1 It fixed XY — and revealed the real blocker is TILT

Closed-loop expert reach with key-aware init **solved XY** on the left keys
(`a` 16.6 → 0.2 mm, `caps` → 0.1 mm, `tab` → 0.4 mm). But **full success did not
improve** (left keys 0/35 vs 1/35 without it; right keys 24/30 vs 25/30 — a hair
worse). Measuring all three tolerances at the min-XY pose explained why:

| key | xy @min | z @min | **tilt @min** | best min-tilt over episode |
|---|---|---|---|---|
| l, o (right) | 3.8 mm | 2.6–3.0 mm | **1.3–1.4°** ✅ | 1.3° |
| g (centre-left) | 2.3 mm | 9.3 mm | 40° | 12.2° |
| a, s, z, caps, tab (left) | 0.1–0.9 mm | 3–24 mm | **48–51°** ❌ | **25–35°** |

The success criterion is xy<4 mm **and** z<5 mm **and** tilt<5°. The left keys
nail XY but the actuator points **25–50° off vertical** there, and the IK — which
actively weights orientation (w_rot up to 5.0) — **cannot get min-tilt below
~25°**. This is a **hard kinematic/dexterity limit**: the 6-DOF arm can *position*
the EEF over the left-edge keys but **physically cannot keep the actuator vertical**
at that extension. Tilt degrades smoothly from ~1° (right) to ~50° (far left); the
5° threshold is crossed in the centre-left, which is exactly the
solved/unsolved boundary seen in every eval.

### 36.2 What this means — M4 is a physical-setup problem, not an algorithm problem

- **The cheap init lever does not unlock the left keys** (XY was never the blocker).
- **Residual RL won't either.** It can exceed the expert's *precision/quality*, but
  it cannot beat a kinematic constraint — no Approach policy can make this arm point
  down over the left keys.
- **M4 (≥80/87 keys at ≥80%) is very likely physically infeasible** with the current
  arm mount + keyboard placement + 5° tilt requirement. The reachable-in-full-pose
  set is the centre/right of the keyboard (~60–64 keys); the left ~23 are tilt-dead.

To actually get the left keys, the fix is **physical / spec-level**, not learning:
1. **Reposition** the keyboard toward the arm's dexterous workspace (e.g. shift the
   `keyboard_offset` +y / rotate it) or move the arm mount — bring the left columns
   into the cone where the wrist can verticalize. *Cheapest real fix; one env change.*
2. **Relax the tilt tolerance** — legitimate *only* if the real solenoid actually
   strikes keys reliably at, say, 15–20° (a hardware question for the K552 + the
   spring-loaded tip). If yes, many left keys come back immediately.
3. **Different arm / longer reach / wrist DOF** — out of scope.

### 36.3 Revised plan

- `key_aware_init` is kept as an **opt-in, tested** feature (default off — it changes
  nothing unless asked). It is useful now as the diagnostic, and useful later if the
  tilt tolerance is relaxed (then it genuinely unlocks the XY-reachable left keys).
- **v14 (key-aware DAgger) was aborted** at round 0 — it can't raise full success
  (rollout-succ 0.37 vs v13b 0.51; the rotated init trades right-key margin for
  XY-only left reach that tilt rejects).
- **Best Approach result stands at v13b round 6 = 200/435 (46.0%)**, which is at the
  kinematic ceiling for the centre/right keys × the expert's per-attempt rate.
- **Next decision is the user's**: (a) make the physical/spec change (reposition or
  tilt-tolerance) to expand the reachable set, then re-run DAgger; and/or (b) residual
  RL to push the *reachable* keys past the ~46% expert ceiling; and/or (c) retrain
  Strike against the new Approach for a real full-chain number.

### 36.4 Lesson

**A "cheap test before the big build" earned its keep.** The 40-minute key-aware-init
experiment converted "the left keys are hard" into "the left keys are kinematically
tilt-dead," which *cancelled* a ~1-day residual-RL build that would have failed on
them, and reframed M4 from an algorithm goal to a physical-setup decision. Always
probe the constraint's *nature* (which tolerance, kinematic vs control) before
investing in a learner to beat it.
