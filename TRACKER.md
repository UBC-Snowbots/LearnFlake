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
| Action space | `JOINT_VELOCITY` (raw 6 joint vels + 1 solenoid) | **OSC_POSE** Cartesian delta (Δx,Δy,Δz,Δrx,Δry,Δrz) + 1 binary solenoid; arm tracks impedance, not raw velocity |
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

## 6. Action space — OSC_POSE + binary solenoid

### 6.1 Cartesian impedance via robosuite OSC_POSE

```python
arm_cfg = suite.load_part_controller_config(default_controller="OSC_POSE")
ctrl_cfg = refactor_composite_controller_config(arm_cfg, "Rover2026", ["right"])
```

- Action: `(dx, dy, dz, droll, dpitch, dyaw) ∈ [−1,1]^6`, scaled to `(±2 cm, ±2 cm, ±2 cm, ±0.1 rad, ±0.1 rad, ±0.1 rad)` per control tick at 20 Hz. These limits cap end-effector velocity at ~40 cm/s and ~2 rad/s — comfortable for the Rover2026.
- Impedance: default fixed Kp/Kd from robosuite. We expose `impedance_mode='variable_kp'` as a future ablation; not enabled in v1.
- Why OSC and not IK: OSC computes joint torques from a Cartesian goal under an inertia model — it's compliant (small forces don't cause large torques), which is exactly what we need when the actuator tip touches the keyboard. Pure IK is rigid; the arm fights any external force.
- Why not impedance directly: OSC *is* impedance in disguise; same math.

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
| OSC Kp gain | ×[0.7, 1.3] | no |
| OSC Kd gain | ×[0.7, 1.3] | no |
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

### Phase 0 — Hygiene (1 day)
- Pin `gymnasium` only (drop `gym==0.23.0`). Pin `stable-baselines3==2.5.0`, `sb3-contrib==2.5.0`, `dm_control>=1.0.20`, `mujoco>=3.3.0`. Bump `requirements.txt`.
- Delete `rl_agent/`, `rl_agent_pranav/`, `skills/{press,reach,traverse,retract}/{train_coarse,train_fine,train_domain_rand}.py`, `skills/hrl/`, `testing/{train_lift_v2,train_lift_v3,bc_to_rl,bc_train,bc_trainer,phase1_verify,phase2_debug,RLAgent,memory,trainer,agent,networks,test,main,simulations}.py`. Move `keyboard_demo.py`, `demo_recorder.py`, `cartesian_keyboard.py`, `env_diagnostics.py`, `mujoco_joint_states.py` into `tools/` (kept).
- Single Python module path: `from rl_autonomy.* import *`. Drop the `sys.path.insert` shim from every script.
- **Artifact**: `pip install -e .` works; `pytest tests/smoke.py` passes (`tests/smoke.py` just imports the package and runs a single env step).

### Phase 1 — Env rewrite (3 days)
- New `rl_autonomy/envs/keyboard_env.py` (replaces current). Single class: `KeyboardEnv`. No skill subclasses. Skill differences are encoded as a `mode: Literal["approach","strike"]` constructor arg that selects: action mask, success criterion, episode horizon, reward function.
- Action: 7-D `OSC_POSE` + solenoid as in §6.
- Observation: §9. Same flat dict observation space; the `gymnasium.Wrapper`s assemble actor obs and critic obs separately on demand.
- Reward: §5 via `dm_control.utils.rewards.tolerance` (add to requirements).
- DR: rewrite `DomainRandWrapper` to subclass `gymnasium.Wrapper` properly and cover §10. Per-step axes implemented in `step`, per-episode in `reset`.
- Demo loader: `rl_autonomy/data/demo_buffer.py` reads HDF5 demos from `demos/` and exposes a `sample(batch_size)` method.
- **Artifact**: `python -m rl_autonomy.tools.env_diagnostics --mode approach --steps 200` runs and prints labelled obs values; success rate of zero-action policy is 0%; success rate of a hand-coded P-controller toward target is ≥ 90%.

### Phase 2 — Algorithm port (3 days)
- `rl_autonomy/algos/rlpd_sac.py`: SB3 SAC subclass that adds (a) LayerNorm critic, (b) symmetric demo+online sampling in the replay buffer, (c) UTD=10 update loop, (d) wider critic (512), (e) privileged critic input.
- `rl_autonomy/algos/bc_pretrain.py`: trivial BC trainer described in §8.2.
- `rl_autonomy/algos/residual_actor.py`: wraps a frozen BC actor, exposes a residual SAC actor, sums their outputs.
- Hyperparameters live in a single `rl_autonomy/configs/rlpd_sac.yaml`. No more scattered `sac_kwargs=dict(...)` per-script.
- **Artifact**: train RLPD-SAC on the standard `Pendulum-v1` for 50k steps and reach return ≥ −150 within 5 minutes on this machine. Sanity check that the algo works *before* we point it at the keyboard env.

### Phase 3 — Curriculum + training script (2 days)
- `rl_autonomy/curricula/state_replay_curriculum.py` (DemoStart-style) and `rl_autonomy/curricula/key_phase_curriculum.py` (manual phases).
- `rl_autonomy/scripts/train_approach.py` and `rl_autonomy/scripts/train_strike.py`. Each is ~80 LOC. They wire env + curriculum + algo + logger.
- WandB logging by default (group=`approach-v1`, name=git-sha+timestamp); fall back to TensorBoard.
- **Artifact**: `python -m rl_autonomy.scripts.train_approach --steps 500_000 --num-envs 8` finishes overnight on this box; eval success rate ≥ 0.85 on Phase B keys.

### Phase 4 — Strike + integration (1 day)
- Strike is a much smaller MDP (~50-step horizon, 1-D action). Train for 100k steps.
- `rl_autonomy/scripts/eval_orchestrator.py`: deterministic chain Approach → Strike → done; runs across the full 87 keys, prints success matrix.
- **Artifact**: 87-key success matrix where ≥ 80 keys succeed.

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

```
src/rl_autonomy/
├── __init__.py
├── pyproject.toml                       # NEW: package metadata + deps (replace requirements.txt)
├── configs/
│   ├── rlpd_sac.yaml                    # algorithm hparams
│   ├── env_keyboard.yaml                # env hparams (control_freq, horizon, DR ranges)
│   └── curriculum.yaml                  # phase boundaries
├── envs/
│   ├── __init__.py
│   ├── keyboard_env.py                  # NEW: single class, mode-switched
│   ├── action_adapter.py                # OSC_POSE + solenoid wrapping
│   ├── obs_adapter.py                   # actor/critic obs builders, EEF-frame conversion
│   ├── domain_rand.py                   # proper gym.Wrapper, full DR axes
│   └── normalizer.py                    # RunningMeanStd
├── algos/
│   ├── __init__.py
│   ├── rlpd_sac.py                      # SAC + LayerNorm + symmetric replay + UTD=10
│   ├── bc_pretrain.py                   # BC trainer
│   ├── residual_actor.py                # frozen base + learnable residual head
│   └── networks.py                      # MLP, GELU, LayerNorm critics
├── curricula/
│   ├── state_replay_curriculum.py
│   └── key_phase_curriculum.py
├── data/
│   ├── demo_buffer.py
│   └── replay_buffer.py                 # symmetric-sampling buffer
├── scripts/
│   ├── train_approach.py
│   ├── train_strike.py
│   ├── eval_orchestrator.py
│   └── render_rollout.py
├── bridge/
│   ├── synthetic_moteus_node.py
│   └── policy_node.py
├── tools/
│   ├── env_diagnostics.py               # ex testing/env_diagnostics.py, kept
│   ├── demo_recorder.py                 # ex testing/, kept
│   ├── cartesian_keyboard.py            # ex testing/, kept
│   ├── keyboard_demo.py                 # ex testing/, kept
│   └── mujoco_joint_states.py           # ex testing/, kept
├── documentation/
│   └── keyboard_typing_pipeline.md      # updated to reflect this rewrite
├── tests/
│   ├── smoke.py
│   ├── test_env_observation_shapes.py
│   ├── test_action_adapter.py
│   └── test_reward_bounds.py
├── demos/                               # checked in only if small (HDF5 ~100 MB)
│   └── approach_v1.h5
├── checkpoints/                         # gitignored
└── logs/                                # gitignored
```

What's gone (under Phase 0):
- `rl_agent/`, `rl_agent_pranav/` — entirely.
- `skills/` — replaced by `scripts/` + `envs/`.
- `testing/` — useful pieces moved to `tools/`, the rest deleted.
- `rl_agent_base/rklb/rlkb_framework/` — superseded by the new structure (it was a half-built skeleton anyway: `BaseAgent`, `Runner`, `MockKeyboardEnv`).

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

### M1 — Env correctness (end of Phase 1)
- Hand-coded P controller in EEF space achieves ≥ 90% Approach success across 20 randomly sampled keys with DR off.
- Env step rate ≥ 100 Hz × 8 envs on this machine.
- Critic obs and actor obs have non-overlapping privileged channels (assertion in tests).
- All reward components ∈ [−1, 2] across 1 M random rollouts.

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
