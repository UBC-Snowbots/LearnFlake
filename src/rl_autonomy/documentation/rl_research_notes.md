# RL for Robotic Manipulation — Research Notes

> Compiled 2026-04-29 during the initial research phase of the LearnFlake `rl_autonomy/` rewrite.
> Audience: a future implementer working on this project who wants a single self-contained reference.
> Scope: roughly 2015–2026, biased toward what's directly applicable to a 6-DOF arm + solenoid pressing keys on a keyboard.
> See `TRACKER.md` §3-§11 for which of these ideas are actually wired into v1, and §19 for why we picked JOINT_POSITION over OSC_POSE on the Rover2026 specifically.

---

## Table of contents

1. [Foundational algorithms (2014–2020)](#1-foundational-algorithms-20142020)
2. [Modern model-free continuous control (2020–2024)](#2-modern-model-free-continuous-control-20202024)
3. [Imitation learning + RL hybrids](#3-imitation-learning--rl-hybrids)
4. [Modern manipulation policies — diffusion, transformer, VLA](#4-modern-manipulation-policies)
5. [Sim-to-real transfer](#5-sim-to-real-transfer)
6. [Reward design](#6-reward-design)
7. [Curriculum learning](#7-curriculum-learning)
8. [Action spaces and low-level control](#8-action-spaces-and-low-level-control)
9. [Observation design](#9-observation-design)
10. [Hierarchical RL and primitives](#10-hierarchical-rl-and-primitives)
11. [Contact-rich manipulation specifically](#11-contact-rich-manipulation-specifically)
12. [Training infrastructure and GPU simulators](#12-training-infrastructure-and-gpu-simulators)
13. [RTX 50-series (Blackwell / sm_120) compatibility notes](#13-rtx-50-series-compatibility)
14. [Practical recipes — what we use, what we considered](#14-practical-recipes)
15. [Glossary](#15-glossary)
16. [Annotated bibliography](#16-annotated-bibliography)

---

## 1. Foundational algorithms (2014–2020)

The "foundation models" of continuous-control RL. Everything modern descends from one of these.

### 1.1 DDPG — Deep Deterministic Policy Gradient (Lillicrap et al. 2015)

- Off-policy actor-critic for continuous actions. Actor is **deterministic**: π(s) → a. Critic is Q(s,a) trained on Bellman targets.
- Exploration via additive Gaussian or Ornstein-Uhlenbeck noise on the actor output.
- Notoriously brittle. Mainly historical interest now; it's the great-great-grandparent of everything below.

### 1.2 TD3 — Twin Delayed Deep Deterministic Policy Gradient (Fujimoto et al. 2018)

- Three improvements over DDPG that fixed most of its instability:
  1. **Twin critics** (`Q_φ1, Q_φ2`), take min of the two for the Bellman target → reduces overestimation.
  2. **Delayed policy update**: actor updates every `d=2` critic updates → stabilizes training.
  3. **Target policy smoothing**: add Gaussian noise to the target action when computing the target → prevents the critic from over-fitting sharp peaks in Q.
- Still deterministic-actor + Gaussian exploration. Replaced by SAC for most modern work.
- Codebases: cleanrl, sb3.

### 1.3 SAC — Soft Actor-Critic (Haarnoja et al. 2018)

- The default for continuous-control RL since ~2019. Builds on TD3 but with a stochastic actor.
- Maximum-entropy RL: optimize `E[Σ_t γ^t (r_t + α H(π(·|s_t)))]`. The entropy bonus pushes the policy to be as random as possible while still solving the task — drastically improves exploration.
- **Auto-tuned temperature α**: keeps entropy near a target value `H_target = -|action_dim|` (or `-0.5 * |action_dim|` in some recipes).
- Twin critics from TD3, soft target update with Polyak coefficient τ ≈ 0.005.
- Actor outputs `(μ, σ)` then samples `a = tanh(μ + σ ⊙ ε)` — the tanh bounds actions to [-1, 1].
- Hyperparameters that "just work": lr=3e-4 for actor/critic/temperature, batch=256, γ=0.99, τ=0.005, replay buffer ~1M, gradient_steps=1 per env step.
- Reference: the original `haarnoja/sac` (TF), `pranz24/pytorch-soft-actor-critic` for PyTorch, sb3's SAC for production.
- **In LearnFlake**: this is the backbone of our chosen algorithm (RLPD-SAC, see §2.4).

### 1.4 PPO — Proximal Policy Optimization (Schulman et al. 2017)

- On-policy actor-critic. Optimizes a clipped surrogate objective that prevents the policy from changing too much per update.
- Strengths: simple, very stable, parallelizes trivially across many envs, great for GPU sims (Isaac Lab uses PPO almost exclusively).
- Weaknesses: sample-inefficient compared to off-policy methods because data can't be reused. Bad fit for real-robot training.
- Hyperparameters: lr=3e-4, n_steps=2048, n_epochs=10, batch_size=64, clip_eps=0.2, entropy_coef=0.0, gamma=0.99, gae_lambda=0.95.
- Used heavily in NVIDIA's Factory/IndustReal pipeline, RoboSuite benchmarks at scale, all of Isaac Gym.

### 1.5 TRPO — Trust Region Policy Optimization (Schulman et al. 2015)

- The conceptual ancestor of PPO. Constrains the policy update to lie within a KL-divergence trust region. Mathematically clean but expensive to implement.
- PPO's clipping is a cheap approximation; nobody uses TRPO directly anymore.

### 1.6 HER — Hindsight Experience Replay (Andrychowicz et al. 2017)

- For *goal-conditioned* RL with sparse reward.
- Idea: when a trajectory fails to reach goal `g`, relabel it as having succeeded at reaching whatever final state `s_T` was actually achieved. Now you have a positive-reward example.
- Goal selection strategies: `final` (always relabel to s_T), `future` (relabel to a random later s in the same episode), `episode` (random s in episode). `future` is the standard.
- HER × n_sampled_goal=4 means 4 relabeled transitions per real transition in the replay buffer.
- Recent variants: MHER (multi-step), 2HER (relabels both effector and object goals), PHER (prioritized), RHER (relay HER for sequential tasks).
- **In LearnFlake**: not used in v1. Our reward is dense via tolerance shaping, so HER's sparse-reward use case doesn't apply. Listed for completeness in case we want goal-conditioned multi-task variants later.

### 1.7 DAPG — Demonstration-Augmented Policy Gradient (Rajeswaran et al. RSS 2018)

- Combine RL with a small set of demos (typically 25 trajectories of dexterous manipulation).
- Two effects:
  1. **BC initialization**: pretrain the policy on demos with behavior cloning before RL starts.
  2. **Auxiliary loss during RL**: every gradient step adds a behavior-cloning term on demonstration actions, weighted by a decaying coefficient `λ_t = λ_0 * γ^t`.
- The `aravindr93/hand_dapg` repo became the standard reference for "RL with a few demos"; built on NPG (natural policy gradient), not SAC.
- Modern descendants: AWAC, RLPD, IBRL, DPPO. We use RLPD's "symmetric demo+online sampling" instead, which is cleaner.

### 1.8 Asymmetric Actor-Critic (Pinto et al. RSS 2018)

- Train the **critic** with privileged state (everything in the simulator: object positions, ground-truth contact forces, mass values). Train the **actor** with only deployable observations (RGB images, joint encoders, noisy sensors).
- The critic learns more accurate value estimates because it sees the truth; the actor never depends on what won't be available at deployment.
- Cheap and dramatic improvement (~3× faster convergence in their experiments).
- Used by OpenAI Dactyl, IndustReal, and pretty much every sim-to-real recipe since 2018.
- **In LearnFlake**: yes — TRACKER §4.2 critic gets ground-truth EEF-to-key vector, contact force, solenoid extension, and current DR knob settings. Actor only gets the deployable observation.

### 1.9 The pre-2020 takeaway

By 2020 the recipe for "decent continuous control on MuJoCo" was:

> SAC + twin critics + Polyak target + auto-temperature + HER if sparse reward + DAPG-style demos if expensive simulation.

Everything below is "what got better since."

---

## 2. Modern model-free continuous control (2020–2024)

The big improvements since 2020 are about **sample efficiency** and **stability**: how do you get good performance in fewer environment steps and with less hyperparameter tuning?

### 2.1 DrQ-v2 — Image-based off-policy RL (Yarats et al. 2021)

- For pixel-based continuous control. Builds on DrQ which adds simple image augmentations (random shift) when sampling from the replay buffer.
- DrQ-v2: improvements include n-step returns, refined exploration schedule, lower batch size. Solves complex humanoid locomotion from pixels — a milestone.
- Throughput ~96 FPS on a single V100. Most DM-Control tasks finish in 8 hours.
- **In LearnFlake**: not used. We synthesize the aruco signal from MuJoCo ground truth instead of rendering pixels (see TRACKER §11.4). DrQ-v2 is the right tool if and only if we ever go end-to-end pixel.

### 2.2 REDQ — Randomized Ensembled Double Q-Learning (Chen et al. ICLR 2021)

- **High update-to-data ratio (UTD=20)**: 20 critic gradient updates per environment step.
- Ensemble of 10 critics, but only sample 2 randomly per Bellman target → low variance, low bias.
- Sample efficiency 5–10× better than SAC on Mujoco.
- Cost: 20× compute per env step. Worth it when sim is slow or you need fewer real-world rollouts.

### 2.3 DroQ — Dropout Q-Functions (Hiraoka et al. 2022)

- Same UTD=20 idea as REDQ, but uses **dropout + LayerNorm** in the critic instead of an ensemble. Achieves REDQ performance at 1/5 the compute.
- Two critics with `dropout_rate=0.01` on every hidden layer + LayerNorm.
- Now folded into the SBX (`stable-baselines-jax`) library; available as a drop-in for SAC.

### 2.4 RLPD — Reinforcement Learning with Prior Data (Ball et al. NeurIPS 2023)

- Three changes on top of SAC:
  1. **High UTD ratio** (10–20).
  2. **LayerNorm on critic hidden layers** — a stability lifesaver.
  3. **Symmetric demo+online sampling**: every batch is half from a demo replay buffer, half from online replay. Demos seed exploration; online data refines the policy.
- Trains from scratch *or* from prior data. SERL (next entry) uses RLPD as its core algorithm.
- **In LearnFlake**: this is our v1 algorithm. TRACKER §3.1 spec.

### 2.5 SERL & HIL-SERL — Sample-Efficient Robotic RL (Luo et al. 2024)

- A **system**, not just an algorithm. Combines:
  - RLPD as the RL backbone
  - JAX-based actor-learner architecture (separate processes for data collection and gradient updates, communicating asynchronously)
  - VICE-style learned reward classifier (RGB images → success probability)
  - Compliance controller that lets the robot move safely even with random actions
- HIL-SERL adds a **human-in-the-loop correction phase**: when the policy fails, a human teleop-guides the correct action; that demo is added to the buffer.
- 100% success rate on multi-stage tasks (RAM insertion, USB pickup, egg flip) within 15–60 minutes of real-world wall-clock training. Best published "RL on a real robot" results.
- Repos: `rail-berkeley/hil-serl` (JAX) and `huggingface/gym-hil` (lo-fi simulator).
- **In LearnFlake**: HIL-SERL is the *real-world fine-tune fallback* if v1 sim-to-real transfer underperforms (TRACKER §11.3). For v1 sim-only we use RLPD only.

### 2.6 CrossQ — Batch Normalization in Deep RL (Bhatt et al. ICLR 2024)

- Replaces target networks with **BatchNorm in the critic**. Training stability comes from BN, not from the slowly-updated target.
- Achieves REDQ-level sample efficiency at **UTD=1** (no ensemble, no high update ratio).
- Wallclock-fastest model-free recipe as of 2024. Implementation: 5 lines on top of SAC.
- Caveats:
  - BN's running stats can drift in off-policy training; the paper handles this carefully.
  - Doesn't compose well with frame stacking (stats interact across stacked frames).
- Repos: `adityab/CrossQ`, `sb3-contrib`, `sbx`.
- **In LearnFlake**: not used in v1; listed as the explicit fall-back if RLPD's UTD=10 burns too much GPU time on this hardware (TRACKER §3.1, §3.2).

### 2.7 TQC — Truncated Quantile Critics (Kuznetsov et al. ICML 2020)

- Critic predicts a *distribution* over Q-values via quantile regression (like QR-DQN), then drops the top `d` quantiles before averaging → controls overestimation bias.
- Uses 5 critics, 25 atoms each; drop top 2–5 atoms per critic.
- Outperforms SAC by 10–25% on the harder benchmark suites (Humanoid in particular).
- Available in `sb3-contrib` as a drop-in.
- **In LearnFlake**: not used. Worth trying as an ablation if RLPD plateaus.

### 2.8 BRO — Bigger, Regularized, Optimistic (Nauman et al. NeurIPS 2024)

- Three findings:
  1. **Critic scale is the bottleneck**, not data efficiency. Bigger critic networks (with strong regularization) consistently improve performance.
  2. **Strong regularization** (LayerNorm + sparse weight init + smooth activations) lets you scale.
  3. **Optimistic exploration** (KL-bonus toward a maximum-entropy reference policy) closes the loop.
- 4× sample efficiency over SAC, ~400% better wallclock performance on hard locomotion tasks.
- Repo: `naumix/BiggerRegularizedOptimistic` (JAX). PyTorch port not yet mainstream.
- **In LearnFlake**: future ablation if RLPD-SAC underperforms.

### 2.9 The 2024 sample-efficiency hierarchy

| Recipe | UTD | Critic | Sample efficiency vs SAC | Wallclock vs SAC |
|---|---|---|---|---|
| SAC (vanilla) | 1 | twin, no LN | 1× | 1× |
| TQC | 1 | 5 quantile critics | 1.2× | 1.5× |
| REDQ | 20 | 10 critics, sample-2 | 5× | 8–20× |
| DroQ | 20 | 2 critics + dropout + LN | 5× | 5× |
| RLPD | 10 | 2 critics + LN + demos | 4× (from scratch) — 10× (with demos) | 3× |
| CrossQ | 1 | 2 critics + BN, no target | 4× | 1.2× |
| BRO | 2 | bigger + LN + optim. | 5–10× | 2–4× |

Recommended default in 2026: **RLPD** if you have demos, **CrossQ** if you want minimal compute and simplest code, **BRO** if you have headroom and want SOTA sample efficiency.

---

## 3. Imitation learning + RL hybrids

Pure RL from scratch on a real robot is glacial. Pure imitation learning from demos doesn't generalize. Hybrids dominate the modern manipulation literature.

### 3.1 Behavior cloning baselines

- **BC**: supervised learning on (state, action) pairs from demos. Surprisingly strong for short-horizon tasks; falls apart on compounding errors over long horizons (covariate shift / DAgger problem).
- **DAgger** (Ross et al. 2011): query the expert at states the policy reaches; train on this online dataset. Mostly impractical for robots.
- **Action chunking** (ACT, see §4.2): predict an action *sequence* of length k=10–100; compounding error per chunk is much lower than per-step.

### 3.2 IL→RL initialization

- **AWAC** (Nair et al. 2020): off-policy actor-critic with KL constraint to a behavior-cloning prior. Useful for finetuning a BC policy with RL.
- **IQL — Implicit Q Learning** (Kostrikov et al. 2021): purely off-policy; the actor never sees its own actions, only demo actions, weighted by advantage. Excellent for offline RL → online RL transitions.
- **CQL — Conservative Q Learning**: regularizes Q to be conservative on out-of-distribution actions. The other big offline RL workhorse.

### 3.3 Residual RL — the modern winning recipe

- Idea: train a base policy (BC or hand-coded controller), freeze it, learn a *residual* on top: `a_total = a_base(o) + a_residual(o)`. The residual is the only thing being optimized.
- Why it wins:
  - Base provides a reasonable trajectory from the start → exploration is bounded.
  - The residual can be very small (low capacity, easy to fit).
  - Safe to deploy: if the residual is zero, you're back to the base policy.

Specific instances:
- **Residual Policy Learning** (Silver et al. 2018): the original. Base = MPC or hand-designed; residual = SAC.
- **ResiP — From Imitation to Refinement** (Ankile et al. 2024): base = action-chunked diffusion BC, residual = PPO. Improves a 0.2 mm peg-in-hole task from 5% → 99% success.
- **Residual Off-Policy RL for Finetuning BC Policies** (Yang et al. 2025): base = any BC policy (treated as black box), residual = SAC. Real-world peg insertion in tens of minutes.

**In LearnFlake**: TRACKER §8.3. Demos are skipped for v1 per user direction; this is the upgrade path for v1.1 if needed.

### 3.4 The replay-buffer trick (RLPD's contribution)

Instead of full BC pretraining → RL fine-tune, just put demos in a separate replay buffer and sample 50/50 demo:online in every gradient batch. The actor learns to imitate demos *and* refine via reward concurrently. Less code, fewer failure modes than BC→RL.

We use this in v1's algorithm (TRACKER §3.1).

### 3.5 GAIL / AIRL / VICE — learned-reward IL

- **GAIL** (Generative Adversarial Imitation Learning, Ho & Ermon 2016): a discriminator distinguishes demo states from policy states; the policy is rewarded for fooling it.
- **AIRL** (Fu et al. 2018): GAIL with explicit reward function — usable for downstream RL.
- **VICE** (Variational Inverse Control with Events, Fu et al. 2018): learn a binary success classifier from positive examples only, use its log-prob as reward.
- HIL-SERL uses a VICE-style classifier as the success signal — way easier than hand-engineering.

**In LearnFlake**: hand-engineered tolerance reward in v1; VICE as a fallback if reward design proves intractable.

---

## 4. Modern manipulation policies

This section is about the **policy parameterization** itself — what kind of neural network maps observations to actions.

### 4.1 The classical MLP actor

- 2-3 hidden layers, 256 units, ReLU/GELU, tanh-squashed Gaussian head. Used by every SAC paper and basically every robosuite benchmark.
- Pros: trivial to implement, fast, well-understood, sufficient for medium-horizon tasks.
- Cons: bad at multi-modal action distributions (pick up cup vs hand it over — actor can only learn the average).

### 4.2 ACT — Action Chunking with Transformers (Zhao et al. RSS 2023)

- Imitation learning method for ALOHA bimanual manipulation.
- Architecture: ResNet image encoder → Transformer encoder (multi-camera + joint pos) → Transformer decoder → action chunk of length k=100.
- **The key trick**: predict 100 actions ahead, execute the first ~30, then re-plan. Reduces effective horizon from 1000 to 10.
- Conditional VAE on top of the transformer to model multi-modal demonstrations.
- 80–90% success on real bimanual tasks (battery insertion, condiment cup) from 50 demos.
- Repo: `tonyzhaozh/aloha`, plus PyTorch reimplementations.

### 4.3 Diffusion Policy (Chi et al. RSS 2023, IJRR 2024)

- Represent the policy as a conditional denoising diffusion model. Action `a` is sampled by iteratively denoising Gaussian noise, conditioned on observations.
- **Properties that matter**:
  - Models multi-modal action distributions natively (mixture without explicit mode count).
  - Predicts action *chunks* like ACT, not single steps.
  - Robust to demonstration noise.
- Two backbones: U-Net (Planning-with-Diffusion-style 1D U-Net) and Transformer (MinGPT-style).
- Average +47% improvement over the previous SOTA across 12 manipulation benchmarks.
- Hyperparameters that matter: `n_obs_steps=2`, `horizon=16` (predict 16 actions), `n_action_steps=8` (execute first 8), `n_diffusion_iters=100`.
- Repo: `real-stanford/diffusion_policy`. Workspace pattern: Hydra configs, EMA model averaging, LinearNormalizer.
- **In LearnFlake**: not in v1. The 1M-parameter diffusion model needs ~hundreds of demos; we have 0. ACT/Diffusion would be the v2 architecture if we collect demos.

### 4.4 DPPO — Diffusion Policy Policy Optimization (Ren et al. ICLR 2025)

- The first reliable recipe for **fine-tuning** a diffusion policy with RL (PPO).
- Key insight: treat the multi-step denoising chain as part of the policy structure; PPO optimizes the noise schedule and final-action distribution jointly.
- Outperforms IDQL, DIPO, QSM, and DQL on Robomimic and Furniture-Bench.
- Pretrain on demos with diffusion BC → fine-tune with PPO.
- Repo: `irom-princeton/dppo`.

### 4.5 Foundation models for robotics — VLA

- **RT-1** (Google 2022): a transformer that takes images + language → action tokens. Pretrained on 130k demonstrations across 13 robots.
- **RT-2** (Google 2023): adds vision-language pretraining → action understanding.
- **OpenVLA** (Kim et al. 2024): open-source 7B-parameter VLA, fine-tunable.
- **π0** (Physical Intelligence 2024): largest VLA to date, deployed on multiple platforms.
- **Octo** (Octo Model Team 2024): smaller (93M), open, trained on the Open X-Embodiment dataset.

**In LearnFlake**: not in v1. VLAs need datasets in the GB-to-TB range. Our 87-key task is much narrower than what VLAs are good at — a 200k-parameter MLP is sufficient. Listed for awareness.

### 4.6 Recurrent policies (LSTM/GRU/Transformer)

- For partially observable MDPs (POMDPs). The policy maintains a memory state across timesteps.
- DRQN (2015) is the classic; modern variants use small Transformers.
- Frame stacking is the cheap alternative (concatenate the last k=3 observations).
- Tradeoff: recurrent policies are slower to train; frame stacking is easier and gets ~80% of the benefit.

**In LearnFlake**: TRACKER §9.1 uses frame stacking k=3, not recurrent. Aruco-detection dropouts and contact-force history are the partial-observability sources we care about; k=3 covers them.

---

## 5. Sim-to-real transfer

Training in simulation and deploying on a real robot is the central challenge of robot learning. Five techniques dominate.

### 5.1 Domain randomization (DR)

Source: Tobin et al. 2017 (visual DR), Peng et al. 2018 (dynamics DR).

- Randomize simulation parameters (mass, friction, motor gains, sensor noise, lighting, textures) every episode.
- The policy learns to be robust *across* this distribution → in expectation it should also work on the real distribution.
- Critical: the real-world distribution must be **inside** the DR distribution. Too narrow DR → overfit to sim. Too wide DR → policy is too conservative or fails to learn.

Typical axes for a 6-DOF arm:
- Joint friction × [0.5, 1.5]
- Joint damping × [0.5, 2.0]
- Link mass × [0.9, 1.1]
- Controller gains × [0.7, 1.3]
- Action latency [0, 100 ms]
- Sensor noise σ tuned to real measurements

**In LearnFlake**: TRACKER §10 lists 18 DR axes. Wider than initial because no real arm exists yet to do system identification.

### 5.2 Automatic Domain Randomization (ADR)

Source: OpenAI 2019 (Solving Rubik's Cube with a Robot Hand, Akkaya et al.).

- Don't fix the DR ranges; expand them automatically as the policy gets better.
- Each axis has a range `[ϕ_low, ϕ_high]`. Run rollouts at the boundary; if success rate > threshold, widen the range; if < threshold, narrow.
- Generates an automatic curriculum from easy (narrow DR) to hard (wide DR).
- Critical for transfer to highly variable real conditions; less critical for our keyboard task where the keyboard itself doesn't change.

**In LearnFlake**: TRACKER §10 uses an "ADR-lite" — automatic *contraction* only. Start wide, narrow whatever fails. Easier to implement than full bidirectional ADR.

### 5.3 Asymmetric Actor-Critic with privileged critic

Source: Pinto et al. RSS 2018 (already covered in §1.8).

The critic sees ground-truth simulator state; the actor sees only deployable observations. Train as normal SAC. The actor's policy is the deployable artifact.

**In LearnFlake**: yes, TRACKER §4.2.

### 5.4 Teacher-student distillation

- Train a **teacher** with privileged inputs (full state).
- Once the teacher reaches expert performance, generate rollouts.
- Train a **student** with deployable inputs (sensor observations) to *imitate* the teacher's actions on those rollouts.
- Used in OpenAI Dactyl, DeepMind locomotion work, RMA (Rapid Motor Adaptation).

Asymmetric AC is "weak teacher-student" — only the value function is privileged. Full teacher-student does it for both Q and π. More expressive but more code.

**In LearnFlake**: not used. Asymmetric AC is sufficient for our complexity level.

### 5.5 RMA — Rapid Motor Adaptation (Kumar et al. 2021)

- Train a teacher with `(state, latent_z)` where `z` encodes physics parameters.
- Train an adapter that infers `z` from a short history of `(state, action)` — the deployable network.
- At deployment, the adapter rapidly converges to the right `z` for the actual physics.
- Originally for legged locomotion; spreading to manipulation.

**In LearnFlake**: not used. Listed for completeness — would help if we observe domain-specific failures during deployment that fixed DR can't cover.

### 5.6 The IndustReal recipe (NVIDIA 2023)

For contact-rich industrial assembly. Specific tricks:
- **SAPU** (Simulation-Aware Policy Update): exclude policy updates from physically-unrealistic transitions (ones that violate friction cones, etc.).
- **SDF reward** (Signed Distance Field): instead of L2 distance to a target, use SDF to the goal manifold. Richer gradient near the goal.
- **SBC** (Sampling-Based Curriculum): sample initial states from a curriculum that interpolates between "near goal" (easy) and "far from goal" (hard).
- **PLAI** (Policy-Level Action Integrator): on the real robot, integrate the policy's action over a longer time window to smooth out steady-state errors.
- 80% peg insertion success rate, fully sim-to-real.
- Repo: `NVlabs/industreallib`, `NVlabs/industrealkit`.

**In LearnFlake**: SBC inspires TRACKER §7's curriculum. SDF reward and SAPU not used because keyboard contact is gentle enough that L2 reward suffices and nothing about MuJoCo physics is unphysical for this task.

### 5.7 DemoStart (DeepMind 2024)

- Demonstration-led auto-curriculum for sim-to-real with multi-fingered hands.
- Build a buffer of demo states. Each episode resets from a sampled demo state, weighted by per-state success rate.
- 100× fewer demos needed than direct real-world IL. 97% sim-to-real success on plug lifting, 64% on plug insertion, 97% on cube reorientation.
- Repo: not public. Algorithm description is enough to implement.

**In LearnFlake**: TRACKER §7 implements a DemoStart-style state-replay curriculum. Demos skipped for v1 means this collapses to "uniform reset across keyboard region" + a manual key-phase curriculum.

### 5.8 Pipeline fidelity (the under-rated technique)

- The simulation must publish on the **same software interfaces** the real robot uses.
- E.g., if the real arm's control input is `/arm/joint_command` over CAN-FD with a particular message format, the simulator publishes the same topic with the same format.
- Whatever sensor processing happens on the real robot (aruco detector, image rectification, EKF) must run unchanged in the simulator on simulated images.
- This eliminates an enormous class of bugs that would otherwise surface only on hardware.

**In LearnFlake**: see `documentation/keyboard_typing_pipeline.md` "Core Principle" at the bottom. We synthesize the aruco signal (same `(dx, dy, visible)` format the aruco_detector produces) rather than rendering pixels and re-detecting.

---

## 6. Reward design

The black art of RL.

### 6.1 Sparse vs dense rewards

- **Sparse**: 1 for success, 0 otherwise. Mathematically clean (the optimal policy is unambiguous), but exploration is hard. Need HER, curriculum, or demos.
- **Dense**: shaping rewards every step (distance to goal, etc.). Easier to learn from, but the optimal policy is now the optimal policy of `r_task + r_shaping`, which may differ from `r_task` alone — you can shape your way into a bad policy.

### 6.2 Potential-based reward shaping (PBRS) — Ng, Harada, Russell ICML 1999

- Add `r'(s, a, s') = r(s,a,s') + γ Φ(s') - Φ(s)` for any potential function Φ.
- **Theorem**: this transformation does not change the optimal policy. Provably safe shaping.
- In practice you pick `Φ = -distance(s, goal)` and immediately get a dense gradient that doesn't bias the policy.
- HPRS (hierarchical PBRS, Frontiers 2024) extends this to tasks with safety / target / comfort priorities.

**In LearnFlake**: TRACKER §5.3 layers PBRS on top of tolerance shaping for the final 30% of training so the optimal policy provably matches the sparse-success-bonus MDP.

### 6.3 dm_control's `tolerance(...)` primitive

`dm_control.utils.rewards.tolerance(x, bounds, margin, sigmoid)`:
- Returns 1 if `x ∈ bounds`, smoothly decays outside.
- Decay shape: `gaussian`, `linear`, `quadratic`, `long_tail`, `cosine`, `tanh_squared`.
- `value_at_margin` (default 0.1): the tolerance value at distance `margin` from the bounds.

Why it's the right primitive:
- Output bounded in [0, 1] → composable across multiple terms.
- Smooth → critic can fit it.
- The `bounds` parameter encodes the success threshold directly; you don't tune a multiplicative weight.

```python
from dm_control.utils import rewards
r_xy = rewards.tolerance(np.linalg.norm(dxy), bounds=(0, 0.004), margin=0.05, sigmoid='gaussian')
# r_xy ≈ 1 when within 4 mm; smoothly decays to ~0 by 5 cm
```

Used by RoboPianist and many DM-Suite tasks. Standard 2024 reward shape.

**In LearnFlake**: TRACKER §5.2 uses tolerance for every shaping term.

### 6.4 Reward magnitudes

- Hand-crafted rewards with magnitudes from −20 to +1000 (as in the original `keyboard_env.py`) are a known pathology. The Bellman target shifts by orders of magnitude across states; the critic struggles to fit a wide-dynamic-range function.
- Best practice: keep reward in a small range (e.g., [-1, 2]) by design. Either bound shaping in [0,1] (tolerance) or normalize via VecNormalize.

### 6.5 Smoothness penalty for sim-to-real

- Add `-||a_t - a_{t-1}||` to the reward (ICRA 2021, Mysore et al.).
- Or: explicit Lipschitz regularization of the actor (gradient penalty on `dπ/dt`).
- Or: first-order action smoothing in the env wrapper: `a_filt = α a_filt + (1-α) a_new`.
- Real servos can't track high-frequency policy outputs without damaging gear trains.

**In LearnFlake**: TRACKER §5.2 includes `r_smooth` in the reward; §6.3 adds a first-order action lag in the env wrapper. Belt and suspenders.

### 6.6 Success classifiers as rewards (VICE/RCE)

- Train a binary classifier from successful states (positive) vs. random states (negative).
- At training time, use `log p(success | s)` as reward.
- Bypasses hand-engineering entirely.
- Practical recipe in HIL-SERL: collect ~100 successful frames, train a small CNN, use as reward.

**In LearnFlake**: not used in v1. Hand-engineered tolerance reward is sufficient.

---

## 7. Curriculum learning

Order training data from easy to hard. Robot RL needs this whenever the task has any ambient difficulty variation.

### 7.1 Reverse curriculum (Florensa et al. 2017)

- For sparse-reward goal tasks. Start the agent right next to the goal; gradually move the start state further away.
- Generates a curriculum of successively harder initial conditions.
- Inspired DemoStart (§5.7).

### 7.2 Goal GAN (Florensa et al. ICLR 2018)

- Train a GAN to generate goals of intermediate difficulty (~50% success rate).
- The generator adapts as the policy improves, always presenting "just-out-of-reach" goals.

### 7.3 ALP-GMM — Absolute Learning Progress with Gaussian Mixture (Portelas et al. 2019)

- Maintains a GMM over task parameters. The mixture is updated to favor parameter regions where learning progress (success-rate change over time) is high.
- Robust replacement for hand-tuned curriculum.

### 7.4 Value disagreement curricula (Zhang et al. NeurIPS 2020)

- Use disagreement among an ensemble of value functions as a curiosity signal.
- High disagreement → the agent doesn't yet know how good a state is → train there.

### 7.5 CurricuLLM (ICRA 2025)

- Use an LLM to design curricula from natural-language task descriptions.
- Surprisingly competitive on AntMaze and manipulation.
- Worth keeping an eye on as LLMs get better at robotics common sense.

### 7.6 Manual phased curricula

- The dumb version: hand-write phases ("first the central 5×3 keys, then home + qwerty, then all 87"). Advance when phase success rate > 0.85.
- Works fine for most projects. Don't over-engineer this until you observe a problem.

**In LearnFlake**: TRACKER §7 = DemoStart-style state replay (skipped for v1 because no demos) + manual phased key curriculum (always on).

---

## 8. Action spaces and low-level control

A surprisingly underdiscussed part of robot RL — and the one that bit us during the spike.

### 8.1 Joint torque

- Action = direct torque command on each joint motor.
- Most flexible (you can do anything any other controller can do).
- Hardest to learn; the policy has to learn the dynamics implicitly.
- Real motors get fried by raw RL torques; needs a current-limiting layer.

### 8.2 Joint velocity

- Action = desired joint velocity. A controller integrates and torque-tracks.
- Better than torque for learning; still bang-bang and noisy.
- This is what the original `keyboard_env.py` used (`JOINT_VELOCITY`). The arm trained but converged slowly.

### 8.3 Joint position (delta)

- Action = small offset in joint space. Internal P-D tracker holds the new position.
- Smooth, well-understood, matches what most hardware servos use natively (Moteus, Dynamixel, Maxon).
- Slightly slower to train than OSC because the policy must implicitly learn forward kinematics.
- **In LearnFlake**: this is what we ended up using (TRACKER §6, §19) after OSC failed on Rover2026.

### 8.4 Operational Space Control — OSC_POSE / OSC_POSITION

- Action = desired EEF pose (or position) delta in Cartesian space.
- Internally: compute desired EEF wrench = Kp·(target - current) + Kd·(0 - velocity); convert to joint torques via `J^T · F` (Jacobian transpose times wrench).
- For redundant arms (DOF > 6): null-space projection keeps a posture preference.
- Strengths: policy reasons in task space, much faster training on most arms, naturally compliant under contact.
- Weaknesses: numerically nasty near singularities; the redundancy resolution can produce large lateral motions if poorly conditioned (this is what bit us on Rover2026 — see §19 of TRACKER).

### 8.5 Inverse kinematics (IK_POSE)

- Action = desired EEF pose. A full IK solver (mink, KDL, pseudo-inverse Jacobian) computes the joint targets, which are then tracked.
- Less compliant than OSC because the IK insists on the exact pose; under contact, the arm fights.
- Practical: good for non-contact reaching, bad for contact tasks.

### 8.6 Variable impedance control (VIC)

- Action = desired pose **and** desired stiffness (Kp).
- The policy can choose to be stiff (force tracking) or compliant (gentle contact).
- Used in: Watch Less Feel More (IROS 2025), SRL-VIC (2024).
- More expressive but doubles the action dimension.

### 8.7 Action chunking

- Predict a sequence of actions instead of a single one (ACT, Diffusion Policy).
- Reduces effective horizon, smooths execution, robust to per-step noise.
- Doesn't change the underlying control modality (you still need to pick joint vs. OSC vs. IK as the per-step action).

**In LearnFlake**: TRACKER §6 uses JOINT_POSITION + binary solenoid; no action chunking in v1 (would require a different policy class).

---

## 9. Observation design

What goes into the policy's observation, in what frame, with what normalization.

### 9.1 Proprioception fundamentals

- **Joint positions**: prefer `(sin q, cos q)` per joint over raw `q` for continuous joints (handles wrap-around).
- **Joint velocities**: raw, after light filtering.
- **EEF position**: in robot base frame.
- **EEF orientation**: never use raw quaternion (double cover, discontinuity). Options:
  - 6-D continuous rotation rep (Zhou et al. CVPR 2019): stack the first two columns of the rotation matrix, recover the third by orthogonalization. Best for general rotations.
  - For task-specific cases (we only care about orientation w.r.t. world up): use the projection of EEF axis onto world Z.

### 9.2 Goal representation

- **Goal in world frame**: forces the policy to learn to subtract its own pose. Bad for DR.
- **Goal in EEF frame**: translation/rotation invariant — much easier to learn and generalize.
- **Goal as a delta**: `target - eef`, expressed in EEF frame.

### 9.3 Sensor observations

- Raw sensor values + a "valid" flag (e.g., aruco produces `(dx, dy, visible)`).
- Always include the valid flag — the policy needs to know to ignore the values when the sensor failed.
- For dropouts modeled by a probability function (visibility decreases with distance/angle), include the inputs to that function (distance, angle) so the policy can predict failures, not just react.

### 9.4 Frame stacking

- Concatenate the last k observations: `o_t = [o_{t-k+1}, …, o_t]`.
- Cheap memory mechanism for partial observability.
- Typical k=3 to k=5.
- Alternative: recurrent policy. More expressive but slower.

### 9.5 Normalization

- Joint angles, EEF position, contact forces, sensor readings all have different scales.
- Always normalize to roughly unit-Gaussian before feeding to the network.
- Best practice: `RunningMeanStd` accumulator that updates online during training, frozen at deployment.
- VecNormalize in stable-baselines3 does this for free; save `obs_rms` alongside the policy weights.

### 9.6 Privileged observations (for the critic only)

When using asymmetric AC:
- Ground-truth values that the actor can't see: object positions, contact forces, true mass/friction values, current DR knob settings.
- Helps the critic correctly attribute reward changes to true state changes vs. sensor noise.

**In LearnFlake**: TRACKER §9 covers all of this for the actor (36-D, frame-stacked k=3) and critic (30-D privileged, single frame).

---

## 10. Hierarchical RL and primitives

Long-horizon tasks (typing a word, assembling a chair) blow up flat RL. Various hierarchical approaches:

### 10.1 Options framework

- High-level policy selects among "options" — temporally-extended actions.
- Each option is itself a policy, plus a termination function.
- Theoretical foundation: Sutton, Precup, Singh 1999.
- Common pitfall: high-level policy tends to converge to picking the same option always (option starvation).

### 10.2 SPiRL — Skill-Prior RL (Pertsch et al. CoRL 2020)

- Train a VAE on demo trajectories to extract a "skill prior" — a distribution over short action sequences.
- High-level RL selects in skill space; the prior guides exploration toward sensible behavior.

### 10.3 MAPLE — Manipulation Primitive-augmented RL (Nasiriany et al. CoRL 2022)

- Library of hand-coded primitives (grasp, push, place).
- High-level policy selects which primitive to invoke and parameterizes it (target position, orientation).
- Augments with a low-level "atomic" RL policy for cases where no primitive fits.
- 70% improvement over flat RL on robosuite tasks.

### 10.4 Plan-Seq-Learn / STAP / Text2Motion

- LLM generates a plan as a sequence of skill names; each skill is a learned policy.
- Bridges symbolic planning and RL.
- Useful when the long-horizon structure is well-decomposed.

### 10.5 PLANRL — classical planner + RL (Sengupta et al. 2024)

- Classical motion planner for free-space motions (RRT*, OMPL) → RL takes over for contact-rich segments.
- "Use the right tool for the job."
- For LearnFlake: this is exactly the right pattern. **Travel** between keys (free-space motion) is solved by classical planning — just call Approach again with the new key. **Approach** and **Strike** are RL policies.

### 10.6 The HRL trap

- HRL can dramatically improve sample efficiency *if* the decomposition is right.
- HRL hurts when:
  - Skills don't compose (the result of skill A is out-of-distribution for skill B).
  - The high-level policy reward is itself sparse (option starvation).
  - Skills have wildly different action spaces (need adapters).
- LearnFlake's original HRL design hit all three; we abandoned it (TRACKER §0, §19, the user's memory: "HRL abandoned").

**In LearnFlake**: 2 RL skills (Approach, Strike) chained by a deterministic state machine. No learned high-level. Closest to PLANRL philosophy.

---

## 11. Contact-rich manipulation specifically

Our task — pressing keys — is a contact-rich problem. Specific literature:

### 11.1 Peg-in-hole

The canonical contact-rich benchmark.

- **Variable Compliance Control** (2020): RL learns a stiffness schedule for assembly.
- **IndustReal** (NVIDIA 2023): full sim-to-real for industrial peg-in-hole. 80% success on a real Franka.
- **ResiP** (2024): residual diffusion + PPO for 0.2 mm clearance peg-in-hole. 5%→99% success.
- **HIL-SERL**: peg-in-hole in 30 minutes of real-world wall clock.
- **Dreamer for tactile insertion** (2024): model-based RL with Gelsight tactile feedback.

### 11.2 Force/torque control via RL

- Actor outputs target wrench; a low-level admittance controller tracks it.
- Gives the policy direct compliance modulation without learning physics from scratch.

### 11.3 Tactile RL

- **ViSk / VISK** (2024): treat tactile skin patches as additional tokens to a vision transformer; +50% success on USB insertion etc.
- **Reactive Diffusion Policy** (2025): two-tier diffusion — slow vision policy + fast tactile reactive policy.
- **VITAL** (2024): VLM for scene-level reasoning, local visuotactile policy for contact.

### 11.4 Force-based contact detection

- Real arms detect contact via motor current/torque spikes — a key surface load increases motor current several mNm above baseline.
- **In LearnFlake**: TRACKER §5.4 + the existing `documentation/keyboard_typing_pipeline.md` use this approach via the `a5_rotation` joint torque on real hardware, and `cfrc_ext` magnitude on the actuator tip in sim. Both go through a synthetic Moteus topic so the policy code is identical sim/real.

### 11.5 RoboPianist (Zakka et al. CoRL 2023) — closest analog to keyboard typing

- Two simulated Shadow Hands play piano on a 88-key piano. 150 piano pieces learned end-to-end with SAC.
- Architecture:
  - Algorithm: SAC with 3-layer (256, 256, 256) GELU MLP critic+actor, target_entropy = -0.5 * dim(action)
  - Reward: `tolerance` based — 0.5 weight on key press accuracy, +0.5 sustain reward, energy penalty 5e-3, optional fingering bonus
  - Observation: joint positions, key states, sustain pedal state, **MIDI lookahead** (next n_steps_lookahead targets), fingering hints
  - Action: full hand joint targets + sustain pedal
  - Training: 5k warmstart steps, 1M total, 256 batch, replay 1M, eval every 10k steps
- Repos: `google-research/robopianist`, `kevinzakka/robopianist-rl`

This is the most directly transferable published work to LearnFlake. The MLP architecture and `tolerance`-based reward shape directly inform our v1 design.

---

## 12. Training infrastructure and GPU simulators

### 12.1 The CPU sim era (pre-2020)

- MuJoCo on CPU, vectorized via SubprocVecEnv (one Python process per env).
- ~100 envs/sec on a 16-core box. Decent for SAC/RLPD where you don't need huge batches.

### 12.2 The GPU sim era (post-2020)

- **Isaac Gym** (NVIDIA 2021): GPU-native physics, end-to-end on GPU including the policy.
- **Isaac Lab** (2024): successor of Isaac Gym, prod-ready, integrated with USD.
- **ManiSkill3** (Hillbot 2024): SAPIEN-based, more memory-efficient than Isaac Lab (3.5 GB vs 14 GB for 128 envs at typical rendering resolutions).
- **Genesis** (2024): newer, ambitious, multi-physics.
- **mujoco_playground** (DeepMind 2024): MuJoCo-on-GPU via MJX, integrates with JAX/Brax.

### 12.3 Performance benchmarks

| Sim | Envs | Throughput (env-steps/s) | RL algo of choice |
|---|---|---|---|
| robosuite + MJX | 8 | ~200 | RLPD |
| robosuite + CPU | 32 | ~3,000 | SAC/RLPD |
| Isaac Gym | 4096 | ~150,000 | PPO |
| Isaac Lab | 4096 | ~120,000 | PPO |
| ManiSkill3 | 1024 | ~80,000 | PPO |

Picking a sim is the single biggest infrastructure decision.

### 12.4 Why robosuite for LearnFlake

- Existing custom `Rover2026` model (URDF + MJCF + composite controller config) is already in the project.
- Porting to Isaac Lab would require re-modeling. Big investment for unclear gain on a small task.
- Throughput on CPU robosuite is sufficient for 1M-step RLPD training overnight.

### 12.5 RL libraries

- **Stable-Baselines3** (PyTorch): production quality, well-documented. SAC, PPO, TD3, A2C, DDPG, DQN. Slow due to PyTorch overhead.
- **sb3-contrib**: experimental algos — TQC, CrossQ, ARS, MaskablePPO, RecurrentPPO.
- **SBX** (`araffin/sbx`): SB3 + JAX. Same API, much faster (~3-5×). DroQ included.
- **CleanRL**: minimal single-file implementations. Great for debugging.
- **RSL-RL** (NVIDIA): PPO optimized for Isaac Lab.
- **rl_games**: PPO + others, used in NVIDIA assembly papers.
- **TorchRL** (PyTorch): newer, modular.

**In LearnFlake**: PyTorch stack (sb3 + sb3-contrib) for v1 because the rover_gpu container already has it and the team's familiarity is there. SBX is the v1.1 speed upgrade if needed.

### 12.6 LayerNorm / BatchNorm in critics

- LayerNorm: per-sample normalization. Composes well with replay buffers, frame stacking, off-policy training. **Default choice in 2024**.
- BatchNorm: per-batch normalization. Trains faster but introduces train/test distribution mismatch in RL where action sampling temperature changes. Use carefully (CrossQ has it figured out).
- The pre-LayerNorm placement (`LN(x) → linear → activation → linear`) tends to work better than post-LN in RL critics.

### 12.7 Optimizer choices

- **Adam(lr=3e-4)**: the default for everything since 2017.
- **AdamW(lr=3e-4, weight_decay=1e-4)**: marginally better, recommended for newer recipes.
- **EMA (exponential moving average)** of policy weights for evaluation: standard in diffusion-policy land, optional in SAC.

### 12.8 Replay buffer memory

- 1M transitions × ~128 dims × float32 = ~512 MB. Fits in CPU RAM trivially.
- For pixel observations (e.g., 64×64×3 frames stacked 3 deep × 1M transitions ≈ 36 GB), needs disk-based or compressed buffer.
- GPU replay buffer (everything stays on GPU during sampling): 5-10× speedup for the gradient step. SBX does this; sb3 does not.

---

## 13. RTX 50-series compatibility

Specific to RTX 5060 / 5070 / 5080 / 5090 (Blackwell architecture, sm_120). Important because LearnFlake runs on a 5060 Laptop GPU.

### 13.1 The compatibility wall

- Stable PyTorch (≤ 2.6) was compiled with `arch_list = [sm_70, sm_75, sm_80, sm_86, sm_90]`. **No sm_120**. Trying to run any GPU op produces:
  ```
  CUDA error: no kernel image is available for execution on the device
  ```
- Three solutions:
  1. **PyTorch nightly cu128** — `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128`. Has sm_120 since early 2025.
  2. **PyTorch 2.7+ stable** (released mid-2025) — has sm_120 in default builds.
  3. **Build PyTorch from source** with `TORCH_CUDA_ARCH_LIST=12.0` — for the patient.

### 13.2 LearnFlake's actual setup (verified 2026-04-29)

```
torch                2.10.0+cu128       arch_list includes sm_120  ✅
robosuite            1.5.1                                          ✅
mujoco               3.6.0                                          ✅
stable-baselines3    2.7.1                                          ✅
gymnasium            1.2.3                                          ✅
NVIDIA driver        590.48.01           CUDA 13.1                  ✅
```

The `rover_gpu` Docker container ships these out of the box. No nightly install needed.

### 13.3 JAX on Blackwell

- Blackwell support added in JAX/jaxlib 25.01 (NVIDIA-distributed JAX containers).
- **Known issue**: JAX RNG is non-deterministic on gamer Blackwell (sm_120, e.g. RTX 5060/5070/5080/5090). Repro experiments may not bit-exact match across runs.
- Driver 570+ recommended.

### 13.4 MuJoCo / robosuite rendering on Blackwell

- MuJoCo 3.x supports EGL, GLFW, OSMesa as `MUJOCO_GL` backends.
- On Blackwell, GLFW (the LearnFlake default in `docker-compose.yml`) works. EGL has occasional issues with `__GLX_VENDOR_LIBRARY_NAME=nvidia`; if it fails, fall back to `MUJOCO_GL=glfw` or `osmesa`.
- The rendering matters for visual debugging only; training without rendering bypasses the issue.

### 13.5 Driver requirements

- Linux: NVIDIA driver 570+ for full Blackwell support. LearnFlake runs 590.48.01 which is more than fine.
- Windows: Game Ready Driver 591.59+ for stability and color reproduction fixes specific to RTX 5060/Ti.

### 13.6 Setuptools / pip nuances on the rover_gpu base image

- The container's base setuptools is 59.6 (old). PEP 660 editable installs (`pip install -e .`) need setuptools ≥ 64.
- Solutions:
  1. `pip install -e . --no-build-isolation` (use already-installed setuptools).
  2. Add a `setup.py` shim with just `from setuptools import setup; setup()` for older-pip compatibility.
- Don't globally upgrade setuptools to ≥ 80 — it breaks `colcon-core` (requires `<80`).

**In LearnFlake**: `setup.py` shim added at repo root in Phase 0; `pip install -e . --no-build-isolation` works. Setuptools 79.0.1 is the safe ceiling for colcon compatibility.

---

## 14. Practical recipes — what we use, what we considered

The synthesis. For each design dimension, what we picked for v1 and why.

| Dimension | LearnFlake v1 | Why |
|---|---|---|
| Algorithm | RLPD-SAC (LayerNorm critic, UTD=10) | Best published "few-shot from prior data" recipe; matches HIL-SERL track record; LayerNorm provides stability. |
| Critic | 3 layers × 512, GELU, LayerNorm, twin (`Q_φ1, Q_φ2`) | BRO/RLPD lesson: bigger critic + regularization. |
| Actor | 3 layers × 256, GELU, tanh-Normal | Standard SAC. Same architecture as RoboPianist. |
| Privileged critic input | yes — ground-truth EEF-to-key, force, solenoid, DR knobs | Asymmetric AC (Pinto 2017). Cheap, big win. |
| Action space | JOINT_POSITION delta + binary solenoid | OSC_POSE failed on Rover2026 in spike (TRACKER §19). JOINT_POSITION matches Moteus native interface. |
| Reward | dm_control `tolerance` + sparse terminal bonus + PBRS layer | Bounded in [-1, 2], composable, smooth, optimal-policy-preserving. |
| Demos | none in v1 (skipped per user direction) | Available in `aaron/more_rl` history if we ever want them. |
| Curriculum | manual key phases + (DemoStart-style state replay when demos exist) | Phased covers the bulk; state replay is the v1.1 upgrade. |
| Domain rand | 18 axes, ADR-lite contraction | Wide because no system identification yet. |
| Sim-to-real | asymmetric AC + DR + identical ROS topic interface | The three layers of defense. |
| Observation | 36-D actor (frame-stacked k=3) + 30-D privileged critic | EEF-frame goal, 6-D rotation rep, sin/cos joint pos, RunningMeanStd. |
| Sim | robosuite (CPU MuJoCo, ~12 envs in container) | Existing Rover2026 model, no Isaac Lab port needed. |
| Skills | 2 RL skills (Approach, Strike) chained by state machine | HRL dropped per user; PLANRL pattern. |
| Backend | PyTorch + sb3 + sb3-contrib | rover_gpu container has it; team familiarity. SBX is the speed upgrade path. |

### What we explicitly considered and rejected

- **Diffusion Policy / DPPO**: needs hundreds of demos; we have zero.
- **OpenVLA / RT-2 / π0**: massively oversized for a 6-DOF + binary-solenoid task.
- **OSC_POSE**: failed on Rover2026 (TRACKER §19). Custom Cartesian impedance is a fallback if JOINT_POSITION ever underperforms.
- **CrossQ / BRO**: documented as v1.1 ablations if RLPD plateaus or burns too much GPU time.
- **HER**: doesn't fit; reward is dense via tolerance.
- **HRL / options framework**: dropped per user; 9 skills → 2 skills.
- **End-to-end pixels (DrQ-v2)**: aruco synth is more accurate and trains faster; pixels not needed.
- **Isaac Lab port**: throughput gain not worth the 1-week port effort given our scale.
- **JAX backend**: planned for v1.1 if PyTorch UTD=10 wallclock is too slow.
- **VICE success classifier**: hand-engineered tolerance reward is sufficient for v1.

### What we picked over the original plan

- **JOINT_POSITION** instead of OSC_POSE (forced by spike).
- **Container-baseline package versions** (sb3 2.7.1, mujoco 3.6.0) instead of TRACKER's original pins (sb3 2.5.0, mujoco 3.3.0). The container's already-working set wins.
- **`pyproject.toml` at repo root** instead of inside `src/rl_autonomy/`. Standard PEP 518 src layout.

---

## 15. Glossary

| Term | Meaning |
|---|---|
| **Actor** | The policy network. Outputs actions given observations. |
| **Critic** | The value network. Outputs Q(s,a) or V(s). |
| **UTD ratio** | Update-To-Data: number of gradient updates per environment step. |
| **Polyak update** | Soft target update: `θ_target ← (1-τ) θ_target + τ θ`. |
| **Tanh-squashed policy** | Actor outputs (μ, σ); samples `a = tanh(μ + σ⊙ε)`. SAC standard. |
| **Replay buffer** | Stores past `(s, a, r, s', done)` transitions for off-policy training. |
| **Bellman target** | `y = r + γ min(Q_φ1(s', a'), Q_φ2(s', a'))` for SAC. |
| **Frame stacking** | Concatenate the last k observations into one input vector. |
| **Asymmetric AC** | Critic sees privileged state, actor sees only deployable observations. |
| **PBRS** | Potential-Based Reward Shaping. Adds `γΦ(s') − Φ(s)` to reward; provably preserves optimal policy. |
| **DR** | Domain Randomization. Randomize sim parameters per episode. |
| **ADR** | Automatic DR. Adapts ranges based on success rate. |
| **OSC** | Operational Space Control. Cartesian-space controller via `J^T·F`. |
| **Jacobian-pseudoinverse** | `J^+ = J^T (J·J^T)^-1` for under-determined systems. |
| **Damped least squares** | Numerically stable pseudoinverse: `J^T (J·J^T + λ²·I)^-1`. |
| **EEF** | End-effector. The "tool" frame at the tip of the robot. |
| **Aruco** | Fiducial marker pattern for camera-based pose estimation. |
| **Hover height** | Z offset above a target where the EEF parks before pressing. |
| **Approach / Strike** | LearnFlake's two RL skills: move + orient, then press. |
| **sm_120** | Blackwell architecture (RTX 50-series). PyTorch 2.7+ required. |
| **CAN-FD** | Controller Area Network with Flexible Data-rate. The Moteus comm bus. |

---

## 16. Annotated bibliography

In rough chronological order. Reading list: anything tagged ★ is recommended.

- ★ Lillicrap et al. **Continuous Control with Deep Reinforcement Learning** (DDPG). 2015. arXiv:1509.02971.
- Schulman et al. **Trust Region Policy Optimization** (TRPO). ICML 2015. arXiv:1502.05477.
- Schulman et al. **Proximal Policy Optimization Algorithms** (PPO). 2017. arXiv:1707.06347. Standard on-policy method.
- Ho & Ermon. **Generative Adversarial Imitation Learning** (GAIL). NeurIPS 2016. arXiv:1606.03476.
- Tobin et al. **Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World**. IROS 2017. arXiv:1703.06907.
- Andrychowicz et al. **Hindsight Experience Replay** (HER). NeurIPS 2017. arXiv:1707.01495.
- Ng, Harada, Russell. **Policy Invariance under Reward Transformations**. ICML 1999. The PBRS paper.
- Rajeswaran et al. **Learning Complex Dexterous Manipulation with Deep RL and Demonstrations** (DAPG). RSS 2018. arXiv:1709.10087. `aravindr93/hand_dapg`.
- Silver et al. **Residual Policy Learning**. 2018. arXiv:1812.06298.
- Pinto et al. **Asymmetric Actor Critic for Image-Based Robot Learning**. RSS 2018. arXiv:1710.06542.
- ★ Fujimoto et al. **Addressing Function Approximation Error in Actor-Critic Methods** (TD3). ICML 2018. arXiv:1802.09477.
- ★ Haarnoja et al. **Soft Actor-Critic** (SAC). ICML 2018. arXiv:1801.01290. Read this first.
- Akkaya et al. **Solving Rubik's Cube with a Robot Hand** (Dactyl/ADR). 2019. arXiv:1910.07113.
- Florensa et al. **Reverse Curriculum Generation for Reinforcement Learning**. CoRL 2017. arXiv:1707.05300.
- Florensa et al. **Automatic Goal Generation for Reinforcement Learning Agents** (Goal GAN). ICML 2018. arXiv:1705.06366.
- Pertsch et al. **Accelerating Reinforcement Learning with Learned Skill Priors** (SPiRL). CoRL 2020.
- Kuznetsov et al. **Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics** (TQC). ICML 2020. arXiv:2005.04269.
- Yarats et al. **DrQ-v2: Improved Data-Augmented RL**. 2021. arXiv:2107.09645. `facebookresearch/drqv2`.
- Chen et al. **Randomized Ensembled Double Q-Learning** (REDQ). ICLR 2021. arXiv:2101.05982.
- Hiraoka et al. **Dropout Q-Functions for Doubly Efficient RL** (DroQ). ICLR 2022. arXiv:2110.02034.
- Kumar et al. **RMA: Rapid Motor Adaptation for Legged Robots**. RSS 2021.
- Kostrikov et al. **Offline RL with Implicit Q Learning** (IQL). 2021. arXiv:2110.06169.
- Nair et al. **AWAC: Accelerating Online RL with Offline Datasets**. 2020. arXiv:2006.09359.
- ★ Nasiriany et al. **MAPLE: Augmenting RL with Behavior Primitives**. CoRL 2022.
- ★ Tang et al. **IndustReal: Transferring Contact-Rich Assembly Tasks from Simulation to Reality** (NVIDIA). RSS 2023. arXiv:2305.17110. `NVlabs/industreallib`.
- ★ Chi et al. **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**. RSS 2023, IJRR 2024. arXiv:2303.04137. `real-stanford/diffusion_policy`.
- ★ Zhao et al. **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware** (ACT/ALOHA). RSS 2023. arXiv:2304.13705.
- ★ Zakka et al. **RoboPianist: Dexterous Piano Playing with Deep RL**. CoRL 2023. `google-research/robopianist`, `kevinzakka/robopianist-rl`. Closest published analog to LearnFlake.
- Ball et al. **Efficient Online Reinforcement Learning with Offline Data** (RLPD). NeurIPS 2023. arXiv:2302.02948.
- Bhatt et al. **CrossQ: Batch Normalization in Deep RL for Greater Sample Efficiency and Simplicity**. ICLR 2024. arXiv:1902.05605. `adityab/CrossQ`.
- ★ Luo et al. **SERL: A Software Suite for Sample-Efficient Robotic RL**. 2024. `rail-berkeley/serl`.
- ★ Luo et al. **Precise and Dexterous Robotic Manipulation via Human-in-the-Loop RL** (HIL-SERL). Science Robotics 2024. arXiv:2410.21845. `rail-berkeley/hil-serl`. Read this for "RL on a real robot" recipes.
- Bauza et al. (DeepMind). **DemoStart: Demonstration-led Auto-Curriculum**. 2024. arXiv:2409.06613.
- Ankile et al. **From Imitation to Refinement: Residual RL for Precise Assembly** (ResiP). 2024. arXiv:2407.16677.
- Ren et al. **Diffusion Policy Policy Optimization** (DPPO). ICLR 2025. arXiv:2409.00588. `irom-princeton/dppo`.
- Yang et al. **Residual Off-Policy RL for Finetuning Behavior Cloning Policies**. 2025. arXiv:2509.19301.
- Nauman et al. **Bigger, Regularized, Optimistic** (BRO). NeurIPS 2024. arXiv:2405.16158. `naumix/BiggerRegularizedOptimistic`.
- Zhou et al. **On the Continuity of Rotation Representations in Neural Networks** (6-D rotation rep). CVPR 2019. arXiv:1812.07035.
- Mysore et al. **Regularizing Action Policies for Smooth Control with RL**. ICRA 2021. The action-smoothness reward.
- Tao et al. **ManiSkill3: GPU Parallelized Robotics Simulation**. RSS 2025. arXiv:2410.00425. `haosulab/ManiSkill`.
- Mittal et al. **Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning** (NVIDIA). 2025. arXiv:2511.04831.
- Open X-Embodiment Collaboration. **Open X-Embodiment: Robotic Learning Datasets and RT-X Models**. 2023. arXiv:2310.08864.

---

## 17. What's NOT in this document

Topics I researched briefly but didn't bring forward because they're orthogonal to v1:

- Model-based RL (Dreamer, MuZero, TD-MPC). Conceptually attractive; deployment on real robots still hard.
- Offline RL pure (CQL, IQL standalone, COMBO). Useful when you have a lot of offline data; we don't.
- Multi-agent RL. Not relevant here.
- Hierarchical reinforcement learning beyond options/MAPLE. Dropped per user instruction.
- Reward learning from preferences (RLHF for robots, e.g. PEBBLE). Could become useful for the keyboard-feel-quality dimension; not a v1 priority.
- Mobile manipulation. Rover2026 is fixed-base for this task.

If any of the above becomes relevant, this document is the seed; the bibliography points to live links.
