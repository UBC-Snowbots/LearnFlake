# LearnFlake

Reinforcement-learning pipeline for the Rover2026 6-DOF arm + solenoid actuator
typing on a Redragon K552 TKL keyboard in MuJoCo (RoboSuite).

> **v1 rewrite is on branch `aaron/rl_rewrite`.** Full engineering log + every
> design decision: [TRACKER.md](TRACKER.md). Skill design docs:
> [`documentation/`](documentation/) (`dagger.md`, `residual_rl.md`).

## Status — M4 PASSED ✅ (in sim)

The Approach→Strike pipeline solves the keyboard in MuJoCo:

| metric | value |
|---|---|
| Approach success (all 87 keys × 5 trials) | **429/435 (98.6%)** |
| **Full chain (Approach→Strike)** | **429/435 (98.6%)** |
| Keys at 100% full chain | **84/87** |
| Keys at ≥80% (the M4 bar is ≥80/87) | **86/87 — PASSED** |

How it was solved (see TRACKER §35–§39 for the full story): **DAgger** on the
hand-coded IK expert (cures behaviour-cloning's covariate shift) → **tuning the
keyboard position** so the whole board is in the arm's dexterous workspace →
**residual RL on the IK** (a tube-clipped learned correction that beats the
expert's ~62% per-attempt ceiling) → **physical strike thresholds** + true
in-process Approach→Strike chaining. The trained model is in
[`models/approach_v16b_residual.pt`](models/) so you can test without retraining.

---

## Quick test (reproduce M4)

All commands run **inside the `rover_gpu` container** (setup below). From
`/LearnFlake`:

```bash
# 0) sanity: the whole test suite (should print "76 passed")
python3 -m pytest tests/ -q

# 1) THE headline: full Approach→Strike chain over all 87 keys (~12 min)
python3 -m rl_autonomy.scripts.eval_orchestrator \
    --approach models/approach_v16b_residual.pt \
    --residual --residual-tube 0.15 --chain --keyboard-offset=-0.10,-0.10 \
    --keys all --trials-per-key 5 \
    --out-md results/m4_fullchain.md
# -> Full chain success: 429/435 (98.6%) ... M4 status: PASSED
```

> **Flag gotchas (must match how the model was trained):**
> - `--residual --residual-tube 0.15` — the checkpoint is a residual-on-IK policy.
> - `--keyboard-offset=-0.10,-0.10` — use the **`=` form** (a leading `-` value
>   confuses argparse without it).
> - `--chain` — true in-process Approach→Strike (no learned Strike policy needed;
>   it extends the solenoid open-loop, which presses every reachable key).

Approach-only matrix (no strike), if you want just the positioning number:

```bash
python3 -m rl_autonomy.scripts.eval_orchestrator \
    --approach models/approach_v16b_residual.pt --strike models/approach_v16b_residual.pt \
    --residual --residual-tube 0.15 --keyboard-offset=-0.10,-0.10 \
    --keys all --trials-per-key 5
# -> Approach success: 428/435 (98.4%)   (--strike is an ignored stand-in here)
```

### Type a string (watch the arm type it out) 🎬

Feed any text and the arm approaches + presses each key in MuJoCo, rendering an
annotated MP4 (overlay shows the target key and the text typed so far). Sample
output: [`media/typing_hello_world.mp4`](media/typing_hello_world.mp4).

```bash
python3 -m rl_autonomy.tools.type_string --text "hello world" --out typing.mp4
# -> typed 11/11 keys successfully; wrote media to typing.mp4
docker cp rover_gpu:/LearnFlake/typing.mp4 .      # copy it out to watch
```

- Each key is a fresh approach-from-home + strike (matches the 98.6% eval), so it
  types **~98% of keys correctly** — e.g. "hello world" types 11/11.
- `--camera agentview|frontview|sideview|birdview`, `--width/--height/--fps`,
  `--continuous` (smoother key→key flow but less reliable — out-of-distribution
  starts), `--interactive` (live X11 viewer instead of MP4).

### Watch a single key live (optional GUI)

```bash
python3 -m rl_autonomy.tools.visualize --policy p_ctrl --key g   # the IK expert
```

---

## Retrain the pipeline from scratch

Each stage is one command. The Approach stack is the interesting part; Strike is
open-loop (no training needed).

```bash
# 1) DAgger Approach (cures BC covariate shift) — ~25 min, all 87 keys
python3 -m rl_autonomy.scripts.train_dagger \
    --keys all --eval-keys stratified --rounds 6 --rollouts-per-round 174 \
    --keyboard-offset=-0.10,-0.10 \
    --save-dir checkpoints/dagger --log-dir logs/dagger
# best all-87 Approach from this stage alone ~ 200/435 (46%); pick the LAST round
# (small in-loop evals are noisy — see TRACKER §35.8).

# 2) Residual RL on the IK — the ceiling-raiser — ~25 min
#    MUST use --reward-mode pbrs_only (success-aligned). xy_focus gets GAMED
#    (TRACKER §38.1). zero-init head means it starts at the IK baseline.
python3 -m rl_autonomy.scripts.train_residual \
    --steps 200000 --tube 0.15 --reward-mode pbrs_only --keyboard-offset=-0.10,-0.10 \
    --save-dir checkpoints/residual --log-dir logs/residual
# -> residual_step_000100000.pt reaches ~99% Approach. This IS the M4 model.

# 3) evaluate it (see "Quick test" above, pointing --approach at your checkpoint)
```

Notes:
- **Demos** (`demos/*.h5`) are committed; DAgger/residual regenerate everything
  else from the live IK expert, so a fresh clone needs no other artifacts.
- `train_strike.py` exists but a learned Strike is **not** needed for M4 — the
  open-loop solenoid extend (used by `--chain`) presses every reachable key once
  the contact thresholds are physical (TRACKER §39).
- Key files: env `src/rl_autonomy/envs/{keyboard_env,residual_ik}.py`; expert
  `src/rl_autonomy/algos/expert_ik.py`; trainers `src/rl_autonomy/scripts/`.

---

## Environment setup

The `rover_gpu` image already ships the full RL stack (torch 2.x+cu128 for
Blackwell/RTX 50xx, mujoco, robosuite, dm-control) **baked in** as of 2026-06-01.
Just bring it up:

```bash
docker compose up -d rover_gpu
docker compose exec rover_gpu bash      # then: cd /LearnFlake
```

> Every `docker compose` command prints harmless host-side `pyenv`/`DISPLAY`
> warnings before the real output — ignore them.

**Setting up on a fresh machine** (image not built / env missing): the exact
install command log + pinned versions are in [RECENT.md](RECENT.md) and
[`docker/rl_env_freeze.txt`](docker/rl_env_freeze.txt). Short version, inside the
container:

```bash
pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128
pip install --ignore-installed sympy   # work around apt's distutils sympy, then re-run torch
pip install numpy scipy "mujoco>=3.6" "gymnasium>=1.0,<2" dm-control h5py tqdm tensorboard PyYAML pytest
pip install termcolor numba "mink==0.0.5" "qpsolvers[quadprog]" Pillow opencv-python-headless pynput
pip install -e src/external_pkgs/RoboSuite --no-deps
pip install -e . --no-deps
# verify:
python3 -c "import torch; print(torch.cuda.is_available())"   # True
python3 -m pytest tests/ -q                                   # 76 passed
# bake it so a container recreate can't wipe it:  (from the host)
#   docker commit rover_gpu learnflake:gpu
```

Visit [Docker.md](Docker.md) for the full Docker install guide, and
[src/rl_autonomy/README.md](src/rl_autonomy/README.md) for the RoboSuite setup.
You also need **X11** on the host for the optional MuJoCo GUI (`visualize`).

## Windows Docker Guide
Use the Windows/WSL compose file: `docker-compose.ubuntu.yml`.

### Prerequisites
- Docker Desktop with WSL2 backend
- WSLg enabled (for GUI apps)

### Build images
```bash
docker compose -f docker-compose.ubuntu.yml build
```

### Start a container
CPU:
```bash
docker compose -f docker-compose.ubuntu.yml run --rm rover_cpu bash
```

RL:
```bash
docker compose -f docker-compose.ubuntu.yml run --rm rover_rl bash
```

GPU:
```bash
docker compose -f docker-compose.ubuntu.yml run --rm rover_gpu bash
```

### Run in background and re-enter
```bash
docker compose -f docker-compose.ubuntu.yml up -d rover_cpu
docker compose -f docker-compose.ubuntu.yml exec rover_cpu bash
```

### Stop everything
```bash
docker compose -f docker-compose.ubuntu.yml down
```
