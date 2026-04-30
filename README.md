# LearnFlake

Reinforcement-learning pipeline for the Rover2026 6-DOF arm + solenoid actuator
typing on a Redragon K552 TKL keyboard in MuJoCo (RoboSuite).

> **v1 rewrite is on branch `aaron/rl_rewrite`.** Design contract:
> [TRACKER.md](TRACKER.md). Background research:
> [`src/rl_autonomy/documentation/rl_research_notes.md`](src/rl_autonomy/documentation/rl_research_notes.md).

## v1 quickstart

All commands run inside the `rover_gpu` docker container (see Docker section
below for setup):

```bash
# from inside rover_gpu, at /LearnFlake/.worktrees/rl_rewrite/
pip install -e . --no-build-isolation     # one-time
pytest tests/                              # 34 tests
```

### Visualize the env

```bash
# Live MuJoCo viewer
python3 -m rl_autonomy.tools.visualize --policy p_ctrl --key g

# Headless PNG sequence
python3 -m rl_autonomy.tools.visualize --save-frames ./viz \
    --policy p_ctrl --key f12 --steps 400 --frame-every 30
```

### Train Approach (1M steps, ~overnight on RTX 5060)

```bash
python3 -m rl_autonomy.scripts.train_approach \
    --steps 1000000 --domain-rand \
    --save-dir checkpoints/approach_v1 \
    --log-dir logs/approach_v1
tensorboard --logdir logs/approach_v1
```

### Train Strike (100k steps, ~30 min)

```bash
python3 -m rl_autonomy.scripts.train_strike \
    --steps 100000 \
    --save-dir checkpoints/strike_v1 \
    --log-dir logs/strike_v1
```

### Evaluate the full pipeline (M4 acceptance)

```bash
python3 -m rl_autonomy.scripts.eval_orchestrator \
    --approach checkpoints/approach_v1/approach_final.pt \
    --strike   checkpoints/strike_v1/strike_final.pt \
    --keys all --trials-per-key 5 \
    --out-md results/m4_success_matrix.md
```

Pass criterion: ≥80/87 keys at ≥80% full-chain success.

### Acceptance gates (TRACKER §15)

| Gate | Status | What it checks |
|---|---|---|
| **M1** env correctness | ✅ PASSED | `python -m rl_autonomy.tools.m1_p_controller` |
| **M2** algorithm correctness | ✅ PASSED | `python -m rl_autonomy.tools.m2_pendulum` |
| **M3** Approach training | ⏳ pending full training run |
| **M4** full pipeline 87-key matrix | ⏳ pending M3 + Strike training |
| **M5** hardware | descoped (no real arm yet) |

---

Visit [Docker.md](Docker.md) for the most recent up-to-date installation guide if you plan on using Docker to manage your environment.

### Continued Installation Guide
After you are done setting up the docker container, the markdown file located at [src/rl_autonomy/README.md](src/rl_autonomy/README.md) contains the installation steps to setup RoboSuite.

Don't forget to install the following system drivers though:
- X11: A windowing system protocol and display server that manages how graphical applications render and display windows. We need this so that Docker knows what display protocol to use when you run RoboSuite in the container.

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
There's a buncha other stuff, like VcXsrv and other stuff that should be ok with my files. In the process of writing code if you come into issues with glfw not running or switching into egl (headless, no GUI), message Pranav.