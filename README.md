# LearnFlake

LearnFlake teaches the UBC Snowbots Rover2026 arm to type. A 6-DOF arm with a
solenoid tip presses keys on a Redragon K552 TKL keyboard, in a MuJoCo
(robosuite) simulation. The repo holds the simulation, the training code, a
trained model, and Docker images for Ubuntu 24.04 + ROS 2 Jazzy.

## What works today (in simulation)

| Test | Result |
|---|---|
| Move to the right key and press it, all 87 keys x 5 tries | **429/435 (98.6%)** |
| Keys that work every time | 84/87 |
| Keys that work at least 80% of the time | 86/87 |

The trained model is in [`models/approach_v16b_residual.pt`](models/), so you can
run everything below without training anything.

How it works, in one line: a hand-written IK controller moves the arm, a small
learned network corrects its mistakes, and the solenoid fires once the tip is
over the key. Details: [`documentation/`](documentation/).

---

## Quick start

You need Docker and an NVIDIA GPU. (No GPU? See [Setup options](#setup-options).)

**1. Build and enter the container** (from the repo root):

```bash
docker compose build rover_gpu
docker compose run --rm rover_gpu bash
```

You are now in `/LearnFlake` inside the container, with ROS 2 Jazzy and the
Python environment ready. The repo folder is shared with your machine, so files
you make here also show up outside.

**2. Check that everything works:**

```bash
python3 -m pytest tests/ -q
```

You should see `80 passed`.

**3. Watch the arm type:**

```bash
python3 -m rl_autonomy.tools.type_string --text "hello world" --out typing.mp4
```

Open `typing.mp4` from the repo folder on your machine. Example output:
[`media/typing_hello_world.mp4`](media/typing_hello_world.mp4).

---

## Common tasks

All of these run inside the container, from `/LearnFlake`.

### Type any text

```bash
python3 -m rl_autonomy.tools.type_string --text "your text here" --out typing.mp4
```

Useful options:
- `--camera agentview` (or `frontview`, `sideview`, `birdview`)
- `--width 640 --height 480 --fps 30`
- `--interactive` opens a live window instead of a video (needs X11, see below)

### Score the model on every key (about 12 minutes)

```bash
python3 -m rl_autonomy.scripts.eval_orchestrator \
    --approach models/approach_v16b_residual.pt \
    --residual --residual-tube 0.15 --chain --keyboard-offset=-0.10,-0.10 \
    --keys all --trials-per-key 5 \
    --out-md results/my_run.md
```

Expected: `Full chain success: 429/435 (98.6%)`. Copy the flags exactly; the
model was trained with them. Keep the `=` in `--keyboard-offset=-0.10,-0.10`.

### Watch one key in a live window

```bash
python3 -m rl_autonomy.tools.visualize --policy p_ctrl --key g
```

### Train the model again

Two steps, about 25 minutes each on an RTX-class GPU:

```bash
# 1) imitation learning from the IK controller
python3 -m rl_autonomy.scripts.train_dagger \
    --keys all --eval-keys stratified --rounds 6 --rollouts-per-round 174 \
    --keyboard-offset=-0.10,-0.10 \
    --save-dir checkpoints/dagger --log-dir logs/dagger

# 2) learn the correction on top of the IK controller
python3 -m rl_autonomy.scripts.train_residual \
    --steps 200000 --tube 0.15 --reward-mode pbrs_only --keyboard-offset=-0.10,-0.10 \
    --save-dir checkpoints/residual --log-dir logs/residual
```

Then score your checkpoint with the command above (change `--approach`). Keep
`--reward-mode pbrs_only`; the other reward modes train worse. The demo files
in `demos/` are all the starting data you need.

---

## Setup options

### Linux with an NVIDIA GPU (recommended)

Use the [Quick start](#quick-start). The host needs the NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Linux without a GPU

```bash
docker compose -f docker-compose.ubuntu.yml build rover_cpu
docker compose -f docker-compose.ubuntu.yml run --rm rover_cpu bash
```

Everything works, but training is slow.

### Windows

Use Docker Desktop with WSL2. Step-by-step guide: [DOCKER_WINDOWS.md](DOCKER_WINDOWS.md).
Short version:

```bash
docker compose -f docker-compose.ubuntu.yml build
docker compose -f docker-compose.ubuntu.yml run --rm rover_gpu bash   # or rover_cpu
```

### Without Docker

Install ROS 2 Jazzy with the team tool
[rovercli](https://github.com/UBC-Snowbots/rovercli), then set up the Python
environment. Exact steps: [setup_scripts/README.md](setup_scripts/README.md).

### Live windows (X11)

`visualize` and `type_string --interactive` open a window, so the host needs
X11. On Linux, run `xhost +local:docker` once before starting the container.
Videos (`--out file.mp4`) do not need X11.

---

## Repo layout

| Path | What is in it |
|---|---|
| `src/rl_autonomy/envs/` | The keyboard simulation (`keyboard_env.py`, `residual_ik.py`) |
| `src/rl_autonomy/algos/` | The IK controller (`expert_ik.py`) and learning code |
| `src/rl_autonomy/scripts/` | Training and scoring commands |
| `src/rl_autonomy/tools/` | `type_string` and `visualize` |
| `src/external_pkgs/` | robosuite and the ROS 2 packages (`serial`, `moteus_ros2`) |
| `models/` | The trained model |
| `demos/` | Starting data for training |
| `results/` | Scoring reports from past runs |
| `documentation/` | How the learning parts work |
| `docker/`, `Dockerfile*`, `docker-compose*.yml` | Container setup |
| `setup_scripts/` | Setup without Docker |
| `tests/` | The test suite |

## Troubleshooting

- **`docker compose` prints `pyenv` or `DISPLAY` warnings.** Safe to ignore.
- **The score is much lower than 98%.** Check that you used all the flags
  from the scoring command, especially `--residual --residual-tube 0.15` and
  `--keyboard-offset=-0.10,-0.10`.
- **`could not select device driver "nvidia"`.** Install the NVIDIA Container
  Toolkit on the host, or use the CPU container.
- **No window opens.** See [Live windows (X11)](#live-windows-x11), or save a
  video with `--out` instead.

## More reading

- [`documentation/dagger.md`](documentation/dagger.md) and
  [`documentation/residual_rl.md`](documentation/residual_rl.md): how the two training steps work
- [TRACKER.md](TRACKER.md): the full engineering log, every experiment and decision
- [RECENT.md](RECENT.md): recent changes to the setup and images
- [Docker.md](Docker.md): Docker basics for the team

Other branches: `nvidia-rl-hackathon` has voice and body-tracking control for
the real rover arm.
