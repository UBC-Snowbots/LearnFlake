# LearnFlake

## Project Overview
- Build reinforcement learning and other autonomous solutions to a variety of tasks for a student design team
- Goal is to learn and build cool things, while maintaining proper coding practices.

## Architecture
- `src/rl_autonomy/` — Reinforcement Learning pipeline for autonomous tasks like clicking a sequence of keys on a keyboard
- `src/external_pkgs/RoboSuite/` — Forked RoboSuite with custom Rover2026 robot
- `docker/` — Container configs for GPU/CPU environments

## HRL Pipeline
<!-- Three-stage hierarchy -->
- **CoarseReach** — move EEF within 3cm XY of target key
- **FineAlign** — precision align to 5mm using ArUco observations
- **PressKey** — extend solenoid, hold contact for 3 steps

## Robot
<!-- Details about the Rover2026 -->
- Custom Rover2026 arm with solenoid end-effector
- Two cameras: `eef_cam` (on flange, 60° FOV) and `eye_in_hand` (wrist, 75° FOV)
- Joint velocity controller, 6-DOF arm + 1-DOF solenoid

## Environment Setup
<!-- How to run locally -->
- Native: `cd src/rl_autonomy && pip install -r requirements.txt`
- Docker: `docker compose up -d rover_gpu && docker compose exec rover_gpu bash`
- Requires: MuJoCo, RoboSuite, Stable Baselines3, PyTorch

## Running
<!-- Common commands -->
- Train CoarseReach: `python train_coarse_reach.py`
- Train FineAlign: `python train_fine_align.py`
- Train PressKey: `python train_press_key.py`
- Headless: add `--no-render`
- Evaluate: `--eval checkpoints/<skill>/best_model.zip`
- TensorBoard: `tensorboard --logdir logs/<skill>`

## Code Style & Conventions
- NumPy-style docstrings
- Clear, clean, readable code — tend towards simplicity
- Use existing code as a reference when writing new code
- No unnecessary code — if it doesn't need to exist, don't write it
- Prefer modular design
- Double check everything


## Things to Know
- RoboSuite is a local fork, not pip-installed
