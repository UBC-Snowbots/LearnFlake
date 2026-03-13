# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LearnFlake is a hierarchical reinforcement learning (HRL) framework for training a Rover2026 robot arm to perform manipulation tasks (primarily lifting objects) in MuJoCo simulation via RoboSuite, with ROS2 integration for real robot control.

## Environment Setup

### Docker (recommended)

Enable X11 forwarding before starting containers:
```bash
xhost +local:docker
```

Build and start the GPU container:
```bash
docker compose build rover_gpu
docker compose --compatibility up rover_gpu -d
docker compose exec rover_gpu bash
```

### RoboSuite (inside container or venv)

Initialize submodules if empty:
```bash
git submodule update --init --recursive
```

Install RoboSuite dependencies:
```bash
cd src/external_pkgs/RoboSuite
pip install -r requirements.txt
pip install -r requirements-extra.txt
```

## Training & Evaluation

All commands run from `src/rl_autonomy/`:

```bash
# Train from scratch
python main.py --train

# Train with options
python main.py --train --num_envs 4 --episodes 2000 --domain_rand --difficulty 2

# Evaluate a checkpoint
python main.py --eval checkpoints/<timestamp>/best_model.pt

# Resume training
python main.py --resume checkpoints/<timestamp>/best_model.pt
```

Key config defaults are in `config.py` — CLI args override them.

### Visualization

```bash
cd src/external_pkgs/RoboSuite
python robosuite/demos/demo_random_action.py
```

## Architecture

### Hierarchical SAC Agent (`agent.py`)

The core agent (`HierarchicalSACAgentV3`) uses two levels:
- **High-level**: `SkillSelectorV3` picks one of 6 discrete skills every 8 steps
- **Low-level**: `SkillConditionedActorV3` outputs 7D joint velocity + gripper commands conditioned on the selected skill

**Skills**: Reach → Grasp → Lift → Hold → Recover → Return

Two separate entropy coefficients are tuned automatically (one per level). Mixed precision (FP16) training with fused AdamW optimizers when CUDA is available.

### Environment (`env_wrapper.py`)

`RoboSuiteEnvV3` wraps RoboSuite's Lift task with:
- **Obs**: ~36 features (joint pos/vel, EEF pose, gripper state, cube pos, skill one-hot)
- **Action**: 7D (6 joint velocities + 1 gripper), range [-1, 1]
- **Curriculum**: 3 difficulty levels (easy/medium/hard) controlling cube randomization range
- **Perturbations**: 2% chance of mid-episode disturbance
- `SubprocVecEnvV3` manages N parallel environments via multiprocessing

### Replay Buffer (`memory.py`)

`GPUReplayBuffer` stores transitions directly on GPU with pinned-memory staging for fast CPU→GPU PCIe transfers via CUDA streams. Capacity: 100k transitions.

### Imitation Learning Pipeline

Data flow: `demo_recorder.py` → HDF5 dataset → `bc_trainer.py` (behavioral cloning) → `bc_to_rl.py` (transfer BC weights to SAC actor with differential learning rates).

`DAgger.py` implements iterative dataset aggregation with a beta schedule controlling expert vs. policy mixture.

### ROS2 Integration

`testing/cartesian_control_ros.py` bridges MuJoCo simulation to ROS2:
- Subscribes to `/arm/sim_command` (7-float action array)
- Publishes joint states and cube position observations

## Key Files

| File | Purpose |
|------|---------|
| `src/rl_autonomy/main.py` | CLI entry point |
| `src/rl_autonomy/config.py` | All hyperparameters (edit here or use CLI) |
| `src/rl_autonomy/agent.py` | `HierarchicalSACAgentV3` |
| `src/rl_autonomy/env_wrapper.py` | MuJoCo environment with curriculum |
| `src/rl_autonomy/networks.py` | All neural network architectures |
| `src/rl_autonomy/memory.py` | GPU replay buffer |
| `src/rl_autonomy/trainer.py` | Training loop and curriculum logic |
| `src/rl_autonomy/DAgger.py` | DAgger imitation learning |
| `src/rl_autonomy/testing/cartesian_control_ros.py` | ROS2↔MuJoCo bridge — subscribes to MoveIt Servo JointTrajectory, mirrors motion in MuJoCo, publishes `/mujoco/observations`, `/mujoco/joint_states`, `/mujoco/actions` |
| `src/rl_autonomy/testing/` | Experimental scripts, BC pipeline, ROS2 bridge |
| `src/rl_autonomy/documentation/keyboard_typing_pipeline.md` | Full design doc for the keyboard typing pipeline (skills, sensors, training order) |

## Related Repository: RoverFlake2

The companion ROS2 repository lives at `../RoverFlake2` (sibling directory). It contains:
- **`src/dev_arm_description_v2/urdf/dev_arm.urdf`** — the physical arm URDF (joint names, limits, EEF flange structure)
- **`src/arm_control/src/joy_arm_control.cpp`** — joystick→MoveIt Servo→JointTrajectory pipeline; firmware axis mapping (shoulder→0 … a5_rotation→4 … a6_rotation→5)
- **`src/arm_control/src/sim_helper_node.cpp`** — translates `ArmCommand` → 7-float `Float64MultiArray` on `/arm/sim_command`
- **`src/rover_msgs/msg/`** — custom message definitions including `ArmCommand.msg` and `MoteusArmStatus.msg`
- **`src/external_pkgs/moteus_ros2/`** — Moteus motor driver; publishes `ControllerState` (with `torque`, `velocity`, `position`) on `id_{N}/state` at ~50 Hz
- **`src/rover_vision/`** and **`src/cameras_cpp/`** — RealSense (`librealsense2`) camera pipeline
- **`src/aruco_detector/`** — ArUco marker detection (used for keyboard pose estimation)

Cross-container ROS2 communication uses `network_mode: host` + `rmw_cyclonedds_cpp` + matching `ROS_DOMAIN_ID` — no bridging needed.

## Docker Services

| Service | Use case |
|---------|----------|
| `rover` | Standard ROS2 development |
| `rover_rl` | MuJoCo with display (MUJOCO_GL=glfw) |
| `rover_gpu` | GPU training (NVIDIA runtime, MUJOCO_GL=egl) |
