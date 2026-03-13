# rl_sar in LearnFlake: Deployment and Inference Role

This note clarifies how `rl_sar` is used in this repository.

## What `rl_sar` is for
`rl_sar` is primarily a **runtime/deployment framework** for robot control policies.

It provides:
- Robot runtime loops (sim/real)
- ROS/Gazebo/MuJoCo integration
- Policy inference with libtorch/TorchScript (`.pt`)
- Joint command publishing / robot control interfaces
- FSM/state machine control flow

In short: **load trained policy -> run inference -> command robot**.

## What `rl_sar` is not (in this repo)
For LearnFlake integration, `rl_sar` is **not** the full end-to-end high-level RL trainer for custom robots.

For training new policies, upstream `rl_sar` docs point to external training stacks such as:
- `robot_lab` (IsaacLab / IsaacSim)
- `legged_gym` / `himloco` (IsaacGym-based)

Then export/copy a TorchScript policy into `rl_sar` policy folders.

## Rover2026-specific usage here
Your rover policy config path is:
- `policy/rover2026/robot_lab/config.yaml`

Expected trained model location:
- `policy/rover2026/robot_lab/policy.pt`

Model load path in code:
- `src/rl_sar/library/core/rl_sdk/rl_sdk.cpp`
  - `model_path = POLICY_DIR + "/" + robot_config_path + "/" + model_name`

## Phase 3 bridge in this workspace
Rover2026 has a rover-only cartesian bridge mode integrated in `rl_sar` runtime:
- Policy output interpreted as `[dx, dy, dz]`
- Published as `geometry_msgs/msg/TwistStamped` to `/servo_node/delta_twist_cmds`
- Native `robot_joint_controller/command` is suppressed in this mode to avoid dual commands

Detailed patch notes:
- `src/robots/rover2026_description/PHASE3_CARTESIAN_BRIDGE.md`

## Practical workflow
1. Train policy externally (or in your own trainer) and export TorchScript `.pt`.
2. Place model at `policy/rover2026/robot_lab/policy.pt`.
3. Run `rl_sar` with rover config for inference/deployment.
4. Use MoveIt worker stack to execute low-level IK/joint control.
