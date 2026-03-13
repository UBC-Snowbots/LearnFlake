# Phase 3 Integration Notes: `rl_sar` Cartesian Manager -> MoveIt Worker Bridge

This note records exactly what was implemented for Rover2026 in Phase 3.

## Goal
Enable a rover-only control mode where:
- RL policy outputs **3D Cartesian deltas** (`dx, dy, dz`)
- `rl_sar` publishes `geometry_msgs/msg/TwistStamped` to MoveIt Servo on `/servo_node/delta_twist_cmds`
- Native `robot_joint_controller/command` publishing is disabled in this rover mode to avoid dual-command conflicts
- Non-rover robots keep default `rl_sar` behavior

## Files Modified

### 1) Rover base policy config
- File: `policy/rover2026/base.yaml`
- Added keys under `rover2026`:
  - `control_bridge_mode: cartesian_moveit`
  - `cartesian_cmd_topic: /servo_node/delta_twist_cmds`
  - `cartesian_cmd_frame: base_link`

### 2) Rover runtime policy config (`robot_lab`)
- File: `policy/rover2026/robot_lab/config.yaml` (new)
- Added rover policy runtime block `rover2026/robot_lab` with:
  - `num_of_dofs: 6` (unchanged)
  - `action_scale: [0.1, 0.1, 0.1]` (length 3)
  - `clip_actions_lower: [-1.0, -1.0, -1.0]` (length 3)
  - `clip_actions_upper: [1.0, 1.0, 1.0]` (length 3)
  - bridge params:
    - `control_bridge_mode: cartesian_moveit`
    - `cartesian_cmd_topic: /servo_node/delta_twist_cmds`
    - `cartesian_cmd_frame: base_link`

### 3) `RL_Sim` bridge wiring (header)
- File: `src/rl_sar/include/rl_sim.hpp`
- Added ROS2 include:
  - `#include <geometry_msgs/msg/twist_stamped.hpp>`
- Added members in ROS2 section:
  - `bool use_cartesian_bridge_ = false;`
  - `std::string cartesian_cmd_topic_ = "/servo_node/delta_twist_cmds";`
  - `std::string cartesian_cmd_frame_ = "base_link";`
  - `rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cartesian_cmd_pub_;`

### 4) `RL_Sim` constructor mode detection + publisher
- File: `src/rl_sar/src/rl_sim.cpp`
- After loading `base.yaml`, added rover-only mode detection:
  - `use_cartesian_bridge_ = (robot_name == "rover2026" && control_bridge_mode == "cartesian_moveit")`
- Read optional params from YAML:
  - `cartesian_cmd_topic`
  - `cartesian_cmd_frame` (fallback to `base_link` if empty)
- In ROS2 publisher setup:
  - Creates `TwistStamped` publisher on `cartesian_cmd_topic_` when mode is enabled

### 5) Disable native joint command publish in cartesian rover mode
- File: `src/rl_sar/src/rl_sim.cpp`
- In `RobotControl()`:
  - If `use_cartesian_bridge_` is true, return before `SetCommand(&robot_command)`
  - This prevents publishing to `.../robot_joint_controller/command` in rover cartesian mode

### 6) RunModel cartesian branch
- File: `src/rl_sar/src/rl_sim.cpp`
- In `RunModel()`:
  1. Run inference (`model_actions = Forward()`)
  2. If `use_cartesian_bridge_`:
     - Keep observation continuity by padding `obs.actions` to length `num_of_dofs` and copying first 3 model outputs
     - Build `TwistStamped`:
       - `header.stamp = now`
       - `header.frame_id = cartesian_cmd_frame_` (fallback `base_link`)
       - `twist.linear.x/y/z = model_actions[0..2]`
       - `twist.angular.* = 0`
     - If action size != 3:
       - publish zero twist
       - throttle warning log
     - publish to `cartesian_cmd_pub_`
     - return early (skip `ComputeOutput(...)` and output queue push)
  3. Else (legacy path): existing joint-control flow remains unchanged

### 7) Legacy safety check
- File: `src/rl_sar/src/rl_sim.cpp`
- Added guard in non-cartesian path:
  - if model action size != `num_of_dofs`, skip tick and log throttled warning

## Behavior Summary
- Rover2026 + `control_bridge_mode: cartesian_moveit`:
  - Publishes `TwistStamped` to `/servo_node/delta_twist_cmds`
  - Does **not** publish native joint commands from `SetCommand()`
- All other robots/modes:
  - unchanged `rl_sar` behavior

## Validation Run Notes
- YAML parse check passed for:
  - `policy/rover2026/base.yaml`
  - `policy/rover2026/robot_lab/config.yaml`
- `colcon build --packages-select rl_sar` in current workspace failed due to missing installed dependencies:
  - `robot_msgs`
  - `robot_joint_controller`

## Quick Runtime Checks
Use these when your full workspace dependencies are built:

```bash
# Confirm cartesian command stream
ros2 topic echo /servo_node/delta_twist_cmds

# Confirm native joint command topic is idle for rover cartesian mode
ros2 topic hz /rover2026_gazebo/robot_joint_controller/command
```
