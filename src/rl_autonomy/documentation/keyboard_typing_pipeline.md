# Keyboard Typing Pipeline — Full Design Document

**Goal**: Train the Rover2026 arm to type on a keyboard using a linear actuator mounted on the end-effector (EEF), with a camera and depth sensor on/near the EEF for visual feedback.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Hardware Model Changes](#2-hardware-model-changes)
3. [Simulation Environment](#3-simulation-environment)
4. [Observation & Sensor Pipeline](#4-observation--sensor-pipeline)
5. [Debugging & Diagnostics](#5-debugging--diagnostics)
6. [Individual Skill Design](#6-individual-skill-design)
7. [Skill Training Strategy](#7-skill-training-strategy)
8. [Skill Orchestration (High-Level Policy)](#8-skill-orchestration-high-level-policy)
9. [ROS2 Bridge Updates](#9-ros2-bridge-updates)
10. [Sim-to-Real Considerations](#10-sim-to-real-considerations)
11. [Implementation Phases (Ordered)](#11-implementation-phases-ordered)

---

## 1. Architecture Overview

The current HRL approach tried to learn everything end-to-end in a single monolithic agent — this is fragile because the reward landscape is too sparse and recovery from any sub-task failure cascades. The new architecture separates concerns completely:

```
High-Level Orchestrator (state machine or lightweight learned policy)
        │
        ├─ Skill: CoarseReach     ← move EEF to region above keyboard
        ├─ Skill: FineAlign       ← camera-guided, align tip above target key
        ├─ Skill: PressKey        ← extend linear actuator, detect contact
        ├─ Skill: Retract         ← retract linear actuator cleanly
        └─ Skill: MoveToNext      ← reposition to next key (loop back to FineAlign)
```

Each skill is:
- Trained independently with its own environment, reward function, and observation space
- A self-contained SAC or BC policy saved as a checkpoint
- Swappable — if one skill regresses, retrain only that one

The high-level orchestrator is initially a **rule-based state machine** (no learning needed for MVP). Once all skills are stable it can optionally be replaced by a learned selector.

---

## 2. Hardware Model Changes

These changes must happen in both the **MuJoCo XML model** for Rover2026 and the **physical robot URDF** in RoverFlake2 (`src/dev_arm_description_v2/urdf/dev_arm.urdf`).

### Arm joint reference (from RoverFlake2 URDF)

| Firmware axis | URDF joint name | Type | Notes |
|---|---|---|---|
| 0 | `shoulder_joint` | continuous | base rotation |
| 1 | `link_1_joint` | revolute | |
| 2 | `link1_link2` | revolute | |
| 3 | `a4_rotation` | revolute | wrist flex |
| 4 | `a5_rotation` | revolute | **best candidate for contact detection** |
| 5 | `a6_rotation` | continuous | wrist roll, nearest to EEF |
| — | `finger_left_joint` | prismatic | **replaced by linear actuator** |
| — | `finger_right_joint` | prismatic | **replaced by linear actuator** |

EEF chain: `a6_EE_holder` → `ee_base_link` (fixed, +0.1m). The linear actuator mounts at `ee_base_link`, which is the physical flange.

### 2a. Linear Actuator on EEF

The linear actuator physically swaps onto the arm's EEF flange (replacing the finger gripper). It is **not Moteus-driven** — it has its own separate actuator (type TBD). In the model it is represented as a **prismatic joint** extending along the EEF Z-axis.

**MuJoCo XML addition** — attach to the `ee_base_link` body (which corresponds to the flange in the URDF):
```xml
<body name="linear_actuator" pos="0 0 0">
    <joint name="actuator_slide" type="slide" axis="0 0 1"
           range="0.0 0.04"      <!-- 4 cm stroke, 0 = retracted -->
           damping="5.0" stiffness="0.0"/>
    <geom name="actuator_tip" type="cylinder"
          size="0.005 0.015"     <!-- 5 mm radius, 1.5 cm half-length -->
          pos="0 0 0.02" rgba="0.3 0.3 0.3 1"/>
    <site name="actuator_tip_site" pos="0 0 0.035" size="0.004"/>
</body>
```

**In RoverFlake2 URDF**: Remove `finger_left_joint` and `finger_right_joint`, add the `actuator_slide` prismatic joint instead. The MoveIt `arm` move group (which already excludes the fingers) does not need to change.

**Action space**: The 7th action dimension was the gripper. It becomes the **linear actuator velocity** (positive = extend, negative = retract), keeping the action space at 7D. The `sim_helper_node.cpp` currently fills `action[6]` from `msg.end_effector` — this slot will now carry the actuator command instead, requiring no structural changes.

### 2b. Contact Detection — Moteus Torque / Current Spike (no touch sensor needed)

Touch sensors are unreliable and not guaranteed to be available. Instead, contact is detected by monitoring the **motor torque** (equivalently, q-axis current) of the linear actuator's Moteus controller. This is a well-established technique in robot force control.

**How it works**:

- When the actuator tip is travelling freely through air, motor torque is low (just overcoming inertia and friction).
- The instant the tip contacts a key surface, the motor load increases → torque spikes.
- A simple threshold: `contact = (torque > CONTACT_THRESHOLD)`.
- Additional confirmation: **velocity stall** — if the commanded velocity is nonzero but measured velocity drops to near-zero, the tip has hit something.
- Combining both signals reduces false positives: `contact = (torque > T_thresh) AND (velocity < V_thresh)`.

**Real robot** — use `a5_rotation` (firmware axis 4, the penultimate wrist joint): Since the linear actuator has no Moteus controller of its own, we piggyback on the nearest arm joint that will feel the press force through the kinematic chain. When the tip presses down, the reaction torque propagates up the arm. `a5_rotation` is a revolute joint whose flex axis is perpendicular to the press direction, meaning a downward press force creates a bending moment on it — this gives the strongest torque signal among the arm joints.

The `moteus_node` already publishes `ControllerState` on `id_{N}/state` (where N is the Moteus controller ID assigned to `a5_rotation`). Subscribe to this topic and run the threshold check. At ~50 Hz this is fast enough for 20 Hz control.

**In simulation** (MuJoCo): MuJoCo computes contact forces internally. Instead of a `touch` sensor XML element, read the external force applied to the actuator tip body directly:

```python
def _get_contact_force(self):
    """Return the magnitude of contact force on the actuator tip body.
    Simulates what the Moteus torque reading would look like on real hardware.
    """
    body_id = self.sim.model.body_name2id('linear_actuator')
    # cfrc_ext: (nbody, 6) — external wrench [torque(3), force(3)] per body
    force_vec = self.sim.data.cfrc_ext[body_id][3:]   # take the force part
    return np.linalg.norm(force_vec)

def _contact_detected(self):
    """Threshold-based contact detection matching real Moteus logic."""
    force = self._get_contact_force()
    actuator_vel = self.sim.data.qvel[self._actuator_qvel_addr]
    return force > CONTACT_FORCE_THRESHOLD and abs(actuator_vel) < STALL_VEL_THRESHOLD
```

**Thresholds to tune** (start with these, adjust from observation):
- `CONTACT_FORCE_THRESHOLD = 2.0` N in simulation (keyboard key requires ~0.5–1.5 N actuation force, set slightly above that)
- `STALL_VEL_THRESHOLD = 0.005` m/s (nearly stopped)
- Real hardware: profile by pressing the actuator against the keyboard surface while logging `id_{a5_id}/state.torque` and `id_{a5_id}/state.velocity`. The crossover point where torque jumps and velocity drops sets `CONTACT_TORQUE_THRESHOLD`. The baseline (arm holding steady, no press) must be subtracted since gravity loads are always present.

**Important**: Because we are reading from an arm joint rather than the actuator motor directly, the signal includes gravity and inertia contributions. To isolate contact: capture `torque_baseline` at the start of each press attempt (actuator stationary, before extending), then `contact = (torque - torque_baseline) > DELTA_THRESHOLD`. The delta removes pose-dependent gravity offsets.

**Why this is better than a touch sensor**: It transfers directly to real hardware with zero additional sensors, produces a continuous signal (not just binary), and lets you infer approximate contact force — useful for not destroying keys.

### 2c. EEF Camera

A small camera is physically mounted on the side of the EEF (near the actuator tip), angled slightly downward to see keys directly below. This gives a close-up view that is always aligned with the tip regardless of arm pose — exactly what FineAlign needs.

**MuJoCo camera definition** — inside the `ee_base_link` body (the flange):
```xml
<!-- Side-mounted, angled ~30° downward toward the tip -->
<camera name="eef_cam" pos="0.02 0 -0.01"
        euler="-60 0 0"
        fovy="60"/>
```
Adjust `pos` and `euler` once the physical mount geometry is known.

**Resolution**: 64×64 for training. The policy only needs to see which key is below and how centered it is — high resolution is wasted compute.

**Key detection from image** (used in FineAlign): ArUco markers on the keyboard corners are detected by the existing `aruco_detector` package in RoverFlake2. Combined with the known keyboard layout, the target key's pixel position in the EEF camera frame is computed, giving a (dx, dy) offset from image center → directly useful as a 2D observation for FineAlign. This avoids end-to-end pixel training.

**Camera mode**: `has_offscreen_renderer=True` and `use_camera_obs=True` only for FineAlign. CoarseReach and PressKey use proprioception only.

**Depth / Rangefinder** — a MuJoCo rangefinder sensor at the actuator tip gives a cheap scalar height-above-surface reading:
```xml
<sensor>
    <rangefinder name="eef_rangefinder" site="actuator_tip_site" cutoff="0.3"/>
</sensor>
```
On the real robot, this can be approximated from the EEF camera depth channel if the camera has it, or omitted and replaced by the actuator position reading (if fully retracted + known keyboard height = known gap).

---

## 3. Simulation Environment

### 3a. Keyboard Scene

Create a new MuJoCo XML scene file `keyboard_scene.xml` (placed alongside the Rover2026 model) containing:

- A flat table body (reuse existing table from Lift task, or a new surface)
- A keyboard body: flat slab geometry, with individual key bodies as children
- Each key is a small box geom, slightly raised, with a unique name `key_<letter>`
- Key positions are defined in a fixed grid (standard QWERTY layout, scaled to ~70% of real keyboard size to fit on the table within arm reach)
- A `target_key_site` is a MuJoCo site that gets updated to the position of whichever key the agent should press next — this is the "goal" fed into the observation

**Key contact detection**: Each key body has a `touch` sensor or simply use `geom contact` detection via MuJoCo's `mjData.contact` array. When the actuator tip geom contacts a key geom, the key is considered "pressed". Optionally add a small spring joint on each key body (range 0–2mm) to simulate key travel.

**Initial arm pose**: The arm starts in a neutral position with the EEF above the keyboard, actuator retracted. Episode resets return to this pose.

### 3b. Environment Class: `KeyboardEnv`

New file: `src/rl_autonomy/keyboard_env.py`

Key design choices:
- Inherits nothing from `RoboSuiteEnvV3` — that environment was Lift-specific
- Uses a custom reward function per skill (see Section 6)
- Supports both proprioceptive-only mode (CoarseReach) and camera mode (FineAlign)
- The `target_key` is set externally by the orchestrator (e.g., `env.set_target_key('a')`)
- Episode terminates on: successful press + retract, or timeout

**Observation modes**:
- `obs_mode='proprio'`: 6D joint pos + 6D joint vel + 7D EEF pose (pos + quat) + 1D actuator pos + 3D target key position + 3D EEF-to-key vector + 1D rangefinder = **~31D**
- `obs_mode='visual'`: proprio + flattened camera image (64×64×3 = 12288D) OR CNN-encoded features (256D)

For initial training, use `proprio` only. Add visual input only for FineAlign once proprioceptive training is stable.

---

## 4. Observation & Sensor Pipeline

### 4a. What goes in the observation vector per skill

| Field | Dim | Skills that use it |
|-------|-----|-------------------|
| Joint positions (6D) | 6 | All |
| Joint velocities (6D) | 6 | All |
| EEF position (3D) | 3 | All |
| EEF quaternion (4D) | 4 | All |
| Actuator extended flag (1D, binary 0/1) | 1 | PressKey, Retract — solenoid has no continuous position |
| Target key position in world (3D) | 3 | All |
| EEF-to-target-key vector (3D) | 3 | CoarseReach, FineAlign |
| Rangefinder reading (1D) | 1 | FineAlign, PressKey |
| Simulated contact force magnitude (1D) | 1 | PressKey — maps to Moteus `torque` on real hardware |
| Actuator velocity (1D) | 1 | PressKey — stall confirmation |
| Camera image (64×64×3, flattened or CNN-encoded) | 12288 or 256 | FineAlign |

### 4b. Camera Image Processing

The EEF camera does not need to be literally rendered every training step. Instead, the ArUco pipeline output `(dx, dy, key_visible)` is **synthesized with realistic noise and failure modes** from MuJoCo ground truth. This matches what the real `aruco_detector` node will produce on hardware without the render cost.

**Synthesized ArUco observation** (implemented in `KeyboardEnv._get_aruco_observation()`):

```python
def _get_aruco_observation(self):
    # Ground truth offset from MuJoCo
    dx_gt, dy_gt = self._compute_eef_to_key_offset()
    dist = np.linalg.norm([dx_gt, dy_gt])

    # Simulate detection failure: marker out of view or EEF tilted too far
    eef_angle_penalty = abs(self._eef_tilt_from_vertical())
    p_visible = 1.0 if dist < 0.05 and eef_angle_penalty < 0.3 else max(0.0, 1.0 - dist / 0.12)
    key_visible = float(np.random.rand() < p_visible)

    if not key_visible:
        return np.array([0.0, 0.0, 0.0])  # dx, dy, visible

    # Realistic ArUco noise: ~1mm std at close range
    noise = np.random.normal(0, 0.001, size=2)
    return np.array([dx_gt + noise[0], dy_gt + noise[1], 1.0])
```

**Why this matters for sim-to-real**: If `(dx, dy)` were injected as perfect ground truth, the FineAlign policy would never learn to handle dropped detections (`key_visible = False`) or noisy measurements — both of which occur routinely on real hardware at close range and non-ideal angles. The synthesized signal ensures the policy is robust to these before it ever touches hardware.

**What the policy sees** (3D observation addition): `[dx, dy, key_visible]` — same format whether running in sim (synthesized) or on real hardware (from `aruco_detector`). No code change at the policy layer when switching to hardware.

**Camera rendering**: The EEF camera is still defined in the MuJoCo XML and can be rendered on demand for diagnostics (`env_diagnostics.py`, `rqt_image_view`). It is not rendered every training step.

### 4c. ROS2 Sensor Topics (new additions to cartesian_control_ros.py or a new node)

| Topic | Type | Content |
|-------|------|---------|
| `/mujoco/eef_camera` | `sensor_msgs/Image` | EEF camera RGB (64×64), published from MuJoCo offscreen render |
| `/mujoco/eef_rangefinder` | `std_msgs/Float32` | Scalar distance below EEF |
| `/mujoco/actuator_pos` | `std_msgs/Float32` | Linear actuator extension (m) |
| `/mujoco/actuator_torque` | `std_msgs/Float32` | Simulated contact force (maps to `id_N/state.torque` on real hardware) |
| `/mujoco/contact_detected` | `std_msgs/Bool` | Thresholded contact state (torque+stall combined) |
| `/mujoco/target_key` | `std_msgs/String` | Current target key name |
| `/mujoco/joint_states` | `sensor_msgs/JointState` | Already exists — include actuator joint |
| `/mujoco/observations` | `std_msgs/Float64MultiArray` | Already exists |

---

## 5. Debugging & Diagnostics

Everything needs to be inspectable at runtime. Here is what you need and how to get it:

### 5a. Terminal: quick observation dump

Add a `--debug` flag to the bridge node. When enabled, every N steps it prints a formatted table:
```
[t=0142] EEF: [0.312, -0.021, 0.412]  Actuator: 0.008m  Touch: False
         Target: key_a @ [0.288, -0.019, 0.350]  Δ: [0.024, -0.002, 0.062]
         Rangefinder: 0.063m  Joints: [0.12, -0.45, 0.33, ...]
```

### 5b. ROS2 topic echo

All observations are published as topics (see 4c). You can inspect them live:
```bash
# Joint states
ros2 topic echo /mujoco/joint_states

# EEF distance to target
ros2 topic echo /mujoco/observations   # look at the EEF-to-key fields

# Actuator state
ros2 topic echo /mujoco/actuator_pos
ros2 topic echo /mujoco/touch
```

### 5c. Camera feed visualization

```bash
# In the RoverFlake2 container or LearnFlake container:
ros2 run rqt_image_view rqt_image_view
# Then select /mujoco/eef_camera or /mujoco/eef_depth
```

Or headlessly record a few frames to disk for inspection:
```bash
ros2 run image_transport republish raw in:=/mujoco/eef_camera out:=/mujoco/eef_camera_raw
```

### 5d. Diagnostic script: `src/rl_autonomy/testing/env_diagnostics.py`

A standalone script (no ROS2 needed) that:
1. Creates the `KeyboardEnv` directly
2. Steps with zero actions
3. Prints the full observation vector with labeled fields
4. Optionally renders and saves camera images to disk
5. Lets you manually set `target_key` and verify the observation vectors update correctly

This is for verifying the environment before starting any training.

### 5e. TensorBoard scalars during training

Each skill training run logs:
- Episode reward
- Phase-specific reward components (reach error, contact events, etc.)
- Actuator position at press time
- Distance to target at episode end
- Success rate (rolling 100-episode window)

---

## 6. Individual Skill Design

### Skill 1: CoarseReach

**Objective**: Move the EEF to a position within ~3 cm of being directly above the target key.

**Observation**: Proprioceptive only — joints + EEF pose + target key position + EEF-to-key vector.

**Action**: 6D joint velocities + actuator velocity locked to 0 (retracted).

**Reward**:
```python
# Dense: exponential pull toward target XY
xy_dist = np.linalg.norm(eef_pos[:2] - target_key_pos[:2])
r_reach = 10.0 * np.exp(-5.0 * xy_dist)

# Height penalty: EEF should be ~5cm above key, not too high
z_error = abs((eef_pos[2] - target_key_pos[2]) - 0.05)
r_height = -2.0 * z_error

# Success: within 3cm XY and correct height
if xy_dist < 0.03 and z_error < 0.015:
    r_success = 100.0
    done = True
```

**Termination**: Success (within threshold) or timeout (300 steps).

**Training**: Pure SAC, proprioceptive input, ~500 episodes with curriculum (start with key in center, then random key positions).

---

### Skill 2: FineAlign

**Objective**: Precisely center the actuator tip directly above the target key, within ~5mm XY error.

**Observation**: Proprioceptive (same as CoarseReach, smaller XY range) + depth rangefinder + optionally camera-derived (dx, dy) offset.

**Action**: 6D joint velocities (small magnitude, this is fine motion) + actuator velocity locked to 0.

**Reward**:
```python
xy_dist = np.linalg.norm(tip_pos[:2] - target_key_pos[:2])
z_dist = abs(tip_pos[2] - (target_key_pos[2] + 0.01))  # 1cm above key surface

r_align = 20.0 * np.exp(-50.0 * xy_dist)   # very tight gradient
r_height = 5.0 * np.exp(-20.0 * z_dist)

# Success: within 5mm XY, correct height
if xy_dist < 0.005 and z_dist < 0.005:
    r_success = 200.0
    done = True

# Penalty for large movements (this is precision work)
r_smooth = -0.5 * np.linalg.norm(action[:6])
```

**Training**: BC first using joystick demonstrations (demo_recorder.py), then SAC fine-tuning. The BC phase gives a good initialization — the SAC phase makes it precise. Use camera input once the proprioceptive version is working.

---

### Skill 3: PressKey

**Objective**: Extend the linear actuator to press the key, hold briefly, then signal completion.

**Observation**: Proprioceptive (joints + EEF) + actuator position + actuator velocity + simulated contact force + rangefinder.

**Action**: All 6 joint velocities locked to ~0 (very small), actuator velocity as the only meaningful dimension. The arm should stay still; only the actuator moves.

**Reward**:
```python
actuator_pos = obs['actuator_pos']       # 0.0 = retracted, 0.04 = fully extended
contact_force = obs['contact_force']     # MuJoCo cfrc_ext magnitude (→ Moteus torque on real HW)
actuator_vel = obs['actuator_vel']

# Progress toward contact (sparse shaping)
r_extend = 5.0 * actuator_pos

# Contact detection: torque spike + velocity stall (mirrors real Moteus detection)
contact = contact_force > CONTACT_FORCE_THRESHOLD and abs(actuator_vel) < STALL_VEL_THRESHOLD

if contact:
    r_contact = 500.0
    # Hold contact for N steps = confirmed key press (debounce)
    contact_steps += 1
    if contact_steps >= 3:
        r_success = 1000.0
        done = True

# Penalty for joint drift (arm should stay still during press)
joint_drift = np.linalg.norm(joint_vel)
r_stability = -2.0 * joint_drift
```

**Termination**: Contact held for 3 steps (success), actuator fully extended without contact (failure — misalignment), or timeout (50 steps).

**Sim-to-real note**: The `contact_force` observation in simulation uses `sim.data.cfrc_ext`. On the real robot, replace this with the `torque` field from `id_{actuator_id}/state` published by `moteus_node`. The threshold values need to be re-tuned once on real hardware by commanding a slow press against a known surface and logging `id_N/state.torque`.

---

### Skill 4: Retract

**Objective**: Retract the solenoid actuator.

**Implementation**: Solenoids are spring-return — cutting power retracts them immediately. This is a **single command**, not a policy:
```python
def retract_policy(obs):
    # Set action[6] = 0 → solenoid de-energized → spring pulls it back
    action = np.zeros(7)
    action[6] = 0.0
    return action  # one step, done immediately
```
No training, no observation needed. The orchestrator simply emits this command for one step and immediately transitions to the next state.

---

### Skill 5: MoveToNext (optional, later)

**Objective**: After pressing one key, reposition to be above the next key. Essentially the same as CoarseReach but initialized from whatever pose PressKey ended in.

In the MVP, this can just re-invoke CoarseReach with the new target key. Only create a separate MoveToNext skill if CoarseReach doesn't generalize well between keys in different parts of the keyboard.

---

## 7. Skill Training Strategy

### Order of training

```
Step 1: CoarseReach  (proprioceptive SAC, simplest, verifies env works)
Step 2: Retract      (hard-coded, no training)
Step 3: PressKey     (SAC with touch reward, builds on stable EEF)
Step 4: FineAlign    (BC demo recording → SAC fine-tune, hardest skill)
Step 5: Integration test (chain all skills, fix transitions)
```

### Per-skill training setup

Each skill gets its own:
- Wrapper class in `keyboard_env.py`: `CoarseReachEnv`, `FineAlignEnv`, `PressKeyEnv`
- Dedicated training script (e.g. `train_coarse_reach.py`)
- Checkpoint directory (e.g. `checkpoints/coarse_reach/`)
- TensorBoard log directory

Shared between all skills:
- `KeyboardEnv` base class (handles MuJoCo setup, sensor reading, action application)
- `flat_sac_baseline.py` or a simplified SAC (no hierarchy needed per skill)
- The same replay buffer (`GPUReplayBuffer`)

### On BC for FineAlign

Record ~50 joystick demonstrations using `demo_recorder.py` for the FineAlign skill specifically. The demo format is the same HDF5 schema defined in `notes.md`. Then:

1. Train BC on the demos: `python bc_trainer.py --skill fine_align --demos fine_align_demos.hdf5`
2. Load BC weights into SAC actor: `python bc_to_rl.py --bc_checkpoint fine_align_bc.pt --output fine_align_rl_init.pt`
3. Run SAC from that initialization: `python train_fine_align.py --pretrained fine_align_rl_init.pt`

---

## 8. Skill Orchestration (High-Level Policy)

### MVP: Rule-based state machine

```python
class KeyboardOrchestrator:
    states = ['coarse_reach', 'fine_align', 'press', 'retract', 'done']

    def step(self, obs):
        if self.state == 'coarse_reach':
            action = coarse_reach_policy(obs)
            if coarse_reach_success(obs):  # within 3cm
                self.state = 'fine_align'

        elif self.state == 'fine_align':
            action = fine_align_policy(obs)
            if fine_align_success(obs):    # within 5mm
                self.state = 'press'

        elif self.state == 'press':
            action = press_key_policy(obs)
            if press_success(obs):         # contact held 3 steps
                self.state = 'retract'

        elif self.state == 'retract':
            action = retract_policy(obs)
            if retract_done(obs):          # actuator pos < 0.001
                self.next_key()
                self.state = 'coarse_reach' if more_keys else 'done'

        return action
```

Success conditions (`coarse_reach_success`, etc.) are the same thresholds used in the reward functions during training — no new logic needed.

### Future: learned selector

Once the rule-based version is working reliably, the orchestrator can be replaced with a small learned policy (2-layer MLP) that takes `[eef_pos, target_key_pos, actuator_pos, touch_sensor, current_skill_one_hot]` and selects the next skill. Train this with simple imitation learning from the rule-based orchestrator's rollouts.

---

## 9. ROS2 Bridge Updates

The existing `cartesian_control_ros.py` needs these additions:

### 9a. New published topics

```python
# Camera image
from sensor_msgs.msg import Image
self.eef_cam_pub = self.create_publisher(Image, '/mujoco/eef_camera', 10)

# Depth image
self.depth_pub = self.create_publisher(Image, '/mujoco/eef_depth', 10)

# Scalar sensors
from std_msgs.msg import Float32, Bool, String
self.rangefinder_pub = self.create_publisher(Float32, '/mujoco/eef_rangefinder', 10)
self.actuator_pub = self.create_publisher(Float32, '/mujoco/actuator_pos', 10)
self.touch_pub = self.create_publisher(Bool, '/mujoco/touch', 10)
self.target_key_pub = self.create_publisher(String, '/mujoco/target_key', 10)
```

### 9b. Actuator joint in JointState

The existing `/mujoco/joint_states` topic should include the linear actuator joint:
```python
js_msg.name = list(self.URDF_JOINT_NAMES) + ['linear_actuator']
js_msg.position = qpos.tolist() + [actuator_qpos]
js_msg.velocity = qvel.tolist() + [actuator_qvel]
```

### 9c. Action subscription change

The incoming action topic `/arm_controller/joint_trajectory` will now control 6 arm joints + linear actuator. The MoveIt Servo config in RoverFlake2 needs to be aware of the actuator joint, or the actuator is controlled via a separate `/actuator/command` topic (simpler to start).

---

## 10. Sim-to-Real Considerations

### EEF Camera
The camera is physically mounted on the EEF. A ROS2 driver node in RoverFlake2 must publish it as `sensor_msgs/Image` — use the existing `cameras_cpp` pipeline as a template.

For keyboard localization: place two ArUco markers on the keyboard corners. The existing `aruco_detector` package in RoverFlake2 returns marker poses in the camera frame. Since the camera is EEF-mounted, the transform from camera frame to world frame is `T_world_eef * T_eef_cam` (both known at runtime from joint states + calibrated mount offset). Combined with the known keyboard layout, all key pixel positions and world positions are computed without any ML. **Decision (Q4)**: Use ArUco from the start rather than pure proprioception — it gives exact key positions and removes a major source of uncertainty for FineAlign.

### Linear actuator
- Add `actuator_slide` joint to `dev_arm.urdf` at `ee_base_link`, remove finger joints
- Write a ROS2 driver node (in RoverFlake2) that subscribes to `/actuator/command` (`std_msgs/Float32`, normalized -1 to +1) and sends the appropriate PWM/serial/CAN command to the physical actuator
- The MoveIt `arm` move group already excludes the fingers, so MoveIt Servo config needs no changes

### Contact detection — `a5_rotation` torque delta
- Before each press attempt, record `torque_baseline` from `id_{a5_id}/state.torque` while the arm is stationary
- During press: `contact = (current_torque - torque_baseline) > DELTA_THRESHOLD AND velocity < STALL_VEL`
- Run the calibration profiling script once to set `DELTA_THRESHOLD`

### Domain randomization for sim-to-real
When training, randomize:
- Keyboard position (±2cm XY from nominal)
- Camera image brightness/contrast ±20% (simulates varying lighting)
- Actuator press speed ±10%
- `a5_rotation` torque noise (add Gaussian noise matching real Moteus measurement noise)

The torque noise term is the most important — prevents the contact detector from overfitting to perfectly clean simulated forces.

---

## 11. Implementation Phases (Ordered)

Each phase produces something runnable and testable before the next phase begins.

---

### Phase 1 — Robot model updates

**Goal**: Rover2026 MuJoCo model has a working linear actuator and EEF camera. Nothing else changes.

**Tasks**:
1. Locate the Rover2026 MuJoCo XML in `src/external_pkgs/RoboSuite/robosuite/models/robots/`
2. Replace finger joint bodies with the linear actuator body (prismatic joint + geom + tip site) at `ee_base_link`
3. Add the EEF-mounted camera inside the `ee_base_link` body
4. Add the rangefinder sensor at `actuator_tip_site`
5. Run `demo_random_action.py` and verify the model loads without errors and `action_dim` is still 7
6. Verify `sim.data.cfrc_ext` reads nonzero when the actuator tip geom overlaps another geom (use a test script)

**Testable output**: `python robosuite/demos/demo_random_action.py` shows the arm with the linear actuator and the EEF camera renders.

---

### Phase 2 — KeyboardEnv base class

**Goal**: A standalone Python environment `keyboard_env.py` that wraps the updated robot model in a keyboard scene, with full sensor readback. No RL, no ROS2 yet.

**Tasks**:
1. Create `keyboard_scene.xml` with a flat table and a simple keyboard (initially just 3–5 keys for testing)
2. Create `KeyboardEnv` class: `reset()`, `step(action)`, `get_obs()`, `set_target_key(name)`
3. Wire up sensors: actuator position, touch sensor, rangefinder, camera image
4. Write `env_diagnostics.py` test script that creates the env and prints all sensor values

**Testable output**: `python testing/env_diagnostics.py` prints sensor values for a zero-action rollout. Camera images saved to `/tmp/eef_cam_test.png`.

---

### Phase 3 — Debugging & Observation Validation

**Goal**: Every observation field is labeled, correct, and visible from ROS2 topics and terminal.

**Tasks**:
1. Update `cartesian_control_ros.py` to publish new sensor topics (camera, depth, actuator, touch)
2. Add `--debug` flag with formatted terminal output
3. Verify all topics appear in `ros2 topic list`
4. View camera feed in `rqt_image_view`
5. Echo joint states and confirm actuator joint is included
6. Confirm observation vector dimensions match what `KeyboardEnv.obs_dim` reports

**Testable output**: Can run the bridge node, echo all topics, and visually inspect camera feed in rqt.

---

### Phase 4 — CoarseReach skill training

**Goal**: A working SAC policy that moves the EEF to within 3cm above any key on the keyboard (no camera, proprioceptive only).

**Tasks**:
1. Create `CoarseReachEnv` subclass with appropriate reward and termination
2. Create `train_coarse_reach.py` training script
3. Train for ~500 episodes
4. Evaluate: success rate > 80% across random key targets
5. Save checkpoint to `checkpoints/coarse_reach/best_model.pt`

**Testable output**: Run evaluation, visually confirm arm moves to hover above randomly selected keys.

---

### Phase 5 — PressKey skill training

**Goal**: SAC policy that extends the actuator to make contact and hold for 3 steps, assuming the arm is already aligned (use CoarseReach to get there first in testing).

**Tasks**:
1. Create `PressKeyEnv` subclass
2. Confirm `cfrc_ext` contact force reads correctly when actuator tip contacts a key (use `env_diagnostics.py`)
3. Tune `CONTACT_FORCE_THRESHOLD` and `STALL_VEL_THRESHOLD` in simulation
4. Train for ~300 episodes
5. Evaluate: contact detection success rate > 90%
6. Save checkpoint

**Testable output**: Combined CoarseReach → PressKey rollout successfully presses a key.

---

### Phase 6 — Demo recording for FineAlign

**Goal**: ~50 high-quality human demonstrations of the FineAlign task recorded to HDF5.

**Tasks**:
1. Update `demo_recorder.py` to use `KeyboardEnv` observation format
2. Run bridge node, drive arm with joystick to perform fine alignment above various keys
3. Record 50 demonstrations, verify HDF5 quality (no NaNs, correct obs_dim)
4. Run BC training: `python bc_trainer.py --skill fine_align --demos fine_align_demos.hdf5`
5. Evaluate BC policy visually

---

### Phase 7 — FineAlign skill training (BC → SAC)

**Goal**: Policy that reliably aligns the EEF tip within 5mm of the target key.

**Tasks**:
1. Load BC weights into SAC actor via `bc_to_rl.py`
2. Run SAC fine-tuning from BC initialization
3. Add camera-derived (dx, dy) observation input (OpenCV key detection)
4. Evaluate: 5mm alignment success rate > 85%

---

### Phase 8 — Orchestrator integration test

**Goal**: String all skills together with the rule-based orchestrator and type a simple word (e.g., "hi").

**Tasks**:
1. Implement `KeyboardOrchestrator` state machine
2. Set target key sequence: `['h', 'i']`
3. Run full end-to-end rollout in simulation
4. Debug transition failures (most common: state machine triggering too early/late)
5. Tune success condition thresholds based on observed behavior

**Testable output**: Arm types "hi" in simulation reliably (> 70% full sequence success).

---

### Phase 9 — Sim-to-real bridge

**Goal**: Orchestrator policy running on real arm via ROS2 bridge.

**Tasks**:
1. Add actuator joint to RoverFlake2 URDF
2. Write ROS2 driver node for real linear actuator
3. Hand-eye calibration for EEF camera
4. Run ArUco-based keyboard pose estimation
5. Test CoarseReach first on real arm
6. Incrementally add FineAlign, PressKey
7. Full end-to-end typing test

---

## Core Principle: Simulation-to-Real Pipeline Fidelity

**The simulation must mirror the exact software pipeline the real robot will use.** Every sensor that exists on the real robot must be rendered in simulation and consumed through the same code path. Skipping a sensor (e.g. not rendering the EEF camera and injecting ground-truth key positions directly) produces a policy that will fail on hardware. Concretely:

- The EEF camera **must** be rendered in MuJoCo every step. The policy must receive the processed camera output (ArUco-derived `(dx, dy)` offset), not injected ground truth.
- ArUco detection must run on the rendered image using the same `aruco_detector` code path used on the real robot — not a lookup table or direct MuJoCo coordinate read.
- Contact detection must go through the synthetic Moteus torque topic, not a direct `cfrc_ext` read. The policy sees the same noisy torque signal it will see on hardware.
- The solenoid command must go through a `/actuator/command` CAN FD topic, same as real hardware.

Any observation field that will exist on the real robot must exist and be populated through the simulated pipeline from day one. Omitting it during training is a non-starter.

---

## Resolved Decisions

| # | Question | Decision |
|---|----------|----------|
| 1 | Keyboard placement | Flat on a level table directly in front of the arm. A 96-key URDF will be created with correct collision geometry (see Phase 1). |
| 2 | Linear actuator type | **Solenoid** — binary state only (retracted / extended). Known stroke length, no position feedback. `actuator_extended` observation is a binary flag (0.0 / 1.0) in both sim and real. Action dim 6 is binary: values > 0 = extend, ≤ 0 = retract. |
| 3 | Moteus ID for `a5_rotation` | Real arm not available yet; ID resolved when hardware is wired. Simulation uses a placeholder `id_5`. The synthetic publisher is the authoritative source for the contact signal — the policy never reads `cfrc_ext` directly. |
| 4 | FineAlign approach | **ArUco from the start, running on the rendered EEF camera image.** ArUco detection runs on every frame using OpenCV + `aruco_detector` code path. Output `(dx, dy, key_visible)` fed into the observation. No ground-truth key injection. |
| 5 | EEF camera placement | Physical mount geometry is not known yet. Anchor a camera body to `ee_base_link` with a reasonable placeholder (`pos="0.02 0 -0.01" euler="-60 0 0"`). The camera **must** be rendered and used from Phase 1 onward. Exact geometry updated when the physical bracket is designed. |
| 6 | Hardware communications bus | **CAN FD everywhere.** All hardware-facing ROS2 nodes (Moteus arm joints, solenoid actuator driver, any new sensors) communicate over CAN FD. The synthetic Moteus publisher and the solenoid command interface are designed to be drop-in replaceable with real CAN FD nodes. No GPIO, PWM, serial, or other bus. |
| 7 | Actuator command interface | **Separate `/actuator/command` topic** (`std_msgs/Float32`, range [-1, 1]). Keeps the solenoid decoupled from MoveIt Servo and avoids JointTrajectory config changes. The ROS2 CAN FD driver node subscribes to this topic and sends the CAN FD frame to the physical solenoid controller. In simulation, `cartesian_control_ros.py` subscribes and clamps to binary 0 / max_extension. |

### Solenoid actuator — model implications

Since the solenoid has no position feedback:
- **Observation**: `actuator_extended` (bool → 0.0 or 1.0), not a continuous position
- **Action dim 6**: treat as binary gate — values > 0 = fire/extend, ≤ 0 = retract
- **In simulation**: the prismatic joint is still physically modelled for realistic contact forces, but the command is clamped to `{0, max_extension}` — no intermediate positions
- **Contact detection**: solenoid extension + synthetic Moteus torque delta (the policy path); `cfrc_ext` is only used internally inside `synthetic_moteus_node.py` to derive the synthetic torque signal

### Synthetic Moteus publisher — design

A standalone ROS2 node `synthetic_moteus_node.py` in `src/rl_autonomy/testing/`:
- Subscribes to `/mujoco/joint_states` to read actual simulated `a5_rotation` state
- Internally reads `sim.data.cfrc_ext` on the actuator body to derive ground-truth contact force
- Adds **Bayesian noise**: Gaussian noise on torque (σ = 0.05 Nm, matching real Moteus noise floor), occasional outlier spikes (1% chance, 3σ), and a slow drift term (simulates temperature effects)
- Publishes `ControllerState` on `id_5/state` at 50 Hz over the same ROS2 topic structure the real Moteus CAN FD driver uses — the policy node subscribes to `id_5/state` in both sim and real, same code, no conditional
- Prints a **verbose readable log** every 0.5s:
  ```
  [SYNTH Moteus id=5 | a5_rotation]
    position  :  0.3142 rad   (raw:  0.3140 + noise: +0.0002)
    velocity  :  0.0023 rad/s (raw:  0.0020 + noise: +0.0003)
    torque    :  1.847  Nm    (raw:  1.800  + noise: +0.047 )  ← CONTACT SPIKE
    mode      :  POSITION (1)
    contact?  :  YES  (delta=0.847 Nm > threshold=0.500 Nm)
  ```
- When `contact_detected` fires, logs a highlighted line so it is unmissable in terminal output

### CAN FD solenoid driver — design (Phase 9, real hardware)

A ROS2 node `solenoid_canfd_node.py` in RoverFlake2:
- Subscribes to `/actuator/command` (`std_msgs/Float32`)
- Sends a CAN FD frame on the designated solenoid CAN ID: value > 0 → energize frame, ≤ 0 → de-energize frame
- Frame format matches whatever CAN FD protocol the solenoid controller uses (to be specified once the controller is selected)
- In simulation, `cartesian_control_ros.py` plays the same role (subscribe to `/actuator/command`, apply to MuJoCo joint) so no code change is needed at the policy layer when moving to hardware
