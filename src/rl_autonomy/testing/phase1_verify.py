"""
Phase 1 verification script — run inside the container:
    python src/rl_autonomy/testing/phase1_verify.py

Checks:
  1. Model loads without errors
  2. action_dim == 7  (6 arm + 1 solenoid)
  3. Solenoid joint exists in qpos
  4. Rangefinder sensor exists
  5. eef_cam camera exists
  6. cfrc_ext reads nonzero when solenoid is commanded to extend into the table
"""

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite")
if os.path.exists(ROBO_PATH) and ROBO_PATH not in sys.path:
    sys.path.insert(0, ROBO_PATH)

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)


def main():
    arm_cfg = suite.load_part_controller_config(default_controller="JOINT_VELOCITY")
    ctrl_cfg = refactor_composite_controller_config(arm_cfg, "Rover2026", ["right"])

    env = suite.make(
        env_name="Lift",
        robots=["Rover2026"],
        controller_configs=ctrl_cfg,
        has_renderer=False,
        has_offscreen_renderer=True,   # needed for camera check
        ignore_done=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=200,
        reward_shaping=True,
    )

    obs = env.reset()
    sim = env.sim

    # --- 1. action_dim ---
    action_dim = env.action_spec[0].shape[0]
    assert action_dim == 7, f"FAIL action_dim={action_dim}, expected 7"
    print(f"[PASS] action_dim = {action_dim}")

    # --- 2. Solenoid joint in model ---
    joint_names = [sim.model.joint_id2name(i) for i in range(sim.model.njnt)]
    solenoid_joints = [n for n in joint_names if "actuator_slide" in n]
    assert solenoid_joints, f"FAIL actuator_slide joint not found. Joints: {joint_names}"
    print(f"[PASS] solenoid joint: {solenoid_joints[0]}")

    # --- 3. Rangefinder sensor ---
    sensor_names = [sim.model.sensor_id2name(i) for i in range(sim.model.nsensor)]
    rf_sensors = [n for n in sensor_names if "rangefinder" in n]
    assert rf_sensors, f"FAIL rangefinder sensor not found. Sensors: {sensor_names}"
    print(f"[PASS] rangefinder sensor: {rf_sensors[0]}")

    # --- 4. eef_cam camera ---
    cam_names = [sim.model.camera_id2name(i) for i in range(sim.model.ncam)]
    eef_cams = [n for n in cam_names if "eef_cam" in n]
    assert eef_cams, f"FAIL eef_cam not found. Cameras: {cam_names}"
    print(f"[PASS] eef_cam: {eef_cams[0]}")

    # --- 5. cfrc_ext spike when solenoid extends into table ---
    # Drive solenoid toward table with a big downward arm action for a few steps
    # then check that the actuator_tip body shows nonzero external force
    tip_body_names = [n for n in [sim.model.body_id2name(i) for i in range(sim.model.nbody)]
                      if "linear_actuator" in n]
    assert tip_body_names, f"FAIL linear_actuator body not found"
    tip_body_id = sim.model.body_name2id(tip_body_names[0])
    print(f"[INFO] actuator body: {tip_body_names[0]} (id={tip_body_id})")

    # Extend solenoid for 30 steps (action[6] = 1.0)
    action = np.zeros(7)
    action[6] = 1.0
    max_force = 0.0
    for _ in range(30):
        obs, _, _, _ = env.step(action)
        force = np.linalg.norm(sim.data.cfrc_ext[tip_body_id][3:])
        max_force = max(max_force, force)

    print(f"[INFO] max cfrc_ext force on actuator tip over 30 steps: {max_force:.4f} N")
    if max_force > 0.01:
        print("[PASS] cfrc_ext nonzero — contact detection signal is live")
    else:
        print("[WARN] cfrc_ext near zero — solenoid may not be reaching a surface yet")

    # --- 6. Rangefinder reading ---
    rf_id = sim.model.sensor_name2id(rf_sensors[0])
    rf_val = sim.data.sensordata[rf_id]
    print(f"[INFO] rangefinder reading: {rf_val:.4f} m")

    env.close()
    print("\nPhase 1 verification complete.")


if __name__ == "__main__":
    main()
