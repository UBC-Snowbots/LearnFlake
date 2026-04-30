"""
Phase 2 diagnostics — run inside the container:
    python src/rl_autonomy/testing/env_diagnostics.py

Verifies:
  1. KeyboardEnv loads and resets without errors
  2. All observation fields are present and correctly dimensioned
  3. set_target_key() changes the target_key_pos observable
  4. ArUco obs updates when target changes
  5. Saves one EEF camera frame to /tmp/eef_cam_phase2.png
  6. Prints a full labeled obs table for a zero-action rollout (10 steps)
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, ".."))                          # src/rl_autonomy
sys.path.insert(0, os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite"))

import numpy as np

from rl_autonomy.envs import KeyboardEnv

# ---------------------------------------------------------------------------
# Expected obs fields and their dimensions
# ---------------------------------------------------------------------------
EXPECTED_FIELDS = {
    "robot0_joint_pos":    6,
    "robot0_joint_vel":    6,
    "robot0_eef_pos":      3,
    "robot0_eef_quat":     4,
    "actuator_extended":   1,
    "target_key_pos":      3,
    "eef_to_key":          3,
    "rangefinder":         1,
    "contact_force":       1,
    "actuator_vel":        1,
    "aruco_obs":           3,   # (dx, dy, visible)
}


def print_obs_table(obs: dict, step: int):
    print(f"\n{'─'*60}")
    print(f"  Step {step}")
    print(f"{'─'*60}")
    for field, expected_dim in EXPECTED_FIELDS.items():
        val = obs.get(field)
        if val is None:
            print(f"  {'MISSING':20s} {field}")
            continue
        val = np.array(val).flatten()
        vals_str = " ".join(f"{v:7.4f}" for v in val)
        dim_ok = "✓" if len(val) == expected_dim else f"✗(got {len(val)})"
        print(f"  {dim_ok:3s} {field:22s}  [{vals_str}]")


def main():
    print("Creating KeyboardEnv …")
    env = KeyboardEnv(render=False, use_camera_obs=False, horizon=50)

    # ---- 1. Reset ----
    obs = env.reset()
    print("[PASS] reset() returned obs dict")

    # ---- 2. Obs field check ----
    all_ok = True
    for field, expected_dim in EXPECTED_FIELDS.items():
        val = obs.get(field)
        if val is None:
            print(f"[FAIL] missing field: {field}")
            all_ok = False
            continue
        dim = len(np.array(val).flatten())
        if dim != expected_dim:
            print(f"[FAIL] {field}: expected dim {expected_dim}, got {dim}")
            all_ok = False
    if all_ok:
        print("[PASS] all observation fields present with correct dimensions")

    # ---- 3. Flat obs dim ----
    flat = env._flat_obs()
    print(f"[INFO] flat obs dim = {len(flat)}  (Section 4a target: ~32)")

    # ---- 4. set_target_key() ----
    env.set_target_key("q")
    obs_q = env.get_obs()
    env.set_target_key("t")
    obs_t = env.get_obs()
    pos_q = np.array(obs_q["target_key_pos"])
    pos_t = np.array(obs_t["target_key_pos"])
    dist  = np.linalg.norm(pos_q - pos_t)
    if dist > 0.01:
        print(f"[PASS] set_target_key() moves target: |q-t| = {dist:.4f} m")
    else:
        print(f"[FAIL] set_target_key() had no effect (dist={dist:.4f})")

    # ---- 5. EEF camera frame ----
    try:
        import cv2
        frame = env.sim.render(
            width=256, height=256,
            camera_name="gripper0_right_eef_cam",
            depth=False,
        )
        # RoboSuite render returns (H, W, 3) RGB; cv2 wants BGR
        bgr = frame[::-1, :, ::-1]   # flip vertically + RGB→BGR
        path = "/tmp/eef_cam_phase2.png"
        cv2.imwrite(path, bgr)
        print(f"[PASS] EEF camera frame saved → {path}")
    except Exception as exc:
        print(f"[WARN] could not save camera frame: {exc}")

    # ---- 6. Zero-action rollout with labeled table ----
    env.set_target_key("e")
    obs = env.reset()
    action = np.zeros(env.action_dim)
    print("\nZero-action rollout (10 steps) — observation table:")
    for step in range(10):
        obs, reward, done, info = env.step(action)
        if step % 5 == 0:
            print_obs_table(obs, step)
        if done:
            break

    # ---- 7. Solenoid extend test ----
    print("\nExtending solenoid for 10 steps …")
    env.set_target_key("e")
    env.reset()
    action = np.zeros(env.action_dim)
    action[-1] = 1.0   # extend
    max_force = 0.0
    for _ in range(10):
        obs, _, _, _ = env.step(action)
        cf = float(np.array(obs["contact_force"]).flatten()[0])
        ae = float(np.array(obs["actuator_extended"]).flatten()[0])
        max_force = max(max_force, cf)

    print(f"  actuator_extended = {ae}  (expect 1.0 after extension)")
    print(f"  max contact_force = {max_force:.4f} N")

    env.close()
    print("\nPhase 2 diagnostics complete.")


if __name__ == "__main__":
    main()
