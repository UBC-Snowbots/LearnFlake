#!/usr/bin/env python3
"""Debug: check EEF orientation at ABOVE_KEYBOARD_QPOS."""
import numpy as np
from keyboard_env import KeyboardEnv, CoarseReachEnv

env = CoarseReachEnv(render=False)
env.reset()

# Teleport to ABOVE_KEYBOARD_QPOS
qpos = env.sim.data.qpos.copy()
qpos[:6] = CoarseReachEnv.ABOVE_KEYBOARD_QPOS
env.sim.data.qpos[:] = qpos
env.sim.forward()

obs = env._get_observations(force_update=True)
pf = env.robots[0].robot_model.naming_prefix

eef_pos = env.sim.data.site_xpos[env._eef_site_id]
eef_quat = obs.get(f"{pf}eef_quat", np.array([1, 0, 0, 0]))

# Rotation matrix
R = KeyboardEnv._quat_to_rot(eef_quat)
eef_z_world = R[:, 2]  # EEF Z-axis in world frame
eef_x_world = R[:, 0]
eef_y_world = R[:, 1]

print(f"EEF position: {eef_pos}")
print(f"EEF quaternion (wxyz): {eef_quat}")
print(f"EEF Z-axis in world: {eef_z_world}")
print(f"EEF X-axis in world: {eef_x_world}")
print(f"EEF Y-axis in world: {eef_y_world}")

# Tilt from vertical
down = np.array([0, 0, -1.0])
cos_a = np.clip(np.dot(eef_z_world, down), -1.0, 1.0)
tilt = np.arccos(cos_a)
print(f"\nTilt from downward (-Z): {np.degrees(tilt):.1f}°")

# Check if maybe the solenoid axis is different from EEF Z
# The solenoid pushes along its local Z. Let's see where the actuator tip is
tip_site_id = env.sim.model.site_name2id("gripper0_right_actuator_tip_site")
tip_pos = env.sim.data.site_xpos[tip_site_id]
grip_site_id = env.sim.model.site_name2id("gripper0_right_grip_site")
grip_pos = env.sim.data.site_xpos[grip_site_id]
print(f"\nGrip site pos: {grip_pos}")
print(f"Tip site pos:  {tip_pos}")
print(f"Tip - Grip:    {tip_pos - grip_pos}")
print(f"Tip is {'below' if tip_pos[2] < grip_pos[2] else 'above'} grip in world Z")

# Check key positions
for key in env.AVAILABLE_KEYS:
    kid = env._key_body_ids[key]
    kpos = env.sim.data.body_xpos[kid]
    xy_dist = np.linalg.norm(eef_pos[:2] - kpos[:2])
    z_diff = eef_pos[2] - kpos[2]
    print(f"  Key '{key}': pos={kpos}, xy_dist={xy_dist*100:.2f}cm, eef_z - key_z = {z_diff*100:.2f}cm")

env.close()
