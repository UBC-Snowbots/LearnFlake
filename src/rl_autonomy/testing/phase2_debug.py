"""Quick debug to identify the solenoid qvel address and rangefinder direction."""
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, ".."))
sys.path.insert(0, os.path.join(ROOT, "..", "..", "external_pkgs", "RoboSuite"))

import numpy as np
from keyboard_env import KeyboardEnv

env = KeyboardEnv(render=False, horizon=20)
env.reset()
sim = env.sim

# --- Joint / DOF map ---
print("=== All joints (id, name, qposadr, dofadr) ===")
for i in range(sim.model.njnt):
    name = sim.model.joint_id2name(i)
    print(f"  {i:2d}  {name:40s}  qposadr={sim.model.jnt_qposadr[i]}  dofadr={sim.model.jnt_dofadr[i]}")

print(f"\nenv._actuator_qpos_addr = {env._actuator_qpos_addr}")
print(f"env._actuator_qvel_addr = {env._actuator_qvel_addr}")
print(f"qpos[_actuator_qpos_addr] = {sim.data.qpos[env._actuator_qpos_addr]:.4f}")
print(f"qvel[_actuator_qvel_addr] = {sim.data.qvel[env._actuator_qvel_addr]:.4f}")
print(f"full qpos = {sim.data.qpos[:10]}")
print(f"full qvel = {sim.data.qvel[:10]}")

# --- Rangefinder sensor details ---
print("\n=== Sensors ===")
for i in range(sim.model.nsensor):
    name = sim.model.sensor_id2name(i)
    val  = sim.data.sensordata[i]
    print(f"  {i:2d}  {name:40s}  val={val:.4f}")

# --- Solenoid site orientation ---
print("\n=== actuator_tip_site xmat (row = x,y,z axes in world) ===")
site_name = None
for i in range(sim.model.nsite):
    n = sim.model.site_id2name(i)
    if "actuator_tip_site" in n:
        site_name = n
        xmat = sim.data.site_xmat[i].reshape(3, 3)
        print(f"  site: {n}")
        print(f"  x-axis (world): {xmat[:, 0]}")
        print(f"  y-axis (world): {xmat[:, 1]}")
        print(f"  z-axis (world): {xmat[:, 2]}  ← rangefinder fires this way")
        break

# --- What ctrl value is being set for the solenoid actuator ---
print("\n=== Actuator ctrl values ===")
for i in range(sim.model.nu):
    name = sim.model.actuator_id2name(i)
    print(f"  {i:2d}  {name:40s}  ctrl={sim.data.ctrl[i]:.4f}")

env.close()
