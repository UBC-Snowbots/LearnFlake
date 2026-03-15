#!/usr/bin/env python3
"""Quick debug: test FineAlign init pose stability with zero actions."""
import numpy as np
from train_fine_align import FineAlignGymEnv

env = FineAlignGymEnv(render=False)
env._env.init_noise = 0.0

print("=== Test 1: Zero noise, zero actions (hold position) ===")
obs, _ = env.reset()
for step in range(200):
    action = np.zeros(6)
    obs, r, done, _, info = env.step(action)
    if step % 50 == 0:
        xy = info["xy_dist"] * 100
        ze = info["z_error"] * 100
        tilt = np.degrees(info["tilt_rad"])
        eef_z = env._env.sim.data.site_xpos[env._env._eef_site_id][2]
        print(f"  step {step:3d}: xy={xy:5.2f}cm z_err={ze:5.2f}cm tilt={tilt:5.1f}deg eef_z={eef_z:.4f}m reward={r:.1f}")
    if done:
        print("  -> terminated early (success)")
        break

print("\n=== Test 2: Zero noise, random actions ===")
env._env.init_noise = 0.0
obs, _ = env.reset()
for step in range(200):
    action = np.random.uniform(-1, 1, size=6).astype(np.float32)
    obs, r, done, _, info = env.step(action)
    if step % 50 == 0:
        xy = info["xy_dist"] * 100
        ze = info["z_error"] * 100
        tilt = np.degrees(info["tilt_rad"])
        eef_z = env._env.sim.data.site_xpos[env._env._eef_site_id][2]
        print(f"  step {step:3d}: xy={xy:5.2f}cm z_err={ze:5.2f}cm tilt={tilt:5.1f}deg eef_z={eef_z:.4f}m reward={r:.1f}")
    if done:
        print("  -> terminated early (success)")
        break

print("\n=== Test 3: With init_noise=0.03, zero actions ===")
env._env.init_noise = 0.03
obs, _ = env.reset()
for step in range(200):
    action = np.zeros(6)
    obs, r, done, _, info = env.step(action)
    if step % 50 == 0:
        xy = info["xy_dist"] * 100
        ze = info["z_error"] * 100
        tilt = np.degrees(info["tilt_rad"])
        eef_z = env._env.sim.data.site_xpos[env._env._eef_site_id][2]
        print(f"  step {step:3d}: xy={xy:5.2f}cm z_err={ze:5.2f}cm tilt={tilt:5.1f}deg eef_z={eef_z:.4f}m reward={r:.1f}")
    if done:
        print("  -> terminated early (success)")
        break

env.close()
print("\nDone.")
