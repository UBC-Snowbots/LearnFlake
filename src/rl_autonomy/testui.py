from keyboard_env import CoarseReachEnv
import numpy as np

# 1. Initialize the environment
env = CoarseReachEnv(render=True, random_key=False)

# 2. Set the initial position (must be a single line or use parens)
env.robots[0].init_qpos = CoarseReachEnv.ABOVE_KEYBOARD_QPOS.copy()

# 3. Reset to apply the position
env.reset()

print("UI should be visible now. Running 100 steps...")

# 4. Loop to keep the physics engine and UI active
for _ in range(100):
    env.step(np.zeros(7))
    env.render()

input('Check the pose in the UI window, then press Enter here to quit')
env.close()