"""
Evaluation Script with Rendering Window
========================================

Visualize a trained PPO agent performing the Lift task in Robosuite.

PREREQUISITES (Docker on Windows):
----------------------------------
1. Install VcXsrv on Windows: https://sourceforge.net/projects/vcxsrv/

2. Launch XLaunch with these settings:
   - Multiple windows
   - Start no client
   - CHECK "Disable access control" (important!)

3. Set DISPLAY in Docker (add to docker-compose.yml or export manually):
   export DISPLAY=host.docker.internal:0.0

4. Ensure the container was started AFTER setting DISPLAY

USAGE:
------
    cd /LearnFlake/src/rl_autonomy/rl_agent
    python evaluate_render.py

CONFIGURATION:
--------------
- MODEL_PATH: Path to your trained .zip model file
- N_EPISODES: Number of evaluation episodes to run
- RENDER_DELAY: Seconds between frames (lower = faster playback)

TROUBLESHOOTING:
----------------
- "Failed to open display": VcXsrv not running or DISPLAY not set
- "Could not initialize GLFW": Try `export MUJOCO_GL=glx` before running
- Black window: Check VcXsrv firewall permissions
"""

import os
# Override to use GLX for window rendering (not EGL which is headless)
os.environ["MUJOCO_GL"] = "glx"

import time
from gym_wrapper import RobosuiteGymWrapper
from stable_baselines3 import PPO

# =============================================================================
# CONFIGURATION - Modify these as needed
# =============================================================================
MODEL_PATH = "/LearnFlake/src/rl_autonomy/rl_agent/ppo_lift_model.zip"
N_EPISODES = 5
RENDER_DELAY = 0.02  # seconds between frames

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
print("Creating environment with renderer enabled...")
env = RobosuiteGymWrapper(
    "Lift",
    robots="Panda",
    has_renderer=True,  # Enable on-screen window
    has_offscreen_renderer=False,
    use_camera_obs=False,
    reward_shaping=True
)

# =============================================================================
# LOAD MODEL
# =============================================================================
print(f"Loading model from: {MODEL_PATH}")
model = PPO.load(MODEL_PATH, env=env, device="cpu")
print(f"Model trained for {model.num_timesteps} timesteps")

# =============================================================================
# RUN EVALUATION
# =============================================================================
print(f"\nRunning {N_EPISODES} evaluation episodes...")
print("=" * 50)

for ep in range(N_EPISODES):
    obs, info = env.reset()
    total_reward = 0
    done = False
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step += 1

        env.render()
        time.sleep(RENDER_DELAY)

    print(f"Episode {ep+1}/{N_EPISODES}: reward={total_reward:.2f}, steps={step}")

print("=" * 50)
env.close()
print("Evaluation complete!")
