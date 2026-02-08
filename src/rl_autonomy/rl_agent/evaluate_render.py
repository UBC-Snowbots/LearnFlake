"""
Evaluation Script with Rendering Window
========================================

Visualize a trained SAC agent performing the Lift task in Robosuite.

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
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

# =============================================================================
# CONFIGURATION - Modify these as needed
# =============================================================================
MODEL_PATH = "best_model/best_model.zip"
N_EPISODES = 5
ROBOT = "Rover2026"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
print("Creating environment with renderer enabled...")
env = RobosuiteGymWrapper(
    "Lift",
    robots=ROBOT,
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    reward_shaping=True
)

# =============================================================================
# LOAD MODEL
# =============================================================================
print(f"Loading model from: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}. Train first!")
    env.close()
    exit(0)

model = SAC.load(MODEL_PATH, env=env, device="cuda")
print(f"Model trained for {model.num_timesteps} timesteps")

# =============================================================================
# RUN EVALUATION WITH RENDERING
# =============================================================================
print(f"\nRunning {N_EPISODES} evaluation episodes with rendering...")
print("=" * 50)

start_time = time.time()
mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=N_EPISODES,
    render=True,
    deterministic=True
)
elapsed = time.time() - start_time

print("=" * 50)
print(f"Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}")
print(f"Evaluation took {elapsed:.2f} seconds")

env.close()
print("Evaluation complete!")
