# Must be set BEFORE any mujoco/robosuite imports
import os

# Rendering backend toggle:
# - Default: glfw (onscreen via X11/VcXsrv)
# - Headless: set MUJOCO_GL_BACKEND = "egl" (or export MUJOCO_GL=egl)
MUJOCO_GL_BACKEND = os.environ.get("MUJOCO_GL", "glfw")
os.environ["MUJOCO_GL"] = MUJOCO_GL_BACKEND

import time

# Importing libraries
from gym_wrapper import RobosuiteGymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Making the environment (fully headless - no rendering at all)
env = RobosuiteGymWrapper(
    "Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=False,  # Disable offscreen too for speed
    use_camera_obs=False,
    reward_shaping=True
)

# Load trained model OR create new one (uncomment one option)
# Option A: Load existing model
model = PPO.load("/LearnFlake/src/rl_autonomy/rl_agent/ppo-lift-model-train1.zip", env=env, device="cpu")
print(f"Model previously trained for {model.num_timesteps} timesteps")

# Option B: Create new model
# model = PPO("MlpPolicy", env, verbose=1)

start_time = time.time()
# Train the model
model.learn(total_timesteps=100000)
elapsed = time.time() - start_time
print(f"Training took {elapsed:.2f} seconds")

# Evaluate
start_time2 = time.time()
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
elapsed2 = time.time() - start_time2
print(f"Evaluation took {elapsed2:.2f} seconds")



# Save the model
model.save("ppo_lift_model")

env.close()
