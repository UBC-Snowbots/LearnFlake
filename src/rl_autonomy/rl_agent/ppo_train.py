# Must be set BEFORE any mujoco/robosuite imports
import os
os.environ["MUJOCO_GL"] = "egl"  # Headless GPU rendering

import time

# Importing libraries
from gym_wrapper import RobosuiteGymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

# Making the environment (fully headless - no rendering here)
env = RobosuiteGymWrapper(
    "Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=False,  # Disabled offscreen too for speed
    use_camera_obs=False,
    reward_shaping=True
)

# Load trained model OR create new one (uncomment one option) depends on whatev we want to do here
# Option A: Load existing model
model = PPO.load("/LearnFlake/src/rl_autonomy/rl_agent/ppo-lift-model-train1.zip", env=env, device="cuda")
print(f"Model previously trained for {model.num_timesteps} timesteps")

# Option B: Create new model
# model = PPO("MlpPolicy", env, verbose=1)

# Save checkpoint every 50k steps so you can evaluate during training 
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="/LearnFlake/src/rl_autonomy/rl_agent/checkpoints/",
    name_prefix="ppo_lift"
)

start_time = time.time()
# Train the model (reset_num_timesteps=False to accumulate across sessions)
model.learn(total_timesteps=1000000, reset_num_timesteps=False, callback=checkpoint_callback)
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
