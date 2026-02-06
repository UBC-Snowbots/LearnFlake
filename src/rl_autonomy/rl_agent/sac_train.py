# Must be set BEFORE any mujoco/robosuite imports
import os
os.environ["MUJOCO_GL"] = "egl"  # Headless GPU rendering

import time

# Importing libraries
from gym_wrapper import RobosuiteGymWrapper
from stable_baselines3 import SAC
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

from stable_baselines3 import SAC
import os

MODEL_PATH = "/LearnFlake/src/rl_autonomy/rl_agent/sac-lift-model-train1.zip"

if (os.path.exists(MODEL_PATH)):
    model = SAC.load(MODEL_PATH, env=env, device="gpu")
    print(f"Model previously trained for {model.num_timesteps} timesteps")
else:
    model = SAC("MlpPolicy", env, verbose=1)

start_time = time.time()
# Train the model
model.learn(total_timesteps=1000)
elapsed = time.time() - start_time
print(f"Training took {elapsed:.2f} seconds")

# # Evaluate
# start_time2 = time.time()
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
# elapsed2 = time.time() - start_time2
# print(f"Evaluation took {elapsed2:.2f} seconds")



# Save the model
model.save("sac_lift_model")

env.close()
