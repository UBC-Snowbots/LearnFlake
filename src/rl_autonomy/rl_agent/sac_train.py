# Must be set BEFORE any mujoco/robosuite imports
import os
os.environ["MUJOCO_GL"] = "egl"  # Headless GPU rendering

import time
from gym_wrapper import RobosuiteGymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_PATH = "sac_Rover2026_V1_model"
SAVE_NAME = "sac_Rover2026_V1_model"
TOTAL_TIMESTEPS = 1_000
CHECKPOINT_FREQ = 100  # Saves 10 checkpoints throughout training
N_EVAL_EPISODES = 10
DEVICE = "cuda"
ROBOT = "Rover2026"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
env = RobosuiteGymWrapper(
    "Lift",
    robots=ROBOT,
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    reward_shaping=True
)

# =============================================================================
# MODEL SETUP - Auto-load or create new
# =============================================================================
if os.path.exists(MODEL_PATH):
    model = SAC.load(MODEL_PATH, env=env, device=DEVICE)
    print(f"Loaded model with {model.num_timesteps} timesteps")
else:
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device=DEVICE,
        tensorboard_log="./logs/"
    )
    print("Created new model")

# =============================================================================
# CALLBACKS
# =============================================================================
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path="./checkpoints/",
    name_prefix="sac_rover"
)

# =============================================================================
# TRAINING
# =============================================================================
start_time = time.time()
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    reset_num_timesteps=False,
    callback=checkpoint_callback
)
elapsed = time.time() - start_time
print(f"Training took {elapsed:.2f} seconds")

# =============================================================================
# FINAL EVALUATION
# =============================================================================
start_time2 = time.time()
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES)
print(f"Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}")
elapsed2 = time.time() - start_time2
print(f"Evaluation took {elapsed2:.2f} seconds")

# =============================================================================
# SAVE MODEL
# =============================================================================
model.save(SAVE_NAME)
print(f"Model saved to {SAVE_NAME}.zip")

env.close()
