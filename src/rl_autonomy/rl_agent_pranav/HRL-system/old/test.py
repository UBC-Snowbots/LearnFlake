#!/usr/bin/env python3
"""
Headless SAC smoke-training entrypoint for goal-conditioned hover navigation.
"""

import os
import argparse

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from UpgradedEnvHRL import UpgradedEnvHRL


def main(render: bool = False, total_timesteps: int = 1000):
    # Must be set before robosuite / mujoco import.
    if render:
        os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "glfw")
    else:
        os.environ["MUJOCO_GL"] = "egl"

    

    env = Monitor(
        UpgradedEnvHRL(
            render=render,
            domain_randomization=True,
            control_freq=20,
            horizon=150,
            controller="OSC_POSE",
            robot_name="Rover2026",
        )
    )

    # Catches API/space edge-cases early.
    check_env(env, warn=True)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        learning_starts=5_000,
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save("sac_rover2026_goal_hover")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Eval reward mean={mean_reward:.3f}, std={std_reward:.3f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Enable on-screen rendering (GLFW).")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of training timesteps.")
    args = parser.parse_args()
    main(render=args.render, total_timesteps=args.timesteps)
