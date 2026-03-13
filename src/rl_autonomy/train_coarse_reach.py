#!/usr/bin/env python3
"""
Phase 4 — CoarseReach skill training.

Trains a SAC policy to move the EEF to within 3 cm XY and correct height
above any keyboard key using proprioceptive observations only.
The solenoid actuator is locked retracted throughout (6-dim effective action).

Usage:
    python train_coarse_reach.py                       # train from scratch
    python train_coarse_reach.py --eval checkpoints/coarse_reach/best_model.zip
    python train_coarse_reach.py --resume checkpoints/coarse_reach/last_model.zip
    python train_coarse_reach.py --timesteps 200000 --no-render
"""

import os
import sys
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "..", "external_pkgs", "RoboSuite")
if os.path.exists(ROBO_PATH) and ROBO_PATH not in sys.path:
    sys.path.insert(0, ROBO_PATH)

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM = "gymnasium"
except ImportError:
    import gym
    from gym import spaces
    _GYM = "gym"

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from keyboard_env import CoarseReachEnv


CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints", "coarse_reach")
LOG_DIR        = os.path.join(ROOT, "logs", "coarse_reach")


# ============================================================================
# Gym wrapper
# ============================================================================

class CoarseReachGymEnv(gym.Env):
    """
    Gym/Gymnasium wrapper around CoarseReachEnv for SB3 compatibility.

    Action space: 6-dim arm joint velocities in [-1, 1].
    The solenoid (action[-1]) is always set to 0 (retracted) internally.
    Observation space: full 32-dim flat obs vector from KeyboardEnv.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, render: bool = False, random_key: bool = True):
        super().__init__()
        self._env = CoarseReachEnv(render=render, random_key=random_key)

        obs_dim = self._env.obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        if seed is not None and hasattr(self._env, 'seed'):
            self._env.seed(seed)
        self._env.reset()
        obs = self._env._flat_obs().astype(np.float32)
        if _GYM == "gymnasium":
            return obs, {}
        return obs

    def step(self, action):
        full_action = np.append(action, 0.0)   # lock solenoid retracted
        _, reward, done, info = self._env.step(full_action)
        obs = self._env._flat_obs().astype(np.float32)
        if _GYM == "gymnasium":
            return obs, float(reward), done, False, info
        return obs, float(reward), done, info

    def render(self, mode='human'):
        self._env.render()

    def close(self):
        self._env.close()


# ============================================================================
# Training
# ============================================================================

def make_env(render: bool = False) -> gym.Env:
    env = CoarseReachGymEnv(render=render)
    return Monitor(env)


def train(timesteps: int, render: bool, resume_path: str | None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    env      = make_env(render=render)
    eval_env = make_env(render=False)

    if resume_path:
        print(f"Resuming from {resume_path}")
        model = SAC.load(resume_path, env=env, tensorboard_log=LOG_DIR)
    else:
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            verbose=1,
            tensorboard_log=LOG_DIR,
            policy_kwargs=dict(net_arch=[256, 256]),
        )

    callbacks = CallbackList([
        EvalCallback(
            eval_env,
            best_model_save_path=CHECKPOINT_DIR,
            log_path=LOG_DIR,
            eval_freq=5_000,
            n_eval_episodes=20,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=20_000,
            save_path=CHECKPOINT_DIR,
            name_prefix="coarse_reach",
            verbose=1,
        ),
    ])

    print(f"\nTraining CoarseReach for {timesteps:,} steps")
    print(f"  obs_dim={env.observation_space.shape[0]}  "
          f"action_dim={env.action_space.shape[0]}")
    print(f"  checkpoints → {CHECKPOINT_DIR}")
    print(f"  tensorboard → tensorboard --logdir {LOG_DIR}\n")

    model.learn(total_timesteps=timesteps, callback=callbacks, reset_num_timesteps=not resume_path)
    model.save(os.path.join(CHECKPOINT_DIR, "last_model"))
    print("Training done. Last model saved.")

    env.close()
    eval_env.close()


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model_path: str, n_episodes: int, render: bool):
    env = CoarseReachGymEnv(render=render)

    model = SAC.load(model_path)
    successes = 0

    for ep in range(n_episodes):
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            result = env.step(action)
            obs, reward, done = result[0], result[1], result[2]
            ep_reward += reward
        success = ep_reward > 50.0   # rough proxy: got the 100-pt bonus
        successes += int(success)
        print(f"  ep {ep+1:3d}/{n_episodes}  reward={ep_reward:7.1f}  {'SUCCESS' if success else 'timeout'}")

    print(f"\nSuccess rate: {successes}/{n_episodes} = {100*successes/n_episodes:.1f}%")
    env.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=150_000,
                        help='Total training steps (default: 150k ≈ 500 episodes × 300 steps)')
    parser.add_argument('--no-render', action='store_true')
    parser.add_argument('--eval', type=str, default=None,
                        metavar='PATH', help='Evaluate a saved model')
    parser.add_argument('--eval-episodes', type=int, default=50)
    parser.add_argument('--resume', type=str, default=None,
                        metavar='PATH', help='Resume training from checkpoint')
    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval, args.eval_episodes, render=not args.no_render)
    else:
        train(args.timesteps, render=not args.no_render, resume_path=args.resume)


if __name__ == '__main__':
    main()
