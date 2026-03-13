#!/usr/bin/env python3
"""
Phase 5 — PressKey skill training.

Trains a SAC policy to extend the solenoid, make contact with the target key,
and hold for 3 consecutive steps.  Each episode starts from an already-aligned
pose provided by the CoarseReach policy.

Action space: 1-dim solenoid command in [-1, 1].
  >0 → energise (extend), ≤0 → de-energise (spring retract).
Arm joints are locked to 0 throughout.

Usage:
    python train_press_key.py                              # train from scratch
    python train_press_key.py --no-render                  # headless
    python train_press_key.py --eval checkpoints/press_key/best_model.zip
    python train_press_key.py --resume checkpoints/press_key/last_model.zip
    python train_press_key.py --cr-checkpoint checkpoints/coarse_reach/best_model.zip
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

from keyboard_env import PressKeyEnv

CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints", "press_key")
LOG_DIR        = os.path.join(ROOT, "logs", "press_key")

DEFAULT_CR_CHECKPOINT = os.path.join(
    ROOT, "checkpoints", "coarse_reach", "best_model.zip"
)


# ============================================================================
# Gym wrapper
# ============================================================================

class PressKeyGymEnv(gym.Env):
    """
    Gym/Gymnasium wrapper around PressKeyEnv.

    Action space: 1-dim solenoid command in [-1, 1].
    Observation space: full 32-dim flat obs vector from KeyboardEnv.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, coarse_reach_checkpoint: str | None = None,
                 render: bool = False, random_key: bool = True):
        super().__init__()
        self._env = PressKeyEnv(
            coarse_reach_checkpoint=coarse_reach_checkpoint,
            render=render,
            random_key=random_key,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._env.obs_dim,), dtype=np.float32,
        )
        # 1-dim: solenoid only
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self._env.reset()
        obs = self._env._flat_obs().astype(np.float32)
        if _GYM == "gymnasium":
            return obs, {}
        return obs

    def step(self, action):
        # Arm locked to 0, only solenoid active
        full_action = np.zeros(self._env.action_dim)
        full_action[-1] = float(action[0])
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

def make_env(cr_checkpoint: str | None, render: bool = False) -> gym.Env:
    env = PressKeyGymEnv(coarse_reach_checkpoint=cr_checkpoint, render=render)
    return Monitor(env)


def train(timesteps: int, render: bool, cr_checkpoint: str | None,
          resume_path: str | None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if cr_checkpoint and not os.path.exists(cr_checkpoint):
        print(f"[WARN] CoarseReach checkpoint not found: {cr_checkpoint}")
        print("       Episodes will start from the default arm home pose.")
        cr_checkpoint = None
    elif cr_checkpoint:
        print(f"Using CoarseReach checkpoint: {cr_checkpoint}")

    env      = make_env(cr_checkpoint, render=render)
    eval_env = make_env(cr_checkpoint, render=False)

    if resume_path:
        print(f"Resuming from {resume_path}")
        model = SAC.load(resume_path, env=env, tensorboard_log=LOG_DIR)
    else:
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=50_000,
            learning_starts=500,
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
            eval_freq=3_000,
            n_eval_episodes=20,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=10_000,
            save_path=CHECKPOINT_DIR,
            name_prefix="press_key",
            verbose=1,
        ),
    ])

    print(f"\nTraining PressKey for {timesteps:,} steps")
    print(f"  obs_dim={env.observation_space.shape[0]}  "
          f"action_dim={env.action_space.shape[0]}")
    print(f"  checkpoints → {CHECKPOINT_DIR}")
    print(f"  tensorboard → tensorboard --logdir {LOG_DIR}\n")

    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        reset_num_timesteps=not resume_path,
    )
    model.save(os.path.join(CHECKPOINT_DIR, "last_model"))
    print("Training done. Last model saved.")

    env.close()
    eval_env.close()


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model_path: str, cr_checkpoint: str | None,
             n_episodes: int, render: bool):
    env = PressKeyGymEnv(
        coarse_reach_checkpoint=cr_checkpoint, render=render
    )
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

        success = ep_reward >= 900.0   # got the 1000-pt hold bonus
        successes += int(success)
        print(f"  ep {ep+1:3d}/{n_episodes}  reward={ep_reward:8.1f}  "
              f"{'SUCCESS' if success else 'timeout'}")

    print(f"\nSuccess rate: {successes}/{n_episodes} = "
          f"{100*successes/n_episodes:.1f}%")
    env.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=90_000,
                        help='Total training steps (default: 90k ≈ 300 ep × 50 steps × ~6 iters)')
    parser.add_argument('--no-render', action='store_true')
    parser.add_argument('--cr-checkpoint', type=str,
                        default=DEFAULT_CR_CHECKPOINT,
                        help='Path to CoarseReach best_model.zip')
    parser.add_argument('--eval', type=str, default=None, metavar='PATH')
    parser.add_argument('--eval-episodes', type=int, default=50)
    parser.add_argument('--resume', type=str, default=None, metavar='PATH')
    args = parser.parse_args()

    cr = args.cr_checkpoint if os.path.exists(args.cr_checkpoint) else None

    if args.eval:
        evaluate(args.eval, cr, args.eval_episodes, render=not args.no_render)
    else:
        train(args.timesteps, render=not args.no_render,
              cr_checkpoint=cr, resume_path=args.resume)


if __name__ == '__main__':
    main()
