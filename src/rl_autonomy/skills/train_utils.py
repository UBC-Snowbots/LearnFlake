"""
Shared SAC training / evaluation utilities for all skill scripts.

Each skill's train_*.py defines a GymEnv wrapper class and calls
`run_cli(...)` which handles argparse, training, evaluation, and checkpointing.
"""

import os
import sys
import argparse
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ROBO_PATH = os.path.join(ROOT, "..", "external_pkgs", "RoboSuite")
if os.path.exists(ROBO_PATH) and ROBO_PATH not in sys.path:
    sys.path.insert(0, ROBO_PATH)

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_TYPE = "gymnasium"
except ImportError:
    import gym
    from gym import spaces
    GYM_TYPE = "gym"

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


def gym_reset(result):
    """Extract obs from gym/gymnasium reset return value."""
    return result[0] if isinstance(result, tuple) else result


def gym_step_result(result):
    """Extract (obs, reward, done, info) from gym/gymnasium step return."""
    return result[0], result[1], result[2], result[-1]


def make_env_fn(env_cls, render: bool = False, **env_kwargs):
    """Return a factory function that creates a Monitored env instance."""
    def _init():
        env = env_cls(render=render, **env_kwargs)
        return Monitor(env)
    return _init


def train(env_cls, skill_name: str, timesteps: int, render: bool,
          resume_path: str | None, num_envs: int = 3,
          sac_kwargs: dict | None = None, eval_freq: int = 5_000,
          checkpoint_freq: int = 20_000, n_eval_episodes: int = 20,
          **env_kwargs):
    """
    Standard SAC training loop.

    Args:
        env_cls: Gym env class to instantiate.
        skill_name: Used for checkpoint/log directory naming.
        timesteps: Total training steps.
        render: Enable rendering (disabled for parallel envs).
        resume_path: Path to resume from, or None.
        num_envs: Number of parallel training envs.
        sac_kwargs: Override default SAC hyperparameters.
        eval_freq: Steps between evaluations.
        checkpoint_freq: Steps between checkpoint saves.
        n_eval_episodes: Episodes per evaluation.
        **env_kwargs: Passed to env_cls constructor.
    """
    checkpoint_dir = os.path.join(ROOT, "checkpoints", skill_name)
    log_dir = os.path.join(ROOT, "logs", skill_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if num_envs > 1:
        env      = SubprocVecEnv([make_env_fn(env_cls, render=False, **env_kwargs)
                                  for _ in range(num_envs)])
        eval_env = SubprocVecEnv([make_env_fn(env_cls, render=False, **env_kwargs)
                                  for _ in range(1)])
    else:
        env      = make_env_fn(env_cls, render=render, **env_kwargs)()
        eval_env = make_env_fn(env_cls, render=False, **env_kwargs)()

    defaults = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=num_envs,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    if sac_kwargs:
        defaults.update(sac_kwargs)

    if resume_path:
        print(f"Resuming from {resume_path}")
        model = SAC.load(resume_path, env=env, tensorboard_log=log_dir)
    else:
        model = SAC(**defaults)

    callbacks = CallbackList([
        EvalCallback(
            eval_env,
            best_model_save_path=checkpoint_dir,
            log_path=log_dir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=checkpoint_dir,
            name_prefix=skill_name.replace("/", "_"),
            verbose=1,
        ),
    ])

    print(f"\nTraining {skill_name} for {timesteps:,} steps ({num_envs} parallel envs)")
    print(f"  obs_dim={env.observation_space.shape[0]}  "
          f"action_dim={env.action_space.shape[0]}  "
          f"gradient_steps={num_envs}")
    print(f"  checkpoints -> {checkpoint_dir}")
    print(f"  tensorboard -> tensorboard --logdir {log_dir}\n")

    model.learn(total_timesteps=timesteps, callback=callbacks,
                reset_num_timesteps=not resume_path)
    model.save(os.path.join(checkpoint_dir, "last_model"))
    print("Training done. Last model saved.")

    env.close()
    eval_env.close()


def evaluate(env_cls, model_path: str, n_episodes: int, render: bool,
             print_fn=None, **env_kwargs):
    """
    Standard evaluation loop.

    Args:
        env_cls: Gym env class.
        model_path: Path to .zip model file.
        n_episodes: Number of evaluation episodes.
        render: Enable rendering.
        print_fn: Optional callable(ep, n_episodes, ep_reward, info, success)
                  for custom per-episode logging. If None, prints reward + success.
        **env_kwargs: Passed to env_cls constructor.
    """
    env = env_cls(render=render, **env_kwargs)
    model = SAC.load(model_path)
    successes = 0

    for ep in range(n_episodes):
        obs = gym_reset(env.reset())
        done = False
        ep_reward = 0.0
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = gym_step_result(env.step(action))
            ep_reward += reward

        success = env._env._check_success()
        successes += int(success)

        if print_fn:
            print_fn(ep, n_episodes, ep_reward, info, success)
        else:
            status = 'SUCCESS' if success else 'timeout'
            print(f"  ep {ep+1:3d}/{n_episodes}  reward={ep_reward:7.1f}  {status}")

    print(f"\nSuccess rate: {successes}/{n_episodes} = "
          f"{100*successes/n_episodes:.1f}%")
    env.close()


def build_parser(skill_name: str, default_timesteps: int = 150_000) -> argparse.ArgumentParser:
    """Build standard argparse for a skill training script."""
    parser = argparse.ArgumentParser(description=f"{skill_name} training")
    parser.add_argument('--timesteps', type=int, default=default_timesteps,
                        help=f'Total training steps (default: {default_timesteps:,})')
    parser.add_argument('--no-render', action='store_true')
    parser.add_argument('--eval', type=str, default=None,
                        metavar='PATH', help='Evaluate a saved model')
    parser.add_argument('--eval-episodes', type=int, default=50)
    parser.add_argument('--resume', type=str, default=None,
                        metavar='PATH', help='Resume training from checkpoint')
    parser.add_argument('--num-envs', type=int, default=3,
                        help='Number of parallel envs (default: 3)')
    return parser


def run_cli(env_cls, skill_name: str, default_timesteps: int = 150_000,
            sac_kwargs: dict | None = None, print_fn=None, **env_kwargs):
    """
    Full CLI entrypoint: parse args, then train or evaluate.

    Usage in a skill script:
        run_cli(MyGymEnv, "reach/coarse", default_timesteps=150_000)
    """
    parser = build_parser(skill_name, default_timesteps)
    args = parser.parse_args()

    if args.eval:
        evaluate(env_cls, args.eval, args.eval_episodes,
                 render=not args.no_render, print_fn=print_fn, **env_kwargs)
    else:
        train(env_cls, skill_name, args.timesteps,
              render=not args.no_render, resume_path=args.resume,
              num_envs=args.num_envs, sac_kwargs=sac_kwargs, **env_kwargs)
