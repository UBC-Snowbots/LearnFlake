#!/usr/bin/env python3
"""
High-level HRL policy training.

Trains a PPO policy that selects which frozen low-level skill to execute
at each decision point. The low-level skills (CoarseReach, FineAlign,
Press, Retract, Traverse) are loaded from checkpoints and run as frozen
options — only the high-level sequencing policy is trained.

The high-level learns:
  - Correct skill ordering (reach -> align -> press -> retract -> traverse)
  - When to transition between skills
  - How to recover from skill failures
  - How to chain skills across multi-key sequences

Action space: Discrete(5) — skill ID
Observation: 42-dim (32 base env + 5 skill one-hot + 5 meta)

Usage:
    # Train on "hello" (default)
    python -m skills.hrl.train_hrl

    # Train on custom string
    python -m skills.hrl.train_hrl --target "the quick brown fox"

    # Evaluate
    python -m skills.hrl.train_hrl --eval checkpoints/hrl/best_model.zip

    # Override individual skill checkpoints
    python -m skills.hrl.train_hrl \\
        --ckpt-reach checkpoints/reach_dr/best_model.zip \\
        --ckpt-align checkpoints/reach_fine/best_model.zip \\
        --ckpt-press checkpoints/press_dr/best_model.zip \\
        --ckpt-retract checkpoints/retract_dr/best_model.zip \\
        --ckpt-traverse checkpoints/traverse_dr/best_model.zip
"""

import os
import sys
import argparse
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
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

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from skills.hrl.hrl_env import (
    HRLEnv, NUM_SKILLS, SKILL_NAMES,
    COARSE_REACH, FINE_ALIGN, PRESS, RETRACT, TRAVERSE,
)


CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints", "hrl")
LOG_DIR        = os.path.join(ROOT, "logs", "hrl")

# Training strings — varied length and key coverage for curriculum
TRAINING_STRINGS = [
    "a",                    # single key (learn the full cycle)
    "ab",                   # two keys (learn traverse)
    "hello",                # short word
    "the",                  # common word
    "quick",                # longer word
    "space bar",            # includes space key
    "hello world",          # two words
    "the quick brown fox",  # long sentence
]


# ============================================================================
# Gym wrapper
# ============================================================================

class HRLGymEnv(gym.Env):
    """
    Gym/Gymnasium wrapper around HRLEnv for SB3 PPO.

    Each episode samples a random target string from the training curriculum.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, render: bool = False, checkpoints: dict | None = None,
                 target_strings: list[str] | None = None,
                 max_skill_steps: int = 200, high_level_horizon: int = 100):
        super().__init__()

        self._target_strings = target_strings or TRAINING_STRINGS
        self._checkpoints = checkpoints

        # Create env with first string (will be randomized on reset)
        self._env = HRLEnv(
            target_string=self._target_strings[0],
            checkpoints=checkpoints,
            max_skill_steps=max_skill_steps,
            high_level_horizon=high_level_horizon,
            render=render,
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._env.obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_SKILLS)

    def reset(self, seed=None, options=None):
        # Sample a random training string for curriculum diversity
        target = np.random.choice(self._target_strings)
        obs = self._env.reset(target_string=target)

        if _GYM == "gymnasium":
            return obs, {'target': target}
        return obs

    def step(self, action):
        skill_id = int(action)
        obs, reward, done, info = self._env.step(skill_id)

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

def train(args):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    checkpoints = _build_checkpoint_dict(args)
    target_strings = [args.target] if args.target else TRAINING_STRINGS

    env = Monitor(HRLGymEnv(
        render=not args.no_render,
        checkpoints=checkpoints,
        target_strings=target_strings,
        max_skill_steps=args.max_skill_steps,
        high_level_horizon=args.hl_horizon,
    ))
    eval_env = Monitor(HRLGymEnv(
        render=False,
        checkpoints=checkpoints,
        target_strings=target_strings,
        max_skill_steps=args.max_skill_steps,
        high_level_horizon=args.hl_horizon,
    ))

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env, tensorboard_log=LOG_DIR)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=256,          # steps per rollout (high-level steps are expensive)
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,        # encourage exploration of skill sequences
            verbose=1,
            tensorboard_log=LOG_DIR,
            policy_kwargs=dict(net_arch=[256, 256]),
        )

    callbacks = CallbackList([
        EvalCallback(
            eval_env,
            best_model_save_path=CHECKPOINT_DIR,
            log_path=LOG_DIR,
            eval_freq=500,         # high-level steps (each is many low-level)
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=2_000,
            save_path=CHECKPOINT_DIR,
            name_prefix="hrl",
            verbose=1,
        ),
    ])

    print(f"\nTraining HRL high-level policy for {args.timesteps:,} steps")
    print(f"  Target strings: {target_strings}")
    print(f"  Max skill steps: {args.max_skill_steps}")
    print(f"  High-level horizon: {args.hl_horizon}")
    print(f"  checkpoints -> {CHECKPOINT_DIR}")
    print(f"  tensorboard -> tensorboard --logdir {LOG_DIR}\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=not args.resume,
    )
    model.save(os.path.join(CHECKPOINT_DIR, "last_model"))
    print("Training done. Last model saved.")

    env.close()
    eval_env.close()


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(args):
    checkpoints = _build_checkpoint_dict(args)
    target = args.target or "hello"

    env = HRLGymEnv(
        render=not args.no_render,
        checkpoints=checkpoints,
        target_strings=[target],
        max_skill_steps=args.max_skill_steps,
        high_level_horizon=args.hl_horizon,
    )

    model = PPO.load(args.eval)
    successes = 0

    for ep in range(args.eval_episodes):
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        done = False
        ep_reward = 0.0
        hl_steps = 0
        skill_sequence = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            result = env.step(action)
            obs, reward, done, info = result[0], result[1], result[2], result[-1]
            ep_reward += reward
            hl_steps += 1
            skill_sequence.append(SKILL_NAMES[int(action)])

        success = info.get('success', False)
        successes += int(success)
        keys_pressed = info.get('keys_pressed', 0)
        keys_total = info.get('keys_total', 0)
        ll_steps = info.get('total_ll_steps', 0)

        status = 'SUCCESS' if success else 'FAIL'
        print(f"  ep {ep+1:3d}/{args.eval_episodes}  '{target}'  "
              f"keys={keys_pressed}/{keys_total}  "
              f"reward={ep_reward:8.1f}  hl_steps={hl_steps}  "
              f"ll_steps={ll_steps}  {status}")

        # Print skill sequence for first few episodes
        if ep < 3:
            seq_str = " -> ".join(skill_sequence[:20])
            if len(skill_sequence) > 20:
                seq_str += f" ... ({len(skill_sequence)} total)"
            print(f"        sequence: {seq_str}")

    print(f"\nSuccess rate: {successes}/{args.eval_episodes} = "
          f"{100 * successes / args.eval_episodes:.1f}%")
    env.close()


# ============================================================================
# Helpers
# ============================================================================

def _build_checkpoint_dict(args) -> dict:
    """Build skill checkpoint dict from CLI args."""
    ckpts = {}
    if args.ckpt_reach:
        ckpts[COARSE_REACH] = args.ckpt_reach
    if args.ckpt_align:
        ckpts[FINE_ALIGN] = args.ckpt_align
    if args.ckpt_press:
        ckpts[PRESS] = args.ckpt_press
    if args.ckpt_retract:
        ckpts[RETRACT] = args.ckpt_retract
    if args.ckpt_traverse:
        ckpts[TRAVERSE] = args.ckpt_traverse
    return ckpts or None


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="HRL high-level policy training")

    # Training
    parser.add_argument('--timesteps', type=int, default=50_000,
                        help='Total high-level training steps (default: 50k)')
    parser.add_argument('--no-render', action='store_true')
    parser.add_argument('--resume', type=str, default=None, metavar='PATH')
    parser.add_argument('--target', type=str, default=None,
                        help='Target string to type (default: curriculum of strings)')

    # Evaluation
    parser.add_argument('--eval', type=str, default=None, metavar='PATH')
    parser.add_argument('--eval-episodes', type=int, default=20)

    # Environment
    parser.add_argument('--max-skill-steps', type=int, default=200,
                        help='Max low-level steps per skill invocation')
    parser.add_argument('--hl-horizon', type=int, default=100,
                        help='Max high-level decisions per episode')

    # Skill checkpoints
    parser.add_argument('--ckpt-reach', type=str, default=None)
    parser.add_argument('--ckpt-align', type=str, default=None)
    parser.add_argument('--ckpt-press', type=str, default=None)
    parser.add_argument('--ckpt-retract', type=str, default=None)
    parser.add_argument('--ckpt-traverse', type=str, default=None)

    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
