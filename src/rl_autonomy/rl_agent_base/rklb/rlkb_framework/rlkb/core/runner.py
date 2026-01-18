import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym

from .types import Transition


@dataclass
class EpisodeStats:
    episode: int
    steps: int
    ep_return: float
    success: bool
    terminated: bool
    truncated: bool
    walltime_s: float


class Runner:
    def __init__(self, env: gym.Env, agent, train: bool = True, train_every: int = 1, max_steps_per_episode: Optional[int] = None):
        self.env = env
        self.agent = agent
        self.train = train
        self.train_every = max(1, int(train_every))
        self.max_steps_per_episode = max_steps_per_episode

    def run_episode(self, episode_idx: int, seed: Optional[int] = None, deterministic: bool = False) -> EpisodeStats:
        start_time = time.time()
        obs, info = self.env.reset(seed=seed)
        done = False
        steps = 0
        ep_return = 0.0

        while not done:
            action = self.agent.act(obs, deterministic=deterministic)
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            transition = Transition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=terminated or truncated,
                info=info,
            )
            self.agent.observe(transition)

            if self.train and steps % self.train_every == 0:
                self.agent.train_step()

            ep_return += reward
            steps += 1
            obs = next_obs
            done = terminated or truncated

            if self.max_steps_per_episode is not None and steps >= self.max_steps_per_episode:
                truncated = True
                done = True

        elapsed = time.time() - start_time
        success = bool(info.get("success", False)) if info is not None else False
        return EpisodeStats(
            episode=episode_idx,
            steps=steps,
            ep_return=ep_return,
            success=success,
            terminated=terminated,
            truncated=truncated,
            walltime_s=elapsed,
        )
