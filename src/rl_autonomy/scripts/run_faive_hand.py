"""
Quick smoke-test script to visualize the FaiveHand manipulator inside a robosuite environment.
"""

import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.append(str(_SRC_ROOT))

from rl_autonomy.env import make_env


def main(episode_length=200):
    env = make_env(render=True, robots="FaiveHand")
    obs, info = env.reset()
    for _ in range(episode_length):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()


if __name__ == "__main__":
    main()
