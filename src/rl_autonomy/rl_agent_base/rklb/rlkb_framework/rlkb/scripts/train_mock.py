from rlkb.agents.heuristic_press_agent import HeuristicPressAgent
from rlkb.core.logging import CSVLogger
from rlkb.core.runner import Runner
from rlkb.envs.mock_keyboard import MockKeyboardEnv


def main():
    env = MockKeyboardEnv()
    agent = HeuristicPressAgent(env.action_space, env.observation_space, training=False)
    runner = Runner(env, agent, train=False)
    logger = CSVLogger("mock_rollouts.csv")

    for ep in range(20):
        stats = runner.run_episode(ep)
        logger.log_episode(stats)
        print(f"Episode {stats.episode}: return={stats.ep_return:.3f}, steps={stats.steps}, success={stats.success}, time={stats.walltime_s:.2f}s")

    logger.close()


if __name__ == "__main__":
    main()
