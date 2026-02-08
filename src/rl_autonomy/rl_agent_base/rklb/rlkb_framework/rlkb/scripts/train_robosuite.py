from rlkb.agents.random_agent import RandomAgent
from rlkb.core.runner import Runner
from rlkb.envs.robosuite_base import RoboSuiteConfig, RoboSuiteGymEnv
from rlkb.envs.robosuite_keyboard import RoboSuiteKeyboardTask


def main():
    try:
        base_env = RoboSuiteGymEnv(RoboSuiteConfig())
    except ImportError:
        print("RoboSuite is not installed. Install it to run this smoke test.")
        return

    env = RoboSuiteKeyboardTask(base_env)
    agent = RandomAgent(env.action_space, env.observation_space)
    runner = Runner(env, agent, train=False)

    stats = runner.run_episode(0)
    print(f"RoboSuite smoke test -> return={stats.ep_return:.3f}, steps={stats.steps}, success={stats.success}")


if __name__ == "__main__":
    main()
