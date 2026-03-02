import os

# Rendering backend toggle:
# - Default: glfw (onscreen via X11/VcXsrv)
# - Headless: set MUJOCO_GL_BACKEND = "egl" (or export MUJOCO_GL=egl)
MUJOCO_GL_BACKEND = os.environ.get("MUJOCO_GL", "glfw")
os.environ["MUJOCO_GL"] = MUJOCO_GL_BACKEND

from rlkb.agents.random_agent import RandomAgent
from rlkb.core.runner import Runner
from rlkb.envs.robosuite_base import RoboSuiteConfig, RoboSuiteGymEnv
from rlkb.envs.robosuite_keyboard import RoboSuiteKeyboardTask


def main():
    try:
        show_gui = os.environ.get("ROBOSUITE_GUI", "1").lower() not in {"0", "false", "no"}
        config = RoboSuiteConfig(has_renderer=show_gui)
        base_env = RoboSuiteGymEnv(config)
    except ImportError:
        print("RoboSuite is not installed. Install it to run this smoke test.")
        return
    except Exception as exc:
        print(f"Failed to create RoboSuite environment (MUJOCO_GL={os.environ.get('MUJOCO_GL')}): {exc}")
        return

    env = RoboSuiteKeyboardTask(base_env)
    agent = RandomAgent(env.action_space, env.observation_space)
    runner = Runner(env, agent, train=False)

    print(f"Starting RoboSuite smoke test with has_renderer={show_gui}, MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    stats = runner.run_episode(0)
    print(f"RoboSuite smoke test -> return={stats.ep_return:.3f}, steps={stats.steps}, success={stats.success}")


if __name__ == "__main__":
    main()
