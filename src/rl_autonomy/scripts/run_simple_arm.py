import argparse
import importlib.util
from pathlib import Path
import time
import numpy as np
from typing import Any

# Path setup
_HELPER_PATH = Path(__file__).resolve().parents[1] / "utils" / "path_setup.py"
_SPEC = importlib.util.spec_from_file_location("rl_autonomy.utils.path_setup", _HELPER_PATH)
_PATH_SETUP = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_PATH_SETUP)
_PATH_SETUP.ensure_src_on_path()

from rl_autonomy.env import make_env


def _maybe_unwrap_reset(reset_out: Any):
    if isinstance(reset_out, tuple):
        return reset_out
    return reset_out, {}


def _zero_action(env):
    if hasattr(env, "action_space"):
        return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    low, _ = env.action_spec
    return np.zeros_like(low)


def _idle_view(env):
    obs, info = _maybe_unwrap_reset(env.reset())
    zero_action = _zero_action(env)

    print("Rendering SimpleArm environment. Close the viewer window or press Ctrl+C to exit.")
    try:
        while True:
            env.step(zero_action)
            if hasattr(env, "render"):
                env.render()
            time.sleep(1.0 / getattr(env, "control_freq", 30))
    except KeyboardInterrupt:
        print("Viewer closed.")
    finally:
        env.close()


def _build_env(render: bool):
    try:
        return make_env(
            env_name="Lift",
            robots="SimpleArm",
            render=render,
            render_offscreen=render,
            wrap_gym=True,
        )
        env.robots[0].robot_model.set_base_xpos([0, 0, 0.9])
        return env
    except ImportError as exc:
        if render and "EGL" in str(exc):
            print("Renderer unavailable, switching to headless mode.")
            return make_env(
                env_name="Lift",
                robots="SimpleArm",
                render=False,
                render_offscreen=False,
                wrap_gym=True,
            )
        raise


def main(render: bool = True):
    env = _build_env(render=render)
    _idle_view(env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimpleArm static viewer.")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run entirely headless (useful on servers without GPU/EGL).",
    )
    args = parser.parse_args()
    main(render=not args.no_render)
