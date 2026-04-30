"""Visualize the new KeyboardEnv — interactive MuJoCo viewer or offscreen PNGs.

Two modes:

  --interactive (default if $DISPLAY is set):
      Open a live MuJoCo passive viewer. Uses GLFW. Needs X11 forwarding
      (the rover_gpu container has it set up via /tmp/.X11-unix mount).

  --save-frames DIR:
      Render to PNG every N steps. Works without a display. Useful inside
      headless containers, ssh sessions, or for embedding in docs.

Common flags:

  --mode {approach,strike}   which env mode (default approach)
  --key NAME                 target key (default 'g'; --random-key for random)
  --steps N                  number of sim steps to run (default 200)
  --policy {zero,p_ctrl}     drive arm with zeros or the M1 P-controller
  --camera NAME              robosuite camera (default 'frontview'; also try
                             'sideview', 'agentview', 'birdview')
  --width / --height         offscreen render size

Examples:
    # Interactive viewer, central key:
    python -m rl_autonomy.tools.visualize --mode approach --key g

    # Watch the M1 P-controller drive the arm to the F12 key (corner):
    python -m rl_autonomy.tools.visualize --policy p_ctrl --key f12 --steps 400

    # Offscreen, dump 20 frames to /tmp/viz/:
    python -m rl_autonomy.tools.visualize --save-frames /tmp/viz \
                                          --steps 200 --frame-every 10
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Callable

import numpy as np

warnings.filterwarnings("ignore")

# robosuite path — inserted by rl_autonomy.envs at import time, but the legacy
# launchers add it explicitly. We do the same so the script also runs from
# outside the package directory.
_ROBO_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "external_pkgs", "RoboSuite")
)
if os.path.exists(_ROBO_PATH) and _ROBO_PATH not in sys.path:
    sys.path.insert(0, _ROBO_PATH)


def _zero_policy(env, _step: int) -> np.ndarray:
    return np.zeros(7, dtype=np.float32)


def _p_ctrl_policy_factory(target_xyz: np.ndarray) -> Callable:
    """Reuses the M1 controller — full-pose adaptive-weight DLS Jacobian step."""
    import mujoco

    def policy(env, step: int) -> np.ndarray:
        sim = env.sim
        ep = sim.data.site_xpos[env._eef_site_id].copy()
        err_pos = target_xyz - ep

        pf = env.robots[0].robot_model.naming_prefix
        obs = env._get_observations(force_update=False)
        eef_quat = obs.get(f"{pf}eef_quat", np.array([1.0, 0.0, 0.0, 0.0]))
        from rl_autonomy.envs.keyboard_env import quat_to_rot
        R = quat_to_rot(eef_quat)
        push_dir = -R[:, 1]
        err_rot = np.cross(push_dir, np.array([0.0, 0.0, -1.0]))

        tilt_mag = float(np.linalg.norm(err_rot))
        w_rot = 5.0 if tilt_mag > 0.087 else (0.5 if tilt_mag < 0.017 else 1.0)

        jacp = np.zeros((3, sim.model.nv))
        jacr = np.zeros((3, sim.model.nv))
        mujoco.mj_jacSite(sim.model._model, sim.data._data, jacp, jacr, env._eef_site_id)
        J = np.vstack([jacp[:, :6], w_rot * jacr[:, :6]])
        err = np.concatenate([err_pos, w_rot * err_rot])
        damping = 0.06
        JJt = J @ J.T + damping**2 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, err) * 0.5

        a = np.zeros(7, dtype=np.float32)
        a[0:6] = np.clip(dq / 0.05, -1.0, 1.0)
        a[6] = -1.0
        return a

    return policy


def _strike_policy(env, _step: int) -> np.ndarray:
    """Always-fire solenoid (mode='strike'). Joints zeroed by the env action wrapper."""
    a = np.zeros(7, dtype=np.float32)
    a[6] = 1.0
    return a


def _make_env(args):
    from rl_autonomy.envs import KeyboardEnv

    use_renderer = args.interactive or args.has_display
    env = KeyboardEnv(
        mode=args.mode,
        random_key=args.random_key,
        horizon=args.steps + 50,
        render=use_renderer,
        use_camera_obs=False,
    )
    env.reset()
    if not args.random_key:
        env.set_target_key(args.key)
    return env


def run_interactive(args):
    """Live viewer — passes through robosuite's render() loop."""
    env = _make_env(args)

    if args.policy == "zero":
        policy = _zero_policy
    elif args.policy == "p_ctrl":
        key_pos = env.sim.data.body_xpos[env._key_body_ids[env.target_key]].copy()
        target = np.array([key_pos[0], key_pos[1], key_pos[2] + env.hover_height])
        policy = _p_ctrl_policy_factory(target)
    elif args.policy == "strike":
        policy = _strike_policy
    else:
        raise ValueError(args.policy)

    print(f"[viz] mode={args.mode}  key={env.target_key}  policy={args.policy}  steps={args.steps}")
    print(f"[viz] press Ctrl+C in the terminal to stop early")
    print(f"[viz] tip: with --policy p_ctrl, watch the arm move toward the highlighted key site (green dot)")

    for step in range(args.steps):
        a = policy(env, step)
        env.step(a)
        env.render()
    env.close()


def run_offscreen(args):
    """Render frames to disk via robosuite's offscreen pipeline."""
    env = _make_env(args)

    out_dir = args.save_frames
    os.makedirs(out_dir, exist_ok=True)

    if args.policy == "zero":
        policy = _zero_policy
    elif args.policy == "p_ctrl":
        key_pos = env.sim.data.body_xpos[env._key_body_ids[env.target_key]].copy()
        target = np.array([key_pos[0], key_pos[1], key_pos[2] + env.hover_height])
        policy = _p_ctrl_policy_factory(target)
    elif args.policy == "strike":
        policy = _strike_policy
    else:
        raise ValueError(args.policy)

    n_saved = 0
    for step in range(args.steps + 1):
        if step % args.frame_every == 0 or step == args.steps:
            frame = env.sim.render(
                width=args.width,
                height=args.height,
                camera_name=args.camera,
            )
            # robosuite returns frames upside-down for some configs
            frame = frame[::-1]
            try:
                from PIL import Image
            except ImportError:
                raise SystemExit("Pillow required for --save-frames. Install: pip install Pillow")
            Image.fromarray(frame).save(os.path.join(out_dir, f"frame_{step:04d}.png"))
            n_saved += 1
        if step < args.steps:
            env.step(policy(env, step))
    env.close()
    print(f"[viz] saved {n_saved} frames to {out_dir}")
    print(f"[viz] view: feh {out_dir}/frame_*.png   (or copy out of the container with `docker cp`)")


def main():
    parser = argparse.ArgumentParser(
        prog="rl_autonomy.tools.visualize",
        description="Visualize the new KeyboardEnv (interactive viewer or offscreen PNGs).",
    )
    parser.add_argument("--mode", choices=("approach", "strike"), default="approach")
    parser.add_argument("--key", default="g", help="target key name (e.g. g, f12, esc, space)")
    parser.add_argument("--random-key", action="store_true", help="ignore --key, sample randomly")
    parser.add_argument("--policy", choices=("zero", "p_ctrl", "strike"), default="p_ctrl")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--camera", default="frontview")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--save-frames", default=None, metavar="DIR",
                        help="render offscreen to PNG sequence in DIR (no live viewer)")
    parser.add_argument("--frame-every", type=int, default=20,
                        help="save one frame every N sim steps (only with --save-frames)")
    parser.add_argument("--interactive", action="store_true",
                        help="force interactive viewer even if $DISPLAY is unset")
    args = parser.parse_args()

    args.has_display = bool(os.environ.get("DISPLAY"))
    if args.save_frames:
        return run_offscreen(args)
    return run_interactive(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
