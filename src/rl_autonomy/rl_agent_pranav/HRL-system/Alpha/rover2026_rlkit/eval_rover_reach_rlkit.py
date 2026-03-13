from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy

# for screen: cd /LearnFlake/src/rl_autonomy/rl_agent_pranav/HRL-system/Alpha/rover2026_rlkit python3 eval_rover_reach_rlkit.py \ --policy-path checkpoints/best_policy_rover2026_reach.pth \   --episodes 20 --horizon 120 \   --render --auto-display

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved Rover2026 RLKit SAC policy")
    parser.add_argument(
        "--policy-path",
        type=str,
        default="checkpoints/best_policy_rover2026_reach.pth",
        help="Path to saved .pth policy checkpoint",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-path", type=str, default="videos/eval_rollout.mp4")
    parser.add_argument("--force-gui", action="store_true", help="Force on-screen X11 rendering")
    parser.add_argument("--auto-display", action="store_true", help="Auto-resolve DISPLAY like pranav scripts")
    return parser.parse_args()


def _is_display_usable(display: str) -> bool:
    if not display:
        return False
    if display.startswith(":"):
        disp = display[1:].split(".", 1)[0]
        return disp.isdigit() and os.path.exists(f"/tmp/.X11-unix/X{disp}")
    if ":" not in display:
        return False
    host, rest = display.split(":", 1)
    screen = rest.split(".", 1)[0]
    if not screen.isdigit():
        return False
    port = 6000 + int(screen)
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _auto_configure_display() -> None:
    cur = os.environ.get("DISPLAY", "")
    if cur and _is_display_usable(cur):
        return
    if os.path.exists("/tmp/.X11-unix/X0"):
        os.environ["DISPLAY"] = ":0"
        return
    candidate = "host.docker.internal:0.0"
    if _is_display_usable(candidate):
        os.environ["DISPLAY"] = candidate
        return
    os.environ.pop("DISPLAY", None)


def main() -> None:
    args = parse_args()
    if args.auto_display:
        _auto_configure_display()

    display = os.environ.get("DISPLAY", "")
    display_ok = _is_display_usable(display)
    want_gui = args.render or args.force_gui
    if want_gui and display_ok:
        os.environ["MUJOCO_GL"] = "glfw"
        print(f"X11 render enabled: DISPLAY={display} MUJOCO_GL=glfw")
    else:
        os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
        print(
            f"Headless/offscreen mode: DISPLAY={display!r} display_ok={display_ok} "
            f"MUJOCO_GL={os.environ['MUJOCO_GL']}"
        )

    from reach_env import Rover2026ReachEnv, RoverReachEnvConfig

    policy_path = Path(args.policy_path)
    if not policy_path.is_absolute():
        cwd_candidate = (Path.cwd() / policy_path).resolve()
        script_candidate = (Path(__file__).resolve().parent / policy_path).resolve()
        if cwd_candidate.exists():
            policy_path = cwd_candidate
        elif script_candidate.exists():
            policy_path = script_candidate
        else:
            raise FileNotFoundError(
                "Policy checkpoint not found. Tried:\n"
                f"  - {cwd_candidate}\n"
                f"  - {script_candidate}"
            )
    else:
        policy_path = policy_path.resolve()
    payload = torch.load(policy_path, map_location="cpu")

    env_cfg = RoverReachEnvConfig(**payload["env_config"])
    env_cfg.horizon = args.horizon
    env_cfg.render = bool(args.render)
    env_cfg.offscreen_render = bool(args.record_video)

    env = NormalizedBoxEnv(Rover2026ReachEnv(env_cfg), reward_scale=1.0)

    policy = TanhGaussianPolicy(
        obs_dim=int(payload["obs_dim"]),
        action_dim=int(payload["action_dim"]),
        hidden_sizes=list(payload["hidden_sizes"]),
    )
    policy.load_state_dict(payload["policy_state_dict"])
    policy.eval()
    eval_policy = MakeDeterministic(policy)

    returns = []
    successes = []
    final_distances = []
    frames = []

    for ep in range(args.episodes):
        obs = env.reset()
        eval_policy.reset()
        ep_ret = 0.0
        ep_success = 0.0
        ep_final_distance = np.nan

        for _ in range(args.horizon):
            if args.record_video and ep == 0:
                frames.append(env.wrapped_env.render(mode="rgb_array"))

            action, _ = eval_policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            ep_ret += float(reward)
            ep_success = max(ep_success, float(info.get("is_success", 0.0)))
            ep_final_distance = float(info.get("distance_to_goal", np.nan))

            if done:
                break

        returns.append(ep_ret)
        successes.append(ep_success)
        final_distances.append(ep_final_distance)
        print(
            f"episode={ep + 1:02d} return={ep_ret:.3f} "
            f"success={ep_success:.1f} final_distance={ep_final_distance:.4f}"
        )

    if args.record_video and frames:
        out = Path(args.video_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out, frames, fps=20)
        print(f"saved video: {out}")

    print("\nEvaluation summary")
    print(f"  avg_return: {np.mean(returns):.3f}")
    print(f"  success_rate: {np.mean(successes):.3f}")
    print(f"  avg_final_distance: {np.nanmean(final_distances):.4f}")

    env.close()


if __name__ == "__main__":
    main()
