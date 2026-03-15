"""
Video runner for Rover2026 reach policy (standoff-aware).

Runs full episodes and records every frame with a live overlay showing:
  - distance to target (cm)
  - standoff error (deviation from the 3 cm shell)
  - progress bar (green = on the standoff shell, red = far from it)

The env itself handles the 3 cm standoff reward — this script just
visualises and reports the error the trained policy achieves.

Usage (from the rover2026_rlkit directory):
    python3 video_runner.py \
        --policy-path checkpoints/best_policy_rover2026_reach.pth \
        --episodes 5 --horizon 200 \
        --video-path videos/standoff_run.mp4
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from alpha_env_utils import config_from_payload, ensure_alpha_import_paths

# ── headless-safe default ────────────────────────────────────────────
os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

ensure_alpha_import_paths()

from reach_env import Rover2026ReachEnv, RoverReachEnvConfig  # noqa: E402
from rlkit.envs.wrappers import NormalizedBoxEnv  # noqa: E402
from rlkit.torch.sac.policies import (  # noqa: E402
    MakeDeterministic,
    TanhGaussianPolicy,
)


# ── overlay drawing ──────────────────────────────────────────────────

def draw_overlay(frame: np.ndarray, distance: float,
                 standoff_error: float, standoff_m: float,
                 tolerance_m: float, max_err: float = 0.20) -> np.ndarray:
    """Draw a standoff-error progress bar + text in the top-right corner."""
    h, w = frame.shape[:2]

    bar_w, bar_h = 150, 16
    margin = 10
    x0 = w - bar_w - margin
    y0 = margin

    # fraction: 1.0 = perfectly on the standoff shell, 0.0 = far off
    frac = float(np.clip(1.0 - standoff_error / max_err, 0.0, 1.0))

    # background
    cv2.rectangle(frame, (x0 - 2, y0 - 2), (x0 + bar_w + 2, y0 + bar_h + 2),
                  (40, 40, 40), -1)
    # filled portion – green when on shell, red when far
    r = int((1.0 - frac) * 220)
    g = int(frac * 220)
    fill_w = max(1, int(frac * bar_w))
    cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y0 + bar_h), (0, g, r), -1)
    # border
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (200, 200, 200), 1)

    # text: raw distance and standoff error
    dist_txt = f"dist {distance * 100:.1f} cm"
    err_txt = f"err  {standoff_error * 100:.2f} cm"
    cv2.putText(frame, dist_txt, (x0, y0 + bar_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, err_txt, (x0, y0 + bar_h + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    # success indicator when within tolerance
    if standoff_error <= tolerance_m:
        cv2.putText(frame, "LOCKED", (x0 + bar_w - 52, y0 + bar_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 80), 1, cv2.LINE_AA)

    return frame


# ── main ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Video runner – visualise standoff reach policy")
    p.add_argument("--policy-path", type=str,
                   default="checkpoints/best_policy_rover2026_reach.pth")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--video-path", type=str, default="videos/standoff_run.mp4")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--camera-name", type=str, default="birdview")
    return p.parse_args()


def main() -> None:
    import cv2

    args = parse_args()

    # ── load policy ──────────────────────────────────────────────────
    policy_path = Path(args.policy_path)
    if not policy_path.is_absolute():
        for candidate in [Path.cwd() / policy_path,
                          Path(__file__).resolve().parent / policy_path]:
            if candidate.resolve().exists():
                policy_path = candidate.resolve()
                break
        else:
            raise FileNotFoundError(f"Policy not found: {args.policy_path}")
    else:
        policy_path = policy_path.resolve()

    payload = torch.load(policy_path, map_location="cpu")

    env_cfg = config_from_payload(RoverReachEnvConfig, payload.get("env_config", {}))
    env_cfg.horizon = args.horizon
    env_cfg.render = False
    env_cfg.offscreen_render = True
    env_cfg.camera_name = args.camera_name
    env_cfg.camera_width = args.cam_width
    env_cfg.camera_height = args.cam_height

    standoff_m = env_cfg.standoff_m
    tolerance_m = env_cfg.standoff_tolerance_m

    env = NormalizedBoxEnv(Rover2026ReachEnv(env_cfg), reward_scale=1.0)

    policy = TanhGaussianPolicy(
        obs_dim=int(payload["obs_dim"]),
        action_dim=int(payload["action_dim"]),
        hidden_sizes=list(payload["hidden_sizes"]),
    )
    policy.load_state_dict(payload["policy_state_dict"])
    policy.eval()
    eval_policy = MakeDeterministic(policy)

    # ── run episodes ─────────────────────────────────────────────────
    all_frames: list[np.ndarray] = []
    episode_results: list[dict] = []

    for ep in range(args.episodes):
        obs = env.reset()
        eval_policy.reset()

        ep_standoff_errors: list[float] = []
        ep_distances: list[float] = []
        ep_successes = 0

        for step in range(args.horizon):
            action, _ = eval_policy.get_action(obs)
            obs, reward, done, info = env.step(action)

            distance = float(info.get("distance_to_goal", 0.0))
            standoff_error = float(info.get("standoff_error", abs(distance - standoff_m)))
            is_success = float(info.get("is_success", 0.0))

            ep_distances.append(distance)
            ep_standoff_errors.append(standoff_error)
            ep_successes += int(is_success > 0.5)

            # render with overlay
            frame = env.wrapped_env.render(mode="rgb_array")
            frame = draw_overlay(frame, distance, standoff_error,
                                 standoff_m, tolerance_m)
            all_frames.append(frame)

            if done:
                break

        # per-episode stats
        final_dist = ep_distances[-1] if ep_distances else float("nan")
        final_err = ep_standoff_errors[-1] if ep_standoff_errors else float("nan")
        mean_err = float(np.mean(ep_standoff_errors)) if ep_standoff_errors else float("nan")
        hold_frac = ep_successes / max(1, len(ep_standoff_errors))

        episode_results.append({
            "episode": ep + 1,
            "final_distance_cm": final_dist * 100,
            "final_standoff_error_cm": final_err * 100,
            "mean_standoff_error_cm": mean_err * 100,
            "hold_fraction": hold_frac,
            "steps": step + 1,
        })
        print(f"  ep {ep + 1:02d}  "
              f"final_dist={final_dist * 100:.2f} cm  "
              f"standoff_err={final_err * 100:.2f} cm  "
              f"mean_err={mean_err * 100:.2f} cm  "
              f"hold={hold_frac:.0%}")

    # ── write video ──────────────────────────────────────────────────
    out_path = Path(args.video_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = all_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))
    for f in all_frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"\nVideo saved: {out_path}  ({len(all_frames)} frames)")

    # ── error report ─────────────────────────────────────────────────
    final_errs = [r["final_standoff_error_cm"] for r in episode_results]
    mean_errs = [r["mean_standoff_error_cm"] for r in episode_results]
    holds = [r["hold_fraction"] for r in episode_results]

    print(f"\n── Standoff Error Report (target = {standoff_m * 100:.1f} cm) ──")
    print(f"  episodes:                  {args.episodes}")
    print(f"  tolerance:                 {tolerance_m * 100:.2f} cm")
    print(f"  mean final standoff err:   {np.mean(final_errs):.2f} cm")
    print(f"  std  final standoff err:   {np.std(final_errs):.2f} cm")
    print(f"  mean episode-avg err:      {np.mean(mean_errs):.2f} cm")
    print(f"  mean hold fraction:        {np.mean(holds):.0%}")
    print(f"  best final standoff err:   {np.min(final_errs):.2f} cm")
    print(f"  worst final standoff err:  {np.max(final_errs):.2f} cm")

    env.close()


if __name__ == "__main__":
    main()
