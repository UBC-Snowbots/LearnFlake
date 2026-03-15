"""
End-to-end Alpha-Omega integration runner.

  1. Loads a trained Alpha reach policy
  2. Creates a KeypadReachEnv (simulation with 10 physical keys)
  3. Wraps it in an AlphaToolkitBackend
  4. Runs Omega's DeciderEngine to press the requested digit code

Usage (from the rover2026_rlkit directory):
    python3 run_alpha_omega.py --code 1230 \
        --policy-path checkpoints/best_policy_rover2026_reach.pth

    python3 run_alpha_omega.py --code 0 --record-video \
        --video-path videos/alpha_omega_key0.mp4
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from alpha_env_utils import ensure_alpha_import_paths

os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))

ensure_alpha_import_paths()

from keypad_env import KeypadReachEnv, KeypadReachEnvConfig
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy

from rlkb.decider.backends import AlphaToolkitBackend
from rlkb.decider.engine import DeciderEngine
from rlkb.decider.layouts import DEFAULT_KEY_LAYOUT
from rlkb.decider.strategy import build_default_registry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alpha-Omega integration runner")
    p.add_argument("--code", required=True, help="Digit code to press (e.g. '1230')")
    p.add_argument("--policy-path", type=str,
                   default="checkpoints/best_policy_rover2026_reach.pth")
    p.add_argument("--max-retries", type=int, default=1)
    p.add_argument("--horizon", type=int, default=2000)
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--video-path", type=str, default="videos/alpha_omega_run.mp4")
    p.add_argument("--fps", type=int, default=20)
    return p.parse_args()


class RecordingAlphaBackend(AlphaToolkitBackend):
    """AlphaToolkitBackend that captures frames for video recording."""

    def __init__(self, *args, record: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.record = record
        self.frames: list[np.ndarray] = []

    def _capture(self) -> None:
        if self.record:
            frame = self.env.render(mode="rgb_array")
            if frame is not None:
                self.frames.append(frame)

    def move_to(self, target):
        # Override to capture frames during move
        target_xyz = np.array(target.position, dtype=np.float32)
        self.env.set_target(target_xyz)
        obs = self.env.current_obs()
        info = dict(self._last_info)

        for step in range(self.max_move_steps):
            self._capture()
            action, _ = self.policy.get_action(obs)
            obs, _, done, info = self.env.step(action)

            if self._is_hover_locked(info):
                stable = 0
                for _ in range(self.stability_window):
                    self._capture()
                    action, _ = self.policy.get_action(obs)
                    obs, _, _, info = self.env.step(action)
                    if self._is_hover_locked(info):
                        stable += 1
                if stable >= self.stability_threshold:
                    break

        self._last_eef_pos = info["eef_xyz"].copy()
        self._last_info = dict(info)
        self.events.append(
            f"move:{target.key_id} dist={info.get('distance_to_goal', '?'):.4f}"
        )

    def press(self, target):
        for _ in range(self.press_steps):
            self._capture()
            action = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            _, _, _, info = self.env.step(action)

        self._last_eef_pos = info["eef_xyz"].copy()
        self._last_info = dict(info)
        self.events.append(f"press:{target.key_id}")


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

    policy = TanhGaussianPolicy(
        obs_dim=int(payload["obs_dim"]),
        action_dim=int(payload["action_dim"]),
        hidden_sizes=list(payload["hidden_sizes"]),
    )
    policy.load_state_dict(payload["policy_state_dict"])
    policy.eval()
    eval_policy = MakeDeterministic(policy)

    # ── create keypad env ────────────────────────────────────────────
    cfg = KeypadReachEnvConfig(
        horizon=args.horizon,
        render=False,
        offscreen_render=args.record_video,
    )
    raw_env = KeypadReachEnv(cfg)
    env = NormalizedBoxEnv(raw_env, reward_scale=1.0)

    # ── create backend ───────────────────────────────────────────────
    backend = RecordingAlphaBackend(
        env=raw_env,
        policy=eval_policy,
        record=args.record_video,
    )

    # ── run Omega ────────────────────────────────────────────────────
    strategy = build_default_registry().get("sequence")
    engine = DeciderEngine(
        code=args.code,
        strategy=strategy,
        backend=backend,
        layout=DEFAULT_KEY_LAYOUT,
        max_retries=args.max_retries,
    )

    # Reset the env before running
    raw_env.reset()

    print(f"Running Omega with code '{args.code}'...")
    result = engine.run()

    # ── results ──────────────────────────────────────────────────────
    status = "SUCCESS" if result.success else "FAILED"
    print(f"\n── Result: {status} ──")
    print(f"  code:     {result.code}")
    print(f"  strategy: {result.strategy_name}")
    print(f"  state:    {result.final_state}")

    for step in result.completed_steps:
        print(f"  pressed key {step.target.key_id} ({step.target.description}) "
              f"in {step.attempts} attempt(s)")

    if result.failed_step:
        step = result.failed_step
        print(f"  FAILED key {step.target.key_id} ({step.target.description}) "
              f"after {step.attempts} attempt(s): {step.message}")

    print(f"\nBackend events:")
    for event in backend.events:
        print(f"  {event}")

    # ── save video ───────────────────────────────────────────────────
    if args.record_video and backend.frames:
        out_path = Path(args.video_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w = backend.frames[0].shape[:2]
        writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))
        for f in backend.frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"\nVideo saved: {out_path}  ({len(backend.frames)} frames)")

    env.close()


if __name__ == "__main__":
    main()
