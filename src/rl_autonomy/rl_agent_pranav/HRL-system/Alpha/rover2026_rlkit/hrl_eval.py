"""
HRL Evaluation Renderer -- Full pipeline visualization of the Alpha-Omega system.

Renders the complete hierarchical RL loop:
  Omega (high-level):  decode digit code → sequence of key targets
  Alpha (low-level):   reach policy servos to 3cm standoff → press down → verify

Every frame is annotated with:
  - Current Omega state (FSM phase)
  - Active key target + key ID
  - Distance / standoff error bar (top-right)
  - Step counter and code progress (top-left)
  - Per-key result annotations

Supports live X11 display (--render --auto-display) and/or MP4 recording.

Usage:
    # Live X11 window (primary mode)
    python3 hrl_eval.py --code 1230 --render --auto-display \
        --policy-path checkpoints/best_policy_rover2026_reach.pth

    # Record to MP4 (headless)
    python3 hrl_eval.py --code 1230 --record-video \
        --video-path videos/hrl_eval.mp4

    # Both: live window + save video
    python3 hrl_eval.py --code 1230 --render --auto-display --record-video
"""
from __future__ import annotations

import argparse
import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch  # noqa: E402 — must be before mujoco-dependent imports

from alpha_env_utils import ensure_alpha_import_paths


# ── X11 display helpers (from eval_rover_reach_rlkit.py) ─────────────

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

ensure_alpha_import_paths()

from rlkb.decider.layouts import DEFAULT_KEY_LAYOUT
from rlkb.decider.strategy import build_default_registry
from rlkb.decider.types import KeyTarget


# ── HUD overlay ──────────────────────────────────────────────────────

@dataclass
class HUDState:
    """Mutable state for the heads-up display overlay."""
    phase: str = "IDLE"
    code: str = ""
    current_key: str = ""
    key_index: int = 0
    total_keys: int = 0
    distance: float = 0.0
    standoff_error: float = 0.0
    step: int = 0
    completed_keys: List[str] = field(default_factory=list)
    failed_key: Optional[str] = None


def draw_hud(frame: np.ndarray, hud: HUDState, cfg_standoff: float) -> np.ndarray:
    """Draw the full HRL evaluation HUD onto a frame."""
    import cv2
    frame = np.ascontiguousarray(frame.copy())
    h, w = frame.shape[:2]

    # ── top-left: code progress ──────────────────────────────────────
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (220, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    code_display = ""
    for i, ch in enumerate(hud.code):
        if ch in hud.completed_keys[:i + 1 if i < len(hud.completed_keys) else 0]:
            code_display += ch
        else:
            code_display += ch
    # Show code with current position highlighted
    progress = f"Code: {hud.code}  [{hud.key_index + 1}/{hud.total_keys}]"
    cv2.putText(frame, progress, (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Phase badge
    phase_colors = {
        "IDLE": (150, 150, 150),
        "MOVING": (50, 180, 255),
        "HOLDING": (50, 255, 180),
        "PRESSING": (255, 100, 50),
        "VERIFYING": (255, 255, 50),
        "SUCCESS": (50, 255, 50),
        "FAILED": (50, 50, 255),
        "COMPLETED": (50, 255, 50),
    }
    color = phase_colors.get(hud.phase, (200, 200, 200))
    cv2.putText(frame, hud.phase, (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Current target
    if hud.current_key:
        cv2.putText(frame, f"Key: {hud.current_key}", (12, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Step: {hud.step}", (12, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)

    # ── top-right: distance bar ──────────────────────────────────────
    bar_w, bar_h = 150, 16
    margin = 10
    x0 = w - bar_w - margin
    y0 = margin

    max_err = 0.15
    frac = float(np.clip(1.0 - hud.standoff_error / max_err, 0.0, 1.0))

    cv2.rectangle(frame, (x0 - 2, y0 - 2), (x0 + bar_w + 2, y0 + bar_h + 2),
                  (40, 40, 40), -1)
    r = int((1.0 - frac) * 220)
    g = int(frac * 220)
    fill_w = max(1, int(frac * bar_w))
    cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y0 + bar_h), (0, g, r), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (200, 200, 200), 1)

    dist_txt = f"dist {hud.distance * 100:.1f} cm"
    err_txt = f"err  {hud.standoff_error * 100:.2f} cm"
    cv2.putText(frame, dist_txt, (x0, y0 + bar_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, err_txt, (x0, y0 + bar_h + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    if hud.standoff_error <= cfg_standoff * 0.2:
        cv2.putText(frame, "LOCKED", (x0 + bar_w - 55, y0 + bar_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 80), 1, cv2.LINE_AA)

    # ── bottom: completed keys ───────────────────────────────────────
    if hud.completed_keys:
        done_txt = "Done: " + " ".join(hud.completed_keys)
        cv2.putText(frame, done_txt, (12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 255, 100), 1, cv2.LINE_AA)

    if hud.failed_key:
        cv2.putText(frame, f"FAILED: key {hud.failed_key}", (w // 2 - 60, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    return frame


# ── HRL runner ───────────────────────────────────────────────────────

def run_hrl_episode(
    env: KeypadReachEnv,
    policy,
    code: str,
    hud: HUDState,
    max_move_steps: int = 200,
    press_steps: int = 15,
    stability_window: int = 10,
    stability_threshold: int = 7,
    live_display: bool = False,
    record_frames: bool = True,
    display_wait_ms: int = 1,
) -> tuple[list[np.ndarray], dict]:
    """
    Run one full HRL episode: decode code → for each key: reach → press → verify.
    Returns (frames, result_dict).
    """
    import cv2

    frames: list[np.ndarray] = []
    window_name = "HRL Eval - Alpha x Omega"
    strategy = build_default_registry().get("sequence")
    targets = strategy.decode(code, DEFAULT_KEY_LAYOUT)

    def is_hover_locked(info: dict) -> bool:
        if "hover_locked" in info:
            return bool(info["hover_locked"])
        xy_tol = float(getattr(env.config, "hover_success_xy_tolerance_m", env.config.standoff_tolerance_m))
        z_tol = float(getattr(env.config, "hover_success_z_tolerance_m", env.config.standoff_tolerance_m))
        return (
            float(info.get("hover_xy_error", float("inf"))) <= xy_tol
            and float(info.get("hover_z_error", float("inf"))) <= z_tol
        )

    hud.code = code
    hud.total_keys = len(targets)
    hud.completed_keys = []
    hud.failed_key = None
    hud.step = 0

    env.reset()
    results: list[dict] = []

    def capture():
        frame = env.render(mode="rgb_array")
        if frame is None:
            return
        annotated = draw_hud(frame, hud, env.config.standoff_m)
        if record_frames:
            frames.append(annotated)
        if live_display:
            bgr = cv2.cvtColor(np.ascontiguousarray(annotated), cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, bgr)
            key = cv2.waitKey(display_wait_ms)
            if key == 27 or key == ord("q"):  # ESC or q to quit
                raise KeyboardInterrupt("User closed display window")

    for idx, target in enumerate(targets):
        hud.key_index = idx
        hud.current_key = target.key_id

        # ── MOVE phase ───────────────────────────────────────────────
        hud.phase = "MOVING"
        target_xyz = np.array(target.position, dtype=np.float32)
        env.set_target(target_xyz)
        obs = env.current_obs()

        converged = False
        for step in range(max_move_steps):
            hud.step += 1
            capture()

            action, _ = policy.get_action(obs)
            obs, _, done, info = env.step(action)
            hud.distance = info.get("distance_to_goal", 0.0)
            hud.standoff_error = info.get("standoff_error", 0.0)

            if is_hover_locked(info):
                hud.phase = "HOLDING"
                stable = 0
                for _ in range(stability_window):
                    hud.step += 1
                    capture()
                    action, _ = policy.get_action(obs)
                    obs, _, _, info = env.step(action)
                    hud.distance = info.get("distance_to_goal", 0.0)
                    hud.standoff_error = info.get("standoff_error", 0.0)
                    if is_hover_locked(info):
                        stable += 1
                if stable >= stability_threshold:
                    converged = True
                    break

        move_dist = hud.distance

        # ── PRESS phase ──────────────────────────────────────────────
        hud.phase = "PRESSING"
        for _ in range(press_steps):
            hud.step += 1
            capture()
            action = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            _, _, _, info = env.step(action)
            hud.distance = info.get("distance_to_goal", 0.0)
            hud.standoff_error = info.get("standoff_error", 0.0)

        eef_pos = info["eef_xyz"]
        press_dist = float(np.linalg.norm(eef_pos - target_xyz))

        # ── VERIFY phase ─────────────────────────────────────────────
        hud.phase = "VERIFYING"
        capture()
        verified = press_dist <= 0.01  # 1cm tolerance

        if verified:
            hud.phase = "SUCCESS"
            hud.completed_keys.append(target.key_id)
            # Hold for a few frames to show success
            for _ in range(15):
                hud.step += 1
                capture()
        else:
            hud.phase = "FAILED"
            hud.failed_key = target.key_id
            for _ in range(20):
                hud.step += 1
                capture()

        results.append({
            "key_id": target.key_id,
            "converged_at_standoff": converged,
            "move_distance": move_dist,
            "press_distance": press_dist,
            "verified": verified,
        })

        if not verified:
            break

    all_ok = all(r["verified"] for r in results)
    hud.phase = "COMPLETED" if all_ok else "FAILED"
    for _ in range(30):
        hud.step += 1
        capture()

    return frames, {
        "code": code,
        "success": all_ok,
        "keys_completed": len([r for r in results if r["verified"]]),
        "keys_total": len(targets),
        "key_results": results,
    }


# ── main ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HRL Evaluation Renderer")
    p.add_argument("--code", required=True, help="Digit code to press")
    p.add_argument("--policy-path", type=str,
                   default="checkpoints/best_policy_rover2026_reach.pth")
    p.add_argument("--episodes", type=int, default=1,
                   help="Number of evaluation episodes")
    p.add_argument("--horizon", type=int, default=3000)
    # display
    p.add_argument("--render", action="store_true",
                   help="Show live X11 window with HUD overlay")
    p.add_argument("--auto-display", action="store_true",
                   help="Auto-resolve DISPLAY env var for Docker/WSLg")
    # recording
    p.add_argument("--record-video", action="store_true",
                   help="Save annotated frames to MP4")
    p.add_argument("--video-path", type=str, default="videos/hrl_eval.mp4")
    p.add_argument("--fps", type=int, default=20)
    # camera
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--camera-name", type=str, default="birdview")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── configure display BEFORE importing MuJoCo-dependent modules ──
    if args.auto_display:
        _auto_configure_display()

    display = os.environ.get("DISPLAY", "")
    display_ok = _is_display_usable(display)
    want_gui = args.render

    if want_gui and display_ok:
        os.environ["MUJOCO_GL"] = "glfw"
        print(f"X11 live display enabled: DISPLAY={display} MUJOCO_GL=glfw")
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")
        if want_gui and not display_ok:
            print(f"WARNING: --render requested but no usable DISPLAY ({display!r}). "
                  f"Falling back to offscreen. Use --auto-display or set DISPLAY.")

    # Now safe to import cv2 and MuJoCo-dependent modules
    import cv2
    from keypad_env import KeypadReachEnv, KeypadReachEnvConfig
    from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy

    live_display = want_gui and display_ok
    record_video = args.record_video
    window_name = "HRL Eval - Alpha x Omega"

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

    # ── create env ───────────────────────────────────────────────────
    cfg = KeypadReachEnvConfig(
        horizon=args.horizon,
        render=False,
        offscreen_render=True,  # always need offscreen for HUD frames
        camera_name=args.camera_name,
        camera_width=args.cam_width,
        camera_height=args.cam_height,
    )
    env = KeypadReachEnv(cfg)

    # ── run episodes ─────────────────────────────────────────────────
    all_frames: list[np.ndarray] = []
    all_results: list[dict] = []

    try:
        if live_display:
            cv2.startWindowThread()
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, args.cam_width, args.cam_height)
        for ep in range(args.episodes):
            print(f"\n── Episode {ep + 1}/{args.episodes} ──")
            hud = HUDState()
            t0 = time.time()
            frames, result = run_hrl_episode(
                env, eval_policy, args.code, hud,
                live_display=live_display,
                record_frames=record_video,
            )
            elapsed = time.time() - t0

            all_frames.extend(frames)
            all_results.append(result)

            status = "SUCCESS" if result["success"] else "FAILED"
            print(f"  {status}  keys={result['keys_completed']}/{result['keys_total']}  "
                  f"frames={len(frames)}  time={elapsed:.1f}s")
            for kr in result["key_results"]:
                v = "OK" if kr["verified"] else "FAIL"
                print(f"    key {kr['key_id']}: {v}  "
                      f"move_dist={kr['move_distance'] * 100:.2f}cm  "
                      f"press_dist={kr['press_distance'] * 100:.2f}cm")
    except KeyboardInterrupt:
        print("\nDisplay closed by user.")
    finally:
        if live_display:
            cv2.destroyAllWindows()

    # ── save video ───────────────────────────────────────────────────
    if record_video and all_frames:
        out_path = Path(args.video_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w = all_frames[0].shape[:2]
        writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))
        for f in all_frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"\nVideo saved: {out_path}  ({len(all_frames)} frames)")

    # ── summary ──────────────────────────────────────────────────────
    if all_results:
        successes = sum(1 for r in all_results if r["success"])
        print(f"\n── HRL Evaluation Summary ──")
        print(f"  code:              {args.code}")
        print(f"  episodes:          {args.episodes}")
        print(f"  success rate:      {successes}/{len(all_results)}")

        all_press_dists = [kr["press_distance"] * 100
                           for r in all_results for kr in r["key_results"]]
        if all_press_dists:
            print(f"  mean press dist:   {np.mean(all_press_dists):.2f} cm")
            print(f"  max  press dist:   {np.max(all_press_dists):.2f} cm")

    env.close()


if __name__ == "__main__":
    main()
