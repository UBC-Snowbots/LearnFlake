from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _add_alpha_path() -> Path:
    here = Path(__file__).resolve()
    alpha_dir = here.parents[1] / "HRL-system" / "Alpha" / "rover2026_rlkit"
    if str(alpha_dir) not in sys.path:
        sys.path.insert(0, str(alpha_dir))
    return alpha_dir


ALPHA_DIR = _add_alpha_path()

from keypad_env import KeypadReachEnv, KeypadReachEnvConfig  # noqa: E402
from keypad_vision import draw_key_bounding_boxes, project_key_bounding_boxes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one sim frame with per-key bounding boxes")
    parser.add_argument("--output-path", type=str, default="debug_frames/keypad_bounding_boxes.png")
    parser.add_argument("--camera-name", type=str, default="birdview")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--random-steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    env = KeypadReachEnv(
        KeypadReachEnvConfig(
            render=False,
            offscreen_render=True,
            camera_name=args.camera_name,
            camera_width=args.cam_width,
            camera_height=args.cam_height,
            horizon=max(50, args.random_steps + 5),
            terminate_on_success=False,
        )
    )

    try:
        env.reset()
        for _ in range(max(0, args.random_steps)):
            action = rng.uniform(low=-1.0, high=1.0, size=env.action_space.shape).astype(np.float32)
            _, _, done, _ = env.step(action)
            if done:
                env.reset()

        frame = env.render(mode="rgb_array")
        if frame is None:
            raise RuntimeError("Offscreen render returned no frame")

        boxes = project_key_bounding_boxes(
            env=env,
            camera_name=args.camera_name,
            camera_width=args.cam_width,
            camera_height=args.cam_height,
        )
        overlay = draw_key_bounding_boxes(frame=frame, boxes=boxes)

        output_path = Path(args.output_path)
        if not output_path.is_absolute():
            output_path = (ALPHA_DIR / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(output_path, overlay)

        print(f"Saved overlay to {output_path}")
        print(f"Detected {len(boxes)} projected key boxes")
        for box in boxes:
            print(
                f"  key={box.key_id} bbox=({box.x_min}, {box.y_min}) -> ({box.x_max}, {box.y_max}) "
                f"desc={box.description}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
