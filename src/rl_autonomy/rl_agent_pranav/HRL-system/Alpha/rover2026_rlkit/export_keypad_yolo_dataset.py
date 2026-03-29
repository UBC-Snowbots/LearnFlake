from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from keypad_env import KeypadReachEnv, KeypadReachEnvConfig
from keypad_vision import (
    append_metadata_row,
    draw_key_bounding_boxes,
    project_key_bounding_boxes,
    save_yolo_label_file,
    write_dataset_yaml,
    yolo_class_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a sim-generated YOLO dataset for keypad key detection")
    parser.add_argument("--output-dir", type=str, default="yolo_datasets/keypad_keys")
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--camera-name", type=str, default="birdview")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--random-steps-min", type=int, default=0)
    parser.add_argument("--random-steps-max", type=int, default=25)
    parser.add_argument("--box-pad-pixels", type=int, default=2)
    parser.add_argument("--min-box-pixels", type=int, default=4)
    parser.add_argument("--image-format", type=str, default="png", choices=["png", "jpg"])
    parser.add_argument("--write-overlays", action="store_true")
    return parser.parse_args()


def split_name(index: int, num_images: int, val_ratio: float) -> str:
    if num_images <= 1:
        return "train"
    val_every = max(1, int(round(1.0 / max(val_ratio, 1e-6))))
    return "val" if val_ratio > 0.0 and ((index + 1) % val_every == 0) else "train"


def maybe_randomize_scene(env: KeypadReachEnv, rng: np.random.Generator, min_steps: int, max_steps: int) -> None:
    obs = env.reset()
    if max_steps <= 0:
        return
    num_steps = int(rng.integers(min_steps, max_steps + 1))
    for _ in range(num_steps):
        action = rng.uniform(low=-1.0, high=1.0, size=env.action_space.shape).astype(np.float32)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    base_dir = Path(__file__).resolve().parent
    output_dir = (base_dir / args.output_dir).resolve()
    image_ext = args.image_format

    for split in ("train", "val"):
        output_dir.joinpath("images", split).mkdir(parents=True, exist_ok=True)
        output_dir.joinpath("labels", split).mkdir(parents=True, exist_ok=True)
        if args.write_overlays:
            output_dir.joinpath("overlays", split).mkdir(parents=True, exist_ok=True)

    cfg = KeypadReachEnvConfig(
        horizon=max(200, args.random_steps_max + 5),
        render=False,
        offscreen_render=True,
        camera_name=args.camera_name,
        camera_width=args.cam_width,
        camera_height=args.cam_height,
        terminate_on_success=False,
    )
    env = KeypadReachEnv(cfg)
    metadata_path = output_dir / "metadata.jsonl"
    if metadata_path.exists():
        metadata_path.unlink()

    print(f"Exporting {args.num_images} images to {output_dir}")
    print(f"Classes: {', '.join(yolo_class_names())}")

    try:
        for index in range(args.num_images):
            split = split_name(index=index, num_images=args.num_images, val_ratio=args.val_ratio)
            maybe_randomize_scene(
                env=env,
                rng=rng,
                min_steps=max(0, args.random_steps_min),
                max_steps=max(args.random_steps_min, args.random_steps_max),
            )

            frame = env.render(mode="rgb_array")
            if frame is None:
                raise RuntimeError("Offscreen render returned no frame")

            boxes = project_key_bounding_boxes(
                env=env,
                camera_name=args.camera_name,
                camera_width=args.cam_width,
                camera_height=args.cam_height,
                pad_pixels=args.box_pad_pixels,
                min_box_pixels=args.min_box_pixels,
            )
            if not boxes:
                continue

            stem = f"keypad_{index:06d}"
            image_path = output_dir / "images" / split / f"{stem}.{image_ext}"
            label_path = output_dir / "labels" / split / f"{stem}.txt"
            imageio.imwrite(image_path, frame)
            save_yolo_label_file(
                path=label_path,
                boxes=boxes,
                image_width=args.cam_width,
                image_height=args.cam_height,
            )

            if args.write_overlays:
                overlay = draw_key_bounding_boxes(frame=frame, boxes=boxes)
                overlay_path = output_dir / "overlays" / split / f"{stem}.{image_ext}"
                imageio.imwrite(overlay_path, overlay)

            append_metadata_row(
                path=metadata_path,
                payload={
                    "image": str(image_path.relative_to(output_dir)),
                    "label": str(label_path.relative_to(output_dir)),
                    "split": split,
                    "camera_name": args.camera_name,
                    "camera_width": args.cam_width,
                    "camera_height": args.cam_height,
                    "boxes": [
                        {
                            "key_id": box.key_id,
                            "class_id": box.class_id,
                            "description": box.description,
                            "x_min": box.x_min,
                            "y_min": box.y_min,
                            "x_max": box.x_max,
                            "y_max": box.y_max,
                        }
                        for box in boxes
                    ],
                },
            )

            if (index + 1) % 25 == 0 or index == 0 or (index + 1) == args.num_images:
                print(f"  [{index + 1:04d}/{args.num_images:04d}] wrote {image_path.relative_to(output_dir)}")
    finally:
        env.close()

    write_dataset_yaml(output_dir)
    print(f"Dataset ready: {output_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
