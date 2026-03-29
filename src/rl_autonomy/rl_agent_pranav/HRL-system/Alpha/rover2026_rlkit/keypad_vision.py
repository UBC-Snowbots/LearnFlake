from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from alpha_env_utils import ensure_alpha_import_paths

ensure_alpha_import_paths()

import robosuite.utils.camera_utils as camera_utils
import robosuite.utils.transform_utils as transform_utils

from keypad_lift_env import KEY_HALF_SIZE
from layout_compat import DEFAULT_KEY_LAYOUT


@dataclass(frozen=True)
class KeyBoundingBox:
    key_id: str
    class_id: int
    description: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float = 1.0

    @property
    def width(self) -> int:
        return int(self.x_max - self.x_min)

    @property
    def height(self) -> int:
        return int(self.y_max - self.y_min)

    @property
    def center_xy(self) -> tuple[float, float]:
        return (
            float(self.x_min + self.width / 2.0),
            float(self.y_min + self.height / 2.0),
        )

    def to_yolo_row(self, image_width: int, image_height: int) -> str:
        center_x, center_y = self.center_xy
        width = float(self.width) / float(image_width)
        height = float(self.height) / float(image_height)
        return (
            f"{self.class_id} "
            f"{center_x / float(image_width):.6f} "
            f"{center_y / float(image_height):.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )


def sorted_key_ids() -> list[str]:
    return sorted(DEFAULT_KEY_LAYOUT.keys(), key=int)


def yolo_class_names() -> list[str]:
    return sorted_key_ids()


def key_color_bgr(key_id: str) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(key_id))
    color = rng.integers(low=48, high=255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def key_box_corners_world(center_xyz: np.ndarray) -> np.ndarray:
    center_xyz = np.asarray(center_xyz, dtype=np.float32).reshape(3)
    half_size = np.asarray(KEY_HALF_SIZE, dtype=np.float32).reshape(3)
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append(center_xyz + np.array([sx, sy, sz], dtype=np.float32) * half_size)
    return np.asarray(corners, dtype=np.float32)


def _env_camera_params(env, camera_name: str | None, camera_width: int | None, camera_height: int | None) -> tuple[str, int, int]:
    name = camera_name or env.config.camera_name
    width = int(camera_width or env.config.camera_width)
    height = int(camera_height or env.config.camera_height)
    return name, width, height


def _world_to_camera_pose(sim, camera_name: str) -> np.ndarray:
    camera_pose_world = camera_utils.get_camera_extrinsic_matrix(sim=sim, camera_name=camera_name)
    return transform_utils.pose_inv(camera_pose_world)


def _camera_space_points(sim, camera_name: str, world_points: np.ndarray) -> np.ndarray:
    world_to_camera_pose = _world_to_camera_pose(sim=sim, camera_name=camera_name)
    world_points = np.asarray(world_points, dtype=np.float32).reshape(-1, 3)
    homogeneous = np.concatenate(
        [world_points, np.ones((world_points.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    camera_points = (world_to_camera_pose @ homogeneous.T).T
    return camera_points[:, :3]


def project_key_bounding_boxes(
    env,
    camera_name: str | None = None,
    camera_width: int | None = None,
    camera_height: int | None = None,
    pad_pixels: int = 2,
    min_box_pixels: int = 4,
) -> list[KeyBoundingBox]:
    camera_name, camera_width, camera_height = _env_camera_params(
        env=env,
        camera_name=camera_name,
        camera_width=camera_width,
        camera_height=camera_height,
    )
    sim = env.env.sim
    world_to_camera = camera_utils.get_camera_transform_matrix(
        sim=sim,
        camera_name=camera_name,
        camera_height=camera_height,
        camera_width=camera_width,
    )

    boxes: list[KeyBoundingBox] = []
    for key_id in sorted_key_ids():
        center_xyz = np.asarray(env.env.get_key_position(key_id), dtype=np.float32)
        corners_world = key_box_corners_world(center_xyz)
        corners_camera = _camera_space_points(sim=sim, camera_name=camera_name, world_points=corners_world)
        if not np.any(corners_camera[:, 2] > 0.0):
            continue

        pixels_rc = camera_utils.project_points_from_world_to_camera(
            points=corners_world,
            world_to_camera_transform=world_to_camera,
            camera_height=camera_height,
            camera_width=camera_width,
        )
        rows = pixels_rc[:, 0]
        cols = pixels_rc[:, 1]

        x_min = max(0, int(cols.min()) - pad_pixels)
        y_min = max(0, int(rows.min()) - pad_pixels)
        x_max = min(camera_width - 1, int(cols.max()) + pad_pixels)
        y_max = min(camera_height - 1, int(rows.max()) + pad_pixels)

        if (x_max - x_min) < min_box_pixels or (y_max - y_min) < min_box_pixels:
            continue

        target = DEFAULT_KEY_LAYOUT[key_id]
        boxes.append(
            KeyBoundingBox(
                key_id=key_id,
                class_id=int(key_id),
                description=target.description,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            )
        )

    return boxes


def draw_key_bounding_boxes(frame: np.ndarray, boxes: Sequence[KeyBoundingBox], show_labels: bool = True) -> np.ndarray:
    import cv2

    annotated = np.ascontiguousarray(frame.copy())
    for box in boxes:
        color = key_color_bgr(box.key_id)
        cv2.rectangle(annotated, (box.x_min, box.y_min), (box.x_max, box.y_max), color, 2)
        if show_labels:
            label = f"{box.key_id}:{box.description}"
            text_y = max(14, box.y_min - 6)
            cv2.putText(
                annotated,
                label,
                (box.x_min, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    return annotated


def save_yolo_label_file(
    path: Path,
    boxes: Sequence[KeyBoundingBox],
    image_width: int,
    image_height: int,
) -> None:
    rows = [box.to_yolo_row(image_width=image_width, image_height=image_height) for box in boxes]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def write_dataset_yaml(path: Path) -> None:
    names = yolo_class_names()
    lines = [
        f"path: {path.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx, name in enumerate(names):
        lines.append(f"  {idx}: '{name}'")
    path.joinpath("dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_metadata_row(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


class SimKeyDetectionBridge:
    """
    Sim-only bridge between image-space detections and Alpha targets.

    For the first iteration, detections gate *which* key is considered visible,
    but the returned 3D position is still the simulator's exact key pose.
    This keeps perception and control decoupled while the detector is being trained.
    """

    def __init__(self, env, min_confidence: float = 0.25):
        self.env = env
        self.min_confidence = float(min_confidence)

    def select_detection(self, target_key_id: str, detections: Iterable[KeyBoundingBox]) -> KeyBoundingBox | None:
        target_key_id = str(target_key_id)
        candidates = [
            det
            for det in detections
            if str(det.key_id) == target_key_id and float(det.confidence) >= self.min_confidence
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda det: (float(det.confidence), det.width * det.height))

    def resolve_target_xyz(self, target_key_id: str, detections: Iterable[KeyBoundingBox]) -> np.ndarray | None:
        if self.select_detection(target_key_id=target_key_id, detections=detections) is None:
            return None
        return np.asarray(self.env.env.get_key_position(str(target_key_id)), dtype=np.float32)
