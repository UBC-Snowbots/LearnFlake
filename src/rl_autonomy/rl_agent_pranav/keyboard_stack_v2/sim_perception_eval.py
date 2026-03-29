from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .perception import KeyDetection, KeyDetector, PixelBBox, TargetKeySelector
except ImportError:
    from perception import KeyDetection, KeyDetector, PixelBBox, TargetKeySelector


def _alpha_dir() -> Path:
    here = Path(__file__).resolve()
    return here.parents[1] / "HRL-system" / "Alpha" / "rover2026_rlkit"


def ensure_alpha_path() -> Path:
    alpha_dir = _alpha_dir()
    alpha_dir_str = str(alpha_dir)
    if alpha_dir.exists() and alpha_dir_str not in sys.path:
        sys.path.insert(0, alpha_dir_str)
    return alpha_dir


def _lazy_alpha_imports():
    ensure_alpha_path()
    from keypad_env import KeypadReachEnv, KeypadReachEnvConfig
    from keypad_vision import project_key_bounding_boxes

    return KeypadReachEnv, KeypadReachEnvConfig, project_key_bounding_boxes


@dataclass(frozen=True)
class SimEvalConfig:
    num_frames: int = 10
    seed: int = 7
    camera_name: str = "birdview"
    camera_width: int = 640
    camera_height: int = 480
    random_steps_min: int = 0
    random_steps_max: int = 10
    horizon: int = 200
    box_pad_pixels: int = 2
    min_box_pixels: int = 4
    iou_threshold: float = 0.5
    min_confidence: float = 0.25


@dataclass(frozen=True)
class DetectionMetrics:
    total_frames: int
    total_ground_truth_boxes: int
    total_predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_matched_iou: float
    exact_frame_match_rate: float
    visible_target_selection_accuracy: float
    evaluated_target_selections: int
    correct_target_selections: int


@dataclass(frozen=True)
class _FrameMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_ious: tuple[float, ...]
    exact_match: bool


def key_detection_from_box(box) -> KeyDetection:
    return KeyDetection(
        key_id=str(box.key_id),
        confidence=float(box.confidence),
        bbox=PixelBBox(
            x_min=int(box.x_min),
            y_min=int(box.y_min),
            x_max=int(box.x_max),
            y_max=int(box.y_max),
        ),
        source="sim_oracle",
    )


def bbox_iou(lhs: PixelBBox, rhs: PixelBBox) -> float:
    x_left = max(lhs.x_min, rhs.x_min)
    y_top = max(lhs.y_min, rhs.y_min)
    x_right = min(lhs.x_max, rhs.x_max)
    y_bottom = min(lhs.y_max, rhs.y_max)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = float((x_right - x_left) * (y_bottom - y_top))
    union = float(lhs.area + rhs.area) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def match_detections(
    ground_truth: Iterable[KeyDetection],
    predictions: Iterable[KeyDetection],
    iou_threshold: float,
) -> _FrameMetrics:
    gt_list = list(ground_truth)
    pred_list = sorted(predictions, key=lambda item: float(item.confidence), reverse=True)
    matched_gt_indices: set[int] = set()
    matched_ious: list[float] = []
    true_positives = 0
    false_positives = 0

    for prediction in pred_list:
        best_gt_index = None
        best_iou = 0.0
        for gt_index, truth in enumerate(gt_list):
            if gt_index in matched_gt_indices or truth.key_id != prediction.key_id:
                continue
            iou = bbox_iou(prediction.bbox, truth.bbox)
            if iou >= iou_threshold and iou > best_iou:
                best_gt_index = gt_index
                best_iou = iou

        if best_gt_index is None:
            false_positives += 1
            continue

        matched_gt_indices.add(best_gt_index)
        matched_ious.append(best_iou)
        true_positives += 1

    false_negatives = len(gt_list) - len(matched_gt_indices)
    exact_match = false_positives == 0 and false_negatives == 0 and true_positives == len(gt_list)
    return _FrameMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        matched_ious=tuple(matched_ious),
        exact_match=exact_match,
    )


class OracleProjectionDetector(KeyDetector):
    def __init__(
        self,
        env,
        camera_name: str,
        camera_width: int,
        camera_height: int,
        pad_pixels: int = 2,
        min_box_pixels: int = 4,
    ):
        _, _, project_key_bounding_boxes = _lazy_alpha_imports()
        self._project_key_bounding_boxes = project_key_bounding_boxes
        self._env = env
        self._camera_name = camera_name
        self._camera_width = int(camera_width)
        self._camera_height = int(camera_height)
        self._pad_pixels = int(pad_pixels)
        self._min_box_pixels = int(min_box_pixels)

    def detect(self, frame) -> list[KeyDetection]:
        boxes = self._project_key_bounding_boxes(
            env=self._env,
            camera_name=self._camera_name,
            camera_width=self._camera_width,
            camera_height=self._camera_height,
            pad_pixels=self._pad_pixels,
            min_box_pixels=self._min_box_pixels,
        )
        return [key_detection_from_box(box) for box in boxes]


def maybe_randomize_scene(env, rng: np.random.Generator, min_steps: int, max_steps: int) -> None:
    env.reset()
    if max_steps <= 0:
        return

    num_steps = int(rng.integers(min_steps, max_steps + 1))
    for _ in range(num_steps):
        action = rng.uniform(low=-1.0, high=1.0, size=env.action_space.shape).astype(np.float32)
        _, _, done, _ = env.step(action)
        if done:
            env.reset()


def build_env(config: SimEvalConfig):
    KeypadReachEnv, KeypadReachEnvConfig, _ = _lazy_alpha_imports()
    env_config = KeypadReachEnvConfig(
        horizon=max(int(config.horizon), int(config.random_steps_max) + 5),
        render=False,
        offscreen_render=True,
        camera_name=config.camera_name,
        camera_width=config.camera_width,
        camera_height=config.camera_height,
        terminate_on_success=False,
    )
    return KeypadReachEnv(env_config)


def collect_oracle_ground_truth(env, config: SimEvalConfig) -> list[KeyDetection]:
    _, _, project_key_bounding_boxes = _lazy_alpha_imports()
    boxes = project_key_bounding_boxes(
        env=env,
        camera_name=config.camera_name,
        camera_width=config.camera_width,
        camera_height=config.camera_height,
        pad_pixels=config.box_pad_pixels,
        min_box_pixels=config.min_box_pixels,
    )
    return [key_detection_from_box(box) for box in boxes]


def evaluate_detector_on_sim(
    detector: KeyDetector,
    env,
    config: SimEvalConfig,
) -> DetectionMetrics:
    rng = np.random.default_rng(config.seed)
    selector = TargetKeySelector(min_confidence=config.min_confidence)
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    exact_frame_matches = 0
    matched_ious: list[float] = []
    total_ground_truth_boxes = 0
    total_predictions = 0
    evaluated_target_selections = 0
    correct_target_selections = 0

    for _ in range(config.num_frames):
        maybe_randomize_scene(
            env=env,
            rng=rng,
            min_steps=max(0, int(config.random_steps_min)),
            max_steps=max(int(config.random_steps_min), int(config.random_steps_max)),
        )
        frame = env.render(mode="rgb_array")
        if frame is None:
            raise RuntimeError("Offscreen render returned no frame")

        ground_truth = collect_oracle_ground_truth(env=env, config=config)
        predictions = list(detector.detect(frame))

        frame_metrics = match_detections(
            ground_truth=ground_truth,
            predictions=predictions,
            iou_threshold=config.iou_threshold,
        )
        true_positives += frame_metrics.true_positives
        false_positives += frame_metrics.false_positives
        false_negatives += frame_metrics.false_negatives
        exact_frame_matches += int(frame_metrics.exact_match)
        matched_ious.extend(frame_metrics.matched_ious)
        total_ground_truth_boxes += len(ground_truth)
        total_predictions += len(predictions)

        for truth in ground_truth:
            evaluated_target_selections += 1
            selection = selector.select(requested_key_id=truth.key_id, detections=predictions)
            if selection.detection is None:
                continue
            if selection.detection.key_id != truth.key_id:
                continue
            if bbox_iou(selection.detection.bbox, truth.bbox) >= config.iou_threshold:
                correct_target_selections += 1

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = float(true_positives / precision_denominator) if precision_denominator else 0.0
    recall = float(true_positives / recall_denominator) if recall_denominator else 0.0
    f1_denominator = precision + recall
    f1 = float((2.0 * precision * recall) / f1_denominator) if f1_denominator else 0.0

    return DetectionMetrics(
        total_frames=config.num_frames,
        total_ground_truth_boxes=total_ground_truth_boxes,
        total_predictions=total_predictions,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_matched_iou=(float(np.mean(matched_ious)) if matched_ious else 0.0),
        exact_frame_match_rate=float(exact_frame_matches / config.num_frames) if config.num_frames else 0.0,
        visible_target_selection_accuracy=(
            float(correct_target_selections / evaluated_target_selections) if evaluated_target_selections else 0.0
        ),
        evaluated_target_selections=evaluated_target_selections,
        correct_target_selections=correct_target_selections,
    )


def format_detection_report(metrics: DetectionMetrics, detector_name: str) -> str:
    return "\n".join(
        [
            f"Detector: {detector_name}",
            f"Frames: {metrics.total_frames}",
            f"Ground-truth boxes: {metrics.total_ground_truth_boxes}",
            f"Predicted boxes: {metrics.total_predictions}",
            f"Precision/recall/F1: {metrics.precision:.1%}/{metrics.recall:.1%}/{metrics.f1:.1%}",
            f"False positives / false negatives: {metrics.false_positives} / {metrics.false_negatives}",
            f"Mean matched IoU: {metrics.mean_matched_iou:.3f}",
            f"Exact frame match rate: {metrics.exact_frame_match_rate:.1%}",
            f"Visible target selection accuracy: {metrics.visible_target_selection_accuracy:.1%} "
            f"({metrics.correct_target_selections}/{metrics.evaluated_target_selections})",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a key detector on live sim-rendered keypad frames")
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--camera-name", type=str, default="birdview")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--random-steps-min", type=int, default=0)
    parser.add_argument("--random-steps-max", type=int, default=10)
    parser.add_argument("--box-pad-pixels", type=int, default=2)
    parser.add_argument("--min-box-pixels", type=int, default=4)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimEvalConfig(
        num_frames=args.num_frames,
        seed=args.seed,
        camera_name=args.camera_name,
        camera_width=args.cam_width,
        camera_height=args.cam_height,
        random_steps_min=args.random_steps_min,
        random_steps_max=args.random_steps_max,
        box_pad_pixels=args.box_pad_pixels,
        min_box_pixels=args.min_box_pixels,
        iou_threshold=args.iou_threshold,
        min_confidence=args.min_confidence,
    )

    env = build_env(config)
    try:
        detector = OracleProjectionDetector(
            env=env,
            camera_name=config.camera_name,
            camera_width=config.camera_width,
            camera_height=config.camera_height,
            pad_pixels=config.box_pad_pixels,
            min_box_pixels=config.min_box_pixels,
        )
        metrics = evaluate_detector_on_sim(detector=detector, env=env, config=config)
    finally:
        env.close()

    print(format_detection_report(metrics, detector_name="oracle_projection"))


if __name__ == "__main__":
    main()
