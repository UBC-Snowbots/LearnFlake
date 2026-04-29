#!/usr/bin/env python3
"""
YOLOv8 keyboard key detector.

Detects individual keyboard keys with bounding boxes using a YOLOv8 model,
replacing the ArUco marker system. Designed to work with the EEF-mounted
RealSense camera on the real robot and with rendered camera images in sim.

Modes:
  - Live inference from RealSense (--source realsense)
  - Inference on a single image (--source path/to/image.png)
  - Inference on MuJoCo sim camera feed (--source sim)
  - Training on labeled key dataset (--train)

The detector outputs per-key bounding boxes + class labels matching the
87 keys defined in keyboard_env.TKL_KEYS, plus a get_key_position() helper
that returns the pixel centroid and estimated 3D offset for a named key.

Usage:
    # Run inference on an image
    python detect_keys_yolo.py --source path/to/image.png

    # Run live from RealSense
    python detect_keys_yolo.py --source realsense

    # Run on sim camera (requires running KeyboardEnv)
    python detect_keys_yolo.py --source sim

    # Train on labeled dataset
    python detect_keys_yolo.py --train --data data/key_detection/dataset.yaml

    # Export model for deployment
    python detect_keys_yolo.py --export checkpoints/key_detection/best.pt
"""

import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Optional

ROOT = os.path.dirname(os.path.abspath(__file__))

# Key names from keyboard_env — the 87 TKL key classes
KEY_CLASSES = [
    # F-row
    "esc", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10",
    "f11", "f12", "prtsc", "scrlk", "pause",
    # Number row
    "grave", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "minus", "equal", "backspace", "ins", "home", "pgup",
    # QWERTY row
    "tab", "q", "w", "e", "r", "t", "y", "u", "i", "o", "p",
    "lbracket", "rbracket", "backslash", "del", "end", "pgdn",
    # Home row
    "caps", "a", "s", "d", "f", "g", "h", "j", "k", "l",
    "semicolon", "quote", "enter",
    # Shift row
    "lshift", "z", "x", "c", "v", "b", "n", "m", "comma", "period",
    "slash", "rshift", "up",
    # Space row
    "lctrl", "win", "lalt", "space", "ralt", "fn", "menu", "rctrl",
    "left", "down", "right",
]

CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints", "key_detection")
DATA_DIR       = os.path.join(ROOT, "data", "key_detection")


# ============================================================================
# Detection result
# ============================================================================

class KeyDetection:
    """Single detected key with bounding box and confidence."""

    def __init__(self, key_name: str, bbox_xyxy: np.ndarray,
                 confidence: float, class_id: int):
        self.key_name = key_name
        self.bbox_xyxy = bbox_xyxy       # [x1, y1, x2, y2] in pixels
        self.confidence = confidence
        self.class_id = class_id

    @property
    def center_px(self) -> np.ndarray:
        """Bounding box center in pixels [cx, cy]."""
        return np.array([
            (self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2,
            (self.bbox_xyxy[1] + self.bbox_xyxy[3]) / 2,
        ])

    @property
    def width_px(self) -> float:
        return float(self.bbox_xyxy[2] - self.bbox_xyxy[0])

    @property
    def height_px(self) -> float:
        return float(self.bbox_xyxy[3] - self.bbox_xyxy[1])

    def __repr__(self):
        cx, cy = self.center_px
        return (f"KeyDetection({self.key_name}, conf={self.confidence:.2f}, "
                f"center=({cx:.0f},{cy:.0f}), "
                f"size={self.width_px:.0f}x{self.height_px:.0f})")


# ============================================================================
# Detector
# ============================================================================

class KeyboardKeyDetector:
    """
    YOLOv8-based keyboard key detector.

    Wraps ultralytics YOLO for inference and training on the 87-key TKL layout.
    """

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.5,
                 device: str = "auto"):
        """
        Args:
            model_path: Path to trained .pt weights. If None, uses yolov8n.pt
                        (pretrained COCO — fine-tune with --train).
            conf_threshold: Minimum confidence for detections.
            device: "auto", "cpu", "cuda", or "cuda:0" etc.
        """
        from ultralytics import YOLO

        if model_path and os.path.exists(model_path):
            print(f"Loading trained model: {model_path}")
            self.model = YOLO(model_path)
        else:
            print("Loading base YOLOv8n — fine-tune with --train for key detection")
            self.model = YOLO("yolov8n.pt")

        self.conf_threshold = conf_threshold
        self.device = device
        self.key_classes = KEY_CLASSES

    def detect(self, image: np.ndarray, verbose: bool = False) -> list[KeyDetection]:
        """
        Run inference on a single image.

        Args:
            image: BGR or RGB numpy array (H, W, 3).
            verbose: Print YOLO inference logs.

        Returns:
            List of KeyDetection objects sorted by confidence (descending).
        """
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=verbose,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()

                if class_id < len(self.key_classes):
                    key_name = self.key_classes[class_id]
                else:
                    key_name = f"unknown_{class_id}"

                detections.append(KeyDetection(
                    key_name=key_name,
                    bbox_xyxy=xyxy,
                    confidence=conf,
                    class_id=class_id,
                ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def get_key_position(self, detections: list[KeyDetection],
                         target_key: str) -> Optional[KeyDetection]:
        """Find a specific key in the detection results."""
        for det in detections:
            if det.key_name == target_key:
                return det
        return None

    def pixel_to_camera_offset(self, detection: KeyDetection,
                               image_shape: tuple,
                               camera_fov_h: float = 69.4,
                               camera_fov_v: float = 42.5,
                               estimated_distance: float = 0.05
                               ) -> np.ndarray:
        """
        Convert pixel detection to approximate 3D offset in camera frame.

        Uses pinhole camera model with known FOV (RealSense D435 defaults).
        Returns [dx, dy] in metres — same format as the ArUco pipeline's
        aruco_obs[:2], so it can be a drop-in replacement.

        Args:
            detection: Detected key.
            image_shape: (H, W, C) of the source image.
            camera_fov_h: Horizontal field of view in degrees.
            camera_fov_v: Vertical field of view in degrees.
            estimated_distance: Estimated distance to keyboard surface (m).

        Returns:
            np.ndarray [dx, dy] offset in camera frame (metres).
        """
        h, w = image_shape[:2]
        cx, cy = detection.center_px

        # Normalize to [-1, 1] from image center
        nx = (cx - w / 2) / (w / 2)
        ny = (cy - h / 2) / (h / 2)

        # Convert to angular offset then to metric offset at estimated distance
        half_fov_h = np.radians(camera_fov_h / 2)
        half_fov_v = np.radians(camera_fov_v / 2)

        dx = estimated_distance * np.tan(nx * half_fov_h)
        dy = estimated_distance * np.tan(ny * half_fov_v)

        return np.array([dx, dy])

    def train(self, data_yaml: str, epochs: int = 100, imgsz: int = 640,
              batch: int = 16, name: str = "key_detection"):
        """
        Fine-tune the model on a labeled key detection dataset.

        Args:
            data_yaml: Path to YOLO-format dataset.yaml with train/val splits.
            epochs: Number of training epochs.
            imgsz: Input image size.
            batch: Batch size.
            name: Run name for logging.
        """
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=name,
            project=CHECKPOINT_DIR,
            exist_ok=True,
            device=self.device,
        )
        print(f"\nTraining complete. Best model: {CHECKPOINT_DIR}/{name}/weights/best.pt")
        return results

    def export(self, model_path: str, format: str = "onnx"):
        """Export model for deployment (ONNX, TensorRT, etc.)."""
        from ultralytics import YOLO
        model = YOLO(model_path)
        model.export(format=format)
        print(f"Exported {model_path} to {format}")


# ============================================================================
# Visualization
# ============================================================================

def draw_detections(image: np.ndarray, detections: list[KeyDetection],
                    target_key: Optional[str] = None) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image.

    Args:
        image: BGR numpy array.
        detections: List of KeyDetection objects.
        target_key: If set, highlight this key in green, others in blue.

    Returns:
        Annotated image copy.
    """
    import cv2

    annotated = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy.astype(int)

        if target_key and det.key_name == target_key:
            color = (0, 255, 0)   # green for target
            thickness = 3
        else:
            color = (255, 128, 0)  # blue for others
            thickness = 2

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = f"{det.key_name} {det.confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - label_size[1] - 6),
                      (x1 + label_size[0], y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated


# ============================================================================
# Source readers
# ============================================================================

def read_realsense():
    """Yield frames from a RealSense D435 camera."""
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            yield np.asanyarray(color_frame.get_data())
    finally:
        pipeline.stop()


def read_sim_camera():
    """Yield frames from the MuJoCo sim eef_cam."""
    ROBO_PATH = os.path.join(ROOT, "..", "external_pkgs", "RoboSuite")
    if os.path.exists(ROBO_PATH) and ROBO_PATH not in sys.path:
        sys.path.insert(0, ROBO_PATH)

    from keyboard_env import KeyboardEnv

    env = KeyboardEnv(render=False, use_camera_obs=True)
    env.reset()

    try:
        while True:
            obs = env._get_observations(force_update=True)
            # RoboSuite camera obs key format: {camera_name}_image
            for key in obs:
                if "image" in key and isinstance(obs[key], np.ndarray):
                    frame = obs[key]
                    if frame.ndim == 3:
                        yield frame
                        break
            # Step with zero action to advance sim
            action = np.zeros(env.action_dim)
            env.step(action)
    finally:
        env.close()


def read_image(path: str):
    """Yield a single image from disk."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    yield img


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 keyboard key detector"
    )
    parser.add_argument('--source', type=str, default=None,
                        help='Input source: "realsense", "sim", or path to image/video')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained .pt model weights')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--target-key', type=str, default=None,
                        help='Highlight a specific key in the output')
    parser.add_argument('--no-display', action='store_true',
                        help='Skip OpenCV display (print detections to stdout)')

    parser.add_argument('--train', action='store_true',
                        help='Train/fine-tune the model')
    parser.add_argument('--data', type=str,
                        default=os.path.join(DATA_DIR, "dataset.yaml"),
                        help='Path to dataset.yaml for training')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)

    parser.add_argument('--export', type=str, default=None,
                        metavar='PATH', help='Export model to ONNX')

    args = parser.parse_args()

    # --- Export mode ---
    if args.export:
        detector = KeyboardKeyDetector()
        detector.export(args.export)
        return

    # --- Train mode ---
    if args.train:
        detector = KeyboardKeyDetector(model_path=args.model, device="auto")
        detector.train(
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
        )
        return

    # --- Inference mode ---
    if args.source is None:
        parser.error("--source is required for inference (use 'realsense', 'sim', or a file path)")

    detector = KeyboardKeyDetector(
        model_path=args.model,
        conf_threshold=args.conf,
    )

    # Select frame source
    if args.source == "realsense":
        frames = read_realsense()
    elif args.source == "sim":
        frames = read_sim_camera()
    else:
        frames = read_image(args.source)

    import cv2

    for frame in frames:
        detections = detector.detect(frame)

        # Print detections
        print(f"\n--- {len(detections)} keys detected ---")
        for det in detections:
            print(f"  {det}")

        if args.target_key:
            target = detector.get_key_position(detections, args.target_key)
            if target:
                offset = detector.pixel_to_camera_offset(target, frame.shape)
                print(f"  Target '{args.target_key}': offset dx={offset[0]*100:.2f}cm "
                      f"dy={offset[1]*100:.2f}cm")
            else:
                print(f"  Target '{args.target_key}' not found in frame")

        if not args.no_display:
            annotated = draw_detections(frame, detections, args.target_key)
            cv2.imshow("Key Detection", annotated)
            key = cv2.waitKey(1 if args.source in ("realsense", "sim") else 0)
            if key == ord('q') or key == 27:
                break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
