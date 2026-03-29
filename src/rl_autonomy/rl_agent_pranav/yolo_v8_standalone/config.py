from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_CLASS_NAMES: tuple[str, ...] = tuple(str(idx) for idx in range(10))


@dataclass(frozen=True)
class YoloV8DatasetSpec:
    """
    YOLO detection dataset description.

    The paths match Ultralytics' expected dataset YAML structure:
    - `path`: dataset root
    - `train`: train image directory relative to `path`
    - `val`: validation image directory relative to `path`
    - `names`: class-id to class-name mapping
    """

    root: Path
    train_images: str = "images/train"
    val_images: str = "images/val"
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES

    def normalized_root(self) -> Path:
        return Path(self.root).expanduser().resolve()


@dataclass(frozen=True)
class YoloV8TrainSpec:
    """
    Minimal standalone training configuration for a YOLOv8 detection model.

    `model_yaml` defaults to `yolov8n.yaml`, which Ultralytics uses to
    construct a fresh detection model from architecture config rather than
    loading pretrained weights.
    """

    dataset_yaml: Path | None = None
    model_yaml: str = "yolov8n.yaml"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    project: str = "runs/keypad_yolov8"
    run_name: str = "train"
    device: str | int | None = None
    workers: int = 8
    seed: int = 0
    pretrained: bool = False
    extra_overrides: dict[str, object] = field(default_factory=dict)

    def train_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "epochs": int(self.epochs),
            "imgsz": int(self.imgsz),
            "batch": int(self.batch),
            "project": str(self.project),
            "name": str(self.run_name),
            "workers": int(self.workers),
            "seed": int(self.seed),
            "pretrained": bool(self.pretrained),
        }
        if self.dataset_yaml is not None:
            kwargs["data"] = str(Path(self.dataset_yaml).expanduser().resolve())
        if self.device is not None:
            kwargs["device"] = self.device
        kwargs.update(self.extra_overrides)
        return kwargs
