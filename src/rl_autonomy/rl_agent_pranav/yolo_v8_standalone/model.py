from __future__ import annotations

from pathlib import Path

try:
    from .config import YoloV8TrainSpec
except ImportError:
    from config import YoloV8TrainSpec


class UltralyticsNotInstalledError(RuntimeError):
    pass


def _load_yolo_class():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise UltralyticsNotInstalledError(
            "Ultralytics is not installed. Install it with `pip install -U ultralytics` "
            "to build or train the standalone YOLOv8 model."
        ) from exc
    return YOLO


class StandaloneYoloV8:
    """
    Standalone wrapper around Ultralytics YOLO for an unintegrated YOLOv8 build.

    The default path uses `yolov8n.yaml`, which creates a fresh model from
    architecture config. This keeps the model empty and ready for later
    training on a custom dataset.
    """

    def __init__(self, train_spec: YoloV8TrainSpec | None = None):
        self.train_spec = train_spec or YoloV8TrainSpec()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = self.build_empty_model()
        return self._model

    def build_empty_model(self):
        YOLO = _load_yolo_class()
        return YOLO(self.train_spec.model_yaml)

    def train(self, dataset_yaml: Path | None = None, **overrides):
        if dataset_yaml is not None:
            self.train_spec = YoloV8TrainSpec(
                dataset_yaml=Path(dataset_yaml),
                model_yaml=self.train_spec.model_yaml,
                epochs=self.train_spec.epochs,
                imgsz=self.train_spec.imgsz,
                batch=self.train_spec.batch,
                project=self.train_spec.project,
                run_name=self.train_spec.run_name,
                device=self.train_spec.device,
                workers=self.train_spec.workers,
                seed=self.train_spec.seed,
                pretrained=self.train_spec.pretrained,
                extra_overrides=dict(self.train_spec.extra_overrides),
            )

        kwargs = self.train_spec.train_kwargs()
        kwargs.update(overrides)
        if "data" not in kwargs:
            raise ValueError("Training requires a dataset YAML path.")
        return self.model.train(**kwargs)

    def validate(self, dataset_yaml: Path | None = None, **overrides):
        kwargs = {}
        if dataset_yaml is not None:
            kwargs["data"] = str(Path(dataset_yaml).expanduser().resolve())
        elif self.train_spec.dataset_yaml is not None:
            kwargs["data"] = str(Path(self.train_spec.dataset_yaml).expanduser().resolve())
        else:
            raise ValueError("Validation requires a dataset YAML path.")
        kwargs.update(overrides)
        return self.model.val(**kwargs)
