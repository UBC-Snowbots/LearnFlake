from __future__ import annotations

import sys
from pathlib import Path

import pytest


PKG_DIR = Path(__file__).resolve().parents[2] / "src" / "rl_autonomy" / "rl_agent_pranav" / "yolo_v8_standalone"
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from config import YoloV8TrainSpec
from model import StandaloneYoloV8, UltralyticsNotInstalledError


def test_build_empty_model_raises_clear_error_without_ultralytics() -> None:
    builder = StandaloneYoloV8(YoloV8TrainSpec(model_yaml="yolov8n.yaml"))
    with pytest.raises(UltralyticsNotInstalledError, match="pip install -U ultralytics"):
        builder.build_empty_model()


def test_train_requires_dataset_yaml() -> None:
    trainer = StandaloneYoloV8(YoloV8TrainSpec(dataset_yaml=None))
    with pytest.raises(ValueError, match="dataset YAML"):
        trainer.train()
