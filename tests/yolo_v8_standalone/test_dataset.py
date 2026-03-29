from __future__ import annotations

import sys
from pathlib import Path


PKG_DIR = Path(__file__).resolve().parents[2] / "src" / "rl_autonomy" / "rl_agent_pranav" / "yolo_v8_standalone"
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from config import DEFAULT_CLASS_NAMES, YoloV8DatasetSpec
from dataset import write_dataset_yaml


def test_write_dataset_yaml_uses_expected_ultralytics_shape(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset.yaml"
    spec = YoloV8DatasetSpec(root=tmp_path / "keypad_dataset", class_names=DEFAULT_CLASS_NAMES)
    written = write_dataset_yaml(spec=spec, output_path=output_path)
    text = written.read_text(encoding="utf-8")

    assert written == output_path.resolve()
    assert f"path: {spec.normalized_root()}" in text
    assert "train: images/train" in text
    assert "val: images/val" in text
    assert "0: '0'" in text
    assert "9: '9'" in text
