from __future__ import annotations

from pathlib import Path

try:
    from .config import YoloV8DatasetSpec
except ImportError:
    from config import YoloV8DatasetSpec


def write_dataset_yaml(spec: YoloV8DatasetSpec, output_path: Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"path: {spec.normalized_root()}",
        f"train: {spec.train_images}",
        f"val: {spec.val_images}",
        "names:",
    ]
    for index, name in enumerate(spec.class_names):
        lines.append(f"  {index}: '{name}'")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
