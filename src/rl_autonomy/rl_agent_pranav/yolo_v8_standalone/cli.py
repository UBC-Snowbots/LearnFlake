from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .config import DEFAULT_CLASS_NAMES, YoloV8DatasetSpec, YoloV8TrainSpec
    from .dataset import write_dataset_yaml
    from .model import StandaloneYoloV8
except ImportError:
    from config import DEFAULT_CLASS_NAMES, YoloV8DatasetSpec, YoloV8TrainSpec
    from dataset import write_dataset_yaml
    from model import StandaloneYoloV8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone YOLOv8 scaffolding for keypad detection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_dataset = subparsers.add_parser("init-dataset", help="write a dataset.yaml template")
    init_dataset.add_argument("--dataset-root", type=Path, required=True)
    init_dataset.add_argument("--output", type=Path, required=True)

    build_model = subparsers.add_parser("build-empty", help="instantiate an empty YOLOv8 model")
    build_model.add_argument("--model-yaml", type=str, default="yolov8n.yaml")

    train = subparsers.add_parser("train", help="train the YOLOv8 model on a dataset")
    train.add_argument("--dataset-yaml", type=Path, required=True)
    train.add_argument("--model-yaml", type=str, default="yolov8n.yaml")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--project", type=str, default="runs/keypad_yolov8")
    train.add_argument("--run-name", type=str, default="train")
    train.add_argument("--device", type=str, default=None)
    train.add_argument("--workers", type=int, default=8)
    train.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init-dataset":
        path = write_dataset_yaml(
            YoloV8DatasetSpec(root=args.dataset_root, class_names=DEFAULT_CLASS_NAMES),
            output_path=args.output,
        )
        print(f"Wrote dataset template to {path}")
        return

    if args.command == "build-empty":
        model = StandaloneYoloV8(YoloV8TrainSpec(model_yaml=args.model_yaml))
        yolo = model.build_empty_model()
        print(f"Built empty YOLO model from {args.model_yaml}: {type(yolo).__name__}")
        return

    if args.command == "train":
        trainer = StandaloneYoloV8(
            YoloV8TrainSpec(
                dataset_yaml=args.dataset_yaml,
                model_yaml=args.model_yaml,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                project=args.project,
                run_name=args.run_name,
                device=args.device,
                workers=args.workers,
                seed=args.seed,
            )
        )
        results = trainer.train()
        print(results)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
