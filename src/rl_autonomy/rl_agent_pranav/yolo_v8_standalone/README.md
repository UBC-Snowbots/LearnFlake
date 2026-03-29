# Standalone YOLOv8

This package is intentionally separate from `keyboard_stack_v2`.

It is a small Ultralytics-based YOLOv8 scaffold for:
- writing a dataset YAML template
- instantiating an empty YOLOv8 detection model from `yolov8n.yaml`
- training later once a dataset exists

The implementation follows the official Ultralytics usage pattern:
- install with `pip install -U ultralytics`
- create a model from YAML for a fresh model
- train with `model.train(data="path/to/dataset.yaml", ...)`

Quick usage:

```bash
python /LearnFlake/src/rl_autonomy/rl_agent_pranav/yolo_v8_standalone/cli.py init-dataset \
  --dataset-root /path/to/dataset \
  --output /path/to/dataset.yaml

python /LearnFlake/src/rl_autonomy/rl_agent_pranav/yolo_v8_standalone/cli.py build-empty

python /LearnFlake/src/rl_autonomy/rl_agent_pranav/yolo_v8_standalone/cli.py train \
  --dataset-yaml /path/to/dataset.yaml
```

This module does not integrate with Alpha, Omega, or the current perception stack yet.
