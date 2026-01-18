# rlkb framework

Minimal reinforcement learning scaffold for keyboard-style control of a rover arm. The framework keeps clear boundaries:
- Gymnasium env layer defines task semantics and rewards.
- Agent layer implements policies (with optional learning hooks).
- Runner layer orchestrates rollouts and training schedules.

Quickstart:
```bash
pip install -e .
python -m rlkb.scripts.train_mock
```

RoboSuite and ROS2 integrations are optional; imports are lazy so the mock environment and core package run without them.
