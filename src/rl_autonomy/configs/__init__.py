"""Static configuration files (controller JSON, future YAML hparams)."""

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent
CONTROLLER_JP_PATH = str(CONFIG_DIR / "controller_jp.json")
