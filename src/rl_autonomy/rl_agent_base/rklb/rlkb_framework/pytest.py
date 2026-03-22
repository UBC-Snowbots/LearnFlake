from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

here = Path(__file__).resolve().parent
sys.path = [path for path in sys.path if path not in {"", str(here)}]

runpy.run_module("pytest", run_name="__main__", alter_sys=True)
