from __future__ import annotations

import sys
from pathlib import Path


def _keyboard_stack_v2_path() -> Path:
    here = Path(__file__).resolve()
    return here.parents[6] / "src" / "rl_autonomy" / "rl_agent_pranav" / "keyboard_stack_v2"


def load_default_key_layout():
    try:
        from rlkb.decider.layouts import DEFAULT_KEY_LAYOUT

        return DEFAULT_KEY_LAYOUT
    except ModuleNotFoundError:
        v2_path = _keyboard_stack_v2_path()
        v2_path_str = str(v2_path)
        if v2_path.exists() and v2_path_str not in sys.path:
            sys.path.insert(0, v2_path_str)
        from layout import DEFAULT_KEY_LAYOUT

        return DEFAULT_KEY_LAYOUT


DEFAULT_KEY_LAYOUT = load_default_key_layout()
