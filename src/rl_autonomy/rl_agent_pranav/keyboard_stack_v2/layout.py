from __future__ import annotations

try:
    from .task_types import KeyLayout, KeyTarget
except ImportError:
    from task_types import KeyLayout, KeyTarget


DEFAULT_KEY_LAYOUT: KeyLayout = {
    "0": KeyTarget("0", (0.11, -0.04, 0.828), "zero"),
    "1": KeyTarget("1", (0.05, -0.08, 0.828), "top-left"),
    "2": KeyTarget("2", (0.07, -0.08, 0.828), "top-middle"),
    "3": KeyTarget("3", (0.09, -0.08, 0.828), "top-right"),
    "4": KeyTarget("4", (0.05, -0.06, 0.828), "mid-left"),
    "5": KeyTarget("5", (0.07, -0.06, 0.828), "mid-middle"),
    "6": KeyTarget("6", (0.09, -0.06, 0.828), "mid-right"),
    "7": KeyTarget("7", (0.05, -0.04, 0.828), "bottom-left"),
    "8": KeyTarget("8", (0.07, -0.04, 0.828), "bottom-middle"),
    "9": KeyTarget("9", (0.09, -0.04, 0.828), "bottom-right"),
}
