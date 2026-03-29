from __future__ import annotations

import sys
from pathlib import Path


V2_DIR = Path(__file__).resolve().parents[2] / "src" / "rl_autonomy" / "rl_agent_pranav" / "keyboard_stack_v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from accuracy_report import generate_accuracy_report


def test_accuracy_report_contains_expected_summary_lines() -> None:
    report = generate_accuracy_report()
    assert "Selection controlled exact-match accuracy: 100.0% (5/5)" in report
    assert "Selection controlled precision/recall: 100.0%/100.0%" in report
    assert "Selection noisy exact-match accuracy: 60.0% (3/5)" in report
    assert "Selection noisy precision/recall: 66.7%/66.7%" in report
    assert "Engine success rate: 75.0% (3/4)" in report
    assert "Engine key completion rate: 75.0% (6/8)" in report
    assert "Engine first-pass/retry recovery: 50.0%/25.0%" in report
    assert "Engine expected-outcome accuracy: 100.0% (4/4)" in report
