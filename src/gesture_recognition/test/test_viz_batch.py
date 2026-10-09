"""Overlay drawing and batch-test helpers (no model needed)."""
from pathlib import Path

import numpy as np

from gesture_recognition.batch import clip_label
from gesture_recognition.pipeline import GesturePipeline
from gesture_recognition.synthetic import as_persons, generate
from gesture_recognition.viz import Overlay, _why


def test_clip_label_from_filename():
    assert clip_label(Path("beckon_alice_5m.mp4")) == "beckon"
    assert clip_label(Path("Wave-bob.MOV")) == "wave"
    assert clip_label(Path("none.mp4")) == "none"


def test_overlay_draws_and_holds_banner():
    t, k, c, b = generate(right="beckon_up", duration=6.0, fps=15, noise_px=1.0)
    pipe = GesturePipeline(estimator=None)
    overlay = Overlay(hold_s=2.0)
    frame = np.zeros((720, 1280, 3), np.uint8)
    fired_at, banner_frames, why = None, 0, ""
    for ti, p in zip(t, as_persons(k, c, b)):
        out = pipe.process_poses([p], float(ti), (1280, 720))
        img = overlay(frame, out, float(ti))
        assert img.shape == frame.shape and img.any()
        if out.events and fired_at is None:
            fired_at = ti
        if overlay._recent:
            banner_frames += 1
        if out.decisions:
            why = _why(next(iter(out.decisions.values())))
    assert fired_at is not None
    assert banner_frames >= 15  # ~2 s at 15 fps, not a single frame
    assert "R: beckon" in why and "L:" in why
