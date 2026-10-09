"""Rule detector on synthetic skeletons: every beckon style fires, everything else stays quiet."""
import numpy as np
import pytest

from gesture_recognition.detector import BECKON, NONE, WAVE, DetectorConfig, analyze_window
from gesture_recognition.synthetic import generate

CFG = DetectorConfig()
BECKONS = ["beckon_up", "beckon_depth", "beckon_palmdown"]
NEGATIVES = ["rest", "walk_swing", "static_raise", "talk", "slow_raise"]
# (px per metre, keypoint noise px): near, mid, and far. 1080p 70deg HFOV: 200 px/m ~ 7 m, 80 px/m ~ 17 m.
RANGES = [(200, 1.0), (80, 1.5)]
SEEDS = range(10)


def classify(right="rest", left="rest", **kw):
    t, k, c, _ = generate(right=right, left=left, duration=CFG.window_s, fps=15, **kw)
    return analyze_window(t, k, c, CFG)


@pytest.mark.parametrize("motion", BECKONS)
@pytest.mark.parametrize("px,noise", RANGES)
def test_beckon_styles_detected(motion, px, noise):
    hits = [classify(right=motion, px_per_m=px, noise_px=noise, seed=s).label == BECKON for s in SEEDS]
    assert sum(hits) >= 9, f"{motion} at {px}px/m: {sum(hits)}/10"


@pytest.mark.parametrize("freq", [0.9, 1.5, 2.5])
def test_beckon_across_speeds(freq):
    assert classify(right="beckon_up", freq=freq, noise_px=1.0).label == BECKON


def test_left_arm_beckon_reports_left():
    r = classify(left="beckon_up", noise_px=1.0)
    assert r.label == BECKON and r.arm == "left"


@pytest.mark.parametrize("px,noise", RANGES)
def test_wave_is_not_beckon(px, noise):
    labels = [classify(right="wave", px_per_m=px, noise_px=noise, seed=s).label for s in SEEDS]
    assert BECKON not in labels
    assert labels.count(WAVE) >= 9


@pytest.mark.parametrize("motion", NEGATIVES)
@pytest.mark.parametrize("px,noise", RANGES + [(40, 1.5)])  # also the very-far, very-noisy case
def test_negatives_stay_quiet(motion, px, noise):
    labels = [classify(right=motion, left=motion, px_per_m=px, noise_px=noise, seed=s).label for s in SEEDS]
    assert BECKON not in labels, f"{motion} at {px}px/m false beckon: {labels}"


def test_walking_person_with_swinging_arms():
    r = classify(right="walk_swing", left="walk_swing", walk_speed_px=150, noise_px=1.0)
    assert r.label == NONE


def test_beckon_while_walking():
    assert classify(right="beckon_up", left="walk_swing", walk_speed_px=100, noise_px=1.0).label == BECKON


def test_occluded_arm_is_none():
    t, k, c, _ = generate(right="beckon_up", duration=CFG.window_s)
    c[:, [6, 8, 10]] = 0.05  # right shoulder/elbow/wrist invisible
    assert analyze_window(t, k, c, CFG).arms["right"].reason == "arm not visible"


def test_keypoint_dropouts_tolerated():
    assert classify(right="beckon_up", noise_px=1.0, dropout=0.1).label == BECKON


def test_too_short_window():
    t, k, c, _ = generate(right="beckon_up", duration=0.5)
    assert analyze_window(t, k, c, CFG).label == NONE


def test_confidence_bounded():
    r = classify(right="beckon_up")
    assert 0.0 < r.confidence <= 1.0
    assert np.isfinite(r.confidence)
