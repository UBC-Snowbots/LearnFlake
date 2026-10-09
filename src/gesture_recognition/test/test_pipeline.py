"""Tracker, debounce, bearing, and the full pipeline driven by a fake pose estimator."""
import json
import math

import numpy as np
import pytest

from gesture_recognition.events import DebounceConfig, Debouncer, bearing_deg
from gesture_recognition.keypoints import PersonPose
from gesture_recognition.pipeline import GesturePipeline
from gesture_recognition.synthetic import as_persons, generate
from gesture_recognition.tracker import Tracker

W, H = 1280, 720


class ScriptedEstimator:
    """Plays back pre-generated per-frame persons, ignoring the image."""

    def __init__(self, frames: list[list[PersonPose]]):
        self.frames = frames
        self.i = 0

    def __call__(self, frame):
        out = self.frames[self.i]
        self.i += 1
        return [PersonPose(p.bbox.copy(), p.keypoints.copy(), p.confidence.copy()) for p in out]


def scene(people: list[dict], duration: float, fps: float = 15.0):
    """people: kwargs for generate(); returns (times, per-frame person lists)."""
    tracks = []
    for i, kw in enumerate(people):
        t, k, c, b = generate(duration=duration, fps=fps, seed=i, **kw)
        tracks.append(as_persons(k, c, b))
    return t, [list(ps) for ps in zip(*tracks)]


def run(people, duration=6.0, **pipe_kw):
    t, frames = scene(people, duration)
    pipe = GesturePipeline(ScriptedEstimator(frames), **pipe_kw)
    blank = np.zeros((H, W, 3), np.uint8)
    events, outs = [], []
    for ti in t:
        out = pipe.process(blank, float(ti))
        outs.append(out)
        events += out.events
    return events, outs


def test_bearing_sign_and_scale():
    assert bearing_deg(W / 2, W, 70) == pytest.approx(0.0)
    assert bearing_deg(0, W, 70) == pytest.approx(35.0)  # image left edge = +half HFOV (REP-103 left)
    assert bearing_deg(W, W, 70) == pytest.approx(-35.0)


def test_debounce_needs_votes_and_respects_cooldown():
    d = Debouncer(DebounceConfig(votes_needed=3, votes_window=5, cooldown_s=4.0))
    fired = [d.update(0, t, "beckon", 0.9) for t in [0.0, 0.2, 0.4, 0.6, 0.8]]
    assert fired == [False, False, True, False, False]
    assert d.update(0, 4.5, "beckon", 0.9)  # cooldown over, votes still there


def test_debounce_ignores_low_confidence():
    d = Debouncer(DebounceConfig(min_confidence=0.5))
    assert not any(d.update(0, t, "beckon", 0.3) for t in np.arange(0, 2, 0.2))


def test_tracker_keeps_ids_for_two_people():
    _, frames = scene([{"center_x": 300, "noise_px": 1.0}, {"center_x": 900, "noise_px": 1.0}], 2.0)
    tr = Tracker()
    ids = [tuple(sorted((round(p.center[0], -2), p.track_id) for p in tr.update(i / 15, ps)))
           for i, ps in enumerate(frames)]
    assert len(set(ids)) == 1 and len(tr.tracks) == 2


def test_tracker_drops_missing_people():
    _, frames = scene([{}], 0.5)
    tr = Tracker(max_missing_s=0.5)
    tr.update(0.0, frames[0])
    tr.update(1.0, [])
    assert not tr.tracks


def test_pipeline_fires_one_event_for_the_beckoner():
    people = [{"right": "beckon_up", "center_x": 300, "noise_px": 1.0},
              {"right": "wave", "center_x": 950, "noise_px": 1.0},
              {"right": "talk", "left": "talk", "center_x": 640, "noise_px": 1.0}]
    events, _ = run(people, duration=5.0)
    assert len(events) == 1  # 5 s < first-fire time + 4 s cooldown
    ev = events[0]
    assert ev.label == "beckon" and ev.arm == "right"
    assert ev.bearing_deg > 0  # person on the left half of the image
    assert ev.stamp < 3.5  # latency: window (2 s) + debounce votes
    json.loads(ev.to_json())


def test_pipeline_cooldown_refires_on_continued_beckon():
    events, _ = run([{"right": "beckon_up", "noise_px": 1.0}], duration=10.0)
    assert len(events) == 3  # ~1.6 s, then once per 4 s cooldown
    assert all(b.stamp - a.stamp >= 4.0 for a, b in zip(events, events[1:]))


def test_pipeline_quiet_scene():
    people = [{"right": "walk_swing", "left": "walk_swing", "walk_speed_px": 80, "noise_px": 1.0},
              {"right": "static_raise", "center_x": 1000, "noise_px": 1.0}]
    events, outs = run(people, duration=6.0)
    assert events == []
    assert outs[-1].decisions  # it did evaluate people


def test_pipeline_far_person():
    events, _ = run([{"right": "beckon_depth", "px_per_m": 80, "noise_px": 1.5}], duration=4.0)
    assert len(events) == 1
    assert math.isfinite(events[0].bearing_deg)
