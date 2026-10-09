"""Frame in, gesture events out. No ROS here so it runs (and is tested) anywhere."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detector import NONE, DetectorConfig, GestureResult, analyze_window
from .events import DebounceConfig, Debouncer, GestureEvent, bearing_deg
from .keypoints import PersonPose
from .pose import PoseEstimator
from .tracker import Tracker


@dataclass
class FrameOutput:
    persons: list[PersonPose]
    decisions: dict[int, GestureResult] = field(default_factory=dict)  # latest decision per track
    events: list[GestureEvent] = field(default_factory=list)  # fired on THIS frame


class GesturePipeline:
    def __init__(self, estimator: PoseEstimator, detector: DetectorConfig | None = None,
                 debounce: DebounceConfig | None = None, hfov_deg: float = 70.0, eval_hz: float = 5.0):
        self.estimator = estimator
        self.det_cfg = detector or DetectorConfig()
        self.debouncer = Debouncer(debounce or DebounceConfig())
        self.tracker = Tracker(history_s=self.det_cfg.window_s + 1.0)
        self.hfov_deg = hfov_deg
        self.eval_period = 1.0 / eval_hz
        self._last_eval = -np.inf
        self.decisions: dict[int, GestureResult] = {}

    def process(self, frame_bgr: np.ndarray, t: float) -> FrameOutput:
        persons = self.estimator(frame_bgr)
        return self.process_poses(persons, t, (frame_bgr.shape[1], frame_bgr.shape[0]))

    def process_poses(self, persons: list[PersonPose], t: float, image_size: tuple[int, int]) -> FrameOutput:
        persons = self.tracker.update(t, persons)
        live = set(self.tracker.tracks)
        self.decisions = {k: v for k, v in self.decisions.items() if k in live}
        self.debouncer.forget(live)
        out = FrameOutput(persons)
        if t - self._last_eval >= self.eval_period:
            self._last_eval = t
            for tid, tr in self.tracker.tracks.items():
                if tr.last_seen != t:
                    continue  # not in this frame; don't vote on stale data
                tw, kw, cw = tr.window(t, self.det_cfg.window_s)
                res = analyze_window(tw, kw, cw, self.det_cfg)
                self.decisions[tid] = res
                if self.debouncer.update(tid, t, res.label, res.confidence) and res.label != NONE:
                    p = tr.last
                    out.events.append(GestureEvent(
                        label=res.label, confidence=round(res.confidence, 3), track_id=tid, arm=res.arm,
                        stamp=t, bbox=[round(float(v), 1) for v in p.bbox], image_size=list(image_size),
                        bearing_deg=round(bearing_deg(p.center[0], image_size[0], self.hfov_deg), 2),
                        person_height_px=round(p.height_px, 1)))
        out.decisions = dict(self.decisions)
        return out
