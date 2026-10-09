"""Minimal multi-person tracker: greedy IoU matching, keeps a short keypoint history per person."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .keypoints import PersonPose


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    history_s: float
    times: deque = field(default_factory=deque)
    poses: deque = field(default_factory=deque)
    last_seen: float = 0.0

    def add(self, t: float, pose: PersonPose) -> None:
        self.times.append(t)
        self.poses.append(pose)
        self.last_seen = t
        while self.times and t - self.times[0] > self.history_s:
            self.times.popleft()
            self.poses.popleft()

    @property
    def last(self) -> PersonPose:
        return self.poses[-1]

    def window(self, now: float, window_s: float):
        """(t, kpts, conf) arrays for poses within the last window_s seconds."""
        idx = [i for i, ti in enumerate(self.times) if now - ti <= window_s]
        if not idx:
            return np.zeros(0), np.zeros((0, 17, 2)), np.zeros((0, 17))
        t = np.array([self.times[i] for i in idx])
        k = np.stack([self.poses[i].keypoints for i in idx])
        c = np.stack([self.poses[i].confidence for i in idx])
        return t, k, c


class Tracker:
    def __init__(self, history_s: float = 3.0, max_missing_s: float = 0.7, min_iou: float = 0.2):
        self.history_s = history_s
        self.max_missing_s = max_missing_s
        self.min_iou = min_iou
        self.tracks: dict[int, Track] = {}
        self._next_id = 0

    def update(self, t: float, persons: list[PersonPose]) -> list[PersonPose]:
        """Assign track ids to this frame's persons (in place) and return them."""
        pairs = []
        for pi, p in enumerate(persons):
            for tid, tr in self.tracks.items():
                score = iou(p.bbox, tr.last.bbox)
                if score < self.min_iou:
                    # fallback for small/fast boxes: centre distance relative to box height
                    (px, py), (qx, qy) = p.center, tr.last.center
                    d = np.hypot(px - qx, py - qy) / max(tr.last.height_px, 1.0)
                    score = max(0.0, 0.5 - d) * self.min_iou * 2 if d < 0.5 else 0.0
                if score > 0:
                    pairs.append((score, pi, tid))
        pairs.sort(reverse=True)
        used_p, used_t = set(), set()
        for score, pi, tid in pairs:
            if pi in used_p or tid in used_t:
                continue
            used_p.add(pi)
            used_t.add(tid)
            persons[pi].track_id = tid
            self.tracks[tid].add(t, persons[pi])
        for pi, p in enumerate(persons):
            if pi not in used_p:
                tid = self._next_id
                self._next_id += 1
                p.track_id = tid
                self.tracks[tid] = Track(tid, self.history_s)
                self.tracks[tid].add(t, p)
        for tid in [k for k, tr in self.tracks.items() if t - tr.last_seen > self.max_missing_s]:
            del self.tracks[tid]
        return persons
