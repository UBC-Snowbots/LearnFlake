"""Debug overlay: skeletons, track ids, current decision (and why), and a banner when an event fires."""
from __future__ import annotations

import cv2
import numpy as np

from .detector import BECKON, NONE, WAVE, GestureResult
from .keypoints import ARMS, SKELETON_EDGES
from .pipeline import FrameOutput

COLORS = {BECKON: (0, 200, 0), WAVE: (0, 170, 255)}
SHORT_ARM = {"left": "L", "right": "R"}


def _why(dec: GestureResult) -> str:
    """One line per person: each arm's verdict, or why it was rejected, plus hand height."""
    parts = []
    for arm in ARMS:
        r = dec.arms.get(arm)
        if r is None:
            continue
        raise_ = r.details.get("raise")
        h = f" h={raise_:.2f}" if raise_ is not None else ""
        parts.append(f"{SHORT_ARM[arm]}: {r.label if r.label != NONE else r.reason}{h}")
    return " | ".join(parts)


def _put(img, text, org, scale, color, th):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2)  # outline
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th)


def draw(frame: np.ndarray, out: FrameOutput, min_conf: float = 0.3, banner: list[str] | None = None,
         show_why: bool = True) -> np.ndarray:
    img = frame.copy()
    th = max(1, img.shape[0] // 400)
    for p in out.persons:
        dec = out.decisions.get(p.track_id)
        label = dec.label if dec else "none"
        color = COLORS.get(label, (200, 200, 200))
        x1, y1, x2, y2 = p.bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, th)
        text = f"#{p.track_id} {label}" + (f" {dec.confidence:.2f}" if dec and dec.confidence else "")
        text += " [zoom]" if p.zoomed else ""
        _put(img, text, (x1, max(12, y1 - 6)), 0.5 * th, color, th)
        if show_why and dec is not None:
            below = y2 + int(16 * th)  # under the box, or just inside its top if the person fills the frame
            org = (x1 + 4, below) if below < img.shape[0] - 4 else (x1 + 4, y1 + int(16 * th))
            _put(img, _why(dec), org, 0.4 * th, color, th)
        for a, b in SKELETON_EDGES:
            if p.confidence[a] >= min_conf and p.confidence[b] >= min_conf:
                cv2.line(img, tuple(p.keypoints[a].astype(int)), tuple(p.keypoints[b].astype(int)), color, th)
    lines = banner if banner is not None else [
        f"EVENT: {ev.label} #{ev.track_id} bearing {ev.bearing_deg:+.1f} deg" for ev in out.events]
    for i, line in enumerate(lines):
        _put(img, line, (10, int(30 * th) + int(30 * th) * i), 0.8 * th, (0, 0, 255), 2 * th)
    return img


class Overlay:
    """draw() plus a banner that stays up for `hold_s` after each event (one frame is invisible on video)."""

    def __init__(self, hold_s: float = 2.0):
        self.hold_s = hold_s
        self._recent: list[tuple[float, str]] = []

    def __call__(self, frame: np.ndarray, out: FrameOutput, t: float) -> np.ndarray:
        for ev in out.events:
            self._recent.append((t, f"EVENT: {ev.label} #{ev.track_id} bearing {ev.bearing_deg:+.1f} deg"))
        self._recent = [(ts, s) for ts, s in self._recent if t - ts <= self.hold_s]
        return draw(frame, out, banner=[s for _, s in self._recent])
