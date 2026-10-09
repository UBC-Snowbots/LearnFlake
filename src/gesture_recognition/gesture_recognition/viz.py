"""Debug overlay: skeletons, track ids, current decision, and a banner when an event fires."""
from __future__ import annotations

import cv2
import numpy as np

from .detector import BECKON, WAVE
from .keypoints import SKELETON_EDGES
from .pipeline import FrameOutput

COLORS = {BECKON: (0, 200, 0), WAVE: (0, 170, 255)}


def draw(frame: np.ndarray, out: FrameOutput, min_conf: float = 0.3) -> np.ndarray:
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
        cv2.putText(img, text, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * th, color, th)
        for a, b in SKELETON_EDGES:
            if p.confidence[a] >= min_conf and p.confidence[b] >= min_conf:
                cv2.line(img, tuple(p.keypoints[a].astype(int)), tuple(p.keypoints[b].astype(int)), color, th)
    for i, ev in enumerate(out.events):
        cv2.putText(img, f"EVENT: {ev.label} #{ev.track_id} bearing {ev.bearing_deg:+.1f} deg",
                    (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * th, (0, 0, 255), 2 * th)
    return img
