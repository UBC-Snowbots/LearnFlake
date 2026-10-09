"""COCO-17 body keypoint layout and the per-person pose record shared by every stage."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# COCO-17 indices (what YOLO-pose / RTMPose / most body-pose models output)
NOSE = 0
L_EYE, R_EYE, L_EAR, R_EAR = 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16
NUM_KEYPOINTS = 17

# (shoulder, elbow, wrist) per arm. "left"/"right" are the PERSON's sides, as the model labels them.
ARMS = {
    "left": (L_SHOULDER, L_ELBOW, L_WRIST),
    "right": (R_SHOULDER, R_ELBOW, R_WRIST),
}

SKELETON_EDGES = [
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP), (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
]


@dataclass
class PersonPose:
    """One detected person in one frame, in full-frame pixel coordinates (x right, y down)."""

    bbox: np.ndarray  # (4,) x1, y1, x2, y2
    keypoints: np.ndarray  # (17, 2)
    confidence: np.ndarray  # (17,) per-keypoint confidence in [0, 1]
    score: float = 1.0  # person detection score
    track_id: int = -1
    zoomed: bool = False  # keypoints came from the high-res crop pass
    extra: dict = field(default_factory=dict)

    @property
    def height_px(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> tuple[float, float]:
        return float(self.bbox[0] + self.bbox[2]) / 2.0, float(self.bbox[1] + self.bbox[3]) / 2.0
