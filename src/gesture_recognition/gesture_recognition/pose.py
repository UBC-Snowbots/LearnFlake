"""Body-pose front end: YOLO-pose on the full frame, plus a "zoom" pass on far-away people.

Why the zoom pass: the model shrinks the whole frame to imgsz (640 px) before it looks at it, so a
person 120 px tall in a 1920x1080 frame is only ~40 px tall to the model and their elbows and
wrists blur out. Published long-range gesture work (Bamani et al. 2023, 96.6% at 25 m) recovers
range by classifying an enlarged crop of the person. We do the cheap version: crop the person from
the full-resolution frame and run pose again on just that crop.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .keypoints import ARMS, NUM_KEYPOINTS, PersonPose
from .tracker import iou

ARM_JOINTS = sorted({j for arm in ARMS.values() for j in arm})


class PoseEstimator(Protocol):
    def __call__(self, frame_bgr: np.ndarray) -> list[PersonPose]: ...


class YoloPoseEstimator:
    def __init__(self, model: str = "yolo11n-pose.pt", device: str | None = None, imgsz: int = 640,
                 conf: float = 0.25, zoom: bool = True, zoom_min_px: float = 160.0,
                 zoom_imgsz: int = 320, zoom_pad: float = 0.3, max_zoom: int = 4, threads: int = 4):
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(threads)  # host CPU cooler limit: keep <= 8
        self.model = YOLO(model)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.zoom = zoom
        self.zoom_min_px = zoom_min_px  # zoom anyone the model would see shorter than this
        self.zoom_imgsz = zoom_imgsz
        self.zoom_pad = zoom_pad
        self.max_zoom = max_zoom  # crops per frame, biggest-benefit first

    def _predict(self, images, imgsz):
        return self.model.predict(images, imgsz=imgsz, conf=self.conf, device=self.device, verbose=False)

    @staticmethod
    def _to_persons(result, offset=(0.0, 0.0)) -> list[PersonPose]:
        if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        kxy = result.keypoints.xy.cpu().numpy()
        kc = result.keypoints.conf
        kc = kc.cpu().numpy() if kc is not None else np.ones(kxy.shape[:2])
        ox, oy = offset
        out = []
        for b, s, k, c in zip(boxes, scores, kxy, kc):
            if k.shape[0] != NUM_KEYPOINTS:
                continue
            out.append(PersonPose(bbox=b + np.array([ox, oy, ox, oy]), keypoints=k + np.array([ox, oy]),
                                  confidence=c.astype(float), score=float(s)))
        return out

    def __call__(self, frame_bgr: np.ndarray) -> list[PersonPose]:
        h, w = frame_bgr.shape[:2]
        persons = self._to_persons(self._predict(frame_bgr, self.imgsz)[0])
        if not self.zoom or not persons:
            return persons

        shrink = self.imgsz / max(h, w)
        small = sorted((p.height_px * shrink, i) for i, p in enumerate(persons)
                       if p.height_px * shrink < self.zoom_min_px)[: self.max_zoom]
        if not small:
            return persons
        crops, offsets, idxs = [], [], []
        for _, i in small:
            x1, y1, x2, y2 = persons[i].bbox
            bw, bh = x2 - x1, y2 - y1
            side = max(bw, bh) * (1 + 2 * self.zoom_pad)  # square crop around the person
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            cx1, cy1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            cx2, cy2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                continue
            crops.append(frame_bgr[cy1:cy2, cx1:cx2])
            offsets.append((float(cx1), float(cy1)))
            idxs.append(i)
        if not crops:
            return persons
        for res, off, i in zip(self._predict(crops, self.zoom_imgsz), offsets, idxs):
            cands = self._to_persons(res, off)
            if not cands:
                continue
            best = max(cands, key=lambda p: iou(p.bbox, persons[i].bbox))
            if iou(best.bbox, persons[i].bbox) < 0.3:
                continue
            if best.confidence[ARM_JOINTS].mean() >= persons[i].confidence[ARM_JOINTS].mean():
                best.zoomed = True
                best.bbox = persons[i].bbox  # keep the full-frame box for tracking stability
                persons[i] = best
        return persons
