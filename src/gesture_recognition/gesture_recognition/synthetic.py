"""Synthetic stick-figure keypoints for tests and for sanity-checking thresholds.

A person faces the camera; 3-D joint positions (metres; X = camera-right, Y = up, Z = toward the
camera) are projected orthographically to pixels. Arm motions are simple joint-angle trajectories.
This is NOT a substitute for real recordings; it pins down the geometry the rules rely on.
"""
from __future__ import annotations

import numpy as np

from .keypoints import (L_ANKLE, L_EAR, L_ELBOW, L_EYE, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, NOSE,
                        NUM_KEYPOINTS, R_ANKLE, R_EAR, R_ELBOW, R_EYE, R_HIP, R_KNEE, R_SHOULDER,
                        R_WRIST, PersonPose)

UPPER_ARM, FOREARM = 0.30, 0.27

# body points relative to the hip centre, for the person's LEFT side (mirrored for the right)
_BODY = {
    (L_HIP, R_HIP): (0.12, 0.0), (L_SHOULDER, R_SHOULDER): (0.20, 0.50), (L_KNEE, R_KNEE): (0.10, -0.45),
    (L_ANKLE, R_ANKLE): (0.10, -0.90), (L_EYE, R_EYE): (0.03, 0.75), (L_EAR, R_EAR): (0.07, 0.73),
}

MOTIONS = ("rest", "beckon_up", "beckon_depth", "beckon_palmdown", "wave", "walk_swing",
           "static_raise", "talk", "slow_raise")


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _sagittal(angle_from_down: float) -> np.ndarray:
    """Direction in the body's forward plane: 0 = straight down, 90 = toward the camera, 180 = up."""
    a = np.radians(angle_from_down)
    return np.array([0.0, -np.cos(a), np.sin(a)])


def arm_joints(motion: str, t: float, side: float, freq: float, rng: np.random.Generator,
               phase: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Elbow and wrist offsets from the shoulder (metres) at time t. side = +1 left, -1 right."""
    w = 2 * np.pi * freq * t + phase
    s = np.sin(w)
    if motion == "rest":
        up = _unit([side * 0.08, -1, 0.02])
        e = UPPER_ARM * up
        return e, e + FOREARM * up
    if motion == "beckon_up":  # palm-up: forearm flexes up toward the face and back
        e = UPPER_ARM * _unit(_sagittal(50) + [side * -0.1, 0, 0])
        return e, e + FOREARM * _sagittal(50 + 70 + 40 * s)
    if motion == "beckon_depth":  # upper arm forward, forearm swings between forward and up
        e = UPPER_ARM * _sagittal(80)
        return e, e + FOREARM * _sagittal(80 + 35 + 35 * s)
    if motion == "beckon_palmdown":  # forearm stays level, whole arm pulls back toward the body
        e = UPPER_ARM * _sagittal(50 + 20 * s)
        return e, e + FOREARM * _sagittal(90)
    if motion == "wave":  # upper arm out to the side, upright forearm pivots side to side
        e = UPPER_ARM * _unit([side * 0.8, -0.4, 0.2])
        psi = np.radians(25) * s
        return e, e + FOREARM * np.array([np.sin(psi), np.cos(psi), 0.0])
    if motion == "walk_swing":  # hanging arm swinging forward/back
        d = _sagittal(20 * s)
        return UPPER_ARM * d, UPPER_ARM * d + FOREARM * _sagittal(20 * s + 15)
    if motion == "static_raise":  # hand held up like "stop"
        e = UPPER_ARM * _unit([side * 0.6, -0.3, 0.3])
        return e, e + FOREARM * np.array([0.0, 1.0, 0.0])
    if motion == "talk":  # hands around the belly, small irregular movements
        e = UPPER_ARM * _sagittal(20)
        jitter = 0.04 * np.array([np.sin(w * 1.3 + 1), np.sin(w * 0.7 + 2), np.sin(w * 2.1)])
        return e, e + FOREARM * _sagittal(85) + jitter
    if motion == "slow_raise":  # one slow lift of the arm (stretching, pointing): no rhythm
        frac = min(t / 2.0, 1.0)
        e = UPPER_ARM * _sagittal(10 + 80 * frac)
        return e, e + FOREARM * _sagittal(10 + 100 * frac)
    raise ValueError(motion)


def generate(left: str = "rest", right: str = "rest", duration: float = 3.0, fps: float = 15.0,
             freq: float = 1.5, px_per_m: float = 200.0, center_x: float = 640.0, ground_y: float = 600.0,
             walk_speed_px: float = 0.0, noise_px: float = 0.0, dropout: float = 0.0, seed: int = 0,
             t0: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns t (N,), keypoints (N,17,2) px, confidence (N,17), bboxes (N,4)."""
    rng = np.random.default_rng(seed)
    n = int(round(duration * fps))
    t = t0 + np.arange(n) / fps
    kpts = np.zeros((n, NUM_KEYPOINTS, 2))
    conf = np.full((n, NUM_KEYPOINTS), 0.9)
    phase = rng.uniform(0, 2 * np.pi)
    for i, ti in enumerate(t):
        hip = np.array([center_x + walk_speed_px * (ti - t0), ground_y - 0.95 * px_per_m])
        p3 = {NOSE: np.array([0.0, 0.72, 0.05])}
        for (li, ri), (x, y) in _BODY.items():
            p3[li] = np.array([x, y, 0.0])
            p3[ri] = np.array([-x, y, 0.0])
        for side, motion, (si, ei, wi) in ((1.0, left, (L_SHOULDER, L_ELBOW, L_WRIST)),
                                           (-1.0, right, (R_SHOULDER, R_ELBOW, R_WRIST))):
            e, w = arm_joints(motion, ti - t0, side, freq, rng, phase)
            p3[ei], p3[wi] = p3[si] + e, p3[si] + w
        bob = 0.02 * np.sin(2 * np.pi * 1.8 * ti) if walk_speed_px else 0.0
        for j, p in p3.items():
            kpts[i, j] = hip + px_per_m * np.array([p[0], -(p[1] + bob)])
    if noise_px:
        kpts += rng.normal(0, noise_px, kpts.shape)
    if dropout:
        conf[rng.random(conf.shape) < dropout] = 0.05
    lo, hi = kpts.min(axis=1), kpts.max(axis=1)
    pad = 0.08 * px_per_m
    bboxes = np.concatenate([lo - pad, hi + pad], axis=1)
    return t, kpts, conf, bboxes


def as_persons(kpts: np.ndarray, conf: np.ndarray, bboxes: np.ndarray) -> list[PersonPose]:
    return [PersonPose(bbox=b, keypoints=k, confidence=c) for k, c, b in zip(kpts, conf, bboxes)]
