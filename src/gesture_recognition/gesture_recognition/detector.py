"""Rule-based beckon / wave detector on a window of body keypoints (Phase 1 baseline).

Idea (see README "How the rule detector works"):
  * Look at the arm only: the wrist position relative to the shoulder, divided by the person's
    body size. This cancels out the person walking, the camera shaking, and distance.
  * Split that motion into three channels:
      vertical  - wrist goes up/down relative to the shoulder (palm-up "come here" flexes the
                  forearm up toward the face; palm-down "scoop" pulls the hand down/in)
      depth     - apparent forearm (elbow->wrist) length changes: the forearm swinging toward /
                  away from the camera, which is what a beckon looks like when the person faces us
      lateral   - wrist goes side to side (a wave pivots the upright forearm at the elbow)
  * beckon = vertical/depth motion dominates, wave = lateral dominates. Both need a repeated,
    rhythmic motion (>= min_cycles back-and-forths at a hand-gesture frequency) with the hand
    raised above mid-torso, so arm swing while walking and talking-with-hands mostly drop out.

Known gaps: a beckon made only with the fingers or wrist (forearm still) is invisible to body
keypoints; that needs hand landmarks at close range. "go away" (shooing) moves on the same axis as a beckon. Separating it needs the
direction of the emphasised stroke or palm orientation; that is left to the learned model (Phase 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .keypoints import ARMS, L_HIP, L_SHOULDER, R_HIP, R_SHOULDER

BECKON = "beckon"
WAVE = "wave"
NONE = "none"


@dataclass
class DetectorConfig:
    window_s: float = 2.0  # how much history each decision looks at
    min_duration_s: float = 1.0  # need at least this much tracked arm inside the window
    resample_hz: float = 30.0
    min_kpt_conf: float = 0.3
    min_valid_fraction: float = 0.6  # fraction of window frames with shoulder+elbow+wrist visible
    min_amplitude: float = 0.25  # peak-to-peak forearm motion, in shoulder widths
    min_cycles: float = 1.5  # 1.5 = toward, away, toward, away (3 direction changes)
    min_freq_hz: float = 0.6
    max_freq_hz: float = 4.0
    dominance: float = 1.3  # winning channel amplitude / losing channel amplitude
    min_raise: float = 0.5  # wrist height: 0 = hip line, 1 = shoulder line (tune on real data)
    smooth_s: float = 0.15
    min_periodicity: float = 0.6  # R^2 of the best sine fit at a gesture-like frequency


@dataclass
class ArmResult:
    arm: str
    label: str
    confidence: float
    reason: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class GestureResult:
    label: str
    confidence: float
    arm: str | None
    arms: dict[str, ArmResult] = field(default_factory=dict)


def _body_scale(kpts: np.ndarray, conf: np.ndarray, cfg: DetectorConfig) -> float:
    """Median body size in px: shoulder width, but not less than 0.6x torso (person turned sideways)."""
    sh_ok = (conf[:, L_SHOULDER] >= cfg.min_kpt_conf) & (conf[:, R_SHOULDER] >= cfg.min_kpt_conf)
    hip_ok = (conf[:, L_HIP] >= cfg.min_kpt_conf) & (conf[:, R_HIP] >= cfg.min_kpt_conf)
    sw = np.linalg.norm(kpts[:, L_SHOULDER] - kpts[:, R_SHOULDER], axis=1)
    mid_sh = (kpts[:, L_SHOULDER] + kpts[:, R_SHOULDER]) / 2
    mid_hip = (kpts[:, L_HIP] + kpts[:, R_HIP]) / 2
    torso = np.linalg.norm(mid_sh - mid_hip, axis=1)
    cand = []
    if sh_ok.any():
        s = sw.copy()
        both = sh_ok & hip_ok
        s[both] = np.maximum(sw[both], 0.6 * torso[both])
        cand = s[sh_ok]
    elif hip_ok.any():
        cand = 0.6 * torso[hip_ok]
    return float(np.median(cand)) if len(cand) else 0.0


def _hysteresis_transitions(x: np.ndarray, t: np.ndarray, h: float) -> np.ndarray:
    """Times at which a zero-mean signal flips between the +h and -h bands."""
    state = 0
    times = []
    for xi, ti in zip(x, t):
        new = 1 if xi > h else (-1 if xi < -h else state)
        if new != state:
            if state != 0:
                times.append(ti)
            state = new
    return np.asarray(times)


def _sine_fit(x: np.ndarray, t: np.ndarray, min_freq: float, max_freq: float) -> tuple[float, float]:
    """Best single-sine fit over gesture frequencies: (R^2, freq). R^2 ~1 for a steady back-and-forth,
    low for jitter or one-off movements."""
    sst = float(np.sum((x - x.mean()) ** 2))
    if sst <= 0:
        return 0.0, 0.0
    best = (0.0, 0.0)
    for f in np.arange(min_freq, max_freq + 1e-9, 0.05):
        w = 2 * np.pi * f * t
        a = np.stack([np.sin(w), np.cos(w), np.ones_like(t)], axis=1)
        coef, *_ = np.linalg.lstsq(a, x, rcond=None)
        r2 = 1.0 - float(np.sum((x - a @ coef) ** 2)) / sst
        if r2 > best[0]:
            best = (r2, float(f))
    return best


def _channel_stats(x: np.ndarray, t: np.ndarray, cfg: DetectorConfig) -> dict:
    lo, hi = np.percentile(x, [5, 95])
    amp = float(hi - lo)
    tr = _hysteresis_transitions(x - np.median(x), t, 0.2 * amp) if amp > 0 else np.array([])
    cycles = len(tr) / 2.0
    freq = float(1.0 / (2.0 * np.mean(np.diff(tr)))) if len(tr) >= 2 else 0.0
    period, fit_freq = _sine_fit(x, t - t[0], cfg.min_freq_hz, cfg.max_freq_hz)
    return {"amp": amp, "cycles": cycles, "freq": freq, "periodicity": period, "fit_freq": fit_freq}


def _detrend_smooth(x: np.ndarray, t: np.ndarray, k: int) -> np.ndarray:
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    if k > 1:
        x = np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    return x


def analyze_arm(t: np.ndarray, kpts: np.ndarray, conf: np.ndarray, arm: str,
                cfg: DetectorConfig, scale: float | None = None) -> ArmResult:
    s_i, e_i, w_i = ARMS[arm]
    valid = conf[:, [s_i, e_i, w_i]].min(axis=1) >= cfg.min_kpt_conf
    if len(t) < 4 or valid.mean() < cfg.min_valid_fraction:
        return ArmResult(arm, NONE, 0.0, "arm not visible")
    tv = t[valid]
    if tv[-1] - tv[0] < cfg.min_duration_s:
        return ArmResult(arm, NONE, 0.0, "too short")
    if scale is None:
        scale = _body_scale(kpts, conf, cfg)
    if scale <= 1e-6:
        return ArmResult(arm, NONE, 0.0, "no body scale")

    arm_vec = (kpts[valid, w_i] - kpts[valid, s_i]) / scale  # shoulder->wrist, body units, y down
    forearm = (kpts[valid, w_i] - kpts[valid, e_i]) / scale  # elbow->wrist
    sho_y = kpts[valid, s_i, 1]
    hip_ok = conf[valid][:, [L_HIP, R_HIP]].min(axis=1) >= cfg.min_kpt_conf
    hip_y = np.where(hip_ok, (kpts[valid, L_HIP, 1] + kpts[valid, R_HIP, 1]) / 2, sho_y + 1.6 * scale)
    raise_frac = float(np.median((hip_y - kpts[valid, w_i, 1]) / np.maximum(hip_y - sho_y, 1e-6)))

    tg = np.arange(tv[0], tv[-1], 1.0 / cfg.resample_hz)
    k = max(1, int(round(cfg.smooth_s * cfg.resample_hz)))
    lateral = _detrend_smooth(np.interp(tg, tv, arm_vec[:, 0]), tg, k)
    vertical = _detrend_smooth(np.interp(tg, tv, -arm_vec[:, 1]), tg, k)
    depth = _detrend_smooth(np.interp(tg, tv, np.linalg.norm(forearm, axis=1)), tg, k)

    ch = {"lateral": _channel_stats(lateral, tg, cfg), "vertical": _channel_stats(vertical, tg, cfg),
          "depth": _channel_stats(depth, tg, cfg)}
    toward = max(("vertical", "depth"), key=lambda c: ch[c]["amp"])
    details = {"channels": ch, "raise": raise_frac, "scale_px": scale, "valid_frac": float(valid.mean()),
               "toward_channel": toward}

    if raise_frac < cfg.min_raise:
        return ArmResult(arm, NONE, 0.0, "hand not raised", details)

    a_tow, a_lat = ch[toward]["amp"], ch["lateral"]["amp"]
    if a_tow >= cfg.dominance * a_lat:
        label, main, ratio = BECKON, ch[toward], a_tow / max(a_lat, 1e-6)
    elif a_lat >= cfg.dominance * a_tow:
        label, main, ratio = WAVE, ch["lateral"], a_lat / max(a_tow, 1e-6)
    else:
        return ArmResult(arm, NONE, 0.0, "motion direction ambiguous", details)

    if main["amp"] < cfg.min_amplitude:
        return ArmResult(arm, NONE, 0.0, "motion too small", details)
    if main["cycles"] < cfg.min_cycles:
        return ArmResult(arm, NONE, 0.0, "not repeated enough", details)
    if not cfg.min_freq_hz <= main["freq"] <= cfg.max_freq_hz:
        return ArmResult(arm, NONE, 0.0, "wrong rhythm", details)
    if main["periodicity"] < cfg.min_periodicity:
        return ArmResult(arm, NONE, 0.0, "no steady rhythm (jitter?)", details)

    parts = [
        min(main["amp"] / (2 * cfg.min_amplitude), 1.0),
        min(main["cycles"] / (cfg.min_cycles + 1.0), 1.0),
        min((ratio - 1.0) / (2 * (cfg.dominance - 1.0)), 1.0),
        min(main["periodicity"] / 0.8, 1.0),
        float(valid.mean()),
    ]
    confidence = float(np.prod(parts) ** (1 / len(parts)))
    return ArmResult(arm, label, confidence, "", details)


def analyze_window(t: np.ndarray, kpts: np.ndarray, conf: np.ndarray, cfg: DetectorConfig) -> GestureResult:
    """Classify one person's keypoint window. Returns the most confident arm's label."""
    scale = _body_scale(kpts, conf, cfg)
    arms = {a: analyze_arm(t, kpts, conf, a, cfg, scale) for a in ARMS}
    best = max(arms.values(), key=lambda r: (r.label != NONE, r.confidence))
    if best.label == NONE:
        return GestureResult(NONE, 0.0, None, arms)
    return GestureResult(best.label, best.confidence, best.arm, arms)
