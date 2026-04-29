"""Domain randomization wrapper — proper gymnasium.Wrapper this time.

Implements the per-episode subset of TRACKER §10's 18 axes. Per-step
sensor-noise axes (joint encoder noise, aruco position noise, synthetic
torque noise) are handled inside ``KeyboardEnv._build_obs_dict`` and
``_aruco_synth`` already; they don't need a wrapper.

Disabled by default in v1 (``make_env(..., domain_rand=False)``). Enabled
in Phase 4 once core RL training is verified.

The wrapper attaches active DR sample values to ``info['dr']`` per step so
the asymmetric critic can be conditioned on them (TRACKER §9.2 #4).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

import gymnasium as gym


@dataclass
class DRSample:
    """Per-episode DR sample.

    Each field is the *multiplier* or *delta* applied this episode. Default
    values represent "no randomization" so a sample of all defaults reproduces
    the nominal sim.
    """
    keyboard_dx: float = 0.0       # m
    keyboard_dy: float = 0.0       # m
    keyboard_dz: float = 0.0       # m
    keyboard_yaw: float = 0.0      # rad
    base_dx: float = 0.0           # m
    base_dy: float = 0.0           # m
    base_dz: float = 0.0           # m
    joint_friction_mul: float = 1.0
    joint_damping_mul: float = 1.0
    link_mass_mul: float = 1.0
    controller_kp_mul: float = 1.0
    controller_damping_mul: float = 1.0
    action_latency_tau: float = 0.0  # seconds (1st-order lag time constant)
    gravity_z: float = -9.81        # m/s^2
    solenoid_stroke: float = 0.04   # m
    solenoid_extension_time: float = 0.07  # s

    def as_array(self) -> np.ndarray:
        return np.array(list(asdict(self).values()), dtype=np.float32)


# Sampling ranges from TRACKER §10. Multiplicative ranges are sampled
# log-uniformly so e.g. [0.5, 2.0] is symmetric in log space.
DR_RANGES: dict[str, tuple[float, float] | tuple[float, float, str]] = {
    "keyboard_dx":              (-0.02, 0.02, "uniform"),
    "keyboard_dy":              (-0.02, 0.02, "uniform"),
    "keyboard_dz":              (-0.01, 0.01, "uniform"),
    "keyboard_yaw":             (-np.deg2rad(5), np.deg2rad(5), "uniform"),
    "base_dx":                  (-0.005, 0.005, "uniform"),
    "base_dy":                  (-0.005, 0.005, "uniform"),
    "base_dz":                  (-0.005, 0.005, "uniform"),
    "joint_friction_mul":       (0.5, 1.5, "log_uniform"),
    "joint_damping_mul":        (0.5, 2.0, "log_uniform"),
    "link_mass_mul":            (0.9, 1.1, "log_uniform"),
    "controller_kp_mul":        (0.7, 1.3, "log_uniform"),
    "controller_damping_mul":   (0.7, 1.3, "log_uniform"),
    "action_latency_tau":       (0.0, 0.1, "uniform"),
    "gravity_z":                (-10.0, -9.65, "uniform"),
    "solenoid_stroke":          (0.035, 0.045, "uniform"),
    "solenoid_extension_time":  (0.04, 0.10, "uniform"),
}


def _sample(name: str, rng: np.random.Generator) -> float:
    spec = DR_RANGES[name]
    lo, hi = spec[0], spec[1]
    kind = spec[2] if len(spec) > 2 else "uniform"
    if kind == "uniform":
        return float(rng.uniform(lo, hi))
    if kind == "log_uniform":
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    raise ValueError(f"unknown sampling kind {kind!r}")


def _sample_dr(rng: np.random.Generator) -> DRSample:
    return DRSample(**{k: _sample(k, rng) for k in DR_RANGES})


class DomainRandWrapper(gym.Wrapper):
    """Per-episode DR. Mutates simulator parameters at reset()."""

    def __init__(self, env: gym.Env, *, seed: int | None = None, enabled: bool = True):
        super().__init__(env)
        self.enabled = bool(enabled)
        self._rng = np.random.default_rng(seed)
        self._sample: DRSample = DRSample()
        # Cache nominal physics so we can restore between episodes if mutating
        # mass/friction/etc. in-place. Discovered at first reset().
        self._nominal_cached = False
        self._nominal_geom_friction: np.ndarray | None = None
        self._nominal_dof_damping: np.ndarray | None = None
        self._nominal_body_mass: np.ndarray | None = None

        # Per-step latency buffer for action_latency_tau.
        self._action_buf: list[np.ndarray] = []
        self._lag_steps = 0  # derived from tau and control_freq

    # ---- gym API ----

    def reset(self, **kwargs):
        if self.enabled:
            self._sample = _sample_dr(self._rng)
        else:
            self._sample = DRSample()
        # Apply the keyboard-position part *before* env.reset() so the env
        # picks up the new offset when it builds the model. The robosuite env
        # rebuilds on hard_reset=True.
        underlying = self._find_underlying()
        underlying.keyboard_offset = np.array([
            -0.15 + self._sample.keyboard_dx,
            0.0 + self._sample.keyboard_dy,
        ])
        underlying.keyboard_height = 0.15 + self._sample.keyboard_dz
        result = self.env.reset(**kwargs)
        # Now that the sim exists, apply physics randomizations.
        self._apply_post_reset_dr()
        # Reset action lag buffer
        control_freq = getattr(underlying, "control_freq", 20)
        self._lag_steps = int(round(self._sample.action_latency_tau * control_freq))
        self._action_buf = []
        return result

    def step(self, action):
        # Action-latency model: first-order discrete delay
        if self._lag_steps > 0:
            self._action_buf.append(np.asarray(action, dtype=np.float32))
            if len(self._action_buf) > self._lag_steps:
                action = self._action_buf.pop(0)
            else:
                action = np.zeros_like(self._action_buf[0])
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["dr"] = asdict(self._sample)
        return obs, reward, terminated, truncated, info

    # ---- internals ----

    def _find_underlying(self):
        """Walk wrapper stack to the inner KeyboardEnv (robosuite native)."""
        env = self.env
        while hasattr(env, "env"):
            if hasattr(env, "underlying"):
                return env.underlying
            env = env.env
        if hasattr(env, "underlying"):
            return env.underlying
        return env  # last resort

    def _apply_post_reset_dr(self):
        if not self.enabled:
            return
        underlying = self._find_underlying()
        sim = getattr(underlying, "sim", None)
        if sim is None:
            return
        s = self._sample

        # Cache nominal values once
        if not self._nominal_cached:
            self._nominal_geom_friction = sim.model.geom_friction.copy()
            self._nominal_dof_damping = sim.model.dof_damping.copy()
            self._nominal_body_mass = sim.model.body_mass.copy()
            self._nominal_cached = True

        sim.model.geom_friction[:] = self._nominal_geom_friction * np.array(
            [s.joint_friction_mul, 1.0, 1.0]
        )
        sim.model.dof_damping[:] = self._nominal_dof_damping * s.joint_damping_mul
        sim.model.body_mass[:] = self._nominal_body_mass * s.link_mass_mul

        # Gravity
        sim.model.opt.gravity[2] = s.gravity_z

        # Recompute derived quantities after mass/inertia changes.
        try:
            import mujoco
            mujoco.mj_setConst(sim.model._model, sim.data._data)
            mujoco.mj_forward(sim.model._model, sim.data._data)
        except Exception:
            sim.forward()

        # Note: controller_kp_mul / controller_damping_mul are not applied here
        # because the JOINT_POSITION controller's internal Kp/Kd live in
        # robosuite's controller object, not the MuJoCo model. They'll be wired
        # up in a future pass once we add a small accessor on KeyboardEnv.
