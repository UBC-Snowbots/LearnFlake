from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol

import numpy as np

from .types import KeyTarget


class KeyPressBackend(Protocol):
    def move_to(self, target: KeyTarget) -> None:
        ...

    def press(self, target: KeyTarget) -> None:
        ...

    def verify(self, target: KeyTarget) -> bool:
        ...


@dataclass
class MockKeyPressBackend:
    """
    Mock backend used to exercise Omega before simulator or robot wiring.

    `verification_plan` maps key ids to a queue of booleans. If the queue is empty,
    verification defaults to True for that key.
    """

    verification_plan: Dict[str, List[bool]] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)
    current_target: KeyTarget | None = None

    def __post_init__(self) -> None:
        self._plan = defaultdict(list, {key: list(values) for key, values in self.verification_plan.items()})

    def move_to(self, target: KeyTarget) -> None:
        self.current_target = target
        self.events.append(f"move:{target.key_id}@{target.position}")

    def press(self, target: KeyTarget) -> None:
        self.current_target = target
        self.events.append(f"press:{target.key_id}")

    def verify(self, target: KeyTarget) -> bool:
        self.events.append(f"verify:{target.key_id}")
        if self._plan[target.key_id]:
            return bool(self._plan[target.key_id].pop(0))
        return True


class AlphaToolkitBackend:
    """
    Backend that uses a trained Alpha reach policy to actuate the Rover2026
    arm inside a KeypadReachEnv simulation.

    Omega owns code decoding and state progression. This backend translates
    each key target into physical motion via the RL policy:
      move_to  → servo EEF to 3 cm standoff from the key
      press    → drive EEF straight down to contact the key
      verify   → check EEF-to-key distance is within tolerance
    """

    _AXIS_SCALES_M = np.array([0.0078, 0.0082, 0.0089], dtype=np.float32)
    _EEF_SITE_CANDIDATES = (
        "robot0_grip_site",
        "gripper0_right_grip_site",
        "gripper0_grip_site",
    )

    def __init__(
        self,
        env: Any,
        policy: Any,
        max_move_steps: int = 200,
        press_steps: int = 15,
        stability_window: int = 10,
        stability_threshold: int = 7,
        verify_tolerance_m: float = 0.01,
        xy_verify_tolerance_m: float = 0.015,
        surface_verify_tolerance_m: float = 0.02,
        policy_mix: float = 0.0,
    ):
        self.env = env
        self.policy = policy
        self.max_move_steps = max_move_steps
        self.press_steps = press_steps
        self.stability_window = stability_window
        self.stability_threshold = stability_threshold
        self.verify_tolerance_m = verify_tolerance_m
        self.xy_verify_tolerance_m = xy_verify_tolerance_m
        self.surface_verify_tolerance_m = surface_verify_tolerance_m
        self.policy_mix = float(np.clip(policy_mix, 0.0, 1.0))
        self._last_eef_pos = np.zeros(3, dtype=np.float32)
        self._last_info: Dict[str, Any] = {}
        self.events: List[str] = []
        self._eef_site_id = self._resolve_eef_site_id()

    def move_to(self, target: KeyTarget) -> None:
        """Servo the EEF to 3 cm standoff from the key using the trained policy."""
        target_xyz = np.array(target.position, dtype=np.float32)
        self.env.set_target(target_xyz)
        obs = self.env.current_obs()
        info: Dict[str, Any] = dict(self._last_info)
        desired_hover = target_xyz + np.array([0.0, 0.0, self._hover_height_m()], dtype=np.float32)

        for step in range(self.max_move_steps):
            action = self._blended_hover_action(obs, desired_hover)
            obs, _, done, info = self.env.step(action)

            eef = np.asarray(info.get("eef_xyz", obs[:3]), dtype=np.float32)
            hover_error = float(np.linalg.norm(desired_hover - eef))
            if self._is_hover_locked(info) or hover_error <= self.xy_verify_tolerance_m:
                # Check stability: hold within tolerance for several consecutive steps
                stable = 0
                for _ in range(self.stability_window):
                    action = self._blended_hover_action(obs, desired_hover)
                    obs, _, _, info = self.env.step(action)
                    eef = np.asarray(info.get("eef_xyz", obs[:3]), dtype=np.float32)
                    hover_error = float(np.linalg.norm(desired_hover - eef))
                    if self._is_hover_locked(info) or hover_error <= self.xy_verify_tolerance_m:
                        stable += 1
                if stable >= self.stability_threshold:
                    break

        self._last_eef_pos = info["eef_xyz"].copy()
        self._last_info = dict(info)
        self.events.append(
            f"move:{target.key_id} dist={info.get('distance_to_goal', '?'):.4f}"
        )

    def press(self, target: KeyTarget) -> None:
        """Drive the EEF straight down to contact the key surface."""
        info: Dict[str, Any] = dict(self._last_info)
        for _ in range(self.press_steps):
            eef = np.asarray(info.get("eef_xyz", self._last_eef_pos), dtype=np.float32)
            target_xyz = np.array(target.position, dtype=np.float32)
            delta = target_xyz - eef
            action = np.array(
                self._action_for_eef_delta(
                    np.array([delta[0], delta[1], delta[2] - 0.005], dtype=np.float32)
                ),
                dtype=np.float32,
            )
            _, _, _, info = self.env.step(action)

        self._last_eef_pos = info["eef_xyz"].copy()
        self._last_info = dict(info)
        self.events.append(f"press:{target.key_id}")

    def verify(self, target: KeyTarget) -> bool:
        """Check that the EEF is within tolerance of the key position."""
        target_xyz = np.array(target.position, dtype=np.float32)
        delta = self._last_eef_pos - target_xyz
        xy_distance = float(np.linalg.norm(delta[:2]))
        z_error = float(abs(delta[2]))
        distance = float(np.linalg.norm(delta))
        ok = (
            distance <= self.verify_tolerance_m
            or (xy_distance <= self.xy_verify_tolerance_m and z_error <= self.surface_verify_tolerance_m)
        )
        self.events.append(
            f"verify:{target.key_id} dist={distance:.4f} xy={xy_distance:.4f} z={z_error:.4f} ok={ok}"
        )
        return ok

    def _blended_hover_action(self, obs: np.ndarray, desired_hover: np.ndarray) -> np.ndarray:
        eef = np.asarray(obs[:3], dtype=np.float32)
        delta = desired_hover - eef
        heuristic = self._action_for_eef_delta(delta)
        if self.policy is None:
            return heuristic.astype(np.float32)
        policy_action, _ = self.policy.get_action(obs)
        policy_action = np.asarray(policy_action, dtype=np.float32).reshape(3)
        action = self.policy_mix * policy_action + (1.0 - self.policy_mix) * heuristic
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _action_for_eef_delta(self, desired_delta: np.ndarray) -> np.ndarray:
        world_delta = np.asarray(desired_delta, dtype=np.float32).reshape(3)
        rotation = self._eef_rotation_world()
        local_delta = rotation.T @ world_delta
        action = local_delta / self._AXIS_SCALES_M
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _resolve_eef_site_id(self) -> int | None:
        sim = getattr(getattr(self.env, "env", None), "sim", None)
        if sim is None:
            return None
        for site_name in self._EEF_SITE_CANDIDATES:
            try:
                return int(sim.model.site_name2id(site_name))
            except Exception:
                continue
        return None

    def _eef_rotation_world(self) -> np.ndarray:
        sim = getattr(getattr(self.env, "env", None), "sim", None)
        if sim is None or self._eef_site_id is None:
            return np.eye(3, dtype=np.float32)
        xmat = np.asarray(sim.data.site_xmat[self._eef_site_id], dtype=np.float32).reshape(3, 3)
        return xmat

    def _hover_height_m(self) -> float:
        return float(getattr(self.env.config, "hover_height_m", self.env.config.standoff_m))

    def _is_hover_locked(self, info: Dict[str, Any]) -> bool:
        if "hover_locked" in info:
            return bool(info["hover_locked"])
        xy_tol = float(getattr(self.env.config, "hover_success_xy_tolerance_m", self.xy_verify_tolerance_m))
        z_tol = float(getattr(self.env.config, "hover_success_z_tolerance_m", self.surface_verify_tolerance_m))
        xy_error = float(info.get("hover_xy_error", float("inf")))
        z_error = float(info.get("hover_z_error", float("inf")))
        return xy_error <= xy_tol and z_error <= z_tol
