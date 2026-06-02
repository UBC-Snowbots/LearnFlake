"""The M1 hand-coded Jacobian-IK expert, as a reusable interactive policy.

This is the single source of truth for the adaptive-weight damped-least-squares
(DLS) Jacobian controller that solves the Approach task analytically. It was
originally inlined in ``tools/gen_demos.py`` (``_jacobian_step``) and
``tools/m1_p_controller.py``; both now import from here.

Why factor it out: DAgger (TRACKER §35) needs to **query the expert at
arbitrary states the policy visits**, not just along expert rollouts. That is
the whole point of DAgger — label the learner's own state distribution. Having
the expert as a callable ``IKExpert.action(kb)`` (deterministic function of the
current sim state + the env's active target key) makes that a one-liner inside
the rollout loop.

The controller is a deterministic function of:
  - the current end-effector site pose (position + orientation), read live
    from ``kb.sim.data``;
  - the manipulator Jacobian at the current joint configuration
    (``mujoco.mj_jacSite``);
  - the target = active key world position + hover height.

It returns a 7-D policy-space action in ``[-1, 1]``: 6 clipped joint deltas +
the solenoid held retracted (Approach never strikes).
"""
from __future__ import annotations

import numpy as np
import mujoco


# DLS / adaptive-weight constants — identical to the original gen_demos and
# m1_p_controller values (TRACKER §29.4 stage 1). Do not retune without
# regenerating demos: the saved demo actions were produced with these.
_DAMPING = 0.06
_DQ_SCALE = 0.5            # step gain on the DLS solution
_ACTION_SCALE = 0.05      # joint delta → [-1,1] normalization (a = dq / 0.05)
_TILT_HI = 0.087          # rad (~5°): above this, prioritize orientation hard
_TILT_LO = 0.017          # rad (~1°): below this, de-prioritize orientation
_W_ROT_HI = 5.0
_W_ROT_MID = 1.0
_W_ROT_LO = 0.5


def ik_step(kb, target_xyz: np.ndarray) -> np.ndarray:
    """One adaptive-weight DLS Jacobian step toward ``target_xyz``.

    Args:
        kb: a ``KeyboardEnv`` (robosuite-native). Read-only — we only inspect
            ``kb.sim``, ``kb._eef_site_id``, ``kb.robots``.
        target_xyz: desired EEF site position in world frame (3,).

    Returns:
        7-D action in [-1, 1]: ``a[0:6]`` = clipped joint deltas, ``a[6]`` =
        -1.0 (solenoid retracted).

    This is byte-for-byte the logic of the old ``gen_demos._jacobian_step`` /
    ``m1_p_controller._jacobian_step``, just relocated.
    """
    # Local import to avoid a circular import (keyboard_env imports algos lazily
    # in some paths) and to keep this module importable without robosuite when
    # only the constants are wanted.
    from rl_autonomy.envs.keyboard_env import quat_to_rot

    sim = kb.sim
    ep = sim.data.site_xpos[kb._eef_site_id].copy()
    err_pos = target_xyz - ep

    pf = kb.robots[0].robot_model.naming_prefix
    obs = kb._get_observations(force_update=False)
    eef_quat = obs.get(f"{pf}eef_quat", np.array([1.0, 0.0, 0.0, 0.0]))
    R = quat_to_rot(eef_quat)
    push_dir = -R[:, 1]
    err_rot = np.cross(push_dir, np.array([0.0, 0.0, -1.0]))

    tilt_mag = float(np.linalg.norm(err_rot))
    w_rot = _W_ROT_HI if tilt_mag > _TILT_HI else (
        _W_ROT_LO if tilt_mag < _TILT_LO else _W_ROT_MID
    )
    w_pos = 1.0

    jacp = np.zeros((3, sim.model.nv))
    jacr = np.zeros((3, sim.model.nv))
    mujoco.mj_jacSite(sim.model._model, sim.data._data, jacp, jacr, kb._eef_site_id)
    J = np.vstack([w_pos * jacp[:, :6], w_rot * jacr[:, :6]])
    err = np.concatenate([w_pos * err_pos, w_rot * err_rot])

    JJt = J @ J.T + _DAMPING**2 * np.eye(6)
    dq = J.T @ np.linalg.solve(JJt, err) * _DQ_SCALE

    a = np.zeros(7, dtype=np.float32)
    a[0:6] = np.clip(dq / _ACTION_SCALE, -1.0, 1.0)
    a[6] = -1.0
    return a


class IKExpert:
    """Interactive DLS-IK expert callable at any state.

    Wraps :func:`ik_step` with target derivation from the env's *currently
    active* target key, so a caller (DAgger rollout, gen_demos) only needs the
    ``KeyboardEnv`` handle::

        expert = IKExpert()
        a = expert.action(kb)        # uses kb.target_key + kb.hover_height

    The expert is stateless; one instance can label many envs/keys.
    """

    def target_for(self, kb) -> np.ndarray:
        """World-frame target = active key position + hover height."""
        key_pos = kb.sim.data.body_xpos[kb._key_body_ids[kb._target_key]].copy()
        return np.array(
            [key_pos[0], key_pos[1], key_pos[2] + kb.hover_height], dtype=np.float64
        )

    def action(self, kb, target_xyz: np.ndarray | None = None) -> np.ndarray:
        """Return the 7-D expert action for ``kb``'s current state.

        Args:
            kb: ``KeyboardEnv``.
            target_xyz: optional explicit world target; if ``None`` it is
                derived from the env's active target key.
        """
        if target_xyz is None:
            target_xyz = self.target_for(kb)
        return ik_step(kb, target_xyz)
