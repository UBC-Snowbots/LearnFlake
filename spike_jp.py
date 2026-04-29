"""
Spike v6 — JOINT_POSITION fallback. Simpler than OSC: commands joint angles directly.
No Jacobian, no redundancy resolution, no orientation tracker fighting.
We compute target joint angles via a one-shot inverse kinematics (mink) and command them.
"""
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src", "external_pkgs", "RoboSuite"))
sys.path.insert(0, os.path.join(ROOT, "src", "rl_autonomy"))

import robosuite as suite  # noqa: E402
from robosuite.environments.manipulation.manipulation_env import (  # noqa: E402
    ManipulationEnv as MEnv,
)
from keyboard_env import KeyboardEnv  # noqa: E402

CTRL_DIR = "/LearnFlake/src/external_pkgs/RoboSuite/robosuite/controllers/config"


class KeyboardEnvWithCfg(KeyboardEnv):
    def __init__(self, controller_configs, **kwargs):
        self.keyboard_offset = np.array(kwargs.pop("keyboard_offset", (-0.15, 0.0)))
        self.keyboard_height = kwargs.pop("keyboard_height", 0.15)
        self._target_key = "g"
        self._contact_steps = 0
        kwargs.setdefault("horizon", 300)
        kwargs.setdefault("render", False)
        kwargs.setdefault("use_camera_obs", False)
        MEnv.__init__(
            self,
            robots=["Rover2026"],
            controller_configs=controller_configs,
            has_renderer=kwargs.pop("render"),
            has_offscreen_renderer=True,
            use_camera_obs=kwargs.pop("use_camera_obs"),
            render_camera="frontview",
            control_freq=20,
            horizon=kwargs.pop("horizon"),
            ignore_done=False,
            hard_reset=True,
            **kwargs,
        )


def eef(env):
    return env.sim.data.site_xpos[env._eef_site_id].copy()


def main():
    # Build JOINT_POSITION composite config from default.
    with open(f"{CTRL_DIR}/robots/default_rover2026.json") as f:
        cfg = json.load(f)
    cfg["body_parts"]["arms"]["right"] = {
        "type": "JOINT_POSITION",
        "input_max": 1.0,
        "input_min": -1.0,
        "output_max": 0.05,        # 0.05 rad max joint delta per timestep
        "output_min": -0.05,
        "kp": 100,
        "damping_ratio": 1.5,
        "impedance_mode": "fixed",
        "kp_limits": [0, 300],
        "damping_ratio_limits": [0, 10],
        "qpos_limits": None,
        "interpolation": None,
        "ramp_ratio": 0.2,
        "gripper": {"type": "GRIP", "input_max": 1, "input_min": -1},
    }
    out_path = "/tmp/rover2026_jp.json"
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)

    cfg = suite.load_composite_controller_config(controller=out_path)
    env = KeyboardEnvWithCfg(controller_configs=cfg, horizon=300)
    print(f"action_dim={env.action_dim}  obs_dim={env.obs_dim}")

    # T1: zero-action holds pose
    env.reset()
    p0 = eef(env)
    for _ in range(40):
        env.step(np.zeros(env.action_dim))
    drift = np.linalg.norm(eef(env) - p0)
    print(f"[T1] zero-action drift = {drift*1000:.1f}mm  ({'PASS' if drift < 0.01 else 'FAIL'})")

    # T2: per-joint excitation. Command +0.5 on each joint individually for 30 steps,
    # see if EEF moves (any direction is fine — just check responsiveness).
    print("\n[T2] per-joint response:")
    for j in range(6):
        env.reset()
        p_before = eef(env)
        a = np.zeros(env.action_dim); a[j] = 0.5
        for _ in range(30):
            env.step(a)
        delta = eef(env) - p_before
        print(f"  joint {j}: Δ_eef = [{delta[0]:+.3f}, {delta[1]:+.3f}, {delta[2]:+.3f}]  |Δ|={np.linalg.norm(delta)*100:.1f}cm")

    # T3: Numerical IK to reach 'g' key. Use mujoco's IK via mink (already in deps),
    # or a hand-coded pseudo-inverse-Jacobian step toward target.
    print("\n[T3] Pseudoinverse-Jacobian IK toward 'g':")
    env.reset()
    env.set_target_key("g")
    key_pos = env.sim.data.body_xpos[env._key_body_ids["g"]].copy()
    target = np.array([key_pos[0], key_pos[1], key_pos[2] + 0.05])
    print(f"  init_eef={eef(env).round(3)}  target={target.round(3)}")

    sim = env.sim
    eef_id = env._eef_site_id

    best_xy, best_z = np.inf, np.inf
    for step in range(200):
        ep = eef(env)
        err = target - ep

        # Compute Jacobian for EEF site position (3xN)
        import mujoco
        jacp = np.zeros((3, sim.model.nv))
        jacr = np.zeros((3, sim.model.nv))
        mujoco.mj_jacSite(sim.model._model, sim.data._data, jacp, jacr, eef_id)
        # Restrict to arm joints (first 6 dofs of robot)
        # The Rover2026's arm joint indices are 0..5 in qpos for this single-arm setup
        J = jacp[:, :6]  # 3x6 Jacobian
        # Damped least-squares pseudo-inverse for stability near singularities
        damping = 0.05
        JJt = J @ J.T + damping**2 * np.eye(3)
        dq = J.T @ np.linalg.solve(JJt, err) * 0.5  # scale to per-step delta

        # JOINT_POSITION action: scaled in [-1,1] over [-output_max, output_max]
        # output_max = 0.05 rad/step. We want to move dq rad/step.
        a = np.zeros(env.action_dim)
        a[0:6] = np.clip(dq / 0.05, -1.0, 1.0)
        a[-1] = -1.0
        env.step(a)

        ep = eef(env)
        xy = np.linalg.norm(ep[:2] - target[:2])
        z = abs(ep[2] - target[2])
        if xy < best_xy: best_xy = xy
        if z < best_z: best_z = z
        if step % 20 == 0:
            print(f"    step={step:>3}  xy={xy*1000:>7.1f}mm  z={z*1000:>7.1f}mm  |dq|={np.linalg.norm(dq):.3f}")

    print(f"  BEST: xy={best_xy*1000:.1f}mm  z={best_z*1000:.1f}mm  "
          f"FINAL: xy={np.linalg.norm(eef(env)[:2] - target[:2])*1000:.1f}mm  "
          f"z={abs(eef(env)[2] - target[2])*1000:.1f}mm")
    success = best_xy < 0.005 and best_z < 0.005
    print(f"  {'PASS' if success else 'FAIL'} (best xy<5mm AND best z<5mm)")


if __name__ == "__main__":
    main()
