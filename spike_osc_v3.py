"""
Spike v3 — try three controllers, report which converges to 'g' key:

  Variant A: OSC_POSE with proper orientation tracking (P-ctrl to vertical)
  Variant B: OSC_POSITION (3-DOF position-only)
  Variant C: IK_POSE

Picks whichever produces xy<10mm and z<10mm in 150 steps.
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


def make_env(controller_cfg_path):
    cfg = suite.load_composite_controller_config(controller=controller_cfg_path)
    env = KeyboardEnvWithCfg(controller_configs=cfg, horizon=200)
    return env


class KeyboardEnvWithCfg(KeyboardEnv):
    def __init__(self, controller_configs, **kwargs):
        self.keyboard_offset = np.array(kwargs.pop("keyboard_offset", (-0.15, 0.0)))
        self.keyboard_height = kwargs.pop("keyboard_height", 0.15)
        self._target_key = "g"
        self._contact_steps = 0
        kwargs.setdefault("horizon", 200)
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


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def osc_pose_action(env, target_xyz, kp_xy=4.0, kp_z=4.0, kp_ori=1.5):
    """OSC_POSE: 6-DoF action — position + orientation."""
    pf = env.robots[0].robot_model.naming_prefix
    obs = env._get_observations(force_update=False)
    ep = eef(env)
    eef_quat = obs.get(f"{pf}eef_quat", np.array([1.0, 0.0, 0.0, 0.0]))

    err = target_xyz - ep
    R = quat_to_rot(eef_quat)
    push_dir = -R[:, 1]
    down = np.array([0.0, 0.0, -1.0])
    rot_axis = np.cross(push_dir, down)

    a = np.zeros(7)
    a[0:2] = np.clip(err[:2] * kp_xy, -1.0, 1.0)
    a[2] = np.clip(err[2] * kp_z, -1.0, 1.0)
    a[3:6] = np.clip(rot_axis * kp_ori, -1.0, 1.0)
    a[6] = -1.0
    return a


def osc_position_action(env, target_xyz, kp_xy=4.0, kp_z=4.0):
    """OSC_POSITION: 3-DoF action — position only. Total dim = 3 + gripper = 4."""
    ep = eef(env)
    err = target_xyz - ep
    a = np.zeros(env.action_dim)
    a[0:2] = np.clip(err[:2] * kp_xy, -1.0, 1.0)
    a[2] = np.clip(err[2] * kp_z, -1.0, 1.0)
    a[-1] = -1.0
    return a


def run_variant(name, ctrl_cfg_path, action_fn):
    print(f"\n{'='*60}\n[{name}] {ctrl_cfg_path}\n{'='*60}")
    try:
        env = make_env(ctrl_cfg_path)
    except Exception as e:
        print(f"  LOAD FAIL: {e}")
        return False

    print(f"  action_dim={env.action_dim}  obs_dim={env.obs_dim}")
    env.reset()
    env.set_target_key("g")
    key_pos = env.sim.data.body_xpos[env._key_body_ids["g"]].copy()
    target = np.array([key_pos[0], key_pos[1], key_pos[2] + 0.05])
    print(f"  target_xyz={target.round(3)}  init_eef={eef(env).round(3)}")

    best_xy, best_z = np.inf, np.inf
    for step in range(150):
        a = action_fn(env, target)
        env.step(a)
        ep = eef(env)
        xy = np.linalg.norm(ep[:2] - target[:2])
        z = abs(ep[2] - target[2])
        if xy < best_xy:
            best_xy = xy
        if z < best_z:
            best_z = z
        if step % 30 == 0:
            print(f"    step={step:>3}  eef={ep.round(3)}  xy={xy*1000:.1f}mm  z={z*1000:.1f}mm  a[:3]=[{a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f}]")

    ep = eef(env)
    xy_final = np.linalg.norm(ep[:2] - target[:2])
    z_final = abs(ep[2] - target[2])
    print(f"  FINAL   xy={xy_final*1000:.1f}mm  z={z_final*1000:.1f}mm")
    print(f"  BEST    xy={best_xy*1000:.1f}mm  z={best_z*1000:.1f}mm")
    success = best_xy < 0.01 and best_z < 0.01
    print(f"  {'PASS' if success else 'fail'}  (best xy<10mm and best z<10mm)")
    return success


def main():
    rover_default = f"{CTRL_DIR}/robots/default_rover2026.json"
    rover_ik = f"{CTRL_DIR}/robots/default_rover2026_ik.json"

    # Build a customized OSC_POSITION composite config for Rover2026 by editing
    # the default config in-memory.
    with open(rover_default) as f:
        rover_osc_pos_cfg = json.load(f)
    rover_osc_pos_cfg["body_parts"]["arms"]["right"]["type"] = "OSC_POSITION"
    rover_osc_pos_cfg["body_parts"]["arms"]["right"]["output_max"] = [0.05, 0.05, 0.05]
    rover_osc_pos_cfg["body_parts"]["arms"]["right"]["output_min"] = [-0.05, -0.05, -0.05]
    out_path = "/tmp/rover2026_osc_position.json"
    with open(out_path, "w") as f:
        json.dump(rover_osc_pos_cfg, f, indent=2)

    results = {}
    results["A_OSC_POSE"] = run_variant("A", rover_default, osc_pose_action)
    results["B_OSC_POSITION"] = run_variant("B", out_path, osc_position_action)
    results["C_IK_POSE"] = run_variant("C", rover_ik, osc_pose_action)

    print("\n" + "=" * 60)
    print("SUMMARY")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
