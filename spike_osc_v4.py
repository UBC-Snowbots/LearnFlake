"""
Spike v4 — tune OSC for the actual config we'll use for RL:
  output_max = 0.02 m, 0.2 rad (smaller per-step than default 0.05/0.5)
  Lower P-controller gains to avoid action saturation.

Pass criterion: 1 episode reaches target_g with xy<5mm AND z<5mm in 200 steps.
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


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def gentle_p_action(env, target_xyz, kp_xy=15.0, kp_z=15.0, kp_ori=2.0, max_action=0.5):
    """
    Gentle P-controller: bounded action so OSC doesn't saturate.
    With OSC output_max=0.02, action 0.5 → 1cm/step reference, well within stability.
    """
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
    a[0:2] = np.clip(err[:2] * kp_xy, -max_action, max_action)
    a[2] = np.clip(err[2] * kp_z, -max_action, max_action)
    a[3:6] = np.clip(rot_axis * kp_ori, -max_action, max_action)
    a[6] = -1.0
    return a


def main():
    # Build a custom OSC_POSE config with tighter output_max, lower kp
    with open(f"{CTRL_DIR}/robots/default_rover2026.json") as f:
        cfg = json.load(f)
    arm = cfg["body_parts"]["arms"]["right"]
    arm["output_max"] = [0.02, 0.02, 0.02, 0.2, 0.2, 0.2]
    arm["output_min"] = [-0.02, -0.02, -0.02, -0.2, -0.2, -0.2]
    arm["kp"] = 80           # was 150 — less aggressive tracking
    arm["damping_ratio"] = 1.5  # over-damped: no overshoot
    out_path = "/tmp/rover2026_osc_tuned.json"
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)

    cfg = suite.load_composite_controller_config(controller=out_path)
    env = KeyboardEnvWithCfg(controller_configs=cfg, horizon=300)
    print(f"action_dim={env.action_dim}  obs_dim={env.obs_dim}")

    # Test on multiple keys: easy (g, h, j) + medium (q, p) + hard (esc, f12)
    test_keys = ["g", "h", "j", "f", "d", "k", "l", "s", "a", "i", "u",
                 "q", "p", "z", "m", "0", "1", "esc", "f12", "space"]

    n_pass = 0
    print(f"\n{'idx':>3}  {'key':<6}  {'best_xy':>8}  {'best_z':>8}  {'final_xy':>9}  {'final_z':>8}  result")
    print("-" * 65)
    for i, key in enumerate(test_keys):
        env.reset()
        env.set_target_key(key)
        key_pos = env.sim.data.body_xpos[env._key_body_ids[key]].copy()
        target = np.array([key_pos[0], key_pos[1], key_pos[2] + 0.05])

        best_xy, best_z = np.inf, np.inf
        for _ in range(220):
            a = gentle_p_action(env, target)
            env.step(a)
            ep = eef(env)
            xy = np.linalg.norm(ep[:2] - target[:2])
            z = abs(ep[2] - target[2])
            if xy < best_xy:
                best_xy = xy
            if z < best_z:
                best_z = z

        ep = eef(env)
        final_xy = np.linalg.norm(ep[:2] - target[:2])
        final_z = abs(ep[2] - target[2])
        ok = best_xy < 0.005 and best_z < 0.005
        n_pass += int(ok)
        print(f"{i+1:>3}  {key:<6}  {best_xy*1000:>8.2f}  {best_z*1000:>8.2f}  "
              f"{final_xy*1000:>9.2f}  {final_z*1000:>8.2f}  {'PASS' if ok else 'fail'}")

    print("-" * 65)
    print(f"Tuned OSC_POSE: {n_pass}/{len(test_keys)} ({100*n_pass/len(test_keys):.0f}%) "
          f"reached <5mm best xy AND <5mm best z")
    if n_pass >= len(test_keys) // 2:
        print("\nSPIKE PASSED — OSC_POSE viable for RL with output_max=0.02, kp=80, damping=1.5")
    else:
        print("\nSPIKE FAILED — OSC_POSE not converging even at conservative settings.")


if __name__ == "__main__":
    main()
