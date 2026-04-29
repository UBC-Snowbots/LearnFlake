"""
Spike v5 — try OSC_POSE in absolute input mode.
With input_type="absolute", action[0:3] is the target xyz directly, not a delta.
We send the *same* target every step; OSC does its own trajectory generation.
This is far closer to how a learned policy would behave (output target pose).
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
    # Try absolute input + delta orientation, larger workspace
    with open("/LearnFlake/src/external_pkgs/RoboSuite/robosuite/controllers/config/robots/default_rover2026.json") as f:
        cfg = json.load(f)
    arm = cfg["body_parts"]["arms"]["right"]
    arm["input_type"] = "absolute"
    arm["input_max"] = [1.0, 1.0, 1.5, 1.0, 1.0, 1.0]
    arm["input_min"] = [-1.0, -1.0, 0.0, -1.0, -1.0, -1.0]
    # output range still applies as scale; for absolute input the action is treated as
    # the desired absolute pose in workspace
    arm["output_max"] = [1.0, 1.0, 1.5, np.pi, np.pi, np.pi]
    arm["output_min"] = [-1.0, -1.0, 0.0, -np.pi, -np.pi, -np.pi]
    arm["kp"] = 100
    arm["damping_ratio"] = 1.5
    out_path = "/tmp/rover2026_osc_abs.json"
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)

    cfg = suite.load_composite_controller_config(controller=out_path)
    env = KeyboardEnvWithCfg(controller_configs=cfg, horizon=300)
    print(f"action_dim={env.action_dim}  obs_dim={env.obs_dim}")

    test_keys = ["g", "h", "j", "f", "d", "k", "l", "s", "a", "u",
                 "i", "q", "p", "z", "m", "0", "1", "esc", "space"]
    n_pass = 0
    print(f"\n{'idx':>3}  {'key':<6}  {'best_xy':>8}  {'best_z':>8}  {'final_xy':>9}  {'final_z':>8}  result")
    print("-" * 65)
    for i, key in enumerate(test_keys):
        env.reset()
        env.set_target_key(key)
        key_pos = env.sim.data.body_xpos[env._key_body_ids[key]].copy()
        target = np.array([key_pos[0], key_pos[1], key_pos[2] + 0.05])

        # Absolute OSC: action is target_xyz + zero orientation (defaults to current)
        a = np.zeros(7)
        a[0:3] = target
        # leave a[3:6] = 0  → no orientation change reference (delta orientation)
        a[6] = -1.0

        best_xy, best_z = np.inf, np.inf
        for _ in range(220):
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
    print(f"Absolute OSC_POSE: {n_pass}/{len(test_keys)} reached best xy<5mm AND best z<5mm")


if __name__ == "__main__":
    main()
