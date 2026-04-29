"""
OSC_POSE spike — verify Rover2026 loads with its default OSC_POSE controller
and that a P-controller in EEF space can reach a hover above arbitrary keys.

Pass criterion: ≥10/20 episodes hit XY <4 mm, |Z error| <5 mm, tilt <5°.
Fail = need a custom Cartesian impedance controller (TRACKER §16 row 1).

Run inside rover_gpu container:
    cd /LearnFlake/.worktrees/rl_rewrite && python3 spike_osc_pose.py
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Match the path setup the existing keyboard_env.py uses
ROOT = os.path.dirname(os.path.abspath(__file__))
ROBO_PATH = os.path.join(ROOT, "src", "external_pkgs", "RoboSuite")
sys.path.insert(0, ROBO_PATH)
sys.path.insert(0, os.path.join(ROOT, "src", "rl_autonomy"))

import robosuite as suite  # noqa: E402
from robosuite.controllers.composite.composite_controller_factory import (  # noqa: E402
    refactor_composite_controller_config,
)

# Import the existing env class (we won't modify it; we'll subclass)
from keyboard_env import KeyboardEnv, TKL_KEYS  # noqa: E402


class OSCKeyboardEnv(KeyboardEnv):
    """KeyboardEnv but with the default Rover2026 OSC_POSE controller."""

    def __init__(self, **kwargs):
        # Skip the parent's __init__ controller setup — we want defaults.
        # Easiest: monkey-patch suite.load_part_controller_config before super().
        # But cleaner: rewrite __init__. We do that directly here.
        from robosuite.environments.manipulation.manipulation_env import (
            ManipulationEnv as MEnv,
        )

        self.keyboard_offset = np.array(kwargs.pop("keyboard_offset", (-0.15, 0.0)))
        self.keyboard_height = kwargs.pop("keyboard_height", 0.15)
        self._target_key = "g"
        self._contact_steps = 0

        # Load the default Rover2026 controller config (= OSC_POSE)
        # Pass robot="Rover2026" so robosuite's loader picks default_rover2026.json
        ctrl_cfg = suite.load_composite_controller_config(robot="Rover2026")
        # robosuite 1.5: the loader returns the composite-shaped dict directly.

        kwargs.setdefault("horizon", 200)
        kwargs.setdefault("render", False)
        kwargs.setdefault("use_camera_obs", False)

        MEnv.__init__(
            self,
            robots=["Rover2026"],
            controller_configs=ctrl_cfg,
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


def p_controller_action(env, kp_xy=8.0, kp_z=8.0, kp_ori=2.0, hover=0.05):
    """
    Hand-coded P-controller: drive EEF toward `target_key` at hover height.

    Action layout for OSC_POSE: (dx, dy, dz, droll, dpitch, dyaw, gripper).
    All in [-1, 1]; controller scales by output_max from default_rover2026.json.
    """
    pf = env.robots[0].robot_model.naming_prefix
    obs = env._get_observations(force_update=False)

    eef_pos = env.sim.data.site_xpos[env._eef_site_id]
    key_pos = env.sim.data.body_xpos[env._key_body_ids[env._target_key]]
    eef_quat = obs.get(f"{pf}eef_quat", np.array([1.0, 0.0, 0.0, 0.0]))

    # Goal in base frame
    goal_xyz = np.array([key_pos[0], key_pos[1], key_pos[2] + hover])
    err_xyz = goal_xyz - eef_pos

    # Tilt from vertical: rotate EEF -Y axis (push direction) toward world -Z
    R = KeyboardEnv._quat_to_rot(eef_quat)
    push_dir = -R[:, 1]
    down = np.array([0.0, 0.0, -1.0])
    # Cross product = small-angle rotation that aligns push_dir with down
    rot_correction = np.cross(push_dir, down) * kp_ori

    a = np.zeros(7)
    a[0:3] = np.clip(err_xyz * kp_xy * np.array([1.0, 1.0, kp_z / kp_xy]), -1.0, 1.0)
    a[3:6] = np.clip(rot_correction, -1.0, 1.0)
    a[6] = -1.0  # gripper / solenoid retracted
    return a


def run_one_episode(env, target_key=None, max_steps=200):
    obs_dict = env.reset()
    if target_key is None:
        target_key = np.random.choice(env.AVAILABLE_KEYS)
    env.set_target_key(target_key)

    for _ in range(max_steps):
        a = p_controller_action(env)
        env.step(a)

        # Check success criteria
        eef_pos = env.sim.data.site_xpos[env._eef_site_id]
        key_pos = env.sim.data.body_xpos[env._key_body_ids[env._target_key]]
        pf = env.robots[0].robot_model.naming_prefix
        eef_quat = env._get_observations(force_update=False).get(
            f"{pf}eef_quat", np.array([1.0, 0.0, 0.0, 0.0])
        )

        xy_dist = float(np.linalg.norm(eef_pos[:2] - key_pos[:2]))
        z_err = float(abs((eef_pos[2] - key_pos[2]) - 0.05))
        tilt = float(KeyboardEnv._eef_tilt_from_vertical(eef_quat))

        if xy_dist < 0.004 and z_err < 0.005 and tilt < np.deg2rad(5):
            return True, target_key, xy_dist, z_err, tilt

    return False, target_key, xy_dist, z_err, tilt


def main():
    print("=" * 60)
    print("Loading Rover2026 with default OSC_POSE controller…")
    env = OSCKeyboardEnv(render=False, horizon=200)
    print(f"  action_dim = {env.action_dim}")
    print(f"  obs_dim    = {env.obs_dim}")
    print(f"  controller = {env.robots[0].composite_controller}")
    print()

    # Sample 20 keys uniformly from the 87 available, but keep alphanumeric for
    # interpretable output. Also include 2 hard keys (corners) explicitly.
    rng = np.random.default_rng(42)
    candidates = [k for k in env.AVAILABLE_KEYS if len(k) == 1] + ["esc", "f12"]
    target_keys = list(rng.choice(candidates, size=18, replace=False)) + ["esc", "f12"]

    n_success = 0
    print(f"{'idx':>3}  {'key':<10}  {'xy(mm)':>8}  {'z(mm)':>8}  {'tilt(°)':>8}  result")
    print("-" * 60)
    for i, key in enumerate(target_keys):
        ok, k, xy, z, tilt = run_one_episode(env, target_key=key)
        n_success += int(ok)
        marker = "PASS" if ok else "fail"
        print(
            f"{i+1:>3}  {k:<10}  {xy*1000:>8.2f}  {z*1000:>8.2f}  "
            f"{np.rad2deg(tilt):>8.2f}  {marker}"
        )

    print("-" * 60)
    print(f"Success rate: {n_success}/{len(target_keys)} "
          f"({100*n_success/len(target_keys):.0f}%)")
    print()
    if n_success >= 10:
        print("SPIKE PASSED — OSC_POSE is viable. Continue with TRACKER plan.")
        return 0
    else:
        print("SPIKE FAILED — need custom Cartesian impedance controller.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
