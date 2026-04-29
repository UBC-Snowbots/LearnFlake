"""
Tighter OSC_POSE diagnostic — separates 'OSC works' from 'P-ctrl tuning':

  Test 1: zero-action — does the arm hold its init pose under OSC?
  Test 2: single-axis +X — does action [0.3,0,0,0,0,0,0] actually move EEF in +X?
  Test 3: single-axis +Z (down) — does action [0,0,-0.3,...] move down?
  Test 4: trajectory toward 'g' key with verbose print every 10 steps
"""
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


class OSCKeyboardEnv(KeyboardEnv):
    def __init__(self, **kwargs):
        self.keyboard_offset = np.array(kwargs.pop("keyboard_offset", (-0.15, 0.0)))
        self.keyboard_height = kwargs.pop("keyboard_height", 0.15)
        self._target_key = "g"
        self._contact_steps = 0
        ctrl_cfg = suite.load_composite_controller_config(robot="Rover2026")
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


def eef(env):
    return env.sim.data.site_xpos[env._eef_site_id].copy()


def main():
    env = OSCKeyboardEnv(render=False, horizon=300)

    # ---- Test 1: zero-action stability ----
    env.reset()
    p0 = eef(env)
    for _ in range(40):
        env.step(np.zeros(7))
    p1 = eef(env)
    drift = np.linalg.norm(p1 - p0)
    print(f"[T1 zero-action] init_eef={p0.round(3)}  after_40_steps={p1.round(3)}  drift={drift*1000:.1f}mm")
    test1 = drift < 0.01  # 1cm tolerance — OSC is critically damped, slight settling is OK
    print(f"     {'PASS' if test1 else 'FAIL'} (drift < 10 mm? {drift*1000:.1f}mm)\n")

    # ---- Test 2: single-axis +X command ----
    env.reset()
    p_before = eef(env)
    a = np.zeros(7); a[0] = 0.5
    for _ in range(20):
        env.step(a)
    p_after = eef(env)
    delta = p_after - p_before
    print(f"[T2 +X cmd]   before={p_before.round(3)}  after={p_after.round(3)}  Δ={delta.round(3)}")
    test2 = delta[0] > 0.05 and abs(delta[1]) < 0.03 and abs(delta[2]) < 0.05
    print(f"     {'PASS' if test2 else 'FAIL'} (Δx>5cm, |Δy|<3cm, |Δz|<5cm)\n")

    # ---- Test 3: single-axis -Z (down) command ----
    env.reset()
    p_before = eef(env)
    a = np.zeros(7); a[2] = -0.5
    for _ in range(20):
        env.step(a)
    p_after = eef(env)
    delta = p_after - p_before
    print(f"[T3 -Z cmd]   before={p_before.round(3)}  after={p_after.round(3)}  Δ={delta.round(3)}")
    test3 = delta[2] < -0.03  # arm goes down at least 3 cm
    print(f"     {'PASS' if test3 else 'FAIL'} (Δz<-3cm)\n")

    # ---- Test 4: P-controller toward 'g' with trajectory log ----
    env.reset()
    env.set_target_key("g")
    pf = env.robots[0].robot_model.naming_prefix
    key_pos = env.sim.data.body_xpos[env._key_body_ids["g"]].copy()
    print(f"[T4 P-ctrl→g]  target_key_pos={key_pos.round(3)}")
    print(f"     {'step':>4}  {'eef_x':>7} {'eef_y':>7} {'eef_z':>7}  {'err_xy':>7} {'err_z':>7}  {'action':>30}")
    for step in range(120):
        ep = eef(env)
        err = np.array([key_pos[0] - ep[0], key_pos[1] - ep[1], (key_pos[2] + 0.05) - ep[2]])
        a = np.zeros(7)
        a[0:3] = np.clip(err * 8.0, -1.0, 1.0)
        # No orientation control here — just see if XY/Z converge
        a[6] = -1.0
        if step % 10 == 0:
            print(f"     {step:>4}  {ep[0]:>7.3f} {ep[1]:>7.3f} {ep[2]:>7.3f}  "
                  f"{np.linalg.norm(err[:2])*1000:>7.1f} {abs(err[2])*1000:>7.1f}  "
                  f"[{a[0]:+.2f}, {a[1]:+.2f}, {a[2]:+.2f}]")
        env.step(a)
    ep = eef(env)
    final_xy = np.linalg.norm(ep[:2] - key_pos[:2])
    final_z = abs(ep[2] - key_pos[2] - 0.05)
    print(f"     final: xy={final_xy*1000:.1f}mm  z={final_z*1000:.1f}mm")
    test4 = final_xy < 0.02 and final_z < 0.02  # 2cm — relaxed; just want to see if it gets close
    print(f"     {'PASS' if test4 else 'FAIL'} (xy<20mm, z<20mm)\n")

    print("=" * 60)
    n_pass = sum([test1, test2, test3, test4])
    print(f"Diagnostic: {n_pass}/4 tests passed.")
    if n_pass >= 3:
        print("OSC_POSE is functionally working. P-controller in spike was just badly tuned.")
    else:
        print("OSC_POSE has a real problem — investigate before committing to TRACKER §6.")


if __name__ == "__main__":
    main()
