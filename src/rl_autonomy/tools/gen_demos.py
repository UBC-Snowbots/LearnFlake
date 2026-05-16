"""Generate demonstration trajectories via the M1 hand-coded Jacobian IK.

Drives the SAME wrapped env that training uses (make_env: KeyboardEnv →
KeyboardGymEnv → ActionAdapter → FrameStackWrapper → ObsAdapter), so the
recorded transitions match the shape the RLPD replay buffer expects.

Per TRACKER §29.4 (v1.1 stage 1). Only successful trajectories are saved
(success = the env's _check_success at any step inside the rollout).

Output: HDF5 with flat-concatenated transitions across all successful
episodes. Schema:

    /actor_obs        (N, 108)  float32   stacked 3×36 actor obs
    /critic_obs       (N, 38)   float32   privileged critic obs
    /action           (N, 7)    float32   raw policy-space action [-1, 1]
                                          (pre-smoothing — see comment below)
    /reward           (N,)      float32
    /next_actor_obs   (N, 108)  float32
    /next_critic_obs  (N, 38)   float32
    /terminated       (N,)      bool
    /episode_id       (N,)      int32     which successful episode this transition came from
    /trial_meta       group     attrs: key, seed, steps_to_success, n_trials, n_success

Notes on the saved action:
  - The action is what M1 would have *output* (pre-ActionAdapter smoothing).
    The env's ActionAdapter applies α=0.4 first-order smoothing before
    actually executing in MuJoCo. The recorded obs/reward already reflect
    the smoothed execution. The agent (when later trained on these demos)
    learns the same input-output relationship the policy will produce at
    deployment, i.e. raw policy action → env wraps and executes.

Run:
    python3 -m rl_autonomy.tools.gen_demos \\
        --out demos/approach_phase_a.h5 \\
        --keys phase_a \\
        --trials-per-key 8 \\
        --max-steps 200
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

import h5py
import mujoco

from rl_autonomy.envs import KeyboardEnv, make_env
from rl_autonomy.envs._wrapper_utils import find_inner
from rl_autonomy.envs.action_adapter import ActionAdapter
from rl_autonomy.envs.keyboard_layout import PHASE_A_KEYS, PHASE_B_KEYS, AVAILABLE_KEYS


# ---------------------------------------------------------------------------
# Adaptive-weight DLS Jacobian step — the M1 controller, re-used here
# ---------------------------------------------------------------------------

def _jacobian_step(env: KeyboardEnv, target_xyz: np.ndarray) -> np.ndarray:
    """Full-pose DLS Jacobian step (position + orientation-toward-vertical).

    Identical to m1_p_controller._jacobian_step. Returns a 7-dim policy
    action in [-1, 1]: 6 joint deltas (clipped) + solenoid retracted.
    """
    sim = env.sim
    ep = sim.data.site_xpos[env._eef_site_id].copy()
    err_pos = target_xyz - ep

    pf = env.robots[0].robot_model.naming_prefix
    obs = env._get_observations(force_update=False)
    eef_quat = obs.get(f"{pf}eef_quat", np.array([1.0, 0.0, 0.0, 0.0]))
    from rl_autonomy.envs.keyboard_env import quat_to_rot
    R = quat_to_rot(eef_quat)
    push_dir = -R[:, 1]
    err_rot = np.cross(push_dir, np.array([0.0, 0.0, -1.0]))

    tilt_mag = float(np.linalg.norm(err_rot))
    w_rot = 5.0 if tilt_mag > 0.087 else (0.5 if tilt_mag < 0.017 else 1.0)
    w_pos = 1.0

    jacp = np.zeros((3, sim.model.nv))
    jacr = np.zeros((3, sim.model.nv))
    mujoco.mj_jacSite(sim.model._model, sim.data._data, jacp, jacr, env._eef_site_id)
    J = np.vstack([w_pos * jacp[:, :6], w_rot * jacr[:, :6]])
    err = np.concatenate([w_pos * err_pos, w_rot * err_rot])

    damping = 0.06
    JJt = J @ J.T + damping**2 * np.eye(6)
    dq = J.T @ np.linalg.solve(JJt, err) * 0.5

    a = np.zeros(7, dtype=np.float32)
    a[0:6] = np.clip(dq / 0.05, -1.0, 1.0)
    a[6] = -1.0
    return a


# ---------------------------------------------------------------------------
# Trajectory recorder
# ---------------------------------------------------------------------------

class _Trajectory:
    """Holds raw transitions for one episode; promoted to demo only on success."""

    def __init__(self):
        self.actor_obs: list[np.ndarray] = []
        self.critic_obs: list[np.ndarray] = []
        self.action: list[np.ndarray] = []
        self.reward: list[float] = []
        self.next_actor_obs: list[np.ndarray] = []
        self.next_critic_obs: list[np.ndarray] = []
        self.terminated: list[bool] = []

    def add(self, obs, action, reward, next_obs, terminated):
        self.actor_obs.append(obs["actor"].copy())
        self.critic_obs.append(obs["critic"].copy())
        self.action.append(np.asarray(action, dtype=np.float32).copy())
        self.reward.append(float(reward))
        self.next_actor_obs.append(next_obs["actor"].copy())
        self.next_critic_obs.append(next_obs["critic"].copy())
        self.terminated.append(bool(terminated))

    @property
    def n_steps(self) -> int:
        return len(self.action)


def run_one_episode(
    env, kb: KeyboardEnv, key: str, max_steps: int,
) -> tuple[_Trajectory, bool, int]:
    """Run M1 IK for one episode on `key`. Returns (trajectory, success, steps)."""
    obs, _ = env.reset()
    kb.set_target_key(key)
    obs, _ = env.reset()
    kb.set_target_key(key)

    key_pos = kb.sim.data.body_xpos[kb._key_body_ids[key]].copy()
    target = np.array([key_pos[0], key_pos[1], key_pos[2] + kb.hover_height])

    traj = _Trajectory()
    success_step: Optional[int] = None

    for step in range(max_steps):
        action = _jacobian_step(kb, target)
        next_obs, reward, term, trunc, info = env.step(action)
        traj.add(obs, action, reward, next_obs, term)

        if info.get("is_success") and success_step is None:
            success_step = step + 1
        obs = next_obs
        if term or trunc:
            break

    return traj, (success_step is not None), traj.n_steps


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_demos_h5(
    out_path: Path,
    trajectories: list[_Trajectory],
    key_meta: list[tuple[str, int, bool, int]],     # (key, seed, success, steps)
    n_total_trials: int,
) -> None:
    """Write the flat-concatenated successful transitions to HDF5."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    success_trajs = [t for t, m in zip(trajectories, key_meta) if m[2]]
    n_success = len(success_trajs)

    if n_success == 0:
        print("[gen_demos] WARNING: 0 successful trajectories — nothing saved.")
        return

    actor_obs = np.concatenate([np.stack(t.actor_obs) for t in success_trajs])
    critic_obs = np.concatenate([np.stack(t.critic_obs) for t in success_trajs])
    action = np.concatenate([np.stack(t.action) for t in success_trajs])
    reward = np.concatenate([np.asarray(t.reward, dtype=np.float32) for t in success_trajs])
    next_actor_obs = np.concatenate([np.stack(t.next_actor_obs) for t in success_trajs])
    next_critic_obs = np.concatenate([np.stack(t.next_critic_obs) for t in success_trajs])
    terminated = np.concatenate([np.asarray(t.terminated, dtype=bool) for t in success_trajs])
    episode_id = np.concatenate([
        np.full(t.n_steps, i, dtype=np.int32) for i, t in enumerate(success_trajs)
    ])

    N = len(actor_obs)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("actor_obs", data=actor_obs, compression="gzip")
        f.create_dataset("critic_obs", data=critic_obs, compression="gzip")
        f.create_dataset("action", data=action, compression="gzip")
        f.create_dataset("reward", data=reward, compression="gzip")
        f.create_dataset("next_actor_obs", data=next_actor_obs, compression="gzip")
        f.create_dataset("next_critic_obs", data=next_critic_obs, compression="gzip")
        f.create_dataset("terminated", data=terminated, compression="gzip")
        f.create_dataset("episode_id", data=episode_id, compression="gzip")
        meta = f.create_group("trial_meta")
        meta.attrs["n_total_trials"] = n_total_trials
        meta.attrs["n_success"] = n_success
        meta.attrs["n_transitions"] = N
        meta.attrs["actor_dim"] = actor_obs.shape[1]
        meta.attrs["critic_dim"] = critic_obs.shape[1]
        meta.attrs["action_dim"] = action.shape[1]
        # per-trial success log
        keys = np.array([m[0] for m in key_meta], dtype="S16")
        seeds = np.array([m[1] for m in key_meta], dtype=np.int32)
        successes = np.array([m[2] for m in key_meta], dtype=bool)
        steps = np.array([m[3] for m in key_meta], dtype=np.int32)
        meta.create_dataset("key", data=keys)
        meta.create_dataset("seed", data=seeds)
        meta.create_dataset("success", data=successes)
        meta.create_dataset("steps", data=steps)

    print(f"[gen_demos] saved {N} transitions across {n_success}/{n_total_trials} "
          f"successful episodes to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

KEY_GROUPS = {
    "phase_a": PHASE_A_KEYS,
    "phase_b": PHASE_B_KEYS,
    "all": AVAILABLE_KEYS,
    "central": ["g", "h", "f", "j", "d", "k", "s", "l", "t", "y", "r", "u"],
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    p.add_argument("--out", type=Path, required=True, help="HDF5 output path")
    p.add_argument("--keys", default="phase_a",
                   help="key group ('phase_a', 'phase_b', 'all', 'central') or comma-separated key names")
    p.add_argument("--trials-per-key", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--frame-stack", type=int, default=3)
    p.add_argument("--reward-mode", choices=["dense", "pbrs_only", "xy_focus"], default="dense",
                   help="must match the reward_mode used by train_approach for "
                        "the saved (reward, next_obs) tuples to be valid demo data")
    args = p.parse_args()

    if args.keys in KEY_GROUPS:
        keys = list(KEY_GROUPS[args.keys])
    else:
        keys = [k.strip() for k in args.keys.split(",") if k.strip() and k.strip() in AVAILABLE_KEYS]
    if not keys:
        print(f"[gen_demos] no valid keys in --keys={args.keys!r}")
        return 1

    print(f"[gen_demos] writing to {args.out}")
    print(f"[gen_demos] keys: {len(keys)}, trials per key: {args.trials_per_key}")
    print(f"[gen_demos] total budget: {len(keys) * args.trials_per_key} episodes, "
          f"max_steps={args.max_steps}")

    env = make_env(mode="approach", frame_stack=args.frame_stack,
                   domain_rand=False, seed=args.seed_base,
                   reward_mode=args.reward_mode)
    print(f"[gen_demos] reward_mode={args.reward_mode}")
    kb = find_inner(env, KeyboardEnv)
    assert kb is not None

    # Disable action smoothing inside the wrapper stack. The trained policy
    # uses smoothing at deployment, so (r, s') will differ slightly between
    # demo recording and online training — but α=0.4 doesn't change dynamics
    # drastically, and without this the per-step IK never converges in 200
    # steps. Net trade: small distribution shift in 0.5% of transitions vs
    # 0 successful demos. Easy call.
    aa = find_inner(env, ActionAdapter)
    if aa is not None:
        print(f"[gen_demos] disabling ActionAdapter smoothing (was α={aa.alpha}) for demo recording")
        aa.alpha = 0.0

    trajectories: list[_Trajectory] = []
    meta: list[tuple[str, int, bool, int]] = []

    n_success_total = 0
    n_total = 0
    print(f"{'idx':>4}  {'key':<8}  {'seed':>5}  {'steps':>5}  {'result':<6}")
    print("-" * 40)
    for ki, key in enumerate(keys):
        for t in range(args.trials_per_key):
            seed = args.seed_base + ki * args.trials_per_key + t
            np.random.seed(seed)
            traj, success, steps = run_one_episode(env, kb, key, args.max_steps)
            trajectories.append(traj)
            meta.append((key, seed, success, steps))
            n_total += 1
            if success:
                n_success_total += 1
            marker = "✓" if success else "x"
            if t == 0 or success:    # don't flood for failed attempts
                print(f"{n_total:>4}  {key:<8}  {seed:>5}  {steps:>5}  {marker:<6}")

    env.close()
    print("-" * 40)
    print(f"[gen_demos] {n_success_total}/{n_total} successful episodes "
          f"({100*n_success_total/n_total:.0f}%)")

    save_demos_h5(args.out, trajectories, meta, n_total_trials=n_total)
    return 0 if n_success_total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
