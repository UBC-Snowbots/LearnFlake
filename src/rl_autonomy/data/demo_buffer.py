"""Load HDF5 demonstration trajectories into RLPD's SymmetricReplayBuffer.

The HDF5 schema produced by `tools.gen_demos`:
    /actor_obs        (N, 108)  float32   — already-stacked actor obs (k=3 × 36-D)
    /critic_obs       (N, 38)   float32   — privileged critic obs
    /action           (N, 7)    float32   — raw policy action [-1, 1]
    /reward           (N,)      float32
    /next_actor_obs   (N, 108)  float32
    /next_critic_obs  (N, 38)   float32
    /terminated       (N,)      bool
    /episode_id       (N,)      int32
    /trial_meta       group     attrs/datasets describing the recording run

Note on normalization:
  The obs in the HDF5 was written through `ObsAdapter` with `training=True`
  but a freshly-initialized RunningMeanStd (count starts at ε). So the saved
  values are *approximately* raw obs (clipped at ±10 by the ObsAdapter).

  At training load time, we have two reasonable options:
    (a) Push the saved obs into the buffer as-is. The trained agent will
        re-normalize via its own ObsAdapter when sampling online transitions;
        demos won't track the agent's evolving RMS, but in practice the RMS
        converges quickly and the drift is small.
    (b) Re-normalize the demo obs at load time using the agent's *current*
        ObsAdapter RMS. This matches the agent's training scale exactly at
        load time. Drifts the same way as online experience.

  We use (b) because it's the principled fix and adds ~3 lines. The agent
  should be allowed a brief RMS warmup before calling this loader, so the
  current RMS is a reasonable approximation of the converged distribution.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ..algos.replay_buffer import ReplayBuffer


def load_demos_into_buffer(
    h5_path: str | Path,
    demo_buffer: ReplayBuffer,
    obs_adapter=None,
    re_normalize: bool = True,
) -> int:
    """Load demos from HDF5 into the supplied ``demo_buffer``.

    Args:
        h5_path: path to the gen_demos.py output file.
        demo_buffer: a ReplayBuffer (typically ``SymmetricReplayBuffer.demos``)
            with matching actor_dim / critic_dim / action_dim.
        obs_adapter: optional ObsAdapter whose ``.rms`` will be used to
            re-normalize the demo observations before pushing into the buffer.
            Pass ``None`` to load raw (no re-normalization).
        re_normalize: when True and ``obs_adapter`` is given, re-normalize.

    Returns:
        Number of transitions loaded.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"demo file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        actor = f["actor_obs"][:]
        critic = f["critic_obs"][:]
        action = f["action"][:]
        reward = f["reward"][:]
        n_actor = f["next_actor_obs"][:]
        n_critic = f["next_critic_obs"][:]
        terminated = f["terminated"][:]
        meta_attrs = dict(f["trial_meta"].attrs)

    N = actor.shape[0]
    if N == 0:
        return 0

    # Shape sanity
    if actor.shape[1] != demo_buffer.actor_obs.shape[1]:
        raise ValueError(
            f"demo actor_obs has dim {actor.shape[1]} but buffer expects "
            f"{demo_buffer.actor_obs.shape[1]}"
        )
    if critic.shape[1] != demo_buffer.critic_obs.shape[1]:
        raise ValueError(
            f"demo critic_obs has dim {critic.shape[1]} but buffer expects "
            f"{demo_buffer.critic_obs.shape[1]}"
        )
    if action.shape[1] != demo_buffer.action.shape[1]:
        raise ValueError(
            f"demo action has dim {action.shape[1]} but buffer expects "
            f"{demo_buffer.action.shape[1]}"
        )

    # Optionally re-normalize obs through the agent's current RMS so the
    # demo scale matches the agent's input expectation.
    if re_normalize and obs_adapter is not None:
        rms = obs_adapter.rms
        # Only meaningful if rms has been initialized to non-default values.
        if rms.count > 100:    # arbitrary "has been warmed up" check
            actor = rms.normalize(actor)
            n_actor = rms.normalize(n_actor)
        # Critic obs is NOT normalized by the agent (per TRACKER §9), so we
        # leave critic / n_critic untouched here.

    for i in range(N):
        demo_buffer.add(
            actor_obs=actor[i],
            critic_obs=critic[i],
            action=action[i],
            reward=float(reward[i]),
            next_actor_obs=n_actor[i],
            next_critic_obs=n_critic[i],
            terminated=bool(terminated[i]),
        )

    return N
