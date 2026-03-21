#!/usr/bin/env python3
"""
Demonstration Recorder for Imitation Learning

Subscribes to the MuJoCo bridge's action and observation topics, pairs them
by timestamp, and saves complete episode trajectories to HDF5 in a format
compatible with robomimic and the BC trainer.

Usage:
    # Make sure both RoverFlake2 teleop AND the MuJoCo bridge are running, then:
    python3 demo_recorder.py                         # interactive recording
    python3 demo_recorder.py --output my_demos.hdf5  # custom filename
    python3 demo_recorder.py --auto-segment          # auto-start new episode on env reset

Controls during recording:
    Enter   — Mark current episode as SUCCESS and start a new one
    Ctrl+C  — Finish recording and save the HDF5 file
    d       — DISCARD current episode (bad demo) and start fresh

HDF5 Schema (robomimic-compatible):
    demo_dataset.hdf5
    ├── data/
    │   ├── demo_0/
    │   │   ├── obs        (T, obs_dim)   float32
    │   │   ├── actions    (T, 7)         float32
    │   │   ├── rewards    (T,)           float32
    │   │   ├── dones      (T,)           bool
    │   │   └── phases     (T,)           int32     # 0=reach 1=grasp 2=lift 3=hold
    │   ├── demo_1/
    │   │   └── ...
    │   └── demo_N/
    └── metadata/
        ├── obs_dim        int
        ├── action_dim     int
        ├── control_freq   int  (20)
        ├── total_demos    int
        └── env_name       str  ("Lift")
"""

import os
import sys
import time
import argparse
import threading
import numpy as np
from datetime import datetime
from collections import deque

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray

# HDF5
try:
    import h5py
except ImportError:
    print("ERROR: h5py is required.  Install with:  pip install h5py")
    sys.exit(1)


# ============================================================================
# Episode buffer — collects (obs, action) pairs for one demonstration
# ============================================================================

class EpisodeBuffer:
    """Accumulates timesteps for a single demonstration episode."""

    def __init__(self):
        self.obs_list: list[np.ndarray] = []
        self.action_list: list[np.ndarray] = []
        self.reward_list: list[float] = []
        self.done_list: list[bool] = []
        self.phase_list: list[int] = []

    def add(self, obs: np.ndarray, action: np.ndarray,
            reward: float = 0.0, done: bool = False, phase: int = 0):
        self.obs_list.append(obs.copy())
        self.action_list.append(action.copy())
        self.reward_list.append(reward)
        self.done_list.append(done)
        self.phase_list.append(phase)

    def __len__(self):
        return len(self.obs_list)

    @property
    def is_empty(self):
        return len(self) == 0

    def to_arrays(self):
        return {
            'obs': np.array(self.obs_list, dtype=np.float32),
            'actions': np.array(self.action_list, dtype=np.float32),
            'rewards': np.array(self.reward_list, dtype=np.float32),
            'dones': np.array(self.done_list, dtype=bool),
            'phases': np.array(self.phase_list, dtype=np.int32),
        }


# ============================================================================
# Recorder Node
# ============================================================================

class DemoRecorderNode(Node):
    """
    Subscribes to:
        /mujoco/actions        — 7-float normalized action (from bridge's Jacobian IK)
        /mujoco/observations   — flat obs array (from cartesian_control_ros bridge)

    Pairs the latest action with each incoming observation and buffers them into
    episodes.  The user presses Enter to finalize an episode, Ctrl+C to save.
    """

    ACTION_TOPIC = "/mujoco/actions"
    OBS_TOPIC = "/mujoco/observations"

    def __init__(self, output_path: str, auto_segment: bool = False):
        super().__init__('demo_recorder')

        self.output_path = output_path
        self.auto_segment = auto_segment

        # State
        self._latest_action: np.ndarray | None = None
        self._latest_obs: np.ndarray | None = None
        self._pending_action = False
        self._pending_obs = False
        self._lock = threading.Lock()
        self._current_episode = EpisodeBuffer()
        self._saved_episodes: list[dict] = []
        self._recording = True
        self._step_count = 0

        # ROS2 subscriptions
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Float64MultiArray, self.ACTION_TOPIC, self._action_cb, qos)
        self.create_subscription(Float64MultiArray, self.OBS_TOPIC, self._obs_cb, qos)

        self.get_logger().info("="*60)
        self.get_logger().info("  DEMO RECORDER — Imitation Learning Data Collection")
        self.get_logger().info("="*60)
        self.get_logger().info(f"  Output:  {self.output_path}")
        self.get_logger().info(f"  Action:  {self.ACTION_TOPIC}")
        self.get_logger().info(f"  Obs:     {self.OBS_TOPIC}")
        self.get_logger().info("")
        self.get_logger().info("  Controls:")
        self.get_logger().info("    Enter  — Save episode & start new one")
        self.get_logger().info("    d      — Discard current episode")
        self.get_logger().info("    Ctrl+C — Finish and write HDF5")
        self.get_logger().info("="*60)

        # Input thread for keyboard controls
        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._input_thread.start()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _action_cb(self, msg: Float64MultiArray):
        with self._lock:
            self._latest_action = np.array(msg.data, dtype=np.float32)
            self._pending_action = True
        self._try_commit_sample()

    def _obs_cb(self, msg: Float64MultiArray):
        with self._lock:
            self._latest_obs = np.array(msg.data, dtype=np.float32)
            self._pending_obs = True
        self._try_commit_sample()

    def _try_commit_sample(self):
        """Buffer one sample only when both a fresh obs and action are present."""
        with self._lock:
            if not (self._pending_action and self._pending_obs):
                return
            if self._latest_action is None or self._latest_obs is None:
                return
            action = self._latest_action.copy()
            obs = self._latest_obs.copy()
            self._pending_action = False
            self._pending_obs = False

        # Determine phase from obs (last 4 elements are the phase one-hot)
        phase_onehot = obs[-4:]
        phase = int(np.argmax(phase_onehot))

        self._current_episode.add(obs=obs, action=action, reward=0.0, done=False, phase=phase)
        self._step_count += 1

        # Print progress every 20 steps (1 second at 20 Hz)
        if self._step_count % 20 == 0:
            n_saved = len(self._saved_episodes)
            n_cur = len(self._current_episode)
            phase_names = ['Reach', 'Grasp', 'Lift', 'Hold']
            p = phase_names[phase] if phase < len(phase_names) else '???'
            print(f"\r  [Recording] Saved: {n_saved} demos | "
                  f"Current: {n_cur} steps | Phase: {p}     ", end='', flush=True)

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def _finalize_episode(self, discard: bool = False):
        """Save current episode to list or discard it."""
        if self._current_episode.is_empty:
            self.get_logger().info("  (empty episode, nothing to save)")
            return

        if discard:
            n = len(self._current_episode)
            self._current_episode = EpisodeBuffer()
            print(f"\n  DISCARDED episode ({n} steps)")
            return

        arrays = self._current_episode.to_arrays()
        # Mark the last timestep as done
        arrays['dones'][-1] = True

        self._saved_episodes.append(arrays)
        n = len(arrays['obs'])
        total = len(self._saved_episodes)
        print(f"\n  SAVED episode #{total} ({n} steps, "
              f"{n / 20.0:.1f}s at 20 Hz)")

        self._current_episode = EpisodeBuffer()

    def _input_loop(self):
        """Background thread reading stdin for episode control."""
        import termios, tty, select
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while self._recording:
                if select.select([sys.stdin], [], [], 0.2)[0]:
                    ch = sys.stdin.read(1)
                    if ch == '\n' or ch == '\r':
                        self._finalize_episode(discard=False)
                    elif ch.lower() == 'd':
                        self._finalize_episode(discard=True)
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    # ------------------------------------------------------------------
    # HDF5 saving
    # ------------------------------------------------------------------

    def save_hdf5(self):
        """Write all saved episodes to HDF5."""
        # Finalize current in-progress episode (if any)
        if not self._current_episode.is_empty:
            print("\n  Finalizing current episode before saving...")
            self._finalize_episode(discard=False)

        if not self._saved_episodes:
            print("\n  No demonstrations recorded. Nothing to save.")
            return

        total = len(self._saved_episodes)
        print(f"\n  Writing {total} demonstrations to {self.output_path} ...")

        with h5py.File(self.output_path, 'w') as f:
            data_grp = f.create_group('data')

            for i, ep in enumerate(self._saved_episodes):
                demo_grp = data_grp.create_group(f'demo_{i}')
                for key, arr in ep.items():
                    demo_grp.create_dataset(key, data=arr, compression='gzip')

            # Metadata
            meta = f.create_group('metadata')
            sample = self._saved_episodes[0]
            meta.attrs['obs_dim'] = sample['obs'].shape[1]
            meta.attrs['action_dim'] = sample['actions'].shape[1]
            meta.attrs['control_freq'] = 20
            meta.attrs['total_demos'] = total
            meta.attrs['env_name'] = 'Lift'
            meta.attrs['robot'] = 'Rover2026'
            meta.attrs['controller'] = 'JOINT_VELOCITY'
            meta.attrs['recorded_at'] = datetime.now().isoformat()

        total_steps = sum(len(ep['obs']) for ep in self._saved_episodes)
        print(f"  Done! {total} demos, {total_steps} total steps "
              f"({total_steps / 20.0:.1f}s of data)")

    def destroy_node(self):
        self._recording = False
        super().destroy_node()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Record demonstrations for Imitation Learning")
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output HDF5 file path (default: demos_YYYYMMDD_HHMMSS.hdf5)')
    parser.add_argument('--auto-segment', action='store_true',
                        help='Auto-start new episode on environment reset')
    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        demo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demos')
        os.makedirs(demo_dir, exist_ok=True)
        args.output = os.path.join(demo_dir, f'demos_{ts}.hdf5')

    rclpy.init()
    node = DemoRecorderNode(output_path=args.output, auto_segment=args.auto_segment)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_hdf5()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
