#!/bin/bash
set -e

# source system ROS 2 (distro-generic; defaults to jazzy)
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
if [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
fi

# activate the RL python venv (ROS already put rclpy on PYTHONPATH above)
if [ -f "${RL_VENV:-/opt/rl-venv}/bin/activate" ]; then
  source "${RL_VENV:-/opt/rl-venv}/bin/activate"
fi

# If workspace install/setup.bash doesn't exist yet, do a build
if [ ! -f "$ROVERFLAKE_ROOT/install/setup.bash" ]; then
  echo "No install/setup.bash found, running colcon build..."
  # only the ROS packages under src/external_pkgs (the repo root's setup.py is the RL package, not ROS)
  if ! colcon build --symlink-install --base-paths src/external_pkgs \
       --packages-up-to serial moteus_msgs moteus_control; then
    echo "WARNING: initial colcon build failed. Container stays up so you can debug." >&2
  fi
fi

# Source workspace overlay
if [ -f "$ROVERFLAKE_ROOT/install/setup.bash" ]; then
  source "$ROVERFLAKE_ROOT/install/setup.bash"
fi

exec "$@"
