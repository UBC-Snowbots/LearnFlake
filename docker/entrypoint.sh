#!/bin/bash
set -e

export ROVERFLAKE_ROOT="${ROVERFLAKE_ROOT:-/LearnFlake}"

# Normalize known rover env scripts to Unix line endings at container start.
for f in \
  "$ROVERFLAKE_ROOT/setup_scripts/rover_env/rover_env_common.sh" \
  "$ROVERFLAKE_ROOT/setup_scripts/rover_env/rover_env_vars.sh" \
  "$ROVERFLAKE_ROOT/setup_scripts/rover_env/rover_aliases_common.sh"
do
  if [ -f "$f" ]; then
    perl -pi -e 's/\r$//' "$f" || true
  fi
done

# Ensure interactive root shells always load rover aliases / env helpers (glfw-on, etc).
if [ -f "/root/.bashrc" ]; then
  if ! grep -q "### LearnFlake Rover Env ###" /root/.bashrc; then
    cat >> /root/.bashrc <<'EOF'
### LearnFlake Rover Env ###
export ROVERFLAKE_ROOT=${ROVERFLAKE_ROOT:-/LearnFlake}
if [ -f "${ROVERFLAKE_ROOT}/setup_scripts/rover_env/rover_env_common.sh" ]; then
  source "${ROVERFLAKE_ROOT}/setup_scripts/rover_env/rover_env_common.sh"
fi
### End LearnFlake Rover Env ###
EOF
  fi
fi

# source system ROS 2
if [ -f "/opt/ros/humble/setup.bash" ]; then
  source /opt/ros/humble/setup.bash
fi

# If workspace install/setup.bash doesn’t exist yet, do a build
if [ ! -f "$ROVERFLAKE_ROOT/install/setup.bash" ]; then
  echo "No install/setup.bash found, running colcon build..."
  colcon build  # should automatically use --symlink-install
fi

# Source workspace overlay
if [ -f "$ROVERFLAKE_ROOT/install/setup.bash" ]; then
  source "$ROVERFLAKE_ROOT/install/setup.bash"
fi

exec "$@"

