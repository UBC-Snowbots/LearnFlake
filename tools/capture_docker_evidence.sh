#!/usr/bin/env bash
set -euo pipefail

# Run this on a machine where Docker Desktop / WSL2 are actually available.
# It does not record video by itself. It runs the exact commands you should
# screen-record in a terminal window.

REPO_ROOT="${1:-$(pwd)}"
cd "$REPO_ROOT"

echo "=== Session 1: Stack Online ==="
docker compose -f docker-compose.ubuntu.yml up -d rover_cpu
docker compose -f docker-compose.ubuntu.yml logs --tail=80 rover_cpu

echo
echo "=== Session 2: WSL2 Health Checks ==="
echo "DISPLAY=$DISPLAY"
docker compose -f docker-compose.ubuntu.yml ps
docker compose -f docker-compose.ubuntu.yml exec rover_cpu bash -lc "source /opt/ros/humble/setup.bash && python - <<'PY'
import robosuite, torch
print('robosuite ok')
print('torch ok')
PY"

echo
echo "Record these terminal sessions with OBS / Xbox Game Bar / another screen recorder."
