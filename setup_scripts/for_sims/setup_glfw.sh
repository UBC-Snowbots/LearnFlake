#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   source setup_scripts/for_sims/setup_glfw.sh
#   bash setup_scripts/for_sims/setup_glfw.sh
#
# Behavior:
# - On host: grants X11 access for root-in-container via xhost.
# - In container: picks a reachable DISPLAY and sets MUJOCO_GL to glfw (or egl fallback).
# - Runs a GLFW smoke test when possible.

GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m"

info() { echo -e "${GREEN}[info]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
err() { echo -e "${RED}[err]${NC} $*"; }

in_container() {
  [[ -f "/.dockerenv" ]]
}

has_local_x_socket() {
  local disp="$1"
  [[ "$disp" =~ ^:([0-9]+)(\..*)?$ ]] || return 1
  local num
  num="$(echo "$disp" | sed -E 's/^:([0-9]+).*/\1/')"
  [[ -S "/tmp/.X11-unix/X${num}" ]]
}

has_tcp_x_display() {
  local disp="$1"
  [[ "$disp" =~ ^[^:]+:[0-9]+(\.[0-9]+)?$ ]] || return 1

  local host screen port
  host="${disp%%:*}"
  screen="${disp##*:}"
  screen="${screen%%.*}"
  port=$((6000 + screen))

  python - "$host" "$port" <<'PY'
import socket, sys
host = sys.argv[1]
port = int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=0.5):
        pass
    print("ok")
except OSError:
    print("no")
PY
}

is_display_reachable() {
  local disp="$1"
  [[ -n "$disp" ]] || return 1

  if has_local_x_socket "$disp"; then
    return 0
  fi

  local tcp_res
  tcp_res="$(has_tcp_x_display "$disp" || true)"
  [[ "$tcp_res" == "ok" ]]
}

glfw_smoke_test() {
  python - <<'PY'
import glfw
ok = glfw.init()
print("glfw.init=", int(bool(ok)))
if ok:
    w = glfw.create_window(100, 100, "glfw-smoke", None, None)
    print("glfw.window=", int(bool(w)))
    if w:
        glfw.destroy_window(w)
    glfw.terminate()
PY
}

setup_on_host() {
  info "Host mode detected."
  if ! command -v xhost >/dev/null 2>&1; then
    err "xhost not found. Install x11-xserver-utils (Linux) or run from WSL with WSLg."
    return 1
  fi

  if [[ -z "${DISPLAY:-}" ]]; then
    warn "DISPLAY is empty on host. X11 may not be active."
  else
    info "Host DISPLAY=${DISPLAY}"
  fi

  # Secure enough for root user inside local Docker container.
  xhost +SI:localuser:root >/dev/null
  info "Granted X11 access to local root user (for container root)."

  cat <<'EOF'

Next:
1) Start container with X11 socket mounted:
   docker compose -f docker-compose.ubuntu.yml run --rm rover_rl bash
2) Inside container, run:
   source /LearnFlake/setup_scripts/for_sims/setup_glfw.sh

EOF
}

setup_in_container() {
  info "Container mode detected."

  local candidates=()
  if [[ -n "${DISPLAY:-}" ]]; then
    candidates+=("${DISPLAY}")
  fi

  # Prefer local unix sockets when mounted.
  for n in 0 1; do
    if [[ -S "/tmp/.X11-unix/X${n}" ]]; then
      candidates+=(":${n}")
    fi
  done

  candidates+=("host.docker.internal:0.0")

  if [[ -r /etc/resolv.conf ]]; then
    local ns
    ns="$(awk '/^nameserver/ {print $2; exit}' /etc/resolv.conf || true)"
    if [[ -n "$ns" ]]; then
      candidates+=("${ns}:0.0")
    fi
  fi

  local chosen=""
  for d in "${candidates[@]}"; do
    if is_display_reachable "$d"; then
      chosen="$d"
      break
    fi
  done

  if [[ -n "$chosen" ]]; then
    export DISPLAY="$chosen"
    export MUJOCO_GL="glfw"
    info "Using DISPLAY=${DISPLAY}"
    info "Set MUJOCO_GL=${MUJOCO_GL}"

    local smoke
    smoke="$(glfw_smoke_test 2>&1 || true)"
    echo "$smoke"
    if echo "$smoke" | rg -q "glfw.init=\s*1"; then
      info "GLFW smoke test passed."
    else
      warn "GLFW smoke test failed; falling back to MUJOCO_GL=egl."
      export MUJOCO_GL="egl"
      warn "Current env: DISPLAY=${DISPLAY} MUJOCO_GL=${MUJOCO_GL}"
      warn "From host, run: xhost +SI:localuser:root"
    fi
  else
    warn "No reachable X display found."
    unset DISPLAY || true
    export MUJOCO_GL="egl"
    warn "Set MUJOCO_GL=egl for headless mode."
    warn "From host, run: xhost +SI:localuser:root and ensure /tmp/.X11-unix is mounted."
  fi

  cat <<EOF

Use this shell env:
  DISPLAY=${DISPLAY:-}
  MUJOCO_GL=${MUJOCO_GL}

Tip: if you executed this script with 'bash ...', vars apply only to this process.
     Use 'source setup_scripts/for_sims/setup_glfw.sh' to persist in your shell.
EOF
}

if in_container; then
  setup_in_container
else
  setup_on_host
fi
