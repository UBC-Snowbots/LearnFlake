# all our common aliases. 
alias rosbuild="cd ${ROVERFLAKE_ROOT} && colcon build --symlink-install"
alias rosclean="cd ${ROVERFLAKE_ROOT} && rm -rf build/ install/ log/"
alias glfwfix="source ${ROVERFLAKE_ROOT}/setup_scripts/for_sims/setup_glfw.sh"

_glfw_on() {
  source "${ROVERFLAKE_ROOT}/setup_scripts/for_sims/setup_glfw.sh"
  export SIM_HEADLESS=0

  if [[ "${MUJOCO_GL:-}" == "glfw" ]]; then
    export FORCE_GUI=1
    echo "[glfw-on] Ready: MUJOCO_GL=glfw FORCE_GUI=1 SIM_HEADLESS=0 DISPLAY=${DISPLAY:-}"
  else
    unset FORCE_GUI
    echo "[glfw-on] DISPLAY not reachable yet, staying headless (MUJOCO_GL=${MUJOCO_GL:-unset})."
    echo "[glfw-on] If needed, run on host first: bash ${ROVERFLAKE_ROOT}/setup_scripts/for_sims/setup_glfw.sh"
  fi
}

_glfw_off() {
  export MUJOCO_GL=egl
  export SIM_HEADLESS=1
  unset FORCE_GUI
  echo "[glfw-off] Headless mode set: MUJOCO_GL=egl SIM_HEADLESS=1"
}

_glfw_status() {
  echo "DISPLAY=${DISPLAY:-}"
  echo "MUJOCO_GL=${MUJOCO_GL:-}"
  echo "FORCE_GUI=${FORCE_GUI:-}"
  echo "SIM_HEADLESS=${SIM_HEADLESS:-}"
}

alias glfw-on="_glfw_on"
alias glfw-off="_glfw_off"
alias glfw-status="_glfw_status"



# Keyboard layouts for the weirdos
alias colemak="setxkbmap us -variant colemak"
alias qwfpgj="setxkbmap us"
