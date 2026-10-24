# 2026-10-23 — Migrated to Ubuntu 24.04 / ROS 2 Jazzy / Python 3.12

Was: Ubuntu 22.04 / ROS 2 Humble / Python 3.10. Branch `prnv/jazzy-24.04` (off main).

**What changed:**
- `Dockerfile`: base `osrf/ros:humble-desktop` -> `osrf/ros:jazzy-desktop`;
  `ros-humble-*` -> `ros-jazzy-*`; `ROS_DISTRO=jazzy`. The RL stack is now baked
  into a venv at `/opt/rl-venv` (24.04 is PEP 668 externally-managed; the venv is
  `--system-site-packages` so rclpy from `/opt/ros/jazzy` stays importable). Same
  pins as `docker/rl_env_freeze.txt` (torch 2.11.0+cu128, numpy 1.26.4, mujoco
  3.9.0). The old `setup_everything_common.sh` call is gone; the image gets ROS
  from the osrf base and ML deps from the venv step.
- `Dockerfile.ubuntu`: `ubuntu:22.04` -> `ubuntu:24.04`; ROS install now uses the
  `ros2-apt-source` .deb (same method as rovercli `install-ros2.sh`); 3.12 pins.
- `docker/entrypoint.sh`: sources `/opt/ros/${ROS_DISTRO}` and activates the venv.
- Native setup now points at the team CLI **rovercli** (`setup --distro jazzy`),
  see `setup_scripts/README.md`. Humble scripts marked DEPRECATED; dead Gazebo
  script removed (we use MuJoCo).
- `.github/workflows/install_and_build.yaml`: CI on `ubuntu-24.04` via rovercli,
  modelled on RoverFlake2's workflow (committed, not run here).

**Note:** mujoco is 3.9.0 (the actual pin in `rl_env_freeze.txt`), not the 3.6.0
mentioned in the pyproject comment. If physics regresses, this is the first thing
to check.

# Added images

Now we need a way to know what changed in the new image (e.g. other than doing a ton of pip installs, you've now added a venv and the person who pulls your image doesn't know that)

### Useful commands
```bash
    # lets you see the dependency-related changes 
    docker history <service>:<tag> 
```

### Simple Documentation
Otherwise you can just add your changes to this file:
```bash
    # Write down the name of the image you pushed and the commands you executed here in order (e.g. python -m venv rl)
```

# 2026-06-01 — RL stack baked into `learnflake:gpu` (FINALLY)

**Problem this fixes:** the torch / mujoco / robosuite / dm-control stack used by
`rl_autonomy` was only ever **pip-installed at runtime into the `rover_gpu`
container's writable layer** — it was never committed to an image. Any
`docker compose up -d rover_gpu` (or `down`) that *recreates* the container
silently wiped the entire training environment. This bit us once on 2026-06-01
(a `compose up` recreated the container and torch vanished).

**What was done (in order):**
```bash
# inside rover_gpu (python3 -> /usr/bin/python3.10, system site-packages, run as root)
pip install --upgrade pip setuptools wheel
pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128   # -> torch 2.11.0+cu128 (sm_120 / Blackwell, RTX 5070 Ti)
# torch first failed: could not uninstall apt's distutils sympy 1.9. Fix:
pip install --ignore-installed sympy mpmath networkx typing-extensions
pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128   # re-run, succeeds
pip install "numpy>=1.24,<3" scipy "mujoco>=3.6.0" "gymnasium>=1.0,<2" dm-control h5py tqdm tensorboard PyYAML pytest pytest-cov ruff
pip install termcolor numba "mink==0.0.5" "qpsolvers[quadprog]>=4.3.1" Pillow opencv-python-headless pynput   # robosuite runtime deps
pip install -e src/external_pkgs/RoboSuite --no-deps      # vendored robosuite 1.5.1
pip install -e . --no-deps                                # rl_autonomy editable
# numba pinned numpy down to 1.26.4 (fine). SB3 is NOT needed — RLPDSAC is from-scratch PyTorch (TRACKER §21.1).
# Then, from the HOST:
docker commit rover_gpu learnflake:gpu     # bakes the env into the image (10.4GB -> 25.3GB)
```
Exact pinned versions captured in `docker/rl_env_freeze.txt`. Verified: 55/55
tests pass, `torch.cuda.is_available()` True, GPU matmul on sm_120 works.

**If the env ever disappears again:** `pip install -r docker/rl_env_freeze.txt`
inside the container won't work directly (torch needs the cu128 index) — follow
the command block above, or just `docker compose up -d rover_gpu` now that the
env is in the image.
