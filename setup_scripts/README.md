# LearnFlake setup

LearnFlake runs on **Ubuntu 24.04 + ROS 2 Jazzy + Python 3.12** (migrated from
22.04 / Humble / 3.10 in 2026-10). Docker is the primary and recommended path.

## Docker (recommended)

From the repo root:

```bash
docker compose build rover_gpu          # GPU (RTX 5080 / Blackwell, torch cu128)
docker compose run --rm rover_gpu bash  # drops you in /LearnFlake with ROS + venv ready
```

CPU-only or Windows/WSL: use `Dockerfile.ubuntu` via
`docker compose -f docker-compose.ubuntu.yml build rover_rl`.

The image sources `/opt/ros/jazzy/setup.bash` and activates the RL venv at
`/opt/rl-venv` on entry (see `docker/entrypoint.sh`). Python pins are in
`docker/rl_env_freeze.txt`.

## Native host setup (use the team CLI, not the scripts here)

The ROS + system setup is done by the team tool **rovercli**
(https://github.com/UBC-Snowbots/rovercli), the same tool RoverFlake2 uses to
install Jazzy. It replaces the per-distro `install-ros2-*.sh` scripts in this
folder.

```bash
git clone https://github.com/UBC-Snowbots/rovercli.git
cd rovercli
python3 main.py setup --distro jazzy --ros-version desktop --git-protocol https --cd-roverflake n
```

Then, from the LearnFlake root, create the Python RL env (rovercli does not
install the ML stack):

```bash
python3 -m venv .venv --system-site-packages   # keeps rclpy from /opt/ros/jazzy importable
source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # GPU / Blackwell
pip install -e src/external_pkgs/RoboSuite
pip install -e .
```

## Deprecated scripts

The following were copied from the old (Humble-era) RoverFlake setup and are
kept only for reference. Do not run them on Jazzy; use rovercli instead:

- `install-ros2-humble.sh` - installs ROS 2 Humble (wrong distro for 24.04)
- `setup_everything_common.sh` - Humble + rover-hardware deps; superseded by rovercli
- `for_sims/install_gazebo_stuff.sh` - removed; LearnFlake uses MuJoCo, not Gazebo
