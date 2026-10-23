# Build with 'docker compose build'
# Create and enter container with 'docker compose run rover bash'
# Enter an existing running container with 'docker compose exec rover bash'
# if you have anymore questions ask Aaron

# Official base image in docs. Jazzy == Ubuntu 24.04 + Python 3.12.
# Migrated from osrf/ros:humble-desktop (Ubuntu 22.04 / Py3.10) 2026-10.
FROM osrf/ros:jazzy-desktop

# consistency with setup files
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV ROVERFLAKE_ROOT=/LearnFlake
ENV ROS_DISTRO=jazzy
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# venv on top of the Jazzy system Python (24.04 is PEP 668 externally-managed).
# --system-site-packages below keeps rclpy (installed under /opt/ros/jazzy and
# put on PYTHONPATH by setup.bash) importable from inside the venv.
ENV RL_VENV=/opt/rl-venv
ENV PATH=$RL_VENV/bin:$PATH

# install base depenencies (recommend not to change if you want to add niche deps)
RUN apt-get update && apt-get install -y \
    tzdata \
    sudo \
    curl \
    git \
    vim \
    btop \
    wget \
    bash \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 \
    build-essential \
    cmake \
    python3-colcon-common-extensions \
    ros-jazzy-rmw-cyclonedds-cpp \
    libegl1 \
    libglvnd0 \
    libgles2 \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# common ROS package dockerdeps (use this for niche deps)
RUN apt-get update && apt-get install -y \
    ros-jazzy-urdf \
    ros-jazzy-image-transport \
    ros-jazzy-cv-bridge \
    ros-jazzy-xacro \
    ros-jazzy-rosidl-default-generators \
    ros-jazzy-rosidl-default-runtime \
    && rm -rf /var/lib/apt/lists/*

# copy root into container
WORKDIR $ROVERFLAKE_ROOT
COPY . $ROVERFLAKE_ROOT

# ROS workspace dependencies (serial + moteus_msgs are vendored, skip their keys).
RUN rosdep update --rosdistro jazzy \
    && . /opt/ros/jazzy/setup.sh \
    && rosdep install --from-paths src --ignore-src -r \
       --skip-keys="serial moteus_msgs" -y 2>/dev/null || true

# Python RL env. Lives in a venv (24.04 blocks system pip). Pins come from
# docker/rl_env_freeze.txt, the snapshot of the env that produced M4 (98.6%).
# torch is cu128 for RTX 5080 / sm_120 (Blackwell) -- see TRACKER on the Blackwell story.
RUN python3 -m venv $RL_VENV --system-site-packages \
    && $RL_VENV/bin/pip install --upgrade pip setuptools wheel \
    && $RL_VENV/bin/pip install --no-cache-dir torch==2.11.0 \
       --index-url https://download.pytorch.org/whl/cu128 \
    && $RL_VENV/bin/pip install --no-cache-dir \
       "numpy==1.26.4" "scipy==1.15.3" "numba==0.65.1" \
       "mujoco==3.9.0" "gymnasium==1.3.0" \
       "stable-baselines3>=2.7,<3" "sb3-contrib>=2.7,<3" \
       "dm-control==1.0.41" "h5py==3.16.0" "tqdm>=4.66" \
       "tensorboard==2.20.0" "PyYAML>=6.0" \
       "mink==0.0.5" "qpsolvers==4.12.0" \
       "pytest>=7.4" "pytest-cov>=4.1" "coverage>=7.10" "ruff>=0.4" \
    && $RL_VENV/bin/pip install --no-cache-dir -e src/external_pkgs/RoboSuite \
    && $RL_VENV/bin/pip install --no-cache-dir -e . \
    && rm -rf /root/.cache/pip

# copy and set entrypoint (sources ROS + venv, then colcon-builds the workspace)
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Mujoco rendering default (can be overridden at runtime via MUJOCO_GL)
ENV MUJOCO_GL=glfw

# Nvidia GPU support (this thing is so hard to integrate)
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia

# changed with COPY
ENTRYPOINT ["/entrypoint.sh"]

CMD ["/bin/bash"]
