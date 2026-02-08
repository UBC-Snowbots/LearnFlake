# Docker Setup for Windows (Ubuntu 22.04 LTS)

This guide covers running LearnFlake (ROS2 Humble + RoboSuite) on Windows using Docker Desktop with WSL2.

## Prerequisites

### 1. Windows Requirements
- Windows 10 version 21H2+ or Windows 11
- WSL2 enabled with a Linux distribution installed
- Docker Desktop for Windows (WSL2 backend)

### 2. Install Docker Desktop
1. Download from [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. During installation, ensure "Use WSL 2 instead of Hyper-V" is selected
3. After installation, go to Settings > Resources > WSL Integration
4. Enable integration with your WSL2 distro

### 3. (Optional) NVIDIA GPU Support
If you have an NVIDIA GPU and want CUDA acceleration:
1. Install the latest NVIDIA drivers on Windows
2. In WSL2, install NVIDIA Container Toolkit:
```bash
# Inside WSL2 terminal
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```
3. Restart Docker Desktop

---

## Quick Start

### Build the Image
```bash
docker compose -f docker-compose.ubuntu.yml build
```

### Run a Container

**CPU only (recommended for most use cases):**
```bash
docker compose -f docker-compose.ubuntu.yml run --rm rover_cpu bash
```

**With GPU support:**
```bash
docker compose -f docker-compose.ubuntu.yml run --rm rover_gpu bash
```

**For RL training:**
```bash
docker compose -f docker-compose.ubuntu.yml run --rm rover_rl bash
```

---

## Available Services

| Service | Description | GPU | GUI |
|---------|-------------|-----|-----|
| `rover_base` | Minimal container, headless | No | No |
| `rover_cpu` | CPU with WSLg GUI support | No | Yes |
| `rover_rl` | RL training focused | No | Yes |
| `rover_gpu` | NVIDIA GPU with CUDA | Yes | Yes |
| `rover_dev` | Development with exposed ports | No | Yes |

---

## Common Workflows

### Start a detached container
```bash
docker compose -f docker-compose.ubuntu.yml up -d rover_cpu
```

### Enter a running container
```bash
docker compose -f docker-compose.ubuntu.yml exec rover_cpu bash
```

### Stop all containers
```bash
docker compose -f docker-compose.ubuntu.yml down
```

### Rebuild without cache
```bash
docker compose -f docker-compose.ubuntu.yml build --no-cache
```

---

## Inside the Container

### Source ROS2 environment (auto-done by entrypoint)
```bash
source /opt/ros/humble/setup.bash
```

### Build the workspace
```bash
cd /LearnFlake
colcon build --symlink-install
source install/setup.bash
```

### Run RoboSuite demo
```bash
cd /LearnFlake/src/external_pkgs/RoboSuite
python -m robosuite.demos.demo_random_action
```

### Run RL training
```bash
cd /LearnFlake/src/rl_autonomy
python scripts/run_ppo.py
```

---

## GUI Applications (WSLg)

Windows 11 and Windows 10 21H2+ include WSLg (Windows Subsystem for Linux GUI) which allows running Linux GUI apps seamlessly.

### Test GUI is working
```bash
# Inside the container
apt-get update && apt-get install -y x11-apps
xeyes  # Should show animated eyes
```

### MuJoCo rendering modes
- `MUJOCO_GL=egl` - Headless rendering (default, fastest)
- `MUJOCO_GL=glfw` - Window-based rendering (requires GUI)
- `MUJOCO_GL=osmesa` - Software rendering (slowest, no GPU)

To change rendering mode:
```bash
export MUJOCO_GL=glfw
python your_script.py
```

---

## Troubleshooting

### "Cannot connect to X server"
Ensure WSLg is working:
```bash
# In WSL2 (not Docker)
echo $DISPLAY  # Should show something like :0
```

If empty, try:
```bash
export DISPLAY=:0
```

### "nvidia-smi not found" in GPU container
1. Verify NVIDIA drivers are installed on Windows
2. Check Docker Desktop > Settings > Resources > GPU
3. Ensure NVIDIA Container Toolkit is installed in WSL2

### Container can't see GPU
```bash
# Verify GPU is visible
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Build fails at RoboSuite installation
The Dockerfile continues even if RoboSuite fails to install (using `|| true`).
Install manually inside the container:
```bash
pip install -e /LearnFlake/src/external_pkgs/RoboSuite
```

### Slow performance
- Use `MUJOCO_GL=egl` for headless rendering
- For GPU containers, ensure NVIDIA runtime is active
- Close unnecessary applications on the host

---

## File Structure

```
LearnFlake/
├── Dockerfile.ubuntu           # Ubuntu 22.04 LTS Dockerfile
├── docker-compose.ubuntu.yml   # Windows-optimized compose file
├── docker/
│   └── entrypoint.sh          # Container startup script
├── src/
│   ├── external_pkgs/
│   │   └── RoboSuite/         # Physics simulation
│   └── rl_autonomy/           # RL algorithms
└── DOCKER_WINDOWS.md          # This file
```

---

## Tips

1. **Live code editing**: The `./:/LearnFlake` volume mount means changes on Windows are immediately reflected in the container.

2. **Persist installed packages**: If you install additional packages, commit the container:
   ```bash
   docker commit learnflake_cpu myusername/learnflake:custom
   ```

3. **Use VS Code Remote**: Install the "Dev Containers" extension in VS Code to develop directly inside the container.

4. **TensorBoard**: The `rover_dev` service exposes port 6006 for TensorBoard:
   ```bash
   # Inside container
   tensorboard --logdir=/LearnFlake/logs --bind_all
   # Access at http://localhost:6006 on Windows
   ```
