# LearnFlake
RL for Arm


Visit [Docker.md](Docker.md) for the most recent up-to-date installation guide if you plan on using Docker to manage your environment.

### Continued Installation Guide
After you are done setting up the docker container, the markdown file located at [src/rl_autonomy/README.md](src/rl_autonomy/README.md) contains the installation steps to setup RoboSuite.

Don't forget to install the following system drivers though:
- X11: A windowing system protocol and display server that manages how graphical applications render and display windows. We need this so that Docker knows what display protocol to use when you run RoboSuite in the container.
# LearnFlake
RL for Arm


Visit [Docker.md](Docker.md) for the most recent up-to-date installation guide if you plan on using Docker to manage your environment.

### Continued Installation Guide
After you are done setting up the docker container, the markdown file located at [src/rl_autonomy/README.md](src/rl_autonomy/README.md) contains the installation steps to setup RoboSuite.

Don't forget to install the following system drivers though:
- X11: A windowing system protocol and display server that manages how graphical applications render and display windows. We need this so that Docker knows what display protocol to use when you run RoboSuite in the container.

# LearnFlake
RL bitch

Visit [Docker.md](Docker.md) for the most recent up-to-date installation guide if you plan on using Docker to manage your environment.

### Continued Installation Guide
After you are done setting up the docker container, the markdown file located at [src/rl_autonomy/README.md](src/rl_autonomy/README.md) contains the installation steps to setup RoboSuite.

Don't forget to install the following system drivers though:
- X11: A windowing system protocol and display server that manages how graphical applications render and display windows. We need this so that Docker knows what display protocol to use when you run RoboSuite in the container.

## Windows Docker Guide
Use the Windows/WSL compose file: `docker-compose.ubuntu.yml`.

### Prerequisites
- Docker Desktop with WSL2 backend
- WSLg enabled (for GUI apps)

### Build images
```bash
docker compose -f docker-compose.ubuntu.yml build
```

### Start a container
CPU:
```bash
docker compose -f docker-compose.ubuntu.yml run --rm rover_cpu bash
```

RL:
```bash
docker compose -f docker-compose.ubuntu.yml run --rm rover_rl bash
```

GPU:
```bash
docker compose -f docker-compose.ubuntu.yml run --rm rover_gpu bash
```

### Run in background and re-enter
```bash
docker compose -f docker-compose.ubuntu.yml up -d rover_cpu
docker compose -f docker-compose.ubuntu.yml exec rover_cpu bash
```

### Stop everything
```bash
docker compose -f docker-compose.ubuntu.yml down
```
There's a buncha other stuff, like VcXsrv and other stuff that should be ok with my files. In the process of writing code if you come into issues with glfw not running or switching into egl (headless, no GUI), message Pranav.