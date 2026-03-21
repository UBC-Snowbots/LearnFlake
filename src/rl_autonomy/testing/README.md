# RL Autonomy

### 1. RoboSuite implementation
Doing a minimalistic Mujoco-RoboSuite physics environment.

If you want to create a virtual environment, go ahead but assuming you are in a docker container, there is no need (as long as you don't stop the container). 

Initial checks (only run if you are using Docker)
> git config --global --add safe.directory /RoverFlake2

Before starting Docker container
```bash
    xhost +local:docker # inside desired directory
```

Starting the docker container (post-build)
```bash
    docker compose --compatibility up rover_gpu -d
    docker compose exec rover_gpu bash
```

```bash
    # Quick python venv setup
    python3 -m venv <venv_name>
    source <venv_name>/bin/activate
```

Then, in the virtual environment or in the docker container:
```bash
    cd src/external_pkgs/RoboSuite
    pip install -r requirements.txt
    pip install -r requirements-extra.txt
```

Note: If any of the submodule folders are empty, just run:
```bash
    git submodule update --init --recursive
```

## Quickstart: visualise the Rover2025 robot (old model)

```bash
# cd into Robosuite (the submodule)
python robosuite/demos/demo_random_action.py
```

