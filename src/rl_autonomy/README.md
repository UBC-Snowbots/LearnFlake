# RL Autonomy

### 1. RoboSuite implementation
Doing a minimalistic Mujoco-RoboSuite physics environment.

If you want to create a virtual environment, go ahead but assuming you are in a docker container, there is no need (as long as you don't stop the container). 

Initial checks (only run if you are using Docker)
```bash
    # Whitelists original directory because you can't create pointers to other repos (submodule creation) while in Docker 
    git config --global --add safe.directory /RoverFlake2
```    

Before starting Docker container
```bash
    xhost +local:docker # inside desired directory
    docker compose exec <rover_compose> bash
```

```bash
    # Quick python venv setup
    python3 -m venv <venv_name>
    source <venv_name>/bin/activate
```

Then, in the virtual environment or in the docker container:
```bash
    pip install -r requirements.txt
```