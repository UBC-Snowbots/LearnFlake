# LearnFlake
RL bitch

Visit Docker.md for the most recent up-to-date installation guide if you plan on using Docker to manage your environment. If you face compatibility issues (docker is currently only setup for x86_64 architectures and won't work on ARM [apple sillicon]), just give up and wait for Aaron to update the compatibility.

TODO list (DevOps side):
- Add basic dependencies to Dockerfile
- Add docker compatibility with ARM

### Continued Installation Guide
After you are done setting up the docker container, the markdown file located at [src/rl_autonomy/README.md](src/rl_autonomy/README.md) contains the installation steps to setup RoboSuite.

Don't forget to install the following system drivers though:
- X11: A windowing system protocol and display server that manages how graphical applications render and display windows. We need this so that Docker knows what display protocol to use when you run RoboSuite in the container.
```bash
    # Native ArchLinux build
    sudo pacman -S xorg-server
```
```bash
    # Native Ubuntu/Debian build
    sudo apt update
    sudo apt install xorg x11-xserver-utils
```
```bash
    # Windows build
    # just chat it until I (Aaron) figure out how this stupid Windows crap works (apparently you need to use a WSL which is what I'm trying to avoid with Docker)
```
