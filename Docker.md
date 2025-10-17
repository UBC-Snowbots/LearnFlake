# Docker setup for RoverFlake2

This guide will help you learn how to build images, run containers, create new containers, and also if you want, learn what happens behind the scenes. Ask Aaron if you need anymore help with using Docker. Keep in mind, I only added `docker-compose.yml` so that you can avoid having to add a million flags after docker run but if you prefer to create containers manually, go ahead. 

# Common workflow (for the lazies)
Running a new container
```bash
  docker compose build <service> # build image
  docker compose run <service> # create/run new container
```
Stopping and exiting the current container
```bash
  exit # type inside the container, stops service
```
Going back in an existing stopped container
```bash
  docker start <service> # start the container again
  docker exec <service> # go into the container
```

### 1. Prerequisites
- Install Docker and Docker Compose on your machine
```bash
  docker --version
  docker compose version
```
- Then, make sure you are in the correct working directory `cd RoverFlake2`

Helpful commands:
```bash 
docker image ls # allows you to see existing images
docker ps # allows you to see running containers
docker ps -a # allows you to see stopped containers
docker build -t <image_name> . # don't forget the period
docker run -it <image_name> bash # create a new container under an image and creates a bash terminal
docker exec -it <image_name> # allows you to go into an existing container
docker rm <image_name> # allows you to remove images
docker rmi <contnainer_name> # allows you to remove containers
docker image prune # allows you to remove dangling image (may prompt to force)
```

Preferred method of image/container removal:
- first check for images
- then check for running/stopped containers
- remove containers first with `docker rm <container_name>
- then remove image with `docker rmi <image_name>
The purpose of this is so that you don't have any dangling containers/images and won't have to prune

### 2. Build the image:
```bash
docker compose build
```

What happens here:
- Uses the Dockerfile to build an image (docker build -t roverflake2:dev .)
- Installs development tools (base dependencies in Dockerfile)
- Copies the [entire] repo into /RoverFlake2 inside the container
- Adds an entrypoint script under [docker/entrypoint.sh] that sets up the ROS env and auto-builds colcon which is the workspace we use for this project

Extra commands:
```bash
docker compose build --no-cache # I use this often bc I'm terrible at managing stuff
docker compose ps # list running images/containers started with compose
docker compose --version
```

### 3. Run the container
```bash
docker compose run <service> bash
```

What is happening here:
- Creates and enters a container

Extra commands:
```bash
docker compose up -d # boots up a detached container (avoid non-detached version)
docker system prune # removes dangling pointers to any images/containers
```

### 4. Go into an existing container
```bash
docker compose exec <service> bash # don't forge to change rover if you are using a diff container
```

What is happening:
- Don't forget to check existing running containers with `docker ps`
- Just an easy way to keep commands consistent with compose
- Can do it manually (which I honestly prefer)

Equivalent commands:
```bash
docker exec -it <container_name>
```

### 4. Exit container (saves any installed dependencies)
```bash
exit # thats it
```

### 5. Stop and remove containers
```bash
docker compose down # remove all containers
docker stop <service> # removes specific container
```

Only use if you are completely done using the container and don't want to save any of the dependencies you thereforth downloaded.