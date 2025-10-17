# This is for the advanced docker features

# Saving, Committing, and Reusing Docker images
If you've installed additional dependencies on a Docker container that aren't included in the Dockerfile, and you want to reuse your image and not have to reinstall all the dependencies all over again, do this (we don't have the feature that allows multiple people to pull images though so for now all of this is just local).

### 1. Run and install all your dependencies in the container
```bash
    # inside the container
    pip install -r requirements.txt
    apt update && apt install -y <other-stuff>
    exit
```

### 2. Commit the container into a new image
```bash
    # You can name the new image whatever you want but just for convention:
    docker commit <service> <myusername>/roverflake2:<TAG>
```

Check existing local images:
```bash
    docker images
```
### 3. Push to registry so you can reuse it
```bash
    docker login
    docker push <myusername>/roverflake2:<TAG>
```

### 4. Now you have different image states
```bash
    docker pull <myusername>/roverflake2:<TAG>
    docker run -it <myusername>/roverflake2:<TAG>
```

When you are satisfied with the dependencies and it works well in the main branch, add them to the actual Dockerfile! :)