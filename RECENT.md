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

### Image: rover_rl

docker commit <service> <myusername>/roverflake2:<TAG>





# aaron's notes
claude --resume 293deeb6-fd5f-4d23-8596-e94ec247f3a9

#### Full 84-key keyboard:

Coarse Reach:
Success rate: 39/50 = 78.0%
Final distances (mean ± std):
  XY:    1.79 ±  0.85 cm
  Z:     0.65 ±  0.53 cm
  Tilt:   4.3 ±   3.0°
