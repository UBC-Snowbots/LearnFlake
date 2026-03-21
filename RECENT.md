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

# aaron's notes
claude --resume 293deeb6-fd5f-4d23-8596-e94ec247f3a9

