WIP

### Development environment steps:

Development is done using docker containers; while in the "SQuAsh" directory, run the following docker commands:

```bash
    docker build -t squash .
```

Check if amd GPU is renderD128 and renderD129 if you have 2 GPUs (or more), in my case it is renderD128

`docker run -it --device /dev/kfd --device /dev/dri/renderD128 -v $(pwd):/workspace squash`

Usage steps: 
1. Run `python3 trainer.py` in directory `rocm_torch_extension` to run the tests and see how to use the python module.