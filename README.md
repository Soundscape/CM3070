# CM3070

Code repository for CM3070.

This repository utilises VSCode and DevContainers. These commands can be executed in the Docker image.

### Lint API module
```sh
pylint ./app
```

### Lint model module
```sh
pylint ./models
```

### Start Tensorboard
```sh
tensorboard --logdir=./lightning_logs/
```

## Run vulnerability scan
```sh
bandit -lll -r .
```
