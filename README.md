WIP

### Development environment steps:

Development is done using docker containers; while in the "SQuAsh" directory, run the following docker commands:

docker build -t squash .

docker run -it --device /dev/kfd --device /dev/dri/renderD129 -v $(pwd):/workspace squash

Then to run the repository, run ./build.bash

This will configure and build the project, then run tests.
