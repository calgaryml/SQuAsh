### Development environment steps:

Development is done using docker containers; while in the "SQuAsh" directory, run the following docker commands:

docker build -t custom_rocm .

docker run -it --rm --device /dev/kfd --device /dev/dri/renderD128 -v $(pwd):/SQuAsh custom_rocm

Then to run the repository, run ./build.bash followed by "Release" or "Debug".
