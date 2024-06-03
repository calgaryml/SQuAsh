### Development environment steps:

Development is done using docker containers; run the following docker commands in the top-level directory:

docker build -t custom_rocm .

docker run -it --rm --device /dev/kfd --device /dev/dri/renderD128 -v $(pwd):/xmc-kernels custom_rocm
