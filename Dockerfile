# TODO: Use build args for the version later
FROM rocm/dev-ubuntu-22.04:6.1

# Set the working directory
WORKDIR /SQuAsh/

RUN apt-get update && \
    apt-get install -y \
    git git-lfs \
    cmake \
    gdb
