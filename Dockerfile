# TODO: Use build args for the version later
FROM rocm/dev-ubuntu-22.04:6.0-complete

RUN apt-get update && \
    apt-get install -y \
    git git-lfs \
    curl \
    && curl -L https://cmake.org/files/v3.29/cmake-3.29.6-linux-x86_64.tar.gz --output /tmp/cmake-3.29.6.tar.gz \
    && tar -xzf /tmp/cmake-3.29.6.tar.gz -C /tmp/ && cd /tmp/cmake-3.29.6-linux-x86_64/ \
    && cp bin/ share/ doc/ /usr/local/ -r && rm -rf /tmp/cmake-3.29.6*
