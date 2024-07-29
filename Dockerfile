FROM rocm/dev-ubuntu-22.04:6.0-complete

WORKDIR /workspace

## cmake + other atp deps
RUN apt-get update && \
    apt-get install -y \
    git git-lfs \
    python3-dev \
    curl \
    && curl -L https://cmake.org/files/v3.29/cmake-3.29.6-linux-x86_64.tar.gz --output /tmp/cmake-3.29.6.tar.gz \
    && tar -xzf /tmp/cmake-3.29.6.tar.gz -C /tmp/ && cd /tmp/cmake-3.29.6-linux-x86_64/ \
    && cp bin/ share/ doc/ /usr/local/ -r && rm -rf /tmp/cmake-3.29.6*


## Python deps
RUN pip install --upgrade pip && pip install sentencepiece protobuf \
    einops lion_pytorch accelerate huggingface_hub[cli] && \
    pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0


# Copy in project files required for building
COPY ./ ./

## Submodules pull and build squash
RUN git config --global --add safe.directory "*" && \
    git submodule init && \
    git submodule update && \
    pip install -e ./third-party/peft && \
    pip install -e ./third-party/torchtune && \
    cmake -DCOMPUTE_BACKEND=hip -S ./third-party/bitsandbytes && \
    make ./third-party/bitsandbytes && \
    pip install -e ./third-party/bitsandbytes && \
    pip install -e ./third-party/sparsimony && \
    pip install -e .


