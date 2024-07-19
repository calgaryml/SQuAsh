FROM rocm/dev-ubuntu-22.04:6.0-complete

RUN apt-get update && \
    apt-get install -y \
    git git-lfs \
    python3-dev \
    curl \
    && curl -L https://cmake.org/files/v3.29/cmake-3.29.6-linux-x86_64.tar.gz --output /tmp/cmake-3.29.6.tar.gz \
    && tar -xzf /tmp/cmake-3.29.6.tar.gz -C /tmp/ && cd /tmp/cmake-3.29.6-linux-x86_64/ \
    && cp bin/ share/ doc/ /usr/local/ -r && rm -rf /tmp/cmake-3.29.6*

RUN python3 -m pip install sentencepiece
RUN python3 -m pip install protobuf
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install einops lion_pytorch accelerate
RUN python3 -m pip install -U "huggingface_hub[cli]"
RUN python3 -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0