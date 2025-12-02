FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip python3-dev \
      git curl ca-certificates build-essential cmake ninja-build pkg-config \
      libgl1 libglib2.0-0 wget && \
    rm -rf /var/lib/apt/lists/*

# Ensure python/pip aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Copy project files first to leverage Docker layer caching
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY data /app/data

# Add models to container
COPY models /app/models

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Upgrade pip and install base deps from pyproject
RUN pip install --upgrade pip wheel setuptools

RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
RUN wget https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl \
 && pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# Install flash-attn compatible with CUDA 11.8
RUN pip install flash-attn==2.7.3 --no-build-isolation

# Install project in editable mode (or standard install)
RUN pip install -e .

# Runtime env for vLLM/local pipeline (vLLM 0.8.5 uses V0 API)
ENV VLLM_USE_V1=0 \
    CUDA_VISIBLE_DEVICES=0

# Expose server port
EXPOSE 8000

ENV HF_HOME=/app/models

# Default command: run FastAPI server
# The app lives at slimgest.server.server:app
CMD ["bash", "-lc", "uvicorn slimgest.server.server:app --host 0.0.0.0 --port 8000 --workers 1"]


