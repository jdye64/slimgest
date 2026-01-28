FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Add deadsnakes PPA for Python 3.12
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev \
      git curl ca-certificates build-essential \
      libgl1 libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Ensure python/pip aliases point to python3.12
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install UV for fast package installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy project files first to leverage Docker layer caching
COPY pyproject.toml README.md /app/
COPY src /app/src

# Copy nemotron-ocr wheel for installation
COPY models/nemotron-ocr-v1/nemotron-ocr/dist/nemotron_ocr-1.0.0-py3-none-any.whl /tmp/
COPY models/nemotron-ocr-v1/checkpoints /app/models/nemotron-ocr-v1/checkpoints

# Note: data/ and models/ directories are excluded via .dockerignore
# Mount them as volumes at runtime:
#   -v /path/to/data:/app/data
#   -v /path/to/models:/app/models

# Create venv with UV using python3.12 and install dependencies
RUN /root/.local/bin/uv venv /opt/venv --python python3.12
ENV PATH="/opt/venv/bin:${PATH}"

# First install torch so we can more directly control the version and CUDA version
RUN /root/.local/bin/uv pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128

# Install nemotron-ocr wheel first
RUN /root/.local/bin/uv pip install /tmp/nemotron_ocr-1.0.0-py3-none-any.whl \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple

# Install project dependencies with UV pip
RUN /root/.local/bin/uv pip install . \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple

# Set model paths environment variable
ENV NEMOTRON_OCR_MODEL_DIR=/app/models/nemotron-ocr-v1/checkpoints

# Expose web server port
EXPOSE 7670

# Default command: run FastAPI web server with 1 workers
CMD ["uvicorn", "slimgest.web:app", "--host", "0.0.0.0", "--port", "7670", "--workers", "1"]
