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

# Install pip for python3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

WORKDIR /app

# Copy project files first to leverage Docker layer caching
COPY pyproject.toml README.md /app/
COPY src /app/src

# Note: data/ and models/ directories are excluded via .dockerignore
# Mount them as volumes at runtime:
#   -v /path/to/data:/app/data
#   -v /path/to/models:/app/models

# Create venv with python3.12
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install project in editable mode
RUN pip install -e .

# Set model paths environment variable
ENV NEMOTRON_OCR_MODEL_DIR=/app/models/nemotron-ocr-v1/checkpoints

# Expose web server port
EXPOSE 7670

# Default command: run FastAPI web server
CMD ["uvicorn", "slimgest.web:app", "--host", "0.0.0.0", "--port", "7670"]


