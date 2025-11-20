# Use Ubuntu 24.04 as base image
FROM ubuntu:24.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv to /root/uv using curl method
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY nemotron-wheels/ ./nemotron-wheels/

# Create venv and install nemotron wheels first, then install project
RUN /root/.local/bin/uv venv .venv && \
    /root/.local/bin/uv pip install nemotron-wheels/*.whl && \
    /root/.local/bin/uv pip install --python .venv/bin/python -e .

# Expose port
EXPOSE 8000 

# Run the server
CMD [".venv/bin/python", "src/api.py", "--host", "0.0.0.0", "--port", "8000"]
