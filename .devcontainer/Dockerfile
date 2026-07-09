FROM nvidia/cuda:12.8.2-devel-ubuntu22.04
 
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/root/.local/bin:/root/.cargo/bin:$PATH"
 
# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*
 
# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
 
WORKDIR /workspaces/cv2
 