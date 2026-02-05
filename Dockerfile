ARG PYTHON_VERSION=3.10-slim
FROM python:${PYTHON_VERSION} AS build

LABEL maintainer="" \
      org.opencontainers.image.title="segmentation-workflows" \
      org.opencontainers.image.source="https://github.com/danieladrianzenSoft/segmentation2dTo3d"

ENV DEBIAN_FRONTEND=noninteractive

# Toggle optional installation of gltf-pipeline (node + npm) via build-arg
ARG INSTALL_GLTF=true

# Install system deps (keep lists cleaned to reduce layer size)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates build-essential git \
       libgl1 libglib2.0-0 libxrender1 libxext6 libx11-6 libsm6 libxrandr2 \
    && rm -rf /var/lib/apt/lists/*

# Optionally install Node and gltf-pipeline (Draco compressor)
RUN if [ "${INSTALL_GLTF}" = "true" ]; then \
      curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
      apt-get update && apt-get install -y --no-install-recommends nodejs && \
      npm install -g gltf-pipeline || true && \
      rm -rf /var/lib/apt/lists/* ; \
    fi

# App directory
WORKDIR /app

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app sources
COPY . /app

# Use a non-root user for better security
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Default entrypoint uses the workflow runner in public mode
ENTRYPOINT ["python", "/app/run.py", "--public"]
CMD ["-h"]
