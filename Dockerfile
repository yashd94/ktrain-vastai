FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    tzdata git wget curl awscli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone outer repo including the ktrain submodule
ARG GITHUB_TOKEN
RUN git clone https://${GITHUB_TOKEN}@github.com/nullset-mit/ktrain.git /workspace/ktrain

# Install dependencies
RUN pip install --upgrade pip && \
    pip install matplotlib medmnist numpy pandas scipy timm \
                tqdm huggingface_hub

# Set PYTHONPATH so ktrain package imports work (mirrors Colab sys.path setup)
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# Create data and results dirs
RUN mkdir -p /workspace/data /workspace/results

# Copy the worker script (created below)
COPY prepare_breakhis.py /workspace/prepare_breakhis.py
COPY run_worker.py /workspace/run_worker.py

ENTRYPOINT ["/bin/bash", "-c", \
  "python3 /workspace/prepare_breakhis.py \
     --magnification ${MAGNIFICATION:-400X} \
     --output /workspace/data/breakhis \
     --download_dir /workspace/data/breakhis_raw \
     --seed ${SEED:-42} \
   && python3 /workspace/run_worker.py"]
