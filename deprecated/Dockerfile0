FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    tzdata python3.11 python3-pip git wget curl awscli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone the repo
RUN git clone https://github.com/nullset-mit/ktrain.git /workspace/ktrain
COPY ktrain/ /workspace/ktrain/


# Install dependencies
RUN pip install --upgrade pip && \
    pip install matplotlib medmnist numpy pandas scipy timm torch \
                torchvision tqdm huggingface_hub

# Set PYTHONPATH so ktrain package imports work
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# Create data and results dirs
RUN mkdir -p /workspace/data /workspace/results

# Copy the worker script (created below)
COPY run_worker.py /workspace/run_worker.py

ENTRYPOINT ["python3", "/workspace/run_worker.py"]

