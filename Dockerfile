# The Vast.ai worker image for kprelogits.
#
# Restored from legacy/Dockerfile (the OCTMNIST study's worker) and rebuilt for
# this package. The base image, the apt set, and the /workspace layout are kept
# verbatim because they are proven on the GPUs this project actually rents.
# What is gone is everything ktrain-shaped: the old image existed to COPY the
# ktrain submodule in and put it on PYTHONPATH so `import ktrain.medmnist...`
# resolved on the box. kprelogits copied those helpers out and adopted them, so
# there is nothing to mount and no submodule to keep in sync with a pin.
#
# Build from the repo root:
#   docker build -t <dockerhub-user>/kprelogits-worker:latest .
#   docker push  <dockerhub-user>/kprelogits-worker:latest
#
# and pass that tag as `image=` to kprelogits.ops.vast.create().

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# awscli is not optional: ops/s3.py shells out to `aws` for every upload,
# listing, and resume check. tzdata keeps state-file timestamps sane.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    tzdata curl ca-certificates awscli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# torch and torchvision come from the base image. These are the rest of what
# kprelogits imports -- nothing else, verified against its actual import set.
#
# Deliberately unpinned, as the legacy image was. Worth knowing: a timm upgrade
# can change which weights a given model_name resolves to, and that identity is
# what `weights_revision` records in every bundle. Bundles from two different
# timm versions are distinguishable after the fact, which is the point of
# recording it -- but they are not interchangeable, so pin here before a run
# that has to match an earlier one.
RUN pip install --upgrade pip && \
    pip install timm medmnist huggingface_hub

# Defaults in ExtractConfig already point here; created so a run with no
# volumes mounted still works.
RUN mkdir -p /workspace/data /workspace/results

COPY kprelogits/ /workspace/kprelogits/
COPY docker/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# So `python -m kprelogits.extract` resolves; the package is not installed
# (packaging is deferred), it just sits on the path.
ENV PYTHONPATH="/workspace:${PYTHONPATH}" \
    PYTHONUNBUFFERED=1

# The entrypoint sources staged credentials before exec'ing the extractor --
# see docker/entrypoint.sh for why that cannot be an ENV or a `-e` flag.
ENTRYPOINT ["/workspace/entrypoint.sh"]
