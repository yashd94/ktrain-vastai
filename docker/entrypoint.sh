#!/usr/bin/env bash
# Worker entrypoint: pick up staged credentials, then run the extractor.
#
# This exists for one reason. AWS keys and the HuggingFace token cannot be
# passed as `vastai create -e KEY=...` or baked into the image: the first puts
# them in the instance's command line, visible in `ps` to anyone else on a host
# we share with strangers, and the second puts them in a layer that outlives
# the run. ops/vast.py::stage_credentials writes them over stdin into a
# mode-600 file instead, and this is the file's only consumer.
#
# Config is otherwise env-driven (ExtractConfig.from_env) with CLI overrides,
# so any argument passed to `docker run` / `--onstart-cmd` lands on the
# extractor unchanged.
set -euo pipefail

SECRETS="${KPRELOGITS_SECRETS:-/workspace/.env-secrets}"

if [ -f "$SECRETS" ]; then
    # `set -a` exports everything the file defines, so boto/awscli and
    # huggingface_hub see it without kprelogits having to read the file.
    set -a
    # shellcheck disable=SC1090
    . "$SECRETS"
    set +a
    echo "entrypoint: loaded credentials from $SECRETS" >&2
else
    # Not fatal: a run against a public bucket with no gated models needs
    # none of this, and preflight reports what is actually missing with far
    # more context than this script could.
    echo "entrypoint: no $SECRETS; continuing without staged credentials" >&2
fi

exec python -m kprelogits.extract "$@"
