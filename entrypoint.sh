#!/usr/bin/env bash
# Container entrypoint for the Qwen3.8-Flash-Next EXL3 server.
#
# 1. fadvise the model files (unified memory: cached pages of a 72 GB pack count
#    against GPU-allocatable RAM and make exllamav3's autosplit refuse to load).
#    This is the unprivileged equivalent of the host's drop-model-cache step.
# 2. optionally pin to the big cores (GB10 pairs 10 Cortex-X925 with 10 A725).
# 3. launch TabbyAPI.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/qwen38-flash-next-unc-3bpw}"
CPUSET="${CPUSET:-}"          # e.g. "5-9,15-19"; empty = no pinning
TABBY_ARGS="${TABBY_ARGS:-}"  # extra args passed to main.py

if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "entrypoint: no model at $MODEL_PATH (mount the pack there; see README)" >&2
  exit 2
fi

echo "entrypoint: advising model pages for $MODEL_PATH"
/usr/local/bin/advise-model-files.py "$MODEL_PATH" || true

cd /opt/tabbyAPI
if [ -n "$CPUSET" ]; then
  echo "entrypoint: taskset -c $CPUSET"
  exec taskset -c "$CPUSET" python main.py ${TABBY_ARGS}
fi
exec python main.py ${TABBY_ARGS}
