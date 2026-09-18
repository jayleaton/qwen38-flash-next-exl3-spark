#!/usr/bin/env bash
# Download the uncensored EXL3 pack for Qwen3.8-Flash-Next.
# 72.4 GB, 7 shards + one 19.8 GB n-gram table. Resumable.
#
# Usage:  ./scripts/download-model.sh [target_dir]
set -euo pipefail

REPO="${REPO:-Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
REVISION="${REVISION:-263be80b7bb1}"
TARGET="${1:-${MODEL_DIR:-$HOME/ai/models/qwen38-flash-next-unc-3bpw}}"

command -v hf >/dev/null 2>&1 || {
  echo "install the Hugging Face CLI first:  pip install -U huggingface_hub[cli]" >&2
  exit 2
}

echo "downloading $REPO@$REVISION -> $TARGET"
# These shards are all under the 50 GB classic-downloader limit, so no Xet needed.
HF_HUB_DISABLE_XET=1 hf download "$REPO" --revision "$REVISION" --local-dir "$TARGET"

echo "done. point MODEL_DIR at: $TARGET"
du -sh "$TARGET"
