#!/usr/bin/env bash
# Stage 1 (7B): train the Talker, audio embeddings, and input merging transformer
# while keeping the Qwen2.5-7B backbone frozen.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/train.sh" "$SCRIPT_DIR/../config/train_stage1_7B.yaml" "$@"
