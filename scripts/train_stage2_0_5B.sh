#!/usr/bin/env bash
# Stage 2 (0.5B): load Stage 1 weights, train the Talker and input modules in full,
# and adapt the Qwen Thinker through LoRA.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/train.sh" "$SCRIPT_DIR/../config/train_stage2_0_5B.yaml" "$@"
