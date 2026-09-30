#!/usr/bin/env bash
# Stage 2 (0.5B): load Stage 1 weights, train the Talker and input modules in full,
# and adapt the Qwen Thinker through LoRA.

set -euo pipefail

# train.sh also defaults SWANLAB_PROJECT to RUN_NAME; set early for clarity.
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-train_stage2_0_5B}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/train.sh" "$SCRIPT_DIR/../config/train_stage2_0_5B.yaml" "$@"
