#!/usr/bin/env bash
# Stage 3 (0.5B): load a merged Stage 2 checkpoint and run full-parameter training
# with DeepSpeed ZeRO-2 by default.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FLEXISLM_LAUNCHER="${FLEXISLM_LAUNCHER:-deepspeed}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$SCRIPT_DIR/../config/ds_config_zero2.json}"
exec "$SCRIPT_DIR/train.sh" "$SCRIPT_DIR/../config/train_stage3_0_5B.yaml" "$@"
