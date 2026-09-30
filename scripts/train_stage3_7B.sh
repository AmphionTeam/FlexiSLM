#!/usr/bin/env bash
# Stage 3 (7B): merge LoRA from the released Stage 2 Hub checkpoint into full
# Thinker weights, then run full-parameter training with DeepSpeed ZeRO-2.
# Stage 3 itself does not use LoRA.

set -euo pipefail

# train.sh also defaults SWANLAB_PROJECT to RUN_NAME; set early for clarity.
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-train_stage3_7B}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export FLEXISLM_LAUNCHER="${FLEXISLM_LAUNCHER:-deepspeed}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$REPO_ROOT/config/ds_config_zero2.json}"

STAGE2_HUB_ID="${STAGE2_HUB_ID:-FlexiSLM/FlexiSLM-7B-Stage2}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-$REPO_ROOT/models/FlexiSLM-7B-Stage2}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-$REPO_ROOT/models/FlexiSLM-7B-Stage2-merged}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python 2>/dev/null || true)}"
if [ -z "$PYTHON_BIN" ]; then
    echo "Error: python not found in PATH. Set PYTHON_BIN." >&2
    exit 1
fi

if [ ! -f "$STAGE2_CHECKPOINT/model.safetensors" ] \
    && [ ! -f "$STAGE2_CHECKPOINT/model.safetensors.index.json" ]; then
    echo "Downloading $STAGE2_HUB_ID -> $STAGE2_CHECKPOINT"
    "$PYTHON_BIN" - "$STAGE2_HUB_ID" "$STAGE2_CHECKPOINT" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
PY
fi

if [ ! -d "$RESUME_CHECKPOINT" ]; then
    echo "Merging Stage 2 LoRA: $STAGE2_CHECKPOINT -> $RESUME_CHECKPOINT"
    "$PYTHON_BIN" "$REPO_ROOT/scripts/merge_lora_checkpoint.py" \
        --input "$STAGE2_CHECKPOINT" \
        --output "$RESUME_CHECKPOINT"
fi

exec "$SCRIPT_DIR/train.sh" "$REPO_ROOT/config/train_stage3_7B.yaml" \
    --resume_from_checkpoint "$RESUME_CHECKPOINT" \
    "$@"
