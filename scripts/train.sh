#!/usr/bin/env bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <config.yaml> [training overrides...]" >&2
    exit 2
fi

CONFIG_PATH="$1"
shift
if [[ "$CONFIG_PATH" != /* ]]; then
    CONFIG_PATH="$REPO_ROOT/$CONFIG_PATH"
fi
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: training config not found: $CONFIG_PATH" >&2
    exit 1
fi

RUN_NAME="${RUN_NAME:-$(basename "$CONFIG_PATH" .yaml)}"
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-$REPO_ROOT/outputs/$RUN_NAME}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# Shared GPU, distributed-training, output-directory, and DeepSpeed setup.
source "$SCRIPT_DIR/env.sh"
cd "$REPO_ROOT"

# Model, encoder, and output paths are declared in the YAML config. Keep this
# shared launcher limited to runtime-specific overrides.
TRAINING_OVERRIDES=(
    --run_name "$RUN_NAME"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
)

if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAINING_OVERRIDES+=(--resume_from_checkpoint "$RESUME_CHECKPOINT")
fi

TRAIN_COMMAND=(
    src/train.py
    --config "$CONFIG_PATH"
    "${TRAINING_OVERRIDES[@]}"
    "$@"
)
FLEXISLM_LAUNCHER="${FLEXISLM_LAUNCHER:-accelerate}"

case "$FLEXISLM_LAUNCHER" in
    accelerate)
        ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate 2>/dev/null || true)}"
        if [ -z "$ACCELERATE_BIN" ]; then
            echo "Error: accelerate not found in PATH. Install accelerate or set ACCELERATE_BIN." >&2
            exit 1
        fi
        echo "Training launcher: accelerate ($NNODES node(s), $NPROC_PER_NODE GPU(s) per node)"
        exec "$ACCELERATE_BIN" launch \
            --num_processes=$((NNODES * NPROC_PER_NODE)) \
            --num_machines="$NNODES" \
            --machine_rank="$NODE_RANK" \
            --main_process_ip="$MASTER_ADDR" \
            --main_process_port="$MASTER_PORT" \
            "${TRAIN_COMMAND[@]}"
        ;;
    deepspeed)
        DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$REPO_ROOT/config/ds_config_zero2.json}"
        if [[ "$DEEPSPEED_CONFIG" != /* ]]; then
            DEEPSPEED_CONFIG="$REPO_ROOT/$DEEPSPEED_CONFIG"
        fi
        if [ ! -f "$DEEPSPEED_CONFIG" ]; then
            echo "Error: DeepSpeed config not found: $DEEPSPEED_CONFIG" >&2
            exit 1
        fi
        has_deepspeed_override=false
        for argument in "$@"; do
            if [[ "$argument" == --deepspeed || "$argument" == --deepspeed=* ]]; then
                has_deepspeed_override=true
                break
            fi
        done
        if [ "$has_deepspeed_override" = false ]; then
            TRAIN_COMMAND+=(--deepspeed "$DEEPSPEED_CONFIG")
        fi
        echo "Training launcher: deepspeed"
        exec "$SCRIPT_DIR/launch_deepspeed.sh" "${TRAIN_COMMAND[@]}"
        ;;
    *)
        echo "Error: unsupported FLEXISLM_LAUNCHER=$FLEXISLM_LAUNCHER (expected accelerate or deepspeed)" >&2
        exit 2
        ;;
esac
