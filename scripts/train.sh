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

ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate 2>/dev/null || true)}"
if [ -z "$ACCELERATE_BIN" ]; then
    echo "Error: accelerate not found in PATH. Install accelerate or set ACCELERATE_BIN." >&2
    exit 1
fi

# Model and encoder paths are declared in the YAML config. Keep this shared
# launcher limited to runtime-specific overrides such as output and resume state.
TRAINING_OVERRIDES=(
    --output_dir "$OUTPUT_DIR"
    --run_name "$RUN_NAME"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
)

if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAINING_OVERRIDES+=(--resume_from_checkpoint "$RESUME_CHECKPOINT")
fi

"$ACCELERATE_BIN" launch \
    --num_processes=$((NNODES * NPROC_PER_NODE)) \
    --num_machines="$NNODES" \
    --machine_rank="$NODE_RANK" \
    --main_process_ip="$MASTER_ADDR" \
    --main_process_port="$MASTER_PORT" \
    src/train.py \
    --config "$CONFIG_PATH" \
    "${TRAINING_OVERRIDES[@]}" \
    "$@"
