#!/bin/bash
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
# Common setup file for training scripts
# This file contains shared environment variables and setup logic
# Source this file in training scripts: source scripts/env.sh

# Resolve repository paths relative to this file, so no user-specific absolute
# path is required.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Set Python path
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"  # Set your key in environment; do not hard-code secrets.

# Create SwanLab global state directory before distributed workers start.
# Older swankit versions race here when several ranks import it concurrently.
export SWANLAB_SAVE_DIR="${SWANLAB_SAVE_DIR:-${HOME}/.swanlab}"
mkdir -p "$SWANLAB_SAVE_DIR"

# NCCL configuration for multi-machine training
export NCCL_CONNECT_TIMEOUT=60
export NCCL_IB_TIMEOUT=60
export OMP_NUM_THREADS=1
export MPI_NUM_THREADS=1
export NCCL_DEBUG=INFO
# Uncomment the following lines if needed for your network setup
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1

# PyTorch CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# Dynamic GPU detection - prioritize CUDA_VISIBLE_DEVICES if set
VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-""}
if [ -z "$VISIBLE_DEVICES" ]; then
    GPU_IDS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    if [ -z "$GPU_IDS" ]; then
        echo "Warning: No GPUs detected via nvidia-smi. GPU_COUNT will be 0."
        GPU_COUNT=0
    else
        GPU_COUNT=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l | xargs)
        echo "Detected GPUs: $GPU_IDS"
    fi
else
    # Handle empty string or comma-separated values
    if [ -z "$VISIBLE_DEVICES" ] || [ "$VISIBLE_DEVICES" = "" ]; then
        GPU_COUNT=0
    else
        GPU_COUNT=$(echo "$VISIBLE_DEVICES" | tr ',' '\n' | wc -l | xargs)
    fi
    echo "Using CUDA_VISIBLE_DEVICES: $VISIBLE_DEVICES"
fi

echo "INFO: This node will use $GPU_COUNT GPUs."
export NPROC_PER_NODE=${GPU_COUNT}

# Set dataloader_num_workers based on GPU count
# Default value can be overridden by setting DATALOADER_NUM_WORKERS_DEFAULT before sourcing
DATALOADER_NUM_WORKERS_DEFAULT=${DATALOADER_NUM_WORKERS_DEFAULT:-4}
if [ "$GPU_COUNT" -gt 1 ]; then
    DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS_DEFAULT}
else
    DATALOADER_NUM_WORKERS=0
fi
echo SET DATALOADER_NUM_WORKERS=$DATALOADER_NUM_WORKERS
# Multi-machine setup detection
# Check if ARNOLD environment variables are set (multi-machine training)
if [ -n "${ARNOLD_WORKER_NUM:-}" ] && [ -n "${ARNOLD_ID:-}" ] && [ -n "${ARNOLD_WORKER_0_HOST:-}" ] && [ "$GPU_COUNT" -gt 1 ]; then
    echo "Multi-machine training detected"
    echo "ARNOLD_WORKER_0_HOST: $ARNOLD_WORKER_0_HOST"
    echo "ARNOLD_WORKER_0_PORT: ${ARNOLD_WORKER_0_PORT:-}"
    echo "ARNOLD_WORKER_NUM: $ARNOLD_WORKER_NUM"
    echo "ARNOLD_WORKER_GPU: ${ARNOLD_WORKER_GPU:-}"
    echo "ARNOLD_ID: $ARNOLD_ID"
    
    # Extract master port from ARNOLD_WORKER_0_PORT (take first port if multiple)
    if [ -z "${HOST_NODE_PORT:-}" ] && [ -n "${ARNOLD_WORKER_0_PORT:-}" ]; then
        HOST_NODE_PORT=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)
    fi
    MASTER_ADDR=${ARNOLD_WORKER_0_HOST}
    MASTER_PORT=${HOST_NODE_PORT:-12345}
    NNODES=${ARNOLD_WORKER_NUM}
    NODE_RANK=${ARNOLD_ID}
    
    echo "Master address: $MASTER_ADDR"
    echo "Master port: $MASTER_PORT"
    echo "Number of nodes: $NNODES"
    echo "Node rank: $NODE_RANK"
    
    MULTI_MACHINE=true
else
    echo "Single-machine training detected"
    MASTER_ADDR="localhost"
    MASTER_PORT=${HOST_NODE_PORT:-12337}
    NNODES=1
    NODE_RANK=0
    MULTI_MACHINE=false
    
    # For single GPU, use a random port to avoid collisions
    # if [ "$GPU_COUNT" -eq 1 ]; then
    #     MASTER_PORT=$((29500 + RANDOM % 1000))
    #     echo "Single GPU detected. Using random port: $MASTER_PORT"
    # fi
fi

# Preserve the configured output directory without adding run-specific suffixes.
if [ -n "${OUTPUT_DIR_BASE:-}" ]; then
    OUTPUT_DIR="$OUTPUT_DIR_BASE"
fi

# Checkpoint resumption logic
# Set RESUME_CHECKPOINT before sourcing this file, or leave empty
# Option 1: Set RESUME_CHECKPOINT to a specific checkpoint path
# Option 2: Set RESUME_CHECKPOINT="auto" to automatically detect the last checkpoint
# Option 3: Leave RESUME_CHECKPOINT empty to start from scratch
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}

# Set overwrite_output_dir based on whether we're resuming
if [ -n "$RESUME_CHECKPOINT" ]; then
    OVERWRITE_OUTPUT_DIR="False"
    if [ "$RESUME_CHECKPOINT" = "auto" ]; then
        echo "Will automatically detect and resume from the last checkpoint in: $OUTPUT_DIR"
    else
        echo "Resuming training from checkpoint: $RESUME_CHECKPOINT"
    fi
else
    OVERWRITE_OUTPUT_DIR="True"
    echo "Starting training from scratch (will overwrite existing output directory)"
fi

# Common paths
WORK_DIR="${WORK_DIR:-$REPO_ROOT}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun 2>/dev/null || true)}"

if [ -z "${TORCHRUN_BIN}" ]; then
    echo "Warning: torchrun not found in PATH. Falling back to 'torchrun'."
    TORCHRUN_BIN="torchrun"
fi

# DeepSpeed is configured exclusively by the training YAML's `deepspeed` key.
# Do not enable it implicitly based on the number of visible GPUs.

# Change to working directory
cd "$WORK_DIR"

# export SWANLAB_MODE=DISABLED
