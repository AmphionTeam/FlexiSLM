#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Launch distributed training with the DeepSpeed CLI. Expects NNODES,
# NPROC_PER_NODE, NODE_RANK, MASTER_ADDR, and MASTER_PORT from env.sh.

set -euo pipefail

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-12337}"

DEEPSPEED_BIN="${DEEPSPEED_BIN:-$(command -v deepspeed 2>/dev/null || true)}"
if [ -z "$DEEPSPEED_BIN" ]; then
    echo "Error: deepspeed not found in PATH. Install deepspeed or set DEEPSPEED_BIN." >&2
    exit 1
fi

exec "$DEEPSPEED_BIN" \
    --num_gpus="${NPROC_PER_NODE}" \
    --num_nodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "$@"
