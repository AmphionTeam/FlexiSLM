#!/usr/bin/env bash
# Validate a 2-node x 4-GPU launch through the shared DeepSpeed launcher.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

export DEEPSPEED_HOSTFILE="${DEEPSPEED_HOSTFILE:-${HOSTFILE:-/opt/kube/hostfile}}"
export DEEPSPEED_NUM_NODES=2
export DEEPSPEED_NUM_GPUS=4
export DEEPSPEED_TIMEOUT_SECONDS="${TEST_TIMEOUT_SECONDS:-300}"
export DEEPSPEED_LOG_DIR="${SMOKE_LOG_DIR:-$REPO_ROOT/outputs/deepspeed_2x4_smoke/$TIMESTAMP}"
export PYTHON_BIN="${PYTHON_BIN:-/F00120260003/flexislm_project/miniconda3/envs/fslm/bin/python}"
export SMOKE_ALLREDUCE_MIB="${SMOKE_ALLREDUCE_MIB:-32}"
export DEEPSPEED_EXPORT_ENV_VARS="${DEEPSPEED_EXPORT_ENV_VARS:-ASR_TTS_DATA_ROOT ASR_TTS_TRAIN_SHARDS S2S_DATA_ROOT S2S_TRAIN_SHARDS WDS_QUARANTINE_PATH SKIP_FINAL_SAVE_MODEL SWANLAB_API_KEY SWANLAB_PROJECT SWANLAB_MODE SWANLAB_LOG_DIR SWANLAB_SAVE_DIR PYTORCH_CUDA_ALLOC_CONF SMOKE_ALLREDUCE_MIB}"

USER_SCRIPT="${1:-scripts/yutong/deepseed/deepspeed_2x4_smoke.py}"
if [ "$#" -gt 0 ]; then
    shift
fi

echo "Running DeepSpeed 2-node x 4-GPU validation"
exec "$REPO_ROOT/scripts/launch_deepspeed.sh" "$USER_SCRIPT" "$@"
