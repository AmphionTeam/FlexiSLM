#!/usr/bin/env bash
# Run one Stage 2 optimizer step against four samples from the local S2S WebDataset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_ROOT=${CONDA_ROOT:-/F00120260003/flexislm_project/miniconda3}
CONDA_ENV_PATH=${CONDA_ENV_PATH:-$CONDA_ROOT/envs/fslm}
DATASET_CONFIG=${DATASET_CONFIG:-$REPO_ROOT/config/datasets/debug_stage2_s2s_webdataset.yaml}
BASE_CONFIG=${BASE_CONFIG:-$REPO_ROOT/config/train_stage2.yaml}

export S2S_DATA_ROOT=${S2S_DATA_ROOT:-/F00120260003/flexislm_project/yutong/data/speech2speech}
export S2S_DEBUG_SHARD=${S2S_DEBUG_SHARD:-$S2S_DATA_ROOT/shards/train-00000-of-01406.tar}
export S2S_DEBUG_INDEX=${S2S_DEBUG_INDEX:-/F00120260003/flexislm_project/yutong/data/FlexiSLM/debug_stage2_s2s_webdataset/index.jsonl}
export RUN_NAME=${RUN_NAME:-debug_stage2_s2s_webdataset}
export OUTPUT_DIR_BASE=${OUTPUT_DIR_BASE:-/F00120260003/flexislm_project/yutong/outputs/$RUN_NAME}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export SWANLAB_MODE=${SWANLAB_MODE:-disabled}
export PYTHONBREAKPOINT=0
export SKIP_FINAL_SAVE_MODEL=1

if [[ ! -d "$S2S_DATA_ROOT" ]]; then
    echo "Error: S2S WebDataset root not found: $S2S_DATA_ROOT" >&2
    exit 1
fi
if [[ ! -f "$S2S_DEBUG_SHARD" ]]; then
    echo "Error: S2S debug shard not found: $S2S_DEBUG_SHARD" >&2
    exit 1
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$(dirname "$S2S_DEBUG_INDEX")" "$OUTPUT_DIR_BASE"
# Always rebuild this four-row index so a shard override cannot reuse stale URIs.
python "$REPO_ROOT/local/precompute_webdataset_index.py" \
    --input "$S2S_DEBUG_SHARD" \
    --output "$S2S_DEBUG_INDEX" \
    --limit 4

exec "$REPO_ROOT/scripts/train.sh" "$BASE_CONFIG" \
    --model_name_or_path /F00120260003/flexislm_project/model/Qwen2.5-7B-Instruct \
    --config_name /F00120260003/flexislm_project/model/Qwen2.5-7B-Instruct \
    --tokenizer_name /F00120260003/flexislm_project/model/Qwen2.5-7B-Instruct \
    --qwen25omni_encoder_path /F00120260003/flexislm_project/model/Qwen2.5-Omni-7B \
    --qwen25omni_encoder_config_path /F00120260003/flexislm_project/model/Qwen2.5-Omni-7B/config.json \
    --flexicodec_config_path /F00120260003/flexislm_project/model/FlexiCodec/12hz_v1_half_config.yaml \
    --flexicodec_ckpt_path /F00120260003/flexislm_project/model/FlexiCodec/nartts_flexicodec_only.safetensors \
    --sensevoice_small_path /F00120260003/flexislm_project/model/SenseVoiceSmall \
    --flow_matching_decoder_ckpt_path /F00120260003/flexislm_project/model/FlexiCodec/nartts.safetensors \
    --flow_matching_vocoder_path /F00120260003/flexislm_project/model/FlexiCodec/vocos_emilia.safetensors \
    --dataset_name "$DATASET_CONFIG" \
    --do_train true \
    --do_eval false \
    --eval_strategy no \
    --max_steps 1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --max_tokens_per_batch 1024 \
    --gradient_accumulation_steps 1 \
    --save_strategy no \
    --logging_steps 1 \
    --dataloader_num_workers 0 \
    --report_to none \
    --run_name "$RUN_NAME" \
    --output_dir "$OUTPUT_DIR_BASE" \
    "$@"
