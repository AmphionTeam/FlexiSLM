#!/usr/bin/env bash
# Initialize Stage 2 from the exported Stage 1 model using weights_only mode.
# Train the Talker and v2 input/frame merging transformer in full, and train the
# Qwen2.5-7B Thinker only through LoRA, on TTS, ASR, and S2S WebDataset streams.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASENAME=train_stage_2_input_merging_transformer_talker_llm_lora_qwen2_5_7b_libritts_mls_gigaspeech_speech2speech_tts_asr_s2s
CONDA_ROOT=${CONDA_ROOT:-/F00120260003/flexislm_project/miniconda3}
CONDA_ENV_PATH=${CONDA_ENV_PATH:-$CONDA_ROOT/envs/fslm}
CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/config/yutong/$BASENAME.yaml}
DATASET_CONFIG=${DATASET_CONFIG:-$REPO_ROOT/config/yutong/datasets/eight_gpu/$BASENAME.yaml}
export RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-/F00120260003/flexislm_project/model/FlexiSLM_Stage1_Talker_InputTransformer_AudioEmbedding}

export ASR_TTS_DATA_ROOT=${ASR_TTS_DATA_ROOT:-/F00120260003/flexislm_project/yutong/data/s2s_webdataset_jiaqi_stage1_asr_tts_dual_mp3}
export ASR_TTS_TRAIN_SHARDS=${ASR_TTS_TRAIN_SHARDS:-$ASR_TTS_DATA_ROOT/data/train-*.tar}
export S2S_DATA_ROOT=${S2S_DATA_ROOT:-/F00120260003/flexislm_project/yutong/data/speech2speech}
export S2S_TRAIN_SHARDS=${S2S_TRAIN_SHARDS:-$S2S_DATA_ROOT/shards/train-*.tar}
export RUN_NAME=${RUN_NAME:-${BASENAME}_8gpu}
export OUTPUT_DIR_BASE=${OUTPUT_DIR_BASE:-/F00120260003/flexislm_project/yutong/outputs/$RUN_NAME}
export WDS_QUARANTINE_PATH=${WDS_QUARANTINE_PATH:-$OUTPUT_DIR_BASE/webdataset_quarantine.jsonl}
export SWANLAB_PROJECT=${SWANLAB_PROJECT:-FlexiSLM}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-/F00120260003/flexislm_project/yutong/logs/FlexiSLM/swanlab/$RUN_NAME}
export SWANLAB_MODE=${SWANLAB_MODE:-cloud}
export PYTHONBREAKPOINT=0
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Keep periodic resumable checkpoints and the latest final model for production training.
export SKIP_FINAL_SAVE_MODEL=${SKIP_FINAL_SAVE_MODEL:-0}

for required_path in \
    "$CONFIG_PATH" \
    "$DATASET_CONFIG" \
    "$RESUME_CHECKPOINT" \
    "$ASR_TTS_DATA_ROOT" \
    "$S2S_DATA_ROOT"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Error: required path not found: $required_path" >&2
        exit 1
    fi
done
if [[ -d "$RESUME_CHECKPOINT" ]]; then
    if [[ ! -f "$RESUME_CHECKPOINT/model.safetensors" \
        && ! -f "$RESUME_CHECKPOINT/model.safetensors.index.json" \
        && ! -f "$RESUME_CHECKPOINT/pytorch_model.bin" \
        && ! -f "$RESUME_CHECKPOINT/pytorch_model.bin.index.json" ]]; then
        echo "Error: model weights not found in checkpoint directory: $RESUME_CHECKPOINT" >&2
        exit 1
    fi
elif [[ "$RESUME_CHECKPOINT" != *.bin && "$RESUME_CHECKPOINT" != *.safetensors ]]; then
    echo "Error: unsupported checkpoint file: $RESUME_CHECKPOINT" >&2
    exit 1
fi
for shard_pattern in "$ASR_TTS_TRAIN_SHARDS" "$S2S_TRAIN_SHARDS"; do
    if ! compgen -G "$shard_pattern" >/dev/null; then
        echo "Error: no training shards matched: $shard_pattern" >&2
        exit 1
    fi
done

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$OUTPUT_DIR_BASE" "$SWANLAB_LOG_DIR" "$(dirname "$WDS_QUARANTINE_PATH")"

# Leave CUDA_VISIBLE_DEVICES untouched. scripts/env.sh uses every detected GPU
# when it is unset, while callers can still select a subset explicitly.
exec "$REPO_ROOT/scripts/train.sh" "$CONFIG_PATH" \
    --dataset_name "$DATASET_CONFIG" \
    --run_name "$RUN_NAME" \
    --output_dir "$OUTPUT_DIR_BASE" \
    "$@"
