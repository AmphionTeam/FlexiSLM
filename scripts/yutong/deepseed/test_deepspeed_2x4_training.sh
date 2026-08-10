#!/usr/bin/env bash
# Run one real FlexiSLM/WebDataset optimizer step on 2 nodes x 4 GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${TRAIN_SMOKE_ROOT:-$REPO_ROOT/outputs/deepspeed_2x4_training_smoke/$TIMESTAMP}"
OUTPUT_DIR="$RUN_ROOT/train_output"
CONFIG_PATH="${TRAIN_SMOKE_CONFIG:-$REPO_ROOT/config/yutong/train_stage_2_input_merging_transformer_talker_llm_lora_qwen2_5_7b_libritts_mls_gigaspeech_speech2speech_tts_asr_s2s.yaml}"
DATASET_CONFIG="${TRAIN_SMOKE_DATASET_CONFIG:-$REPO_ROOT/config/yutong/datasets/eight_gpu/train_stage_2_input_merging_transformer_talker_llm_lora_qwen2_5_7b_libritts_mls_gigaspeech_speech2speech_tts_asr_s2s.yaml}"
DEEPSPEED_CONFIG="${TRAIN_SMOKE_DEEPSPEED_CONFIG:-$REPO_ROOT/config/ds_config_zero2.json}"
CHECKPOINT="${TRAIN_SMOKE_CHECKPOINT:-/F00120260003/flexislm_project/model/FlexiSLM_Stage1_Talker_InputTransformer_AudioEmbedding}"
export ASR_TTS_DATA_ROOT="${ASR_TTS_DATA_ROOT:-/F00120260003/flexislm_project/yutong/data/s2s_webdataset_jiaqi_stage1_asr_tts_dual_mp3}"
export S2S_DATA_ROOT="${S2S_DATA_ROOT:-/F00120260003/flexislm_project/yutong/data/speech2speech}"
export WDS_QUARANTINE_PATH="$OUTPUT_DIR/webdataset_quarantine.jsonl"
export SKIP_FINAL_SAVE_MODEL=1
export SWANLAB_MODE=disabled
export SWANLAB_LOG_DIR="$RUN_ROOT/swanlab"
export SWANLAB_SAVE_DIR="$RUN_ROOT/swanlab_state"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TEST_TIMEOUT_SECONDS="${TEST_TIMEOUT_SECONDS:-1800}"
export SMOKE_LOG_DIR="$RUN_ROOT/launcher_logs"
RUN_NAME="deepspeed_2x4_training_smoke_$TIMESTAMP"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

for path in \
    "$CONFIG_PATH" "$DATASET_CONFIG" "$DEEPSPEED_CONFIG" "$CHECKPOINT" \
    "$ASR_TTS_DATA_ROOT" "$S2S_DATA_ROOT"; do
    [ -e "$path" ] || fail "required path not found: $path"
done
[ ! -e "$RUN_ROOT" ] || fail "refusing to reuse smoke-test directory: $RUN_ROOT"

# Check representative shards before paying the model initialization cost.
compgen -G "$ASR_TTS_DATA_ROOT/data/train-*.tar" >/dev/null \
    || fail "no ASR/TTS shards found under $ASR_TTS_DATA_ROOT/data"
compgen -G "$S2S_DATA_ROOT/shards/train-*.tar" >/dev/null \
    || fail "no S2S shards found under $S2S_DATA_ROOT/shards"

mkdir -p "$RUN_ROOT"
echo "Running one real FlexiSLM optimizer step on 2 nodes x 4 GPUs"
echo "Training output: $OUTPUT_DIR"
echo "Launcher logs: $SMOKE_LOG_DIR"
echo "No periodic or final model checkpoint will be saved."

exec "$SCRIPT_DIR/test_deepspeed_2x4.sh" \
    src/train.py \
    --config "$CONFIG_PATH" \
    --dataset_name "$DATASET_CONFIG" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --resume_from_checkpoint "$CHECKPOINT" \
    --checkpoint_load_mode weights_only \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --max_steps 1 \
    --save_strategy no \
    --logging_steps 1 \
    --report_to none \
    --dataloader_num_workers 1 \
    --overwrite_output_dir true
