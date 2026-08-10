#!/usr/bin/env bash
# WebDataset 4-GPU training test using the production optimized pipeline.
# This script is self-contained and accepts no command-line parameters.

set -euo pipefail

if [[ $# -ne 0 ]]; then
    echo "Error: this script accepts no arguments; edit its internal constants instead" >&2
    exit 2
fi

REPO_ROOT=/F00120260003/flexislm_project/FlexiSLM
TRAIN_LAUNCHER=$REPO_ROOT/scripts/yutong/train_stage_2_input_merging_transformer_talker_llm_lora_qwen2_5_7b_libritts_mls_gigaspeech_speech2speech_tts_asr_s2s.sh
RUN_NAME=webdataset_optimized_4gpu_v1
OUTPUT_DIR_BASE=/F00120260003/flexislm_project/yutong/outputs/webdataset_ab/$RUN_NAME
AB_STEPS=300
AB_LOGGING_STEPS=10

export RUN_NAME OUTPUT_DIR_BASE
export CUDA_VISIBLE_DEVICES=0,1,2,3
export EXPECTED_NUM_GPUS=4
export WDS_QUARANTINE_PATH=$OUTPUT_DIR_BASE/webdataset_quarantine.jsonl
export FLEXISLM_PROFILE_DIR=$OUTPUT_DIR_BASE/profiler
export FLEXISLM_PROFILE_WAIT_STEPS=10
export FLEXISLM_PROFILE_WARMUP_STEPS=5
export FLEXISLM_PROFILE_ACTIVE_STEPS=20
export SKIP_FINAL_SAVE_MODEL=1

if [[ ! -x "$TRAIN_LAUNCHER" ]]; then
    echo "Error: training launcher is not executable: $TRAIN_LAUNCHER" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR_BASE" "$FLEXISLM_PROFILE_DIR"
cat > "$OUTPUT_DIR_BASE/ab_metadata.env" <<EOF
pipeline=optimized
max_steps=$AB_STEPS
logging_steps=$AB_LOGGING_STEPS
profile_dir=$FLEXISLM_PROFILE_DIR
launcher=$TRAIN_LAUNCHER
EOF

printf 'Starting WebDataset 4-GPU training test\n'
printf '  run:       %s\n' "$RUN_NAME"
printf '  steps:     %s\n' "$AB_STEPS"
printf '  output:    %s\n' "$OUTPUT_DIR_BASE"
printf '  profiling: %s\n' "$FLEXISLM_PROFILE_DIR"

set +e
"$TRAIN_LAUNCHER" \
    --max_steps "$AB_STEPS" \
    --logging_steps "$AB_LOGGING_STEPS" \
    --save_strategy no \
    2>&1 | tee "$OUTPUT_DIR_BASE/train.log"
status=${PIPESTATUS[0]}
set -e
exit "$status"
