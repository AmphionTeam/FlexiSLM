# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

if [ "$0" = "bash" ]; then exit; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

RUN_NAME=$(basename "$0" .sh)
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-$REPO_ROOT/outputs/$RUN_NAME}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"  # Set to checkpoint path, "auto", or leave empty.

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
QWEN25O_ENCODER_PATH="${QWEN25O_ENCODER_PATH:-}"
QWEN25O_ENCODER_CONFIG_PATH="${QWEN25O_ENCODER_CONFIG_PATH:-}"
if [ -n "$QWEN25O_ENCODER_PATH" ] && [ -z "$QWEN25O_ENCODER_CONFIG_PATH" ]; then
    QWEN25O_ENCODER_CONFIG_PATH="${QWEN25O_ENCODER_PATH}/audio_config.json"
fi
if [ -z "$QWEN25O_ENCODER_PATH" ] || [ -z "$QWEN25O_ENCODER_CONFIG_PATH" ]; then
    echo "Error: QWEN2.5 Omni encoder paths are required when --use_qwen25o_feature=True."
    echo "Set QWEN25O_ENCODER_PATH and QWEN25O_ENCODER_CONFIG_PATH in your environment."
    exit 1
fi

source "$REPO_ROOT/flexislm/scripts/env.sh"

echo RUN_NAME=$RUN_NAME
echo OUTPUT_DIR_BASE=$OUTPUT_DIR_BASE
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"  # Set your key in environment; do not hard-code secrets.
export USE_LORA="${USE_LORA:-True}"  # Set your key in environment; do not hard-code secrets.

ACCELERATE_BIN="${ACCELERATE_BIN:-$(command -v accelerate 2>/dev/null)}"
if [ -z "$ACCELERATE_BIN" ]; then
    echo "Error: accelerate not found in PATH. Install accelerate or set ACCELERATE_BIN."
    exit 1
fi

"$ACCELERATE_BIN" launch \
    --num_processes=$((NNODES * NPROC_PER_NODE)) \
    --num_machines=${NNODES} \
    --machine_rank=${NODE_RANK} \
    --main_process_ip=${MASTER_ADDR} \
    --main_process_port=${MASTER_PORT} \
    train.py \
    --log_level "info" \
    --do_train \
--do_eval \
    $(if [ -n "$RESUME_CHECKPOINT" ]; then echo "--resume_from_checkpoint \"$RESUME_CHECKPOINT\""; fi) \
    --eval_strategy "steps" \
    --eval_steps 5000 \
    --eval_accumulation_steps 1 \
    --bf16_full_eval True \
        --ddp_timeout 7200 \
    --max_eval_samples 5000 \
    --config_name "$BASE_MODEL" \
    --tokenizer_name "$BASE_MODEL" \
    --model_name_or_path "$BASE_MODEL" \
    --dataset_name "dataset/recipes/dataset_train_stage1.yaml" \
    --dataset_name_eval "dataset/recipes/dataset_eval_stage1.yaml" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 5 \
    --max_tokens_per_batch 4000 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --auto_find_batch_size False \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 120 \
    --max_grad_norm 1.0 \
    --learning_rate 5e-4 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr": 4e-5}' \
    --logging_steps 100 \
    --model_max_length 1024 \
    --gradient_checkpointing False \
    --trust_remote_code True \
    --attn_implementation flash_attention_2 \
    --seed 42 \
    --data_seed 42 \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --bf16 True \
    --fp16 False \
    --report_to "swanlab" \
    --run_name $RUN_NAME \
    --dataloader_pin_memory True \
    --torch_dtype bfloat16 \
    --use_sensevoice_feature False \
    --load_from_stage1 False \
    --use_parallel True \
    --enable_flexible_framerate True \
    --training_framerate_options "0.86,0.87,0.88,0.90,0.91,1.0,1.0" \
    --training_input_framerate_options "8.5,9.0,9.5,10.0,11.0,12.0,12.5" \
    --framerate_min 0.0 \
    --framerate_max 1.0 \
    --use_v3 True \
    --freeze_llm False \
    --use_joint_text_audio_vocab False \
    --early_diverge_talker False \
    --only_train_talker True \
    --use_lora False \
    --force_use_combined_embedding False \
    --no_use_combined_embedding \
   --use_qwen25o_feature True \
    --qwen25o_encoder_path "$QWEN25O_ENCODER_PATH" \
    --qwen25o_encoder_config_path "$QWEN25O_ENCODER_CONFIG_PATH" \
    --talker_concat_lm_text_output True \
    --talker_hidden_size 1024 \
    --talker_num_layers 20 \
    --use_sinusoidal True \
    --per_sample_frame_rate_embed True \
    --text_loss_weight 2 \
    --use_mlp_for_audio_embed True \
    --use_omni_token True \
    --no_pad True \
    --use_input_merging_transformer True \
    --use_learnable_audio_boundary True \
    --extend_lm_head False \
    --finetune_speech_encoder False \
    --use_input_merging_transformer_v2 True \
    --input_merging_transformer_d_model 768
