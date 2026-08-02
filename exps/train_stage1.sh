# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
# Stage1 training: v3 parallel model bootstrap.

if [ "$0" = "bash" ]; then exit; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

RUN_NAME=$(basename "$0" .sh)
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-$REPO_ROOT/outputs/$RUN_NAME}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"  # Set to checkpoint path, "auto", or leave empty.

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Omni-7B}"
QWEN3_ENCODER_PATH="${QWEN3_ENCODER_PATH:-}"
QWEN3_ENCODER_CONFIG_PATH="${QWEN3_ENCODER_CONFIG_PATH:-}"

if [ -z "$QWEN3_ENCODER_PATH" ] || [ -z "$QWEN3_ENCODER_CONFIG_PATH" ]; then
    echo "Error: QWEN3 encoder paths are required when --use_qwen3_feature=True."
    echo "Set QWEN3_ENCODER_PATH and QWEN3_ENCODER_CONFIG_PATH in your environment."
    exit 1
fi

source "$REPO_ROOT/flexislm/scripts/env.sh"

"$TORCHRUN_BIN" \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train.py \
    --log_level "info" \
    --do_train \
    $(if [ -n "$RESUME_CHECKPOINT" ]; then echo "--resume_from_checkpoint \"$RESUME_CHECKPOINT\""; fi) \
    --eval_strategy "no" \
    --eval_steps 50 \
    --eval_accumulation_steps 1 \
    --bf16_full_eval True \
    --per_device_eval_batch_size 1 \
    --ddp_timeout 7200 \
    --max_eval_samples 1000 \
    --config_name "$BASE_MODEL" \
    --tokenizer_name "$BASE_MODEL" \
    --model_name_or_path "$BASE_MODEL" \
    --dataset_name "dataset/jiaqi_recipes/dataset_train_v6_dialogonly_filtered_ourdata_ttsv2.yaml" \
    --dataset_name_eval "dataset/dataset_eval.yaml" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 5 \
    --max_tokens_per_batch 4000 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --auto_find_batch_size False \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 4 \
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
    --use_qwen3_feature True \
    --qwen3_encoder_path "$QWEN3_ENCODER_PATH" \
    --qwen3_encoder_config_path "$QWEN3_ENCODER_CONFIG_PATH" \
    --talker_concat_lm_text_output True \
    --talker_hidden_size 1024 \
    --talker_num_layers 20 \
    --use_sinusoidal True \
    --per_sample_frame_rate_embed True \
    --text_loss_weight 2 \
    --use_mlp_for_audio_embed True \
    --use_omni_token True \
    --no_pad True
