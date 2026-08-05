# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
import json
import os
import time
import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from typing import Optional, Tuple, Union, List
import warnings

# from src.models.modeling_speechLM_one_batch import MultimodalQwen2ForCausalLM
# from src.models.modeling_speechLM_flexicodec_s2s import FlexiCodecS2SForCausalLM as MultimodalQwen2ForCausalLM
import loguru

logger = loguru.logger
from src.processor.constants import (
    AUD_START_TOKEN,
    AUD_END_TOKEN,
    AUD_TAG_TOKEN,
)


def _aud_boundary_token_strings(model_args) -> tuple:
    """Resolve audio start/end string tokens (default vs Omni) from model_args."""
    if getattr(model_args, "use_omni_token", False):
        from src.processor.constants import AUD_START_TOKEN_OMNI, AUD_END_TOKEN_OMNI

        return AUD_START_TOKEN_OMNI, AUD_END_TOKEN_OMNI
    raise NotImplementedError
    return AUD_START_TOKEN, AUD_END_TOKEN
# AUD_TAG_TOKEN = "<|audio|>"
# AUD_START_TOKEN = "<|begin_of_audio|>"
# AUD_END_TOKEN = "<|end_of_audio|>"


def add_audio_tokens_if_needed(tokenizer) -> int:
    """
    Add AUD_START_TOKEN, AUD_END_TOKEN, AUD_TAG_TOKEN to the tokenizer vocabulary
    if they are not already present. Returns the number of tokens added.
    """
    from src.processor.constants import AUD_START_TOKEN_OMNI, AUD_END_TOKEN_OMNI
    special_audio_tokens = [AUD_START_TOKEN_OMNI, AUD_END_TOKEN_OMNI, AUD_TAG_TOKEN]
    vocab = tokenizer.get_vocab()
    tokens_to_add = [t for t in special_audio_tokens if t not in vocab]
    if tokens_to_add:
        num_added = tokenizer.add_tokens(tokens_to_add, special_tokens=True)
        return num_added
    return 0



def _load_talker_checkpoint(model, checkpoint: str) -> None:
    """Load only ``talker_model`` weights from a FlexiSLM checkpoint directory."""
    if not os.path.isdir(checkpoint):
        raise FileNotFoundError(f"Talker checkpoint directory not found: {checkpoint}")

    prefix = "talker_model."
    index_path = os.path.join(checkpoint, "model.safetensors.index.json")
    single_path = os.path.join(checkpoint, "model.safetensors")
    state_dict = {}

    if os.path.isfile(index_path):
        from safetensors import safe_open

        with open(index_path, "r", encoding="utf-8") as index_file:
            weight_map = json.load(index_file).get("weight_map", {})
        selected = {name: shard for name, shard in weight_map.items() if name.startswith(prefix)}
        for shard_name in sorted(set(selected.values())):
            shard_path = os.path.join(checkpoint, shard_name)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"Talker checkpoint shard not found: {shard_path}")
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                for name, mapped_shard in selected.items():
                    if mapped_shard == shard_name:
                        state_dict[name[len(prefix):]] = shard.get_tensor(name)
    elif os.path.isfile(single_path):
        from safetensors import safe_open

        with safe_open(single_path, framework="pt", device="cpu") as checkpoint_file:
            for name in checkpoint_file.keys():
                if name.startswith(prefix):
                    state_dict[name[len(prefix):]] = checkpoint_file.get_tensor(name)
    else:
        raise FileNotFoundError(
            f"No safetensors model found in Talker checkpoint: {checkpoint}"
        )

    if not state_dict:
        raise ValueError(f"No weights with prefix {prefix!r} found in {checkpoint}")

    expected = model.talker_model.state_dict()
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    mismatched = sorted(
        (name, tuple(state_dict[name].shape), tuple(expected[name].shape))
        for name in set(expected) & set(state_dict)
        if state_dict[name].shape != expected[name].shape
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing[:10]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:10]}")
        if mismatched:
            details.append(f"shape_mismatches={mismatched[:10]}")
        raise ValueError(
            "Talker checkpoint is incompatible with the configured architecture: " + "; ".join(details)
        )

    model.talker_model.load_state_dict(state_dict, strict=True, assign=True)
    model.config.talker_checkpoint_path = os.path.abspath(checkpoint)
    logger.info(f"Loaded {len(state_dict)} Talker weights from {checkpoint}")


def load_train_flexislm_model_and_tokenizer(
    qwen2_model_path: str,
    model_args,
    audio_vocab_size: int = 32768,
    max_length_classes: int = 32,
    model_impl_module: str = "src.models.modeling_flexislm",
):
    """
    Load FlexiSLM model and tokenizer for training (single supported model).
    """
    import importlib

    model_module = importlib.import_module(model_impl_module)
    ParallelS2SForCausalLM = getattr(model_module, "ParallelS2SForCausalLM")
    ParallelS2SConfig = getattr(model_module, "ParallelS2SConfig")

    # 1. Load original tokenizer
    print("Loading original tokenizer...")
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
        "model_max_length": model_args.model_max_length,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        qwen2_model_path,
        **tokenizer_kwargs
    )

    logger.info(f"{tokenizer.__class__.__name__=} {len(tokenizer)=}")
    # Add pad tokens and special audio tokens (audio tokens only if not already present)
    token_list = [f"<pad_{i}>" for i in range(151936-151664)]
    tokenizer.add_tokens(token_list, special_tokens=True)
    add_audio_tokens_if_needed(tokenizer)
    logger.info(f"After adding special audio tokens: {tokenizer.__class__.__name__=} {len(tokenizer)=}")
    text_vocab_size = len(tokenizer)

    # if model_args.use_joint_text_audio_vocab:
    #     print(f"Expanding tokenizer with {audio_vocab_size} audio vocab tokens...")
    #     audio_vocab_tokens = [f"<audio_{i}>" for i in range(audio_vocab_size)]
    #     tokenizer.add_tokens(audio_vocab_tokens, special_tokens=False)
    #     logger.info(f"After adding audio vocab tokens: {tokenizer.__class__.__name__=} {len(tokenizer)=}")
    _aud_s, _aud_e = _aud_boundary_token_strings(model_args)
    AUD_START_ID = tokenizer(_aud_s, add_special_tokens=False).input_ids[0]
    AUD_END_ID = tokenizer(_aud_e, add_special_tokens=False).input_ids[0]
    AUD_TAG_ID = tokenizer(AUD_TAG_TOKEN, add_special_tokens=False).input_ids[0]

    # 2. Load base Qwen2 config and extend it
    print("Loading base config and creating FlexiSLM config...")
    base_config = Qwen2Config.from_pretrained(
        qwen2_model_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    if model_args.use_sensevoice_feature:
        base_config = base_config.to_dict()
        base_config['codec_hidden_size'] = 512
    else:
        base_config = base_config.to_dict()
    config = ParallelS2SConfig(
        audio_vocab_size=audio_vocab_size,
        padded_audio_vocab_size=audio_vocab_size,
        max_length_classes=max_length_classes,
        framerate_min=model_args.framerate_min,
        framerate_max=model_args.framerate_max,
        enable_flexible_framerate=(
            (model_args.enable_flexible_framerate if hasattr(model_args, 'enable_flexible_framerate') else False)
            or getattr(model_args, 'uniform_merging', False)
        ),
        uniform_merging=getattr(model_args, 'uniform_merging', False),
        output_uniform_merging=getattr(model_args, 'output_uniform_merging', False),
        use_sinusoidal=getattr(model_args, 'use_sinusoidal', False),
        per_sample_frame_rate_embed=getattr(model_args, 'per_sample_frame_rate_embed', False),
        max_tokens_per_group=model_args.max_tokens_per_group if hasattr(model_args, 'max_tokens_per_group') else 8,
        training_framerate_options=getattr(model_args, 'training_framerate_options', None),
        training_input_framerate_options=getattr(model_args, 'training_input_framerate_options', None),
        use_sensevoice_feature=model_args.use_sensevoice_feature,
        use_qwen3_feature=getattr(model_args, 'use_qwen3_feature', False),
        use_whisper_fetaure=getattr(model_args, 'use_whisper_fetaure', False),
        qwen3_encoder_path=getattr(model_args, 'qwen3_encoder_path', None),
        qwen3_encoder_config_path=getattr(model_args, 'qwen3_encoder_config_path', None),
        whisper_encoder_path=getattr(model_args, 'whisper_encoder_path', None),
        use_qwen25o_feature=getattr(model_args, 'use_qwen25omni_feature', False),
        qwen25o_encoder_path=getattr(model_args, 'qwen25omni_encoder_path', None),
        qwen25o_encoder_config_path=getattr(model_args, 'qwen25omni_encoder_config_path', None),
        use_omni_token=getattr(model_args, 'use_omni_token', False),
        flexicodec_config_path=getattr(model_args, 'flexicodec_config_path', None),
        flexicodec_ckpt_path=getattr(model_args, 'flexicodec_ckpt_path', None),
        sensevoice_small_path=getattr(model_args, 'sensevoice_small_path', None),
        flow_matching_decoder_ckpt_path=getattr(model_args, 'flow_matching_decoder_ckpt_path', None),
        flow_matching_vocoder_path=getattr(model_args, 'flow_matching_vocoder_path', None),
        use_joint_text_audio_vocab=model_args.use_joint_text_audio_vocab,
        text_vocab_size=text_vocab_size,
        add_length_embeddings=model_args.add_length_embeddings,
        use_mlp_for_audio_embed=model_args.use_mlp_for_audio_embed if hasattr(model_args, 'use_mlp_for_audio_embed') else False,
        audio_embed_mlp_hidden_ratio=model_args.audio_embed_mlp_hidden_ratio if hasattr(model_args, 'audio_embed_mlp_hidden_ratio') else 4.0,
        audio_embed_mlp_dropout=model_args.audio_embed_mlp_dropout if hasattr(model_args, 'audio_embed_mlp_dropout') else 0.0,
        early_diverge_talker=model_args.early_diverge_talker if hasattr(model_args, 'early_diverge_talker') else False,
        freeze_llm=model_args.freeze_llm if hasattr(model_args, 'freeze_llm') else False,
        freeze_talker=getattr(model_args, 'freeze_talker', False),
        talker_hidden_size=getattr(model_args, 'talker_hidden_size', None),
        talker_num_layers=getattr(model_args, 'talker_num_layers', 20),
        talker_num_attention_heads=getattr(model_args, 'talker_num_attention_heads', 8),
        talker_intermediate_size=getattr(model_args, 'talker_intermediate_size', None),
        talker_pretrained_model_path=getattr(model_args, 'talker_pretrained_model_path', None),
        speech_delay_tokens=getattr(model_args, 'speech_delay_tokens', 5),
        talker_concat_lm_text_output=getattr(model_args, 'talker_concat_lm_text_output', False),
        use_concat_len_emb=getattr(model_args, 'use_concat_len_emb', False),
        talker_embed_v2=getattr(model_args, 'talker_embed_v2', False),
        extend_lm_head=getattr(model_args, "extend_lm_head", False),
        no_pad=getattr(model_args, "no_pad", False),
        AUD_START_TOKEN=AUD_START_ID,
        AUD_END_TOKEN=AUD_END_ID,
        AUD_TAG_TOKEN=AUD_TAG_ID,
        use_combined_embedding=(
            getattr(model_args, 'use_combined_embedding', True)
        ),
        force_use_combined_embedding=(
            getattr(model_args, 'force_use_combined_embedding', False)
        ),
        text_loss_weight=getattr(model_args, 'text_loss_weight', 1.0),
        use_input_merging_transformer=getattr(model_args, 'use_input_merging_transformer', False),
        input_merging_transformer_num_layers=getattr(model_args, 'input_merging_transformer_num_layers', 4),
        input_merging_transformer_d_model=getattr(model_args, 'input_merging_transformer_d_model', 0),
        input_merging_transformer_num_heads=getattr(model_args, 'input_merging_transformer_num_heads', 8),
        input_merging_transformer_dim_feedforward=getattr(model_args, 'input_merging_transformer_dim_feedforward', 2048),
        input_merging_transformer_context=getattr(model_args, 'input_merging_transformer_context', 32),
        input_merging_transformer_causal=getattr(model_args, 'input_merging_transformer_causal', False),
        use_input_merging_transformer_v2=getattr(model_args, 'use_input_merging_transformer_v2', False),
        use_learnable_audio_boundary=getattr(model_args, 'use_learnable_audio_boundary', False),
        # Second-audio-token ablation (group size 2)
        predict_second_audio_token=getattr(model_args, 'predict_second_audio_token', False),
        thinker_concat_user_speech=getattr(model_args, 'thinker_concat_user_speech', False),
        thinker_duplex_use_both_streams=getattr(model_args, 'thinker_duplex_use_both_streams', True),
        thinker_duplex_query_num_heads=getattr(model_args, 'thinker_duplex_query_num_heads', 8),
        thinker_duplex_supervision_warmup_steps=getattr(model_args, 'thinker_duplex_supervision_warmup_steps', 1000),
        thinker_duplex_supervision_weight=getattr(model_args, 'thinker_duplex_supervision_weight', 0.1),
        assistant_text_start_after_speech=getattr(model_args, 'assistant_text_start_after_speech', True),
        assistant_text_start_delay_tokens=getattr(model_args, 'assistant_text_start_delay_tokens', -1),
        **base_config,
    )
    config.only_train_llm = getattr(model_args, "only_train_llm", False)
    config.finetune_speech_encoder = getattr(model_args, "finetune_speech_encoder", False)

    if hasattr(model_args, 'attn_implementation') and model_args.attn_implementation:
        config._attn_implementation = model_args.attn_implementation

    # 3. Create model
    logger.info(f"Creating FlexiSLM model... w/ {config = }")
    torch_dtype = (
        getattr(torch, model_args.torch_dtype)
        if model_args.torch_dtype not in ["auto", None]
        else torch.bfloat16
    )
    model = ParallelS2SForCausalLM(config)
    logger.info("Loading pretrained Qwen2 weights...")
    pretrained_model = Qwen2ForCausalLM.from_pretrained(
        qwen2_model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        trust_remote_code=model_args.trust_remote_code,
    )
    pretrained_state_dict = pretrained_model.model.state_dict()
    pretrained_embed_size = pretrained_state_dict['embed_tokens.weight'].shape[0]

    # Handle DeepSpeed ZeRO-3 partitioned parameters: use ds_shape for the
    # real (unpartitioned) size and GatheredParameters for writing.
    embed_param = model.model.embed_tokens.weight
    _zero3 = hasattr(embed_param, 'ds_id')
    model_embed_size = embed_param.ds_shape[0] if _zero3 else embed_param.shape[0]

    # Use assign=True to load the bulk of the weights efficiently
    pretrained_state_dict_filtered = {k: v for k, v in pretrained_state_dict.items() if k != 'embed_tokens.weight'}
    logger.info("Applying model.load_state_dict(..., assign=True)")
    model.model.load_state_dict(pretrained_state_dict_filtered, strict=False, assign=True)

    if pretrained_embed_size <= model_embed_size:
        with torch.no_grad():
            if _zero3:
                import deepspeed
                with deepspeed.zero.GatheredParameters([embed_param], modifier_rank=None):
                    embed_param.data[:pretrained_embed_size] = pretrained_state_dict['embed_tokens.weight']
            else:
                if embed_param.device.type == "meta":
                    # Initialize with normal distribution matching pretrained embedding variance
                    # We use a fixed generator to ensure reproducibility across runs/ranks for the new tokens
                    new_embed = torch.empty_like(embed_param, device="cpu", dtype=torch_dtype if torch_dtype not in ["auto", None] else embed_param.dtype)
                    std = getattr(model.config, "initializer_range", 0.02)
                    g = torch.Generator(device="cpu")
                    g.manual_seed(42)
                    new_embed.normal_(mean=0.0, std=std, generator=g)
                    logger.info(f"Initialized embed_tokens weights with std={std:.4f} (seed=42)")
                    # Copy pretrained weights over the corresponding indices
                    new_embed[:pretrained_embed_size] = pretrained_state_dict['embed_tokens.weight']
                    model.model.embed_tokens.weight = torch.nn.Parameter(new_embed, requires_grad=embed_param.requires_grad)
                    embed_param = model.model.embed_tokens.weight  # Update reference
                else:
                    embed_param.data[:pretrained_embed_size] = pretrained_state_dict['embed_tokens.weight']
        logger.info(f"Copied {pretrained_embed_size} text vocab embeddings to embed_tokens ({embed_param.shape}) (model has {model_embed_size} total)")
    else:
        raise ValueError(f"Pretrained model has larger vocab ({pretrained_embed_size}) than expected ({model_embed_size})")

    pretrained_lm_head_size = pretrained_model.lm_head.weight.shape[0]
    lm_head_param = model.lm_head.weight
    _zero3_lm = hasattr(lm_head_param, 'ds_id')
    model_lm_head_size = lm_head_param.ds_shape[0] if _zero3_lm else lm_head_param.shape[0]

    if pretrained_lm_head_size <= model_lm_head_size:
        with torch.no_grad():
            if _zero3_lm:
                import deepspeed
                with deepspeed.zero.GatheredParameters([lm_head_param], modifier_rank=None):
                    lm_head_param.data[:pretrained_lm_head_size] = pretrained_model.lm_head.weight.data
            else:
                if lm_head_param.device.type == "meta":
                    # Initialize with normal distribution matching pretrained head variance
                    # We use a fixed generator to ensure reproducibility across runs/ranks for the new tokens
                    new_lm_head = torch.empty_like(lm_head_param, device="cpu", dtype=torch_dtype if torch_dtype not in ["auto", None] else lm_head_param.dtype)
                    std = getattr(model.config, "initializer_range", 0.02)
                    g = torch.Generator(device="cpu")
                    g.manual_seed(42)
                    new_lm_head.normal_(mean=0.0, std=std, generator=g)
                    logger.info(f"Initialized lm_head weights with std={std:.4f} (seed=42)")
                    # Copy pretrained weights over the corresponding indices
                    new_lm_head[:pretrained_lm_head_size] = pretrained_model.lm_head.weight.data
                    model.lm_head.weight = torch.nn.Parameter(new_lm_head, requires_grad=lm_head_param.requires_grad)
                    lm_head_param = model.lm_head.weight  # Update reference
                else:
                    lm_head_param.data[:pretrained_lm_head_size] = pretrained_model.lm_head.weight.data
        logger.info(f"Copied {pretrained_lm_head_size} lm_head weights to lm_head ({lm_head_param.shape}) (model has {model_lm_head_size} total)")
    else:
        raise ValueError(f"Pretrained lm_head has larger vocab ({pretrained_lm_head_size}) than expected ({model_lm_head_size})")
    del pretrained_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    talker_checkpoint_path = getattr(model_args, "talker_checkpoint_path", None)
    if talker_checkpoint_path:
        _load_talker_checkpoint(model, talker_checkpoint_path)
    new_embedding_size = model.text_vocab_size + model.audio_vocab_size
    if torch_dtype not in ["auto", None]:
        model = model.to(torch_dtype)
        
    logger.info("FlexiSLM model loaded successfully!")
    logger.info(f"  - Text vocab size: {model.text_vocab_size}")
    logger.info(f"  - Audio vocab size: {model.audio_vocab_size}")
    logger.info(f"  - Max length classes: {model.config.max_length_classes}")
    print(f"  - Total embedding size: {new_embedding_size}")
    return model, tokenizer, model.text_vocab_size, new_embedding_size



def load_flexislm_model_and_tokenizer(
    qwen2_model_path: str,
    model_args,
    audio_vocab_size: int = 32768,
    max_length_classes: int = 32,
):
    """Load the single supported FlexiSLM model and tokenizer."""
    return load_train_flexislm_model_and_tokenizer(
        qwen2_model_path=qwen2_model_path,
        model_args=model_args,
        audio_vocab_size=audio_vocab_size,
        max_length_classes=max_length_classes,
        model_impl_module="src.models.modeling_flexislm",
    )
