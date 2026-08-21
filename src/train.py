#!/usr/bin/env python
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
# coding=utf-8

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import fields

import datasets
import torch
import yaml

from src.models.utils import (
    load_flexislm_model_and_tokenizer,
)
# torch.cuda.memory._set_allocator_settings("expandable_segments:False")
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    # Trainer,
    # TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)
from transformers import AutoModel, AutoConfig
from transformers.trainer_utils import get_last_checkpoint

# LoRA-related imports
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. LoRA will not work.")


from src.arguments import ModelArguments, DataTrainingArguments, TrainingArguments
from src.dataset.collator import collate_fn_deepspeed, get_collator
from src.dataset.interleaved import Qwen2Dataset
from src.dataset.interleaved_collator import InterleavedDataCollator
from src.dataset.webdataset.native import (
    build_qwen2_webdataset,
    is_webdataset_stream_config,
)
from src.trainer.self_trainer import ATrainer
# from sampler.TokenBatchSampler import TokenBatchSampler
import loguru

logger = loguru.logger

def _looks_like_hub_repo_id(path: str) -> bool:
    """True for Hub-style ids such as ``org/name`` that are not local paths."""
    if not path or path in ("auto",):
        return False
    if os.path.isdir(path) or os.path.isfile(path):
        return False
    if path.startswith((".", "/", "~")):
        return False
    parts = path.split("/")
    return len(parts) == 2 and all(parts)


def _resolve_checkpoint_path(checkpoint: str, *, process_index: int = 0) -> str:
    """Resolve a local path or Hugging Face Hub repo id to a local directory.

    Hub ids (for example ``FlexiSLM/FlexiSLM-7B-Stage1``) are downloaded into
    ``models/<repo_name>/`` on first use and reused afterwards.
    """
    if checkpoint is None or checkpoint in ("", "auto"):
        return checkpoint
    if os.path.isdir(checkpoint) or os.path.isfile(checkpoint):
        return checkpoint
    if not _looks_like_hub_repo_id(checkpoint):
        return checkpoint

    from huggingface_hub import snapshot_download

    local_dir = os.path.join("models", checkpoint.rsplit("/", 1)[-1])
    weight_markers = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    if os.path.isdir(local_dir) and any(
        os.path.isfile(os.path.join(local_dir, name)) for name in weight_markers
    ):
        logger.info("Using local Hub checkpoint cache at %s", local_dir)
        return local_dir

    if process_index == 0:
        logger.info(
            "Downloading resume checkpoint from Hugging Face Hub: %s -> %s",
            checkpoint,
            local_dir,
        )
        snapshot_download(repo_id=checkpoint, local_dir=local_dir)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(
            f"Failed to resolve Hub checkpoint {checkpoint!r} to {local_dir}"
        )
    return local_dir


def _load_state_dict_from_checkpoint(checkpoint: str) -> dict:
    """Load state dict from checkpoint, supporting single-file and sharded Hugging Face formats.

    Supports:
    - Single: model.safetensors, pytorch_model.bin
    - Sharded safetensors: model.safetensors.index.json + model-00001-of-N.safetensors
    - Sharded pytorch: pytorch_model.bin.index.json + pytorch_model-00001-of-N.bin
    """
    st_index = os.path.join(checkpoint, "model.safetensors.index.json")
    st_single = os.path.join(checkpoint, "model.safetensors")
    bin_index = os.path.join(checkpoint, "pytorch_model.bin.index.json")
    bin_single = os.path.join(checkpoint, "pytorch_model.bin")

    # Sharded safetensors
    if os.path.isfile(st_index):
        from safetensors.torch import load_file as safe_load_file
        with open(st_index, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", index)
        state_dict = {}
        for shard_file in sorted(set(weight_map.values())):
            shard_path = os.path.join(checkpoint, shard_file)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"Shard not found: {shard_path}")
            state_dict.update(safe_load_file(shard_path, device="cpu"))
        logger.info(f"Loaded sharded safetensors ({len(weight_map)} weights from {len(set(weight_map.values()))} shards)")
        return state_dict

    # Single safetensors
    if os.path.isfile(st_single):
        from safetensors.torch import load_file as safe_load_file
        return safe_load_file(st_single, device="cpu")

    # Sharded pytorch
    if os.path.isfile(bin_index):
        with open(bin_index, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", index)
        state_dict = {}
        for shard_file in sorted(set(weight_map.values())):
            shard_path = os.path.join(checkpoint, shard_file)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"Shard not found: {shard_path}")
            state_dict.update(torch.load(shard_path, map_location="cpu"))
        logger.info(f"Loaded sharded pytorch ({len(weight_map)} weights from {len(set(weight_map.values()))} shards)")
        return state_dict

    # Single pytorch
    if os.path.isfile(bin_single):
        return torch.load(bin_single, map_location="cpu")

    raise FileNotFoundError(
        f"No supported model file found in {checkpoint} "
        "(expected model.safetensors, model.safetensors.index.json, "
        "pytorch_model.bin, or pytorch_model.bin.index.json)"
    )


_REQUIRED_TRANSFER_COMPONENTS = {
    "talker_model": ("talker_model.",),
    "input_merging_transformer": ("input_merging_transformer.",),
    "audio_boundaries": ("audio_start_embedding", "audio_end_embedding"),
}


def _keys_with_prefixes(keys, prefixes):
    return {key for key in keys if any(key.startswith(prefix) for prefix in prefixes)}


def _drop_incompatible_state_dict_tensors(model, state_dict: dict) -> list:
    """Drop checkpoint tensors whose shapes do not match the current model.

    ``load_state_dict(..., strict=False)`` still raises on size mismatches. Architecture
    flags such as ``talker_embed_v2`` change ``combined_embed_proj``. Keep the freshly
    initialized tensors when the checkpoint layout cannot transfer.
    """
    current = model.state_dict()
    dropped = []
    for key, tensor in list(state_dict.items()):
        if key not in current:
            continue
        current_shape = tuple(current[key].shape)
        incoming_shape = tuple(tensor.shape)
        if current_shape != incoming_shape:
            dropped.append((key, incoming_shape, current_shape))
            del state_dict[key]
    return dropped


def _validate_weights_only_transfer_components(
    model,
    state_dict: dict,
    *,
    reinitialize_input_merging_transformer: bool = False,
) -> dict:
    """Require complete Stage 1 components while allowing a separately initialized backbone.

    Talker transfer is optional and may be partial. Missing ``talker_model`` tensors keep
    the constructed values, which is required when transferring a 7B Talker onto a 0.5B
    backbone (LLM-width projections do not match). Extra Talker keys are still rejected.
    """
    target_keys = set(model.state_dict())
    source_keys = set(state_dict)
    counts = {}

    for component, prefixes in _REQUIRED_TRANSFER_COMPONENTS.items():
        if component == "input_merging_transformer" and reinitialize_input_merging_transformer:
            continue
        expected = _keys_with_prefixes(target_keys, prefixes)
        provided = _keys_with_prefixes(source_keys, prefixes)
        if component == "talker_model":
            unexpected = sorted(provided - expected)
            if unexpected:
                raise RuntimeError(
                    f"Incomplete weights_only checkpoint component {component!r}: "
                    f"target_count={len(expected)}, checkpoint_count={len(provided)}, "
                    f"missing=[], unexpected={unexpected[:20]}"
                )
            if not provided:
                logger.info(
                    "Checkpoint has no talker_model tensors; keeping the constructed Talker"
                )
                continue
            missing = sorted(expected - provided)
            if missing:
                logger.warning(
                    "Checkpoint is missing {} talker_model tensors; keeping constructed "
                    "values: {}",
                    len(missing),
                    missing,
                )
            counts[component] = len(provided)
            continue
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        if not expected or missing or unexpected:
            raise RuntimeError(
                f"Incomplete weights_only checkpoint component {component!r}: "
                f"target_count={len(expected)}, checkpoint_count={len(provided)}, "
                f"missing={missing[:20]}, unexpected={unexpected[:20]}"
            )
        counts[component] = len(provided)

    return counts


def _validate_ignored_frozen_checkpoint_weights(
    checkpoint: str,
    unexpected_keys,
) -> dict:
    """Allow ignored Stage 1 Thinker weights; still reject a finetuned Omni encoder."""
    ignored = {
        "thinker": _keys_with_prefixes(unexpected_keys, ("model.",)),
        "qwen25o_encoder": _keys_with_prefixes(
            unexpected_keys, ("_qwen25o_encoder.",)
        ),
    }
    ignored = {name: keys for name, keys in ignored.items() if keys}
    if not ignored or not os.path.isdir(checkpoint):
        return {name: len(keys) for name, keys in ignored.items()}

    config_path = os.path.join(checkpoint, "config.json")
    if not os.path.isfile(config_path):
        logger.warning(
            "Cannot verify whether ignored checkpoint backbone weights were frozen: "
            "missing {}",
            config_path,
        )
        return {name: len(keys) for name, keys in ignored.items()}

    with open(config_path, "r", encoding="utf-8") as stream:
        checkpoint_config = json.load(stream)

    if "thinker" in ignored:
        logger.info(
            "Ignoring {} checkpoint Thinker tensors and keeping the initialized "
            "backbone. Stage 1 does not train the Thinker; LoRA Stage 2 does not "
            "remap model.* onto PEFT keys.",
            len(ignored["thinker"]),
        )
    if "qwen25o_encoder" in ignored and checkpoint_config.get(
        "finetune_speech_encoder", False
    ):
        raise RuntimeError(
            "weights_only loading would ignore a finetuned Qwen2.5-Omni encoder. "
            "Materialize and load the encoder before continuing."
        )

    return {name: len(keys) for name, keys in ignored.items()}


def _validate_resume_batch_settings(training_args, checkpoint: str) -> None:
    """Reject full-state resume when batch boundaries would change."""
    saved_args_path = os.path.join(checkpoint, "training_args.bin")
    if not os.path.isfile(saved_args_path):
        raise ValueError(
            "Cannot verify resume batch settings because the checkpoint is missing "
            f"training_args.bin: {checkpoint}"
        )

    try:
        saved_args = torch.load(
            saved_args_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        saved_args = torch.load(saved_args_path, map_location="cpu")

    batch_fields = (
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_tokens_per_batch",
    )
    mismatches = []
    for name in batch_fields:
        saved_value = getattr(saved_args, name, None)
        current_value = getattr(training_args, name, None)
        if saved_value != current_value:
            mismatches.append(
                f"{name}: checkpoint={saved_value!r}, current={current_value!r}"
            )

    if mismatches:
        raise ValueError(
            "Batch settings changed since the checkpoint; checkpoint_load_mode='resume' "
            "requires identical batch boundaries. Restore the checkpoint values or use "
            "checkpoint_load_mode='weights_only' for a new training state. Mismatches: "
            + "; ".join(mismatches)
        )


def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

def parse_training_args(argv=None):
    """Parse a YAML config and allow explicit CLI arguments to override it."""
    argv = list(sys.argv[1:] if argv is None else argv)

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, help="Path to a YAML training config.")
    config_args, remaining_args = config_parser.parse_known_args(argv)

    argument_types = (ModelArguments, DataTrainingArguments, TrainingArguments)
    parser = HfArgumentParser(argument_types)

    if config_args.config:
        config_path = os.path.abspath(config_args.config)
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        if not isinstance(config, dict):
            raise ValueError(f"Training config must contain a YAML mapping: {config_path}")

        # Keep the SwanLab credential out of dataclasses, logs, and the uploaded run config.
        swanlab_api_key = config.pop("swanlab_api_key", None)
        if swanlab_api_key is not None:
            if not isinstance(swanlab_api_key, str) or not swanlab_api_key.strip():
                raise ValueError("swanlab_api_key must be a non-empty string")
            os.environ["SWANLAB_API_KEY"] = swanlab_api_key.strip()
            os.environ["SWANLAB_MODE"] = "cloud"

        valid_keys = {field.name for argument_type in argument_types for field in fields(argument_type)}
        unknown_keys = sorted(set(config) - valid_keys)
        if unknown_keys:
            raise ValueError(
                f"Unknown training config keys in {config_path}: {', '.join(unknown_keys)}"
            )

        # argparse applies explicit CLI values after defaults, so command-line
        # arguments can override values loaded from YAML.
        parser.set_defaults(**config)

    if not config_args.config and len(remaining_args) == 1 and remaining_args[0].endswith(".json"):
        return parser.parse_json_file(json_file=os.path.abspath(remaining_args[0]))

    return parser.parse_args_into_dataclasses(args=remaining_args)


@logger.catch(onerror=lambda _: sys.exit(1))
def main():
    model_args, data_args, training_args = parse_training_args()

    if training_args.do_eval and not is_webdataset_stream_config(data_args.dataset_name):
        raise ValueError(
            "src/train.py only supports in-training validation for native "
            "WebDataset configs with a 'validation' section; run evaluation "
            "through src.eval or inference through src.infer instead of setting do_eval"
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),                          # Output to terminal
            # logging.FileHandler("log/train.log", mode='w', encoding='utf-8')  # Output to file
        ]
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    training_args.gradient_checkpointing_kwargs = {'use_reentrant': False}

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")


    # Detecting last checkpoint.
    # Auto-resume from output_dir only when resume_from_checkpoint is explicitly "auto".
    last_checkpoint = None
    resume_from_checkpoint_value = getattr(training_args, "resume_from_checkpoint", None)
    resume_from_checkpoint_value = resume_from_checkpoint_value.replace('"', '').strip() if isinstance(resume_from_checkpoint_value, str) else resume_from_checkpoint_value
    wants_auto_resume = resume_from_checkpoint_value == "auto"

    if os.path.isdir(training_args.output_dir) and training_args.do_train:
        if wants_auto_resume:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is not None:
                logger.info(
                    f"Resume from checkpoint is 'auto': detected {last_checkpoint} in {training_args.output_dir}, resuming."
                )
                training_args.overwrite_output_dir = False
            else:
                logger.warning(
                    f"Resume from checkpoint is 'auto' but no checkpoint found in {training_args.output_dir}. Starting from scratch."
                )
        else:
            # Not auto mode: use standard HF logic (check when overwrite_output_dir is False)
            if not training_args.overwrite_output_dir:
                last_checkpoint = get_last_checkpoint(training_args.output_dir)
                if last_checkpoint is not None:
                    logger.info(
                        f"Checkpoint detected in {training_args.output_dir}, resuming from {last_checkpoint}."
                    )
                elif len(os.listdir(training_args.output_dir)) > 0:
                    training_args.overwrite_output_dir = True
                    logger.warning(
                        f"Output directory ({training_args.output_dir}) already exists and is not empty, "
                        "but no checkpoints found. Overwriting existing files."
                    )

    if (
        str(getattr(training_args, "checkpoint_load_mode", "weights_only"))
        .strip()
        .lower()
        == "resume"
    ):
        resume_checkpoint = resume_from_checkpoint_value
        if resume_checkpoint in (None, False, "auto"):
            resume_checkpoint = last_checkpoint
        if isinstance(resume_checkpoint, str) and os.path.isdir(resume_checkpoint):
            _validate_resume_batch_settings(training_args, resume_checkpoint)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if getattr(model_args, "finetune_speech_encoder", False):
        if not getattr(model_args, "use_sensevoice_feature", False) and not getattr(
            model_args, "use_qwen3_feature", False
        ) and not getattr(
            model_args, "use_whisper_fetaure", False
        ) and not getattr(
            model_args, "use_qwen25omni_feature", False
        ):
            raise ValueError(
                "finetune_speech_encoder=True requires use_sensevoice_feature, use_qwen3_feature, use_whisper_fetaure, or use_qwen25omni_feature"
            )
        if getattr(model_args, "only_train_llm", False):
            logger.warning(
                "finetune_speech_encoder is ignored when only_train_llm=True (speech encoders stay frozen)."
            )

    user_encoder_flags = [
        bool(getattr(model_args, "use_sensevoice_feature", False)),
        bool(getattr(model_args, "use_qwen3_feature", False)),
        bool(getattr(model_args, "use_whisper_fetaure", False)),
        bool(getattr(model_args, "use_qwen25omni_feature", False)),
    ]
    if sum(user_encoder_flags) > 1:
        raise ValueError(
            "Please enable only one user audio encoder feature flag among: "
            "use_sensevoice_feature / use_qwen3_feature / use_whisper_fetaure / "
            "use_qwen25omni_feature."
        )
    convert_from_lora = getattr(model_args, "convert_from_lora", False)
    if convert_from_lora:
        assert model_args.use_lora, "convert_from_lora=True requires use_lora=True"
        assert PEFT_AVAILABLE, "convert_from_lora requires PEFT (pip install peft)"

    model, tokenizer, embedding_size, new_embedding_size = load_flexislm_model_and_tokenizer(
        qwen2_model_path=model_args.model_name_or_path,
        model_args=model_args,
    )
    # Single model path: wrap the main model's LLM (model.model) with LoRA when enabled.
    if model_args.use_lora:
        logger.info('Applying LoRA to main model.model (FlexiSLM).')
        target_modules = [m.strip() for m in model_args.lora_target_modules.split(',')]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            bias=model_args.lora_bias,
            modules_to_save=None,
        )
        model.model = get_peft_model(model.model, lora_config)
        if model_args.torch_dtype not in ['auto', None]:
            print(f'Converting model to {model_args.torch_dtype}')
            model.model.to(getattr(torch, model_args.torch_dtype))
        model.model.print_trainable_parameters()
        model.config.use_lora = True
        model.config.lora_rank = model_args.lora_rank
        model.config.lora_alpha = model_args.lora_alpha
        model.config.force_use_combined_embedding = model_args.force_use_combined_embedding
    # -----------------------------------------------------------------------------
    # only_train_talker: freeze everything except talker modules
    # -----------------------------------------------------------------------------
    def only_unfreeze_the_talker(model):
        """Freeze all params, then unfreeze the talker branch."""
        for param in model.parameters():
            param.requires_grad = False

        if hasattr(model, "talker_model") and model.talker_model is not None:
            for p in model.talker_model.parameters():
                p.requires_grad = True

        if bool(getattr(model.config, "use_joint_text_audio_vocab", True)):
            if hasattr(model, "lm_head") and model.lm_head is not None:
                for p in model.lm_head.parameters():
                    p.requires_grad = True
            if hasattr(model, "model") and hasattr(model.model, "embed_tokens") and model.model.embed_tokens is not None:
                for p in model.model.embed_tokens.parameters():
                    p.requires_grad = True

    # -----------------------------------------------------------------------------
    # convert_from_lora: unfreeze all parameters for full-parameter training
    # -----------------------------------------------------------------------------
    def unfreeze_all_parameters(model):
        """Unfreeze all parameters for full-parameter training."""
        for param in model.parameters():
            param.requires_grad = True

    # -----------------------------------------------------------------------------
    # freeze_talker: freeze talker modules (talker_model)
    # -----------------------------------------------------------------------------
    def freeze_the_talker(model):
        """Freeze talker-related parameters. Keeps LLM (and LoRA if present) trainable."""
        if hasattr(model, "talker_model") and model.talker_model is not None:
            for p in model.talker_model.parameters():
                p.requires_grad = False
            assert not model_args.use_combined_embedding, "freeze_talker=True requires use_combined_embedding=False"
            # for p in model.talker_model.lm_to_talker_proj.parameters():
            #     p.requires_grad = True
            # for p in model.talker_model.talker_embed_to_hidden.parameters():
            #     p.requires_grad = True
            # for p in model.talker_model.length_embedding.parameters():
            #     p.requires_grad = True

    # -----------------------------------------------------------------------------
    # only_train_llm: freeze everything except main LLM and audio input adapters.
    # -----------------------------------------------------------------------------
    def only_unfreeze_the_llm(model, use_lora: bool = False):
        """Freeze all params, then unfreeze the main LLM and audio input adapters.

        With LoRA, only the LoRA adapter weights inside model are unfrozen.
        Also unfreezes lm_head, audio_embed_transform, and
        input_merging_transformer. Freezes talker_model, combined_embed_proj,
        length components, etc.
        """
        for param in model.parameters():
            param.requires_grad = False

        if hasattr(model, "model") and model.model is not None:
            if use_lora:
                if hasattr(model.model, "enable_adapter_layers"):
                    model.model.enable_adapter_layers()
                for name, p in model.model.named_parameters():
                    if "lora_" in name or "modules_to_save" in name:
                        p.requires_grad = True
            else:
                # Unfreeze main LLM (transformer layers + embed_tokens)
                for p in model.model.parameters():
                    p.requires_grad = True
        if hasattr(model, "framerate_embeddings") and model.framerate_embeddings is not None:
            for p in model.framerate_embeddings.parameters():
                p.requires_grad = True

        # Unfreeze lm_head (text prediction head, may include audio for joint vocab)
        if hasattr(model, "lm_head") and model.lm_head is not None:
            for p in model.lm_head.parameters():
                p.requires_grad = True
        if hasattr(model, "audio_embed_transform") and model.audio_embed_transform is not None:
            for p in model.audio_embed_transform.parameters():
                p.requires_grad = True
        if hasattr(model, "input_merging_transformer") and model.input_merging_transformer is not None:
            for p in model.input_merging_transformer.parameters():
                p.requires_grad = True

    if getattr(model_args, "only_train_llm", False):
        assert not getattr(model_args, "only_train_talker", False), "only_train_llm and only_train_talker cannot be True at the same time"
        assert not model_args.freeze_llm, "only_train_llm and freeze_llm cannot be True at the same time"
        try:
            model.config.only_train_llm = True
        except Exception:
            pass
        setattr(model, "only_train_llm", True)
        llm_train_scope = "main LLM LoRA adapters" if model_args.use_lora else "main LLM (model)"
        logger.info(f"only_train_llm=True: freezing all parameters except {llm_train_scope}, lm_head, audio_embed_transform, and input_merging_transformer...")
        only_unfreeze_the_llm(model, use_lora=model_args.use_lora)
        for name, p in model.named_parameters():
            if p.requires_grad:
                logger.info(f"Trainable (only_train_llm): {name} with shape {tuple(p.shape)}")
    elif getattr(model_args, "only_train_talker", False):
        # Make the flag discoverable by the model code as well.
        model.config.only_train_talker = True
        setattr(model, "only_train_talker", True)

        logger.info("only_train_talker=True: freezing all parameters except talker modules...")
        only_unfreeze_the_talker(model)
        # Log trainable parameters
        for name, p in model.named_parameters():
            if p.requires_grad:
                logger.info(f"Trainable (only_train_talker): {name} with shape {tuple(p.shape)}")
    else:
        model.config.only_train_talker = False
        model.config.only_train_llm = False

    # freeze_talker: freeze talker modules and exclude talker loss from training
    if getattr(model_args, "freeze_talker", False):
        assert not getattr(model_args, "only_train_talker", False), "freeze_talker and only_train_talker cannot be True at the same time"
        try:
            model.config.freeze_talker = True
        except Exception:
            pass
        logger.info("freeze_talker=True: freezing talker modules, excluding talker loss from training (loss = text_loss only)...")
        freeze_the_talker(model)
        for name, p in model.named_parameters():
            if not p.requires_grad and "talker" in name.lower():
                logger.info(f"Frozen (freeze_talker): {name}")

    # When using LoRA, print trainable parameter information
    if model_args.use_lora:
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
        elif hasattr(model, "model") and hasattr(model.model, "print_trainable_parameters"):
            model.model.print_trainable_parameters()
        trainable = [(n, tuple(p.shape)) for n, p in model.named_parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(f"use_lora=True: {len(trainable)} trainable parameter groups, {n_trainable:,} trainable params ({100.0 * n_trainable / n_total:.2f}% of {n_total:,} total)")
        for name, shape in trainable:
            logger.info(f"  Trainable (LoRA): {name} with shape {shape}")
    if getattr(model_args, "early_diverge_talker", False):
        try:
            model.config.early_diverge_talker = True
        except Exception:
            pass
        logger.info("early_diverge_talker=True: talker input hidden state will use the -6th layer of the LLM.")

    if getattr(model_args, "enable_flexible_framerate", False):
        try:
            model.config.enable_flexible_framerate = True
        except Exception:
            pass
        logger.info("enable_flexible_framerate=True: flexible frame rate with SenseVoice feature merging (similarity-based) will be enabled.")

    if getattr(model_args, "uniform_merging", False):
        try:
            model.config.uniform_merging = True
            model.config.enable_flexible_framerate = True
        except Exception:
            pass
        logger.info("uniform_merging=True: input features will use uniform merging with a random 4-12 Hz target rate.")

    if getattr(model_args, "output_uniform_merging", False):
        try:
            model.config.output_uniform_merging = True
        except Exception:
            pass
        logger.info("output_uniform_merging=True: FlexiCodec output-side (assistant) merging will be UNIFORM with a random 4-12 Hz target rate (ablation).")

    if getattr(model_args, "use_sinusoidal", False):
        try:
            model.config.use_sinusoidal = True
        except Exception:
            pass
        logger.info("use_sinusoidal=True: using continuous sinusoidal framerate embedding instead of learnable discrete embeddings.")

    if getattr(model_args, "per_sample_frame_rate_embed", False):
        try:
            model.config.per_sample_frame_rate_embed = True
        except Exception:
            pass
        logger.info("per_sample_frame_rate_embed=True: using per-sample frame rate (code_lens/feature_lens*15, range 0-15 Hz) instead of unified merge threshold embed.")

    if getattr(model_args, "talker_embed_v2", False):
        try:
            model.config.talker_embed_v2 = True
        except Exception:
            pass
        logger.info(
            "talker_embed_v2=True: audio/length/framerate cond stay at talker_hidden_size; "
            "Thinker hidden states stay at hidden_size until talker_cond_proj."
        )

    if hasattr(model_args, "text_loss_weight"):
        try:
            model.config.text_loss_weight = model_args.text_loss_weight
            logger.info(f"text_loss_weight={model_args.text_loss_weight}: text loss multiplier in combined loss.")
        except Exception:
            pass

    if hasattr(model_args, "length_loss_weight"):
        try:
            model.config.length_loss_weight = model_args.length_loss_weight
            logger.info(f"length_loss_weight={model_args.length_loss_weight}: length loss multiplier in combined loss.")
        except Exception:
            pass
    if getattr(model_args, "predict_second_audio_token", False):
        logger.info(
            "predict_second_audio_token=True: replacing the talker length-prediction head "
            "with a second-audio-token prediction head (group size 2 ablation)."
        )

    if model_args.freeze_llm:
        assert not getattr(model_args, "only_train_talker", False), "only_train_talker and freeze_llm cannot be True at the same time"
        assert not getattr(model_args, "only_train_llm", False), "only_train_llm and freeze_llm cannot be True at the same time"
        if not model_args.use_lora:
            logger.info("freeze_llm=True: freezing only the main LLM backbone and text LM head.")
            for param in model.model.parameters():
                param.requires_grad = False
            for param in model.lm_head.parameters():
                param.requires_grad = False
        else:
            logger.info("freeze_llm skipped due to LoRA usage")

    if (
        model_args.freeze_adaptor
        and not getattr(model_args, "only_train_talker", False)
        and not getattr(model_args, "only_train_llm", False)
    ):
        # Handle both Linear and MLP for audio_embed_transform
        if hasattr(model.audio_embed_transform, 'weight'):
            # Linear layer
            model.audio_embed_transform.weight.requires_grad = False
            if hasattr(model.audio_embed_transform, 'bias') and model.audio_embed_transform.bias is not None:
                model.audio_embed_transform.bias.requires_grad = False
        elif hasattr(model.audio_embed_transform, 'fc1'):
            # MLP layer
            model.audio_embed_transform.fc1.weight.requires_grad = False
            if model.audio_embed_transform.fc1.bias is not None:
                model.audio_embed_transform.fc1.bias.requires_grad = False
            model.audio_embed_transform.fc2.weight.requires_grad = False
            if model.audio_embed_transform.fc2.bias is not None:
                model.audio_embed_transform.fc2.bias.requires_grad = False
        else:
            # Fallback: freeze all parameters in audio_embed_transform
            for param in model.audio_embed_transform.parameters():
                param.requires_grad = False

    def parse_only_train_modules():
        raw_names = getattr(model_args, "only_train_modules", None)
        if not raw_names:
            return []
        names = [name.strip() for name in raw_names.split(",") if name.strip()]
        if len(names) != len(set(names)):
            raise ValueError(f"only_train_modules contains duplicate names: {names}")
        return names

    def component_parameters(component_name):
        # ``llm_lora`` is a virtual component for selecting only PEFT adapter
        # parameters without unfreezing the wrapped Qwen backbone.
        if component_name == "llm_lora":
            if not model_args.use_lora:
                raise ValueError("only_train_modules=llm_lora requires use_lora=True")
            parameters = [
                param
                for name, param in model.model.named_parameters()
                if "lora_" in name or "modules_to_save" in name
            ]
            if not parameters:
                raise ValueError("only_train_modules=llm_lora found no PEFT adapter parameters")
            return parameters

        component = getattr(model, component_name, None)
        if component is None:
            raise ValueError(f"only_train_modules references missing component: {component_name}")
        if isinstance(component, torch.nn.Parameter):
            return [component]
        if isinstance(component, torch.nn.Module):
            parameters = list(component.parameters())
            if not parameters:
                raise ValueError(f"only_train_modules component has no parameters: {component_name}")
            return parameters
        raise TypeError(
            f"only_train_modules component {component_name} must be an nn.Module or nn.Parameter, "
            f"got {type(component).__name__}"
        )

    def configure_only_train_modules(component_names):
        """Apply the explicit YAML trainable-component allowlist."""
        for param in model.parameters():
            param.requires_grad = False
        for component_name in component_names:
            for param in component_parameters(component_name):
                param.requires_grad = True

    def validate_only_train_modules(component_names):
        allowed_ids = {
            id(param)
            for component_name in component_names
            for param in component_parameters(component_name)
        }
        unexpected = [
            name for name, param in model.named_parameters()
            if param.requires_grad and id(param) not in allowed_ids
        ]
        frozen = [
            name for name, param in model.named_parameters()
            if id(param) in allowed_ids and not param.requires_grad
        ]
        if unexpected or frozen:
            raise RuntimeError(
                "only_train_modules scope mismatch: "
                f"unexpected trainable={unexpected[:20]}, selected but frozen={frozen[:20]}"
            )
        counts = {
            name: sum(param.numel() for param in component_parameters(name) if param.requires_grad)
            for name in component_names
        }
        logger.info(
            f"Validated only_train_modules={component_names}: counts={counts}, "
            f"total={sum(counts.values()):,}"
        )

    selected_train_modules = parse_only_train_modules()
    if selected_train_modules:
        if getattr(model_args, "only_train_talker", False) or getattr(model_args, "only_train_llm", False):
            raise ValueError("only_train_modules cannot be combined with only_train_talker or only_train_llm")
        if getattr(model_args, "freeze_talker", False) and "talker_model" in selected_train_modules:
            raise ValueError("freeze_talker=True conflicts with talker_model in only_train_modules")
        configure_only_train_modules(selected_train_modules)
        validate_only_train_modules(selected_train_modules)
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"Trainable (only_train_modules): {name} with shape {tuple(param.shape)}")

    if training_args.gradient_checkpointing and not model_args.freeze_llm and not getattr(model_args, "only_train_talker", False):
        try:
            model.enable_input_require_grads()  # Some models provide built-in support.
        except AttributeError:
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # logger.info(f"model {model}")

    def print_grad_status(model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"=> parameter {name} requires_grad is True.")
        f'num_params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}'
        f'num_params_talker: {sum(p.numel() for p in model.talker_model.parameters() if p.requires_grad)}'
        f'num_params_proj: {sum(p.numel() for p in model.audio_embed_transform.parameters() if p.requires_grad)}'
        f'num_params_model: {sum(p.numel() for p in model.model.parameters() if p.requires_grad)}'

    print_grad_status(model)


    # breakpoint()
    # Native WebDataset streams bypass the map-style training dataset.
    native_webdataset = is_webdataset_stream_config(data_args.dataset_name)
    train_dataset_builder = build_qwen2_webdataset if native_webdataset else Qwen2Dataset
    dataset_kwargs = dict(
        max_padding_length=model_args.model_max_length,
        variable_length=data_args.variable_length,
        output_dir=training_args.output_dir,
        training_args=training_args,
        shift_token=False,
        create_position_ids=True,
        create_attention_mask=False,
        create_attention_mask_2d=False,
        create_loss_mask=False,
        max_num_frame=model_args.max_num_frame,
        max_fps=model_args.max_fps,
        reset_position_ids=data_args.reset_position_ids,
        reset_attention_mask=data_args.reset_attention_mask,
        seed=training_args.seed,
        cross_dataset_joint=data_args.cross_dataset_joint,
        dataset_joint=data_args.dataset_joint,
        use_megatron=False,
        use_qwen3_feature=getattr(model_args, "use_qwen3_feature", False),
        use_qwen25o_feature=getattr(model_args, "use_qwen25omni_feature", False),
        use_whisper_fetaure=getattr(model_args, "use_whisper_fetaure", False),
        use_omni_token=getattr(model_args, "use_omni_token", False),
        disable_text_normalize_llm=data_args.disable_text_normalize,
    )
    train_dataset = train_dataset_builder(
        data_args.dataset_name,
        tokenizer,
        **dataset_kwargs,
    )
    eval_dataset = None
    if native_webdataset:
        eval_dataset = build_qwen2_webdataset(
            data_args.dataset_name,
            tokenizer,
            split="validation",
            **dataset_kwargs,
        )
    if training_args.do_eval and eval_dataset is None:
        raise ValueError(
            "do_eval/eval_strategy requires a 'validation' section in the "
            f"dataset YAML: {data_args.dataset_name}"
        )
    if eval_dataset is not None and not training_args.do_eval:
        logger.warning(
            "Dataset YAML defines a validation split but eval_strategy is 'no'; "
            "set eval_strategy=steps and eval_steps to run validation."
        )
    if training_args.do_train:
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            # train_dataset = train_dataset.select(range(max_train_samples))
            train_dataset = train_dataset[:max_train_samples]


    # Training
    if training_args.do_train:
        checkpoint = None
        if convert_from_lora:
            # convert_from_lora: load from convert_from_lora_checkpoint or resume_from_checkpoint
            ckpt_path = getattr(model_args, "convert_from_lora_checkpoint", None)
            if ckpt_path:
                checkpoint = ckpt_path.replace('"', '').strip() if isinstance(ckpt_path, str) else ckpt_path
            else:
                resume_val = getattr(training_args, "resume_from_checkpoint", None)
                resume_val = resume_val.replace('"', '').strip() if isinstance(resume_val, str) else resume_val
                if resume_val and resume_val not in ("", "auto"):
                    checkpoint = resume_val
            assert checkpoint, "convert_from_lora=True requires convert_from_lora_checkpoint or resume_from_checkpoint"
        else:
            resume_val = getattr(training_args, "resume_from_checkpoint", None)
            resume_val = resume_val.replace('"', '').strip() if isinstance(resume_val, str) else resume_val
            if resume_val and resume_val not in ("", "auto"):
                checkpoint = resume_val
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint

        if isinstance(checkpoint, str) and checkpoint not in ("", "auto"):
            checkpoint = _resolve_checkpoint_path(
                checkpoint, process_index=training_args.process_index
            )

        checkpoint_load_mode = str(
            getattr(training_args, "checkpoint_load_mode", "weights_only")
        ).strip().lower()
        if checkpoint_load_mode not in {"resume", "weights_only"}:
            raise ValueError(
                "checkpoint_load_mode must be 'resume' or 'weights_only', got "
                f"{checkpoint_load_mode!r}"
            )
        if convert_from_lora and checkpoint_load_mode == "resume":
            raise ValueError(
                "convert_from_lora is incompatible with checkpoint_load_mode='resume'; "
                "use checkpoint_load_mode='weights_only'"
            )

        trainer_resume_checkpoint = None
        if checkpoint is not None and checkpoint_load_mode == "resume":
            if not os.path.isdir(checkpoint):
                raise ValueError(
                    "checkpoint_load_mode='resume' requires a checkpoint directory, got "
                    f"{checkpoint}"
                )
            required_state_files = ["trainer_state.json", "optimizer.pt", "scheduler.pt"]
            missing_state_files = [
                name
                for name in required_state_files
                if not os.path.isfile(os.path.join(checkpoint, name))
            ]
            rank = training_args.process_index
            rank_rng = os.path.join(checkpoint, f"rng_state_{rank}.pth")
            shared_rng = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rank_rng) and not os.path.isfile(shared_rng):
                missing_state_files.append(os.path.basename(rank_rng))
            if getattr(train_dataset, "is_native_webdataset", False):
                native_state = f"native_dataloader_state_rank{rank}.json"
                if not os.path.isfile(os.path.join(checkpoint, native_state)):
                    missing_state_files.append(native_state)
            if missing_state_files:
                raise ValueError(
                    "checkpoint_load_mode='resume' requires complete Trainer state; "
                    f"missing from {checkpoint}: {', '.join(missing_state_files)}"
                )
            trainer_resume_checkpoint = checkpoint
            logger.info(
                "Checkpoint load mode 'resume': restoring complete Trainer state from %s",
                checkpoint,
            )
        elif checkpoint is not None:
            logger.info(
                "Checkpoint load mode 'weights_only': loading model weights from %s and "
                "starting optimizer, scheduler, step, RNG, and dataloader state from scratch",
                checkpoint,
            )

        if checkpoint is not None and checkpoint_load_mode == "weights_only":
            initial_merging_state = None
            should_reinitialize_merging = bool(
                getattr(training_args, "reinitialize_input_merging_transformer", False)
            )
            if should_reinitialize_merging:
                merging_module = getattr(model, "input_merging_transformer", None)
                if merging_module is None:
                    raise ValueError(
                        "reinitialize_input_merging_transformer=True requires "
                        "model.input_merging_transformer"
                    )
                initial_merging_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in merging_module.state_dict().items()
                }

            # Support single-file and sharded Hugging Face checkpoints
            if os.path.isdir(checkpoint):
                state_dict = _load_state_dict_from_checkpoint(checkpoint)
            elif os.path.isfile(checkpoint):
                if checkpoint.endswith(".bin"):
                    state_dict = torch.load(checkpoint, map_location="cpu")
                    print(f"Loaded state dict from {checkpoint}")
                elif checkpoint.endswith(".safetensors"):
                    try:
                        from safetensors.torch import load_file as safe_load_file
                        state_dict = safe_load_file(checkpoint)
                        print(f"Loaded state dict from {checkpoint}")
                    except ImportError:
                        raise RuntimeError("safetensors is not installed but safetensors checkpoint found.")
                else:
                    raise FileNotFoundError(f"Checkpoint file {checkpoint} does not have a supported extension.")
            else:
                raise FileNotFoundError(f"Checkpoint path {checkpoint} does not exist.")
            incompatible_tensors = _drop_incompatible_state_dict_tensors(model, state_dict)
            if incompatible_tensors:
                logger.warning(
                    "Dropped {} checkpoint tensors with incompatible shapes (keeping freshly "
                    "initialized values): {}",
                    len(incompatible_tensors),
                    ", ".join(
                        f"{key} {src}->{dst}"
                        for key, src, dst in incompatible_tensors[:20]
                    ),
                )
            transfer_counts = _validate_weights_only_transfer_components(
                model,
                state_dict,
                reinitialize_input_merging_transformer=should_reinitialize_merging,
            )
            logger.info(
                "Validated weights_only transfer components from {}: {}",
                checkpoint,
                ", ".join(
                    f"{name}={count}" for name, count in transfer_counts.items()
                ),
            )
            load_result = model.load_state_dict(state_dict, strict=False, assign=True)
            if initial_merging_state is not None:
                model.input_merging_transformer.load_state_dict(
                    initial_merging_state, strict=True, assign=True
                )
                logger.info(
                    "Restored fresh Stage 2 input_merging_transformer initialization "
                    "after loading exported model checkpoint %s",
                    checkpoint,
                )
                initial_merging_state = None

            missing_keys = set(load_result.missing_keys)
            unexpected_keys = set(load_result.unexpected_keys)
            ignored_counts = _validate_ignored_frozen_checkpoint_weights(
                checkpoint,
                unexpected_keys,
            )

            expected_missing_prefixes = [
                "model.base_model.model.",
                # no_talker checkpoints omit these; keep the constructed Talker.
                "talker_model.",
            ]
            expected_missing = _keys_with_prefixes(
                missing_keys,
                expected_missing_prefixes,
            )
            expected_missing |= {key for key, _, _ in incompatible_tensors}
            expected_unexpected = _keys_with_prefixes(
                unexpected_keys,
                ("model.", "_qwen25o_encoder.", "speech_delay_embeddings"),
            )
            remaining_missing = sorted(missing_keys - expected_missing)
            remaining_unexpected = sorted(unexpected_keys - expected_unexpected)

            logger.info(
                "Loaded checkpoint weights from {}; kept {} initialized PEFT Thinker/LoRA "
                "parameters and ignored frozen checkpoint weights: {}",
                checkpoint,
                len(expected_missing),
                ", ".join(
                    f"{name}={count}" for name, count in ignored_counts.items()
                ) or "none",
            )
            if remaining_missing:
                logger.warning(
                    "Checkpoint is missing {} non-backbone parameters: {}",
                    len(remaining_missing),
                    remaining_missing[:20],
                )
            if remaining_unexpected:
                logger.warning(
                    "Checkpoint contains {} additional non-backbone parameters: {}",
                    len(remaining_unexpected),
                    remaining_unexpected[:20],
                )

            # convert_from_lora: merge LoRA into base, then switch to full-parameter training
            if convert_from_lora:
                logger.info("convert_from_lora=True: merging LoRA adapter into base model...")
                if hasattr(model.model, "merge_and_unload"):
                    model.model = model.model.merge_and_unload()
                    logger.info("LoRA merged successfully. Switching to full-parameter training.")
                else:
                    raise RuntimeError(
                        "convert_from_lora=True but model.model has no merge_and_unload. "
                        "Ensure model was created with LoRA (use_lora=True)."
                    )
                # Unset use_lora everywhere so model behaves as full-parameter (no LoRA)
                model.config.use_lora = False
                setattr(model, "use_lora", False)
                model.config.only_train_talker = False
                setattr(model, "only_train_talker", False)
                model_args.only_train_talker = False
                model_args.use_lora = False
                unfreeze_all_parameters(model)
                n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                n_total = sum(p.numel() for p in model.parameters())
                logger.info(f"Full-parameter training: {n_trainable:,} trainable params ({100.0 * n_trainable / n_total:.2f}% of {n_total:,} total)")

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable params: {n_trainable:,}')
        # num_params = lambda model: sum(p.numel() for p in model.parameters()) / 1e6
        # num_params = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        # Eagerly load lazy modules (Qwen3 encoder, FlexiCodec) before DeepSpeed init. 
        # DeepSpeed registers params at init; lazily-loaded modules would cause
        # "failed to find frozen Parameter in named params" when saving checkpoint.
        if getattr(model_args, "use_qwen3_feature", False) and hasattr(model, "qwen3_encoder_and_projection"):
            _ = model.qwen3_encoder_and_projection
            logger.info("Eagerly loaded Qwen3 encoder for DeepSpeed checkpoint compatibility.")
        if getattr(model_args, "use_qwen25omni_feature", False) and hasattr(model, "qwen25o_encoder_and_projection"):
            _ = model.qwen25o_encoder_and_projection
            logger.info("Eagerly loaded Qwen25o encoder for DeepSpeed checkpoint compatibility.")
        if getattr(model_args, "use_whisper_fetaure", False) and hasattr(model, "whisper_encoder_and_projection"):
            _ = model.whisper_encoder_and_projection
            logger.info("Eagerly loaded Whisper encoder for DeepSpeed checkpoint compatibility.")
        if hasattr(model, "flexicodec_dict"):
            _ = model.flexicodec_dict
            logger.info("Eagerly loaded FlexiCodec for DeepSpeed checkpoint compatibility.")

        if selected_train_modules:
            validate_only_train_modules(selected_train_modules)

        # Initialize our Trainer - pass model by reference; Trainer must NOT re-initialize weights
        # (no model_init) so loaded/merged weights are preserved
        trainer = ATrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            processing_class=tokenizer,
            data_collator=InterleavedDataCollator(
                tokenizer,
                use_omni_token=getattr(model_args, "use_omni_token", False),
            ),
        )
        assert trainer.model is model, "Trainer must use the same model object; model_init would re-initialize weights"
        train_result = trainer.train(
            resume_from_checkpoint=trainer_resume_checkpoint
        )

        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        
        # Skip the large final checkpoint for one-step smoke tests when requested.
        if os.environ.get("SKIP_FINAL_SAVE_MODEL", "0") == "1":
            logger.warning("SKIP_FINAL_SAVE_MODEL=1: skipping final trainer.save_model().")
        else:
            # VQAdaptor is saved automatically inside trainer.save_model().
            trainer.save_model()
            if model_args.use_lora and PEFT_AVAILABLE:
                logger.info("LoRA model with InterleavedS2S components saved successfully!")
            else:
                logger.info("Full InterleavedS2S model saved successfully!")

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()




if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()

    
