#!/usr/bin/env python
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
# coding=utf-8
"""
Interleaved Speech-to-Speech (S2S) Inference Script

This script enables inference for the ParallelS2SForCausalLM model, generating
interleaved text and audio responses from audio or text inputs.

Features:
- Interactive mode for real-time conversation
- Batch mode for processing JSONL files
- Multi-round conversation with history
- Audio output using FlexiCodec decoder
- Frame rate control for output speech (0.8-1.0)
"""

MODEL_PATH = None

# If True, prepend system prompt in dataset format (system block + user block).
# Default False: only use system message when calling generate_tts.
USE_SYSTEM_PROMPT = False

import os
import sys
import json
import re
import torch
import argparse
import logging
import random
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from tqdm import tqdm
from accelerate import init_empty_weights
import time

import numpy as np

# Add project paths (repo root + package dir), no hard-coded absolute paths.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# from flexislm.models.modeling_flexislm import ParallelS2SForCausalLM


import importlib.util, sys, os

_file = f'{os.path.dirname(os.path.abspath(__file__))}/models/modeling_flexislm.py'
spec = importlib.util.spec_from_file_location('modeling_flexislm', _file)
mod = importlib.util.module_from_spec(spec)
sys.modules['modeling_flexislm'] = mod
spec.loader.exec_module(mod)

ParallelS2SForCausalLM = mod.ParallelS2SForCausalLM







from flexislm.processor.constants import (
    AUD_START_TOKEN,
    AUD_END_TOKEN,
    AUD_START_TOKEN_OMNI,
    AUD_END_TOKEN_OMNI,
    TTS_PROMPT_LEGACY,
    TTS_PROMPT,
    AUD_TAG_TOKEN,
    DEFAULT_TTS_SYSTEM_PROMPT,
    S2S_TTS_SYSTEM_PROMPT,
    S2T_TTS_SYSTEM_PROMPT,
    T2S_TTS_SYSTEM_PROMPT,
    T2T_TTS_SYSTEM_PROMPT,
    S2S_TTS_SYSTEM_PROMPT_OMNI,
    S2T_TTS_SYSTEM_PROMPT_OMNI,
    T2S_TTS_SYSTEM_PROMPT_OMNI,
    T2T_TTS_SYSTEM_PROMPT_OMNI,
    ASR_PROMPT,
    S2T_ASR_SYSTEM_PROMPT,
)
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
USER = "user"
ASSISTANT = "assistant"
SYSTEM = "system"

# If True, dynamically select system prompt based on user and assistant audio presence.
# Default False: retains old behavior.
USE_DYNAMIC_SYSTEM_PROMPTS = True

from transformers import AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    import soundfile as sf
    import librosa
    import torchaudio
    import torchaudio.transforms as T
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: Audio libraries not available. Audio I/O will be disabled.")

# Prefer the in-repo FlexiCodec implementation used by the model itself.
from flexislm.third_party.flexicodec.flexicodec.infer import (
    prepare_model,
    encode_flexicodec,
)
FLEXICODEC_AVAILABLE = True

try:
    from flexislm.third_party.flexicodec.flexicodec.nar_tts.modeling_voicebox import VoiceboxWrapper
    from flexislm.third_party.flexicodec.flexicodec.feature_extractors import FBankGen
    FLOW_MATCHING_AVAILABLE = True
except:
    FLOW_MATCHING_AVAILABLE = False
    print("Warning: Flow matching decoder not available.")

# Setup logging
import loguru

logger = loguru.logger


def _load_state_dict_from_checkpoint(model_path: str) -> dict:
    """Load state dict from checkpoint, supporting both single-file and sharded formats.

    Supports:
    - Single: model.safetensors, pytorch_model.bin
    - Sharded safetensors: model.safetensors.index.json + model-00001-of-N.safetensors
    - Sharded pytorch: pytorch_model.bin.index.json + pytorch_model-00001-of-N.bin
    """
    st_index = os.path.join(model_path, "model.safetensors.index.json")
    st_single = os.path.join(model_path, "model.safetensors")
    bin_index = os.path.join(model_path, "pytorch_model.bin.index.json")
    bin_single = os.path.join(model_path, "pytorch_model.bin")

    # Sharded safetensors
    if os.path.isfile(st_index):
        import safetensors.torch as safetensors_torch
        with open(st_index, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", index)
        state_dict = {}
        for shard_file in sorted(set(weight_map.values())):
            shard_path = os.path.join(model_path, shard_file)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"Shard not found: {shard_path}")
            state_dict.update(safetensors_torch.load_file(shard_path, device="cpu"))
        logger.info(f"Loaded sharded safetensors ({len(weight_map)} weights from {len(set(weight_map.values()))} shards)")
        return state_dict

    # Single safetensors
    if os.path.isfile(st_single):
        import safetensors.torch as safetensors_torch
        return safetensors_torch.load_file(st_single, device="cpu")

    # Sharded pytorch
    if os.path.isfile(bin_index):
        with open(bin_index, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", index)
        state_dict = {}
        for shard_file in sorted(set(weight_map.values())):
            shard_path = os.path.join(model_path, shard_file)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"Shard not found: {shard_path}")
            state_dict.update(torch.load(shard_path, map_location="cpu"))

        logger.info(f"Loaded sharded pytorch ({len(weight_map)} weights from {len(set(weight_map.values()))} shards)")
        return state_dict

    # Single pytorch
    if os.path.isfile(bin_single):
        return torch.load(bin_single, map_location="cpu")

    raise FileNotFoundError(
        f"No model weights found in {model_path} "
        "(expected model.safetensors, model.safetensors.index.json, "
        "pytorch_model.bin, or pytorch_model.bin.index.json)"
    )


def _load_finetuned_sensevoice_weights_if_needed(model: "ParallelS2SForCausalLM", model_path: str) -> None:
    """Materialize ``sensevoice_finetune_copy`` and load finetuned weights from the checkpoint.

    Training saves SenseVoice finetune tensors under ``sensevoice_finetune_copy.*``, but that
    submodule is created lazily with FlexiCodec, so the initial ``from_pretrained`` / LoRA
    ``load_state_dict`` does not install them. This mirrors training's use of
    ``semantic_model_override`` in FlexiCodec.
    """
    if not getattr(model.config, "finetune_speech_encoder", False):
        return
    if not getattr(model.config, "use_sensevoice_feature", False):
        return
    if getattr(model.config, "only_train_llm", False):
        return

    # Triggers FlexiCodec + deep copy of SenseVoice when finetune_speech_encoder is set.
    _ = model.flexicodec_dict
    sv = getattr(model, "sensevoice_finetune_copy", None)
    if sv is None:
        logger.warning(
            "finetune_speech_encoder=True but sensevoice_finetune_copy was not created; "
            "FlexiCodec may be unavailable or misconfigured."
        )
        return

    try:
        sd = _load_state_dict_from_checkpoint(model_path)
    except FileNotFoundError as e:
        logger.warning(f"Could not load checkpoint tensors for SenseVoice finetune: {e}")
        return

    prefix = "sensevoice_finetune_copy."
    sub = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    if not sub:
        logger.info(
            "No %s* tensors in checkpoint; using SenseVoice weights copied from frozen FlexiCodec.",
            prefix,
        )
    else:
        missing, unexpected = sv.load_state_dict(sub, strict=False)
        if missing:
            logger.info("SenseVoice finetune load: %d missing keys in submodule", len(missing))
        if unexpected:
            logger.info("SenseVoice finetune load: %d unexpected keys in submodule", len(unexpected))
        logger.info("Loaded %d tensors into sensevoice_finetune_copy from checkpoint.", len(sub))

    sv.eval()
    for p in sv.parameters():
        p.requires_grad = False


@dataclass
class InterleavedInferenceConfig:
    """Configuration for Interleaved S2S inference."""
    # Model paths
    model_path: Optional[str] = None
    flexicodec_ckpt_path: Optional[str] = None
    flexicodec_config_path: Optional[str] = None
    sensevoice_path: Optional[str] = None
    
    # Flow matching decoder configuration
    use_flow_matching_decoder: bool = False
    flow_matching_ckpt_path: Optional[str] = None
    flow_matching_vocoder_path: Optional[str] = None
    flow_matching_prompt_audio_path: Optional[str] = None
    flow_matching_n_timesteps: int = 15
    flow_matching_cfg: float = 2.0
    flow_matching_rescale_cfg: float = 0.75
    
    # System prompt
    system_prompt: str = ""
    
    # Model configuration
    audio_vocab_size: int = 32768
    max_length_classes: int = 32
    use_omni_token: bool = False  # Set from checkpoint in InterleavedS2SInference.__init__
    use_sensevoice_feature: Optional[bool] = None  # Will be read from model config
    use_whisper_fetaure: Optional[bool] = None  # Will be read from model config
    # Audio input mode (determined from model config):
    # - False (Mode 1): Encode audio to semantic codes, use audio_embed_tokens with length embeddings
    # - True (Mode 2): Extract semantic features from codes, use audio_embed_transform (no length embeddings)
    
    # Frame rate control
    framerate_min: float = 0.8
    framerate_max: float = 1.0
    default_framerate: float = 1.0  # Default framerate for output generation
    input_framerate: float = 1.0  # Framerate for encoding: <=1.0 = merging_threshold, >1.0 = target_rate (Hz)
    input_base_rate: float = 12.5  # Base frame rate in Hz (used when input_framerate > 1.0 as target_rate)
    dynamic_merging: bool = True  # If True and target_rate set: search threshold to hit target; else uniform merge
    enable_flexible_framerate: bool = False  # Enable flexible frame rate with feature merging
    
    # Text-audio interleaving ratio
    text_audio_interval_ratio: List[int] = field(default_factory=lambda: [5, 10])
    
    # Generation parameters
    max_new_tokens: int = 600
    temperature: float = 0.7
    top_k: int = 20
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.0
    
    # Length prediction parameters
    length_temperature: float = 1.0
    length_top_k: int = 5
    length_top_p: float = 1.0
    
    # Output settings
    output_sample_rate: int = 16000
    decode_audio: bool = True
    
    # Model loading
    use_lora: bool = False
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    low_cpu_mem_usage: bool = True
    attn_implementation: str = "flash_attention_2"
    model_max_length: int = 2048
    cache_dir: str = "cache/"
    model_revision: str = "main"
    token: Optional[str] = None
    use_fast_tokenizer: bool = True


class ModelArgsAdapter:
    """Adapter to make config compatible with load_interleaved_s2s_model_and_tokenizer."""
    def __init__(self, config: InterleavedInferenceConfig):
        self.cache_dir = config.cache_dir
        self.use_fast_tokenizer = config.use_fast_tokenizer
        self.model_revision = config.model_revision
        self.token = config.token
        self.trust_remote_code = config.trust_remote_code
        self.model_max_length = config.model_max_length
        self.attn_implementation = config.attn_implementation
        self.torch_dtype = config.torch_dtype
        self.low_cpu_mem_usage = config.low_cpu_mem_usage


class InterleavedS2SInference:
    """
    Interleaved Speech-to-Speech inference engine.
    
    Supports:
    - Audio input → Interleaved Text + Audio output
    - Text input → Interleaved Text + Audio output
    - Multi-round conversation with history
    """
    
    def __init__(self, config: InterleavedInferenceConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        global AUD_START_TOKEN, AUD_END_TOKEN
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()

        # use_omni = self.model.config.use_omni_token
        use_omni = True  # TODO load from config
        self.config.use_omni_token = use_omni

        # Get special token IDs (Omni boundaries match training when use_omni_token in checkpoint)
        if self.config.use_omni_token:
            AUD_START_TOKEN = AUD_START_TOKEN_OMNI
            AUD_END_TOKEN = AUD_END_TOKEN_OMNI
            self.AUD_START_ID = self.tokenizer(
                AUD_START_TOKEN, add_special_tokens=False
            ).input_ids[0]
            self.AUD_END_ID = self.tokenizer(
                AUD_END_TOKEN, add_special_tokens=False
            ).input_ids[0]
        else:
            self.AUD_START_ID = self.model.config.AUD_START_TOKEN
            self.AUD_END_ID = self.model.config.AUD_END_TOKEN
        self.AUD_TAG_ID = self.model.config.AUD_TAG_TOKEN
        self.model.config.enable_flexible_framerate = True

        # Load flow matching decoder if enabled
        self.flow_matching_model = None
        self.vocoder_decode_func = None
        self.feature_extractor = None
        self.prompt_audio_cache = None  # Cache for prompt audio
        self._prompt_features_cache = {}  # Cache for extracted prompt features (path -> features dict)
        if config.use_flow_matching_decoder:
            self._load_flow_matching_decoder()
            self._load_prompt_audio()
        
        logger.info(f"InterleavedS2S Model loaded successfully on {device} w/ {self.config = }")
        logger.info(f"Text vocab size: {self.model.text_vocab_size}")
        logger.info(f"Audio vocab size: {self.model.audio_vocab_size}")
        logger.info(f"Max length classes: {self.model.max_length_classes}")
        logger.info(
            f"AUD_START_ID: {self.AUD_START_ID}, AUD_END_ID: {self.AUD_END_ID} "
            f"(use_omni_token={self.config.use_omni_token})"
        )
        if getattr(self.model.config, "use_qwen3_feature", False):
            input_mode = "Qwen3 encoder features"
        elif getattr(self.model.config, "use_qwen25o_feature", False):
            input_mode = "Qwen2.5-Omni encoder features"
        elif getattr(self.model.config, "use_whisper_fetaure", False):
            input_mode = "Whisper-large-v3 encoder features"
        elif self.config.use_sensevoice_feature:
            input_mode = "SenseVoice features"
        else:
            input_mode = "Semantic codes"
        logger.info(f"Using audio input mode: {input_mode}")
        logger.info(f"Flexible framerate enabled: {getattr(self.model.config, 'enable_flexible_framerate', False)}")
        logger.info(f"Input framerate: {self.config.input_framerate}, Output framerate: {self.config.default_framerate}")
    
    def _load_model(self) -> Tuple[ParallelS2SForCausalLM, AutoTokenizer]:
        """Load the Interleaved S2S model and tokenizer.

        When `model.config.use_lora` is True (set by train.py), the saved
        checkpoint contains the full state dict with PEFT-style key names because
        training wraps `model.model` with `get_peft_model()` before training and the
        HF Trainer saves the complete outer model via `model.save_pretrained()`.

        The key names in the checkpoint therefore look like:
            model.base_model.model.layers.N.self_attn.q_proj.lora_A.default.weight
        which does NOT match a plain ParallelS2SForCausalLM (model.layers.N.…).

        To handle this correctly we:
          1. Read the saved config to detect use_lora.
          2. Build the skeleton model (no pretrained weights) from that config.
          3. Load the raw checkpoint state dict to peek at the LoRA key names and
             infer `r` (rank) and target modules.
          4. Apply get_peft_model() to model.model – exactly as training did.
          5. Load the full state dict; keys now match.
        """
        logger.info(f"Loading InterleavedS2S model from {self.config.model_path}")
        model_path = self.config.model_path

        # ------------------------------------------------------------------
        # Step 1: read config to determine whether LoRA was used
        # ------------------------------------------------------------------
        from flexislm.models.modeling_flexislm import ParallelS2SConfig
        saved_config = ParallelS2SConfig.from_pretrained(model_path)
        # override
        saved_config.max_tokens_per_group = 16
        use_lora = getattr(saved_config, 'use_lora', False)
        torch_dtype = (
            getattr(torch, self.config.torch_dtype)
            if self.config.torch_dtype != "auto"
            else torch.bfloat16
        )
        # Flash Attention 2 requires fp16/bf16 at init/load time (not just after .to()).
        if self.config.attn_implementation == "flash_attention_2" and torch_dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            logger.warning(
                "Flash Attention 2 requires torch.float16/torch.bfloat16. "
                f"Overriding torch_dtype={torch_dtype} -> torch.bfloat16."
            )
            torch_dtype = torch.bfloat16
        if self.config.attn_implementation:
            saved_config._attn_implementation = self.config.attn_implementation
            logger.info(
                f"Using attention implementation: {self.config.attn_implementation}"
            )

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "Checkpoint was trained with use_lora=True but the `peft` library "
                    "is not installed. Install it with: pip install peft"
                )

            # ------------------------------------------------------------------
            # Step 2: load raw state dict to infer LoRA structure
            # (supports single-file and sharded safetensors/pytorch)
            # ------------------------------------------------------------------
            state_dict = _load_state_dict_from_checkpoint(model_path)

            # ------------------------------------------------------------------
            # Step 3: infer LoRA rank and target module names from key shapes
            # Key format: "model.base_model.model.….<module>.lora_A.default.weight"
            # lora_A shape is [r, in_features], so shape[0] == r
            # ------------------------------------------------------------------
            lora_a_keys = [k for k in state_dict if ".lora_A." in k]
            if not lora_a_keys:
                logger.warning(
                    "use_lora=True in config but no lora_A keys found in checkpoint. "
                    "Falling back to standard from_pretrained load."
                )
                use_lora = False
            else:
                target_modules = sorted({
                    k.split(".lora_A.")[0].rsplit(".", 1)[-1]
                    for k in lora_a_keys
                })
                lora_rank = int(state_dict[lora_a_keys[0]].shape[0])
                # lora_alpha is not recoverable from weights alone; use training default (16)
                lora_alpha = getattr(saved_config, 'lora_alpha', lora_rank // 2) or lora_rank // 2
                # TODO change
                # lora_alpha = 16
                # breakpoint()
                logger.info(
                    f"Detected LoRA checkpoint: r={lora_rank}, "
                    f"lora_alpha={lora_alpha}, target_modules={target_modules}"
                )

        if use_lora:
            # ------------------------------------------------------------------
            # Step 4: build skeleton model and apply LoRA (matches training setup)
            # ------------------------------------------------------------------
            with init_empty_weights():
                model = ParallelS2SForCausalLM._from_config(saved_config, dtype=torch_dtype)
            # model = ParallelS2SForCausalLM(saved_config)

            if getattr(saved_config, "use_qwen3_feature", False):
                pass
                # _ = model.qwen3_encoder_and_projection
            elif getattr(saved_config, "use_qwen25o_feature", False):
                _ = model.qwen25o_encoder_and_projection
            elif getattr(saved_config, "use_whisper_fetaure", False):
                _ = model.whisper_encoder_and_projection

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.0,           # no dropout at inference
                target_modules=target_modules,
                bias="none",
                modules_to_save=None,
            )
            model.model = get_peft_model(model.model, lora_config)
            logger.info("Applied LoRA config to model.model (matches train.py)")

            # ------------------------------------------------------------------
            # Step 5: load the full checkpoint – keys now match the LoRA model
            # ------------------------------------------------------------------
            load_result = model.load_state_dict(state_dict, strict=False, assign=True)
            missing_keys = load_result.missing_keys
            unexpected_keys = load_result.unexpected_keys
            logger.info("LoRA checkpoint loaded successfully.")
            if missing_keys:
                logger.info(f"Parameters NOT loaded (missing in checkpoint): {len(missing_keys)} keys")
                for k in missing_keys[:20]:
                    logger.info(f"  - {k}")
                if len(missing_keys) > 20:
                    logger.info(f"  ... and {len(missing_keys) - 20} more")
            else:
                logger.info("All model parameters were loaded from checkpoint.")
            if unexpected_keys:
                logger.info(f"Parameters in checkpoint NOT loaded (unexpected): {len(unexpected_keys)} keys")
                for k in unexpected_keys[:20]:
                    logger.info(f"  - {k}")
                if len(unexpected_keys) > 20:
                    logger.info(f"  ... and {len(unexpected_keys) - 20} more")

            # ------------------------------------------------------------------
            # Step 5b: materialize any parameters/buffers still on meta device
            # (i.e. keys that were missing from the checkpoint). Without this,
            # `model.to(device)` raises:
            #   NotImplementedError: Cannot copy out of meta tensor; no data!
            # ------------------------------------------------------------------
            text_slice = int(getattr(model, "_combined_embed_proj_text_slice", 0))
            for mod_name, module in model.named_modules():
                for pname, param in list(module._parameters.items()):
                    if param is None or not param.is_meta:
                        continue
                    full_name = f"{mod_name}.{pname}" if mod_name else pname
                    logger.info(f"Materializing missing meta parameter: {full_name}")
                    new_tensor = torch.empty(
                        param.shape, dtype=param.dtype, device="cpu"
                    )
                    with torch.no_grad():
                        if full_name == "combined_embed_proj.weight" and text_slice > 0:
                            new_tensor.zero_()
                            new_tensor[:, :text_slice].copy_(torch.eye(text_slice, dtype=param.dtype))
                            logger.info(
                                f"  -> initialized combined_embed_proj.weight as identity on first {text_slice} input dims"
                            )
                        else:
                            torch.nn.init.normal_(new_tensor, mean=0.0, std=0.02)
                    module._parameters[pname] = torch.nn.Parameter(
                        new_tensor, requires_grad=param.requires_grad
                    )
                for bname, buf in list(module._buffers.items()):
                    if buf is None or not buf.is_meta:
                        continue
                    full_name = f"{mod_name}.{bname}" if mod_name else bname
                    logger.info(f"Materializing missing meta buffer: {full_name}")
                    module._buffers[bname] = torch.zeros(
                        buf.shape, dtype=buf.dtype, device="cpu"
                    )

            logger.info(f"Moving model to {self.device}")
            model = model.to(self.device, dtype=torch_dtype)
            logger.info(f"Setting model evaluation mode")
            model.eval()
        else:
            # Non-LoRA checkpoint: standard HF load + move to dtype
            model = ParallelS2SForCausalLM.from_pretrained(
                model_path,
                device_map="cuda",
                attn_implementation=self.config.attn_implementation,
                dtype=torch_dtype,
            )
            model = model.to(self.device, dtype=torch_dtype)
            model.eval()

        # Read flags from model config (both LoRA and non-LoRA paths land here)
        if self.config.use_sensevoice_feature is None:
            self.config.use_sensevoice_feature = getattr(model.config, 'use_sensevoice_feature', False)
            logger.info(f"Read use_sensevoice_feature={self.config.use_sensevoice_feature} from model config")
        if self.config.use_whisper_fetaure is None:
            self.config.use_whisper_fetaure = getattr(model.config, 'use_whisper_fetaure', False)
            logger.info(f"Read use_whisper_fetaure={self.config.use_whisper_fetaure} from model config")

        if hasattr(model.config, 'enable_flexible_framerate'):
            self.config.enable_flexible_framerate = model.config.enable_flexible_framerate
            logger.info(f"Set enable_flexible_framerate={self.config.enable_flexible_framerate} from model config")

        _load_finetuned_sensevoice_weights_if_needed(model, model_path)

        return model, tokenizer
    
    def _load_flow_matching_decoder(self):
        """Load the flow matching decoder (VoiceBox model)."""
        if not FLOW_MATCHING_AVAILABLE:
            logger.error("Flow matching decoder requested but not available")
            raise ImportError("Flow matching dependencies not available")
        if not self.config.flexicodec_ckpt_path or not self.config.flexicodec_config_path:
            raise ValueError(
                "Flow matching decoder requires --flexicodec_ckpt and --flexicodec_config."
            )
        if not self.config.sensevoice_path:
            raise ValueError(
                "Flow matching decoder requires --sensevoice_path."
            )
        if not self.config.flow_matching_vocoder_path:
            raise ValueError(
                "Flow matching decoder requires --flow_matching_vocoder."
            )
        
        logger.info("Loading flow matching decoder...")
        
        # VoiceBox configuration
        voicebox_config = {
            'mel_dim': 128,
            'hidden_size': 1024,
            'num_layers': 16,
            'num_heads': 16,
            'cfg_scale': 0.2,
            'use_cond_code': True,
            'cond_codebook_size': 32768,
            'cond_scale_factor': 4,
            'cond_dim': 1024,
            'sigma': 1e-5,
            'time_scheduler': "cos"
        }
        
        # Load VoiceBox wrapper with local FlexiCodec checkpoint/config to avoid download
        self.flow_matching_model = VoiceboxWrapper(
            voicebox_config=voicebox_config,
            flexicodec_ckpt_path=self.config.flexicodec_ckpt_path,
            flexicodec_config_path=self.config.flexicodec_config_path,
            sensevoice_path=self.config.sensevoice_path,
        )
        
        # Load checkpoint if provided
        if self.config.flow_matching_ckpt_path:
            logger.info(f"Loading flow matching checkpoint from {self.config.flow_matching_ckpt_path}")
            if self.config.flow_matching_ckpt_path.endswith('.safetensors'):
                import safetensors.torch
                state_dict = safetensors.torch.load_file(self.config.flow_matching_ckpt_path)
            else:
                ckpt = torch.load(self.config.flow_matching_ckpt_path, map_location='cpu')
                state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
            
            load_result = self.flow_matching_model.load_state_dict(state_dict, strict=False)
            missing_keys = load_result.missing_keys
            unexpected_keys = load_result.unexpected_keys
            if missing_keys:
                logger.info(f"Flow matching: parameters NOT loaded (missing in checkpoint): {len(missing_keys)} keys")
                for k in missing_keys[:20]:
                    logger.info(f"  - {k}")
                if len(missing_keys) > 20:
                    logger.info(f"  ... and {len(missing_keys) - 20} more")
            else:
                logger.info("Flow matching: all parameters were loaded from checkpoint.")
            if unexpected_keys:
                logger.info(f"Flow matching: parameters in checkpoint NOT loaded (unexpected): {len(unexpected_keys)} keys")
                for k in unexpected_keys[:20]:
                    logger.info(f"  - {k}")
                if len(unexpected_keys) > 20:
                    logger.info(f"  ... and {len(unexpected_keys) - 20} more")
        
        self.flow_matching_model.eval()
        self.flow_matching_model = self.flow_matching_model.to(self.device)
        
        # Load vocoder
        logger.info("Loading vocoder for flow matching decoder...")
        from flexislm.third_party.flexicodec.flexicodec.nar_tts.inference_voicebox import load_vocoder

        self.vocoder_decode_func, _ = load_vocoder(self.device, vocoder_path=self.config.flow_matching_vocoder_path)
        
        # Load feature extractor for flow matching
        self.feature_extractor = FBankGen(sr=16000)
        
        logger.info("Flow matching decoder loaded successfully")
    
    def _load_prompt_audio(self):
        """Load and cache the prompt audio for flow matching decoder."""
        prompt_path = self.config.flow_matching_prompt_audio_path
        if not prompt_path:
            logger.info("No flow matching prompt audio path provided; using runtime fallback.")
            return
        
        if not os.path.exists(prompt_path):
            logger.warning(f"Prompt audio file not found: {prompt_path}")
            logger.warning("Will use silent placeholder instead")
            return
        try:
            logger.info(f"Loading prompt audio from: {prompt_path}")
            prompt_audio, sr = torchaudio.load(prompt_path)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                resampler = T.Resample(sr, 16000)
                prompt_audio = resampler(prompt_audio)
            
            # Convert to mono if needed
            if prompt_audio.shape[0] > 1:
                prompt_audio = prompt_audio.mean(dim=0, keepdim=True)
            
            # Cache the prompt audio
            self.prompt_audio_cache = prompt_audio
            logger.info(f"Prompt audio loaded: duration = {prompt_audio.shape[-1] / 16000:.2f}s")
            
        except Exception as e:
            logger.error(f"Error loading prompt audio: {e}")
            logger.warning("Will use silent placeholder instead")
            self.prompt_audio_cache = None
    
    def _build_prompt(
        self,
        user_content: str,
        history: str = "",
        system_message: Optional[str] = None,
        has_audio: bool = False,
        use_system_prompt: Optional[bool] = None,
        assistant_has_audio: bool = True,
    ) -> str:
        """Build the input prompt for the model."""
        user_content = user_content.replace('<audio>', '')
        
        has_text = bool(user_content.strip())
        
        if has_audio:
            if self.config.use_omni_token:
                user_content = user_content + AUD_START_TOKEN_OMNI
                prompt1 = f"{AUD_END_TOKEN_OMNI}{IM_END}\n{IM_START}{ASSISTANT}\n"
            else:
                user_content = user_content + AUD_START_TOKEN
                prompt1 = f"{AUD_END_TOKEN}{IM_END}\n{IM_START}{ASSISTANT}\n"
        else:
            prompt1 = f"{IM_END}\n{IM_START}{ASSISTANT}\n"

        use_sys = use_system_prompt if use_system_prompt is not None else USE_SYSTEM_PROMPT
        
        if USE_DYNAMIC_SYSTEM_PROMPTS:
            if getattr(self.config, "use_omni_token", False):
                s2s, s2t, t2s, t2t = (
                    S2S_TTS_SYSTEM_PROMPT_OMNI,
                    S2T_TTS_SYSTEM_PROMPT_OMNI,
                    T2S_TTS_SYSTEM_PROMPT_OMNI,
                    T2T_TTS_SYSTEM_PROMPT_OMNI,
                )
            else:
                s2s, s2t, t2s, t2t = (
                    S2S_TTS_SYSTEM_PROMPT,
                    S2T_TTS_SYSTEM_PROMPT,
                    T2S_TTS_SYSTEM_PROMPT,
                    T2T_TTS_SYSTEM_PROMPT,
                )
            if has_audio and assistant_has_audio:
                sys_prompt = s2s
            elif has_audio and not assistant_has_audio:
                sys_prompt = s2t
            elif not has_audio and assistant_has_audio:
                sys_prompt = t2s
            else:
                sys_prompt = t2t
            if has_audio and not assistant_has_audio and ASR_PROMPT in user_content:
                sys_prompt = S2T_ASR_SYSTEM_PROMPT

            prompt = f"{IM_START}{SYSTEM}\n{sys_prompt}{IM_END}\n{IM_START}{USER}\n{user_content}"
        elif use_sys:
            raise
            # Dataset format: system block + user block (matches dataset_interleaved.py)
            prompt = f"{IM_START}{SYSTEM}\n{DEFAULT_TTS_SYSTEM_PROMPT}{IM_END}\n{IM_START}{USER}\n{user_content}"
        else:
            # No system block; user block only
            prompt = f"{IM_START}{USER}\n{user_content}"
        # print(fr'prompt: {prompt}')
        return prompt, prompt1
    
    def _load_audio(self, audio_path: str) -> Optional[torch.Tensor]:
        """Load and resample audio to 16kHz."""
        wav, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = T.Resample(sr, 16000)
            wav = resampler(wav)
        return wav

    def _semantic_model_override_for_encode(self):
        """FlexiCodec semantic encoder override when training used ``finetune_speech_encoder`` (matches modeling forward)."""
        cfg = self.model.config
        if not getattr(cfg, "finetune_speech_encoder", False):
            return None
        if not getattr(cfg, "use_sensevoice_feature", False):
            return None
        if getattr(cfg, "only_train_llm", False):
            return None
        _ = self.model.flexicodec_dict
        return getattr(self.model, "sensevoice_finetune_copy", None)
    
    def _encode_audio(self, audio_tensor: torch.Tensor, framerate: float = 1.0) -> Union[Dict, torch.Tensor]:
        """Encode audio for user input, matching the model's training-time implementation.

        When `model.config.use_qwen3_feature`, `use_whisper_fetaure`,
        or `use_qwen25o_feature` is True, we:
            - Extract 128-bin Whisper-style mel features
              for Qwen/Whisper paths
            - Run the lazy encoder (`_encode_user_audio_qwen3`, `_encode_user_audio_whisper`,
              or `_encode_user_audio_qwen25o`)
            - Optionally apply similarity-based merging when `enable_flexible_framerate` is enabled
            - Return per-frame encoder features `[T, H_enc]`

        Otherwise, we fall back to FlexiCodec-based encoding and pass `audio_features_lens` to
        `encode_flexicodec` to mirror the new batched training path (`_forward_grouped_batch`).

        Returns:
            For Mode 1 (use_sensevoice_feature=False):
                Dict with 'semantic_codes' and 'token_lengths'
            For Mode 2 / Qwen3 / Qwen2.5-Omni / Whisper
            (use_sensevoice_feature=True, use_qwen3_feature=True,
            use_qwen25o_feature=True, or use_whisper_fetaure=True):
                When enable_flexible_framerate and framerate < 1.0:
                    Dict with 'semantic_features' [T, H] and 'token_lengths' [T]
                Otherwise: Direct semantic features tensor [T, H] (after squeeze and transpose)
        """
        # Ensure audio is [B, T] format
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(1)

        # ------------------------------------------------------------------
        # Qwen3 / Whisper / Qwen2.5-Omni encoder path (no FlexiCodec dependency for user audio)
        # ------------------------------------------------------------------
        use_qwen3 = getattr(self.model.config, "use_qwen3_feature", False)
        use_whisper = getattr(self.model.config, "use_whisper_fetaure", False)
        use_qwen25o = getattr(self.model.config, "use_qwen25o_feature", False)
        if use_qwen3 or use_qwen25o or use_whisper:
            from flexislm.dataset.dataset_override.dataset_interleaved import Qwen3FbankExtractor, Qwen25OFbankExtractor

            # Lazily construct fbank extractor
            if use_qwen3 or use_whisper:
                encoder_name = "Qwen3" if use_qwen3 else "Whisper"
                if not hasattr(self, "_qwen3_fbank_extractor"):
                    logger.info("Initializing 128-bin Whisper-style fbank extractor for inference audio encoding.")
                    self._qwen3_fbank_extractor = Qwen3FbankExtractor()
                fbank_extractor = self._qwen3_fbank_extractor
            if use_qwen25o:
                if not hasattr(self, "_qwen25o_fbank_extractor"):
                    logger.info("Initializing Qwen25OFbankExtractor for inference audio encoding.")
                    self._qwen25o_fbank_extractor = Qwen25OFbankExtractor()
                fbank_extractor = self._qwen25o_fbank_extractor
                encoder_name = "Qwen2.5-Omni"
            logger.info(f"{encoder_name = } loaded")

            # Convert to mono 1D waveform [samples]
            wav = audio_tensor
            if wav.dim() == 2:
                # [C, T] -> average channels
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0)
                else:
                    wav = wav.squeeze(0)
            elif wav.dim() == 1:
                pass
            else:
                raise ValueError(f"Unexpected audio tensor shape for {encoder_name} path: {wav.shape}")

            device = next(self.model.parameters()).device
            dtype = next(self.model.parameters()).dtype

            u_rows = torch.tensor([0], device=device, dtype=torch.long)

            # Encode with the selected encoder (matches training)
            with torch.no_grad():
                if use_qwen3:
                    # Extractor expects 16kHz mono
                    feats = fbank_extractor.extract_features(wav.cpu(), fs=16000)  # [T, 128]
                    user_audio_features = feats.unsqueeze(0).to(device=device, dtype=dtype)  # [1, T, 128]
                    user_audio_features_lens = torch.tensor(
                        [feats.shape[0]], device=device, dtype=torch.long
                    )  # [1]
                    u_codec, u_codec_lens = self.model._encode_user_audio_qwen3(
                        user_audio_features=user_audio_features,
                        user_audio_features_lens=user_audio_features_lens,
                        u_rows=u_rows,
                        dtype=dtype,
                        device=device,
                    )
                if use_whisper:
                    feats = fbank_extractor.extract_features(wav.cpu(), fs=16000)  # [T, 128]
                    user_audio_features = feats.unsqueeze(0).to(device=device, dtype=dtype)  # [1, T, 128]
                    user_audio_features_lens = torch.tensor(
                        [feats.shape[0]], device=device, dtype=torch.long
                    )  # [1]
                    u_codec, u_codec_lens = self.model._encode_user_audio_whisper(
                        user_audio_features=user_audio_features,
                        user_audio_features_lens=user_audio_features_lens,
                        u_rows=u_rows,
                        dtype=dtype,
                        device=device,
                    )
                if use_qwen25o:
                    feats = fbank_extractor.extract_features(wav.cpu(), fs=16000)  # [T, 128]
                    user_audio_features = feats.unsqueeze(0).to(device=device, dtype=dtype)  # [1, T, 128]
                    user_audio_features_lens = torch.tensor(
                        [feats.shape[0]], device=device, dtype=torch.long
                    )  # [1]
                    u_codec, u_codec_lens = self.model._encode_user_audio_qwen25o(
                        user_audio_features=user_audio_features,
                        user_audio_features_lens=user_audio_features_lens,
                        u_rows=u_rows,
                        dtype=dtype,
                        device=device,
                    )

                # Optional flexible frame rate merging (same logic as training)
                if getattr(self.model.config, "enable_flexible_framerate", False):
                    default_base_rate = 25.0 if use_qwen25o else 12.5
                    base_rate = getattr(self.config, "input_base_rate", default_base_rate)
                    # Force override if it defaults to 12.5 but we are using a 25 Hz encoder
                    if use_qwen25o and base_rate == 12.5:
                        logger.warning(f"[{encoder_name}] Found base_rate == 12.5, rewrite with 25.0")
                        base_rate = 25.0
                    if framerate > 1.0:
                        dynamic = getattr(self.config, "dynamic_merging", True)
                        mode = "dynamic" if dynamic else "uniform"
                        logger.info(
                            f"[{encoder_name}] Applying target-rate merging ({mode}): target={framerate:.1f} Hz, base={base_rate} Hz"
                        )
                        u_alignment, sim, u_codec_lens, token_lengths = (
                            self.model._perform_similarity_alignment_vectorized(
                                u_codec,
                                u_codec_lens,
                                target_rate=framerate,
                                base_rate=base_rate,
                                dynamic_merging=dynamic,
                            )
                        )
                    else:
                        logger.info(
                            f"[{encoder_name}] Applying flexible frame rate merging with threshold {framerate:.2f}"
                        )
                        u_alignment, sim, u_codec_lens, token_lengths = (
                            self.model._perform_similarity_alignment_vectorized(
                                u_codec,
                                u_codec_lens,
                                merging_threshold=framerate,
                            )
                        )
                    use_input_merging_v2 = (
                        getattr(self.model.config, "use_input_merging_transformer", False)
                        and getattr(self.model.config, "use_input_merging_transformer_v2", False)
                        and getattr(self.model.config, "input_merging_transformer_num_layers", 0) > 0
                    )
                    if use_input_merging_v2:
                        # v2 path: keep pre-merge frames; the interleaved merging
                        # transformer will perform aggregation via alignment.
                        semantic_features = u_codec.squeeze(0).to(torch.bfloat16)  # [T, H_enc]
                        return {
                            "semantic_features": semantic_features,
                            "token_lengths": token_lengths[0],
                            "sim": sim,
                            "alignment_matrix": u_alignment,  # [1, G, T]
                            "num_segments_per_item": u_codec_lens,  # [1]
                        }
                    u_codec = self.model.aggregate_features(u_codec, u_alignment)  # [1, G, H_enc]
                    u_codec = u_codec  # [1, G, H_enc]
                    semantic_features = u_codec.squeeze(0).to(torch.bfloat16)  # [G, H_enc]
                    # For Qwen/Whisper we do NOT use length embeddings at inference time, so we drop token_lengths.
                    return {
                        "semantic_features": semantic_features,
                        "token_lengths": token_lengths[0],
                        'sim': sim,
                    }

                # No flexible merging: just return encoder features
                semantic_features = u_codec.squeeze(0).to(torch.bfloat16)  # [T, H_enc]
                return semantic_features

        # ------------------------------------------------------------------
        # FlexiCodec path (SenseVoice features or semantic codes)
        # ------------------------------------------------------------------
        if not FLEXICODEC_AVAILABLE:
            raise RuntimeError("FlexiCodec is not available but audio encoding was requested")

        # Extract fbank features first (matching training collator)
        feature_extractor = self.model.flexicodec_dict["feature_extractor"]
        device = next(self.model.flexicodec_dict["model"].parameters()).device

        audio_16k = audio_tensor.to(device)

        features_list = []
        for i in range(audio_16k.shape[0]):
            features_i, _ = feature_extractor.extract_fbank(audio_16k[i : i + 1].cpu().float())
            features_list.append(features_i.squeeze(0))
        audio_features_lens = torch.tensor([x.shape[0] for x in features_list], device=device, dtype=torch.long)
        audio_features = torch.nn.utils.rnn.pad_sequence(features_list, batch_first=True).to(device)

        semantic_override = self._semantic_model_override_for_encode()

        with torch.no_grad():
            if self.model.config.use_sensevoice_feature:
                codec_output = encode_flexicodec(
                    audio_16k,
                    self.model.flexicodec_dict,
                    audio_features=audio_features,
                    audio_features_lens=audio_features_lens,
                    sample_rate=16000,
                    num_quantizers=1,
                    merging_threshold=1.0,
                    return_semantic_feature=True,
                    semantic_model_override=semantic_override,
                )
                semantic_features = codec_output.squeeze(0).transpose(0, 1).to(torch.bfloat16)  # [T, H]
                
                # Apply flexible frame rate merging if enabled (matches training in modeling_flexislm)
                if getattr(self.model.config, 'enable_flexible_framerate', False):
                    semantic_features = semantic_features.unsqueeze(0)  # [1, T, H]
                    base_rate = getattr(self.config, "input_base_rate", 12.5)
                    if framerate > 1.0:
                        dynamic = getattr(self.config, "dynamic_merging", True)
                        mode = "dynamic" if dynamic else "uniform"
                        logger.info(f"Applying target-rate merging ({mode}): target={framerate:.1f} Hz, base={base_rate} Hz")
                        u_alignment, sim, u_codec_lens, token_lengths = self.model._perform_similarity_alignment_vectorized(
                            semantic_features,
                            x_lens=None,
                            target_rate=framerate,
                            base_rate=base_rate,
                            dynamic_merging=dynamic,
                        )
                    else:
                        logger.info(f"Applying flexible frame rate merging with threshold {framerate:.2f}")
                        u_alignment, sim, u_codec_lens, token_lengths = self.model._perform_similarity_alignment_vectorized(
                            semantic_features,
                            x_lens=None,
                            merging_threshold=framerate,
                        )
                    use_input_merging_v2 = (
                        getattr(self.model.config, "use_input_merging_transformer", False)
                        and getattr(self.model.config, "use_input_merging_transformer_v2", False)
                        and getattr(self.model.config, "input_merging_transformer_num_layers", 0) > 0
                    )
                    if use_input_merging_v2:
                        # v2 path: keep pre-merge frames; let the interleaved
                        # merging transformer aggregate via alignment.
                        pre_merge_features = semantic_features.squeeze(0)  # [T, H]
                        logger.info(
                            f"v2 merging: keeping {pre_merge_features.shape[0]} pre-merge frames "
                            f"(will be aggregated to {int(u_codec_lens.max().item())} groups by transformer)"
                        )
                        return {
                            "semantic_features": pre_merge_features,
                            "token_lengths": token_lengths[0],
                            "alignment_matrix": u_alignment,  # [1, G, T]
                            "num_segments_per_item": u_codec_lens,  # [1]
                        }
                    semantic_features = self.model.aggregate_features(semantic_features, u_alignment)
                    semantic_features = semantic_features.squeeze(0)  # [G, H]
                    logger.info(f"Merged features: {semantic_features.shape[0]} groups from {codec_output.shape[-1]} original frames")
                    logger.info(f"Token lengths shape: {token_lengths.shape}, values: {token_lengths[0].tolist()[:10]}...")
                    return {"semantic_features": semantic_features, "token_lengths": token_lengths[0]}
                
                return semantic_features
            else:
                codec_output = encode_flexicodec(
                    audio_16k,
                    self.model.flexicodec_dict,
                    audio_features=audio_features,
                    audio_features_lens=audio_features_lens,
                    sample_rate=16000,
                    num_quantizers=1,
                    merging_threshold=framerate,
                    return_semantic_feature=False,
                    semantic_model_override=semantic_override,
                )
                return {
                    "semantic_codes": codec_output["semantic_codes"],
                    "token_lengths": codec_output["token_lengths"],
                }
    
    def _extract_prompt_audio_tokens(
        self,
        prompt_audio: Union[str, torch.Tensor],
        framerate: float = 1.0,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract FlexiCodec tokens and token lengths from prompt audio for TTS prompting.
        Returns discrete semantic codes and length class indices for teacher forcing.
        
        Args:
            prompt_audio: Path to audio file or [1, T] audio tensor
            framerate: Frame rate for encoding (0.8-1.0), used as merging_threshold
            
        Returns:
            (force_audio_ids, force_length_ids) or None if extraction fails.
            force_audio_ids: [P] codec token indices (0-indexed)
            force_length_ids: [P] length class indices (0 to max_length_classes-1)
        """
        if not FLEXICODEC_AVAILABLE or not hasattr(self.model, "flexicodec_dict"):
            logger.warning("FlexiCodec not available for prompt extraction")
            return None
        try:
            if isinstance(prompt_audio, str):
                audio_tensor = self._load_audio(prompt_audio)
                if audio_tensor is None:
                    return None
            else:
                audio_tensor = prompt_audio
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.dim() == 3:
                audio_tensor = audio_tensor.squeeze(1)
            device = next(self.model.flexicodec_dict["model"].parameters()).device
            audio_16k = audio_tensor.to(device)
            if audio_16k.shape[0] > 1:
                audio_16k = audio_16k.mean(dim=0, keepdim=True)
            feature_extractor = self.model.flexicodec_dict["feature_extractor"]
            features_list = []
            for i in range(audio_16k.shape[0]):
                features_i, _ = feature_extractor.extract_fbank(audio_16k[i : i + 1].cpu().float())
                features_list.append(features_i.squeeze(0))
            audio_features_lens = torch.tensor([x.shape[0] for x in features_list], device=device, dtype=torch.long)
            audio_features = torch.nn.utils.rnn.pad_sequence(features_list, batch_first=True).to(device)

            if framerate > 1.0:
                if framerate >=12.0:
                    framerate = 1.00
                elif framerate >= 8.0:
                    framerate = 0.91
                elif framerate > 7.0:
                    framerate = 0.90
                else:
                    framerate = 0.90
            semantic_override = self._semantic_model_override_for_encode()
            with torch.no_grad():
                codec_output = encode_flexicodec(
                    audio_16k,
                    self.model.flexicodec_dict,
                    audio_features=audio_features,
                    audio_features_lens=audio_features_lens,
                    sample_rate=16000,
                    num_quantizers=1,
                    merging_threshold=framerate,
                    return_semantic_feature=False,
                    semantic_model_override=semantic_override,
                )
            semantic_codes = codec_output["semantic_codes"].squeeze()  # [T]
            token_lengths = codec_output["token_lengths"].squeeze()  # [T]
            max_length_classes = getattr(self.model.config, "max_length_classes", 32)
            return semantic_codes.to(self.device), token_lengths.to(self.device)
        except Exception as e:
            logger.warning(f"Failed to extract prompt audio tokens: {e}")
            return None
    
    def _decode_audio_tokens(
        self,
        audio_tokens: torch.Tensor,
        length_ids: Optional[torch.Tensor] = None,
        prompt_audio: Optional[torch.Tensor] = None,
        prompt_audio_path: Optional[str] = None,
        framerate: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Decode audio tokens to waveform using FlexiCodec or Flow Matching decoder.
        
        Args:
            audio_tokens: [T] audio token indices (already offset by text_vocab_size)
            length_ids: [T] length class indices
            prompt_audio: Optional prompt audio tensor for flow matching decoder [1, T_audio]
            prompt_audio_path: Optional path to prompt audio file for caching features
            framerate: Optional frame rate. If None, uses default framerate.
        Returns:
            Numpy array of audio waveform or None if decoding fails
        """
        if self.config.use_flow_matching_decoder:
            return self._decode_audio_tokens_flow_matching(audio_tokens, length_ids, prompt_audio, prompt_audio_path=prompt_audio_path, framerate=framerate)
        else:
            return self._decode_audio_tokens_flexicodec(audio_tokens, length_ids)
    
    def _decode_audio_tokens_flexicodec(
        self,
        audio_tokens: torch.Tensor,
        length_ids: Optional[torch.Tensor] = None,
    ) -> Optional[np.ndarray]:
        """
        Decode audio tokens to waveform using FlexiCodec.
        
        Args:
            audio_tokens: [T] audio token indices (already offset by text_vocab_size)
            length_ids: [T] length class indices
            
        Returns:
            Numpy array of audio waveform or None if decoding fails
        """
        if not FLEXICODEC_AVAILABLE:
            logger.warning("FlexiCodec not available for decoding")
            return None
        
        try:
            # Remove text_vocab_size offset to get raw codec indices
            semantic_codes = audio_tokens - self.model.text_vocab_size
            semantic_codes = semantic_codes.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
            
            # Convert length_ids to token_lengths (add 1 since classes are 0-indexed)
            if length_ids is not None:
                token_lengths = length_ids.unsqueeze(0) + 1  # [1, T]
            else:
                # Default to length class 1 if not provided
                token_lengths = torch.ones(1, semantic_codes.size(1), dtype=torch.long, device=semantic_codes.device)
            
            # Decode using FlexiCodec
            with torch.no_grad():
                reconstructed_audio = self.model.flexicodec_dict['model'].decode_from_codes(
                    semantic_codes=semantic_codes,
                    token_lengths=token_lengths,
                    acoustic_codes=None,
                )
            
            if reconstructed_audio is not None:
                audio_np = reconstructed_audio.squeeze().cpu().numpy()
                return audio_np
            
            return None
            
        except Exception as e:
            logger.error(f"Error decoding audio tokens: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_or_extract_prompt_features(
        self,
        prompt_audio: torch.Tensor,
        prompt_audio_path: Optional[str] = None,
        framerate: Optional[float] = None,
    ) -> Dict:
        """
        Get prompt features from cache or extract and cache them.
        
        Args:
            prompt_audio: Prompt audio tensor [1, T_audio]
            prompt_audio_path: Optional path to the prompt audio file for cache key
            framerate: Frame rate for feature extraction
            
        Returns:
            Dictionary with 'expanded_prompt_tokens' and 'prompt_mel'
        """
        # Create cache key from path and framerate
        cache_key = None
        if prompt_audio_path is not None:
            cache_key = (prompt_audio_path, framerate)
        
        # Check cache
        if cache_key is not None and cache_key in self._prompt_features_cache:
            return self._prompt_features_cache[cache_key]
        
        # Extract features if not cached
        prompt_audio = prompt_audio.to(self.device)
        
        # Ensure prompt audio is mono
        if prompt_audio.dim() == 1:
            prompt_audio = prompt_audio.unsqueeze(0)
        if prompt_audio.shape[0] > 1:
            prompt_audio = prompt_audio.mean(dim=0, keepdim=True)
        
        # Extract features for prompt audio
        prompt_mel_features, _ = self.feature_extractor.extract_fbank(prompt_audio.cpu())
        prompt_mel_features = prompt_mel_features.to(self.device)
        prompt_x_lens = torch.tensor([prompt_mel_features.shape[1]], dtype=torch.long, device=self.device)
        
        # Extract semantic codes from prompt
        prompt_output = self.flow_matching_model._extract_dualcodec_features(
            prompt_audio, mel=prompt_mel_features, x_lens=prompt_x_lens, manual_threshold=framerate
        )
        prompt_cond_codes = prompt_output['semantic_codes_aggregated'].squeeze(1)  # [1, T_prompt]
        prompt_token_lengths = prompt_output.get('token_lengths')  # Get prompt token lengths if available
        
        # Expand prompt tokens if token_lengths are available
        expanded_prompt_tokens = prompt_cond_codes
        if prompt_token_lengths is not None and prompt_cond_codes.shape[1] > 0:
            expanded_prompt_tokens = torch.repeat_interleave(prompt_cond_codes[0], prompt_token_lengths[0]).unsqueeze(0)
            logger.info(f"Expanded prompt tokens from {prompt_cond_codes.shape[1]} to {expanded_prompt_tokens.shape[1]} using prompt_token_lengths")
        
        # Extract prompt mel features
        prompt_mel = self.flow_matching_model._extract_mel_features(prompt_audio)
        
        # Cache the results
        result = {
            'expanded_prompt_tokens': expanded_prompt_tokens,
            'prompt_mel': prompt_mel,
        }
        if cache_key is not None:
            self._prompt_features_cache[cache_key] = result
        
        return result
    
    def _decode_audio_tokens_flow_matching(
        self,
        audio_tokens: torch.Tensor,
        length_ids: Optional[torch.Tensor] = None,
        prompt_audio: Optional[torch.Tensor] = None,
        prompt_audio_path: Optional[str] = None,
        framerate: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Decode audio tokens using flow matching decoder (VoiceBox).
        
        Args:
            audio_tokens: [T] audio token indices (already offset by text_vocab_size)
            length_ids: [T] length class indices
            prompt_audio: Optional prompt audio tensor [1, T_audio]. If None, uses placeholder.
            prompt_audio_path: Optional path to prompt audio file for caching features
            framerate: Optional frame rate. If None, uses default framerate.
        Returns:
            Numpy array of audio waveform or None if decoding fails
        """
        if not FLOW_MATCHING_AVAILABLE:
            logger.warning("Flow matching decoder not available")
            return None
        
        # Resolve prompt_audio: use cached tensor or load from path if needed
        if prompt_audio is None and prompt_audio_path is not None and os.path.exists(prompt_audio_path):
            try:
                prompt_audio = self._load_audio(prompt_audio_path)
            except Exception as e:
                logger.warning(f"Failed to load prompt audio from {prompt_audio_path}: {e}")
        if prompt_audio is None:
            logger.error(
                "Flow matching decoder requires prompt audio. "
                "Set flow_matching_prompt_audio_path to a valid audio file."
            )
            return None
        
        try:
            # Remove text_vocab_size offset to get raw codec indices
            semantic_codes = audio_tokens
            semantic_codes = semantic_codes.unsqueeze(0)  # [1, T]
            
            # Convert length_ids to token_lengths if provided
            if length_ids is not None:
                token_lengths = length_ids.unsqueeze(0)  # [1, T]
            else:
                token_lengths = torch.ones(1, semantic_codes.size(1), dtype=torch.long, device=semantic_codes.device)
            
            # Expand generated tokens using duration classes (token_lengths)
            # Referencing cli_d2codec_tts_infer_librispeech.py line 372
            expanded_gen_tokens = semantic_codes
            if length_ids is not None:
                # Align lengths: when generation hits the max-token limit, semantic_codes
                # and token_lengths may differ by one. Strip the trailing extra token(s)
                # so repeat_interleave gets matching sizes.
                min_len = min(semantic_codes.size(1), token_lengths.size(1))
                if semantic_codes.size(1) != token_lengths.size(1):
                    logger.warning(
                        f"semantic_codes length ({semantic_codes.size(1)}) and token_lengths length "
                        f"({token_lengths.size(1)}) mismatch; truncating both to {min_len}."
                    )
                    semantic_codes = semantic_codes[:, :min_len]
                    token_lengths = token_lengths[:, :min_len]
                expanded_gen_tokens = torch.repeat_interleave(semantic_codes[0], token_lengths[0]).unsqueeze(0)
                logger.info(f"Expanded semantic tokens from {semantic_codes.shape[1]} to {expanded_gen_tokens.shape[1]} using token_lengths")
            
            # Get or extract prompt features (cached for efficiency)
            prompt_features = self._get_or_extract_prompt_features(
                prompt_audio=prompt_audio,
                prompt_audio_path=prompt_audio_path,
                framerate=framerate,
            )
            expanded_prompt_tokens = prompt_features['expanded_prompt_tokens']
            prompt_mel = prompt_features['prompt_mel']
            
            # Concatenate expanded prompt and expanded generated codes
            cond_codes = torch.cat([expanded_prompt_tokens, expanded_gen_tokens], dim=1)  # [1, T_prompt_expanded + T_expanded]
            
            # Get conditioning features
            voicebox_model = self.flow_matching_model.voicebox_model
            try:
                cond_feature = voicebox_model.cond_emb(cond_codes)
            except:
                logger.exception("Failed to build flow-matching conditioning features.")
                raise
            cond_feature = torch.nn.functional.interpolate(
                cond_feature.transpose(1, 2),
                scale_factor=voicebox_model.cond_scale_factor,
            ).transpose(1, 2)
            
            # Run reverse diffusion (prompt_mel is already extracted and cached)
            with torch.no_grad():
                predicted_mel = voicebox_model.reverse_diffusion(
                    cond=cond_feature,
                    prompt=prompt_mel,
                    n_timesteps=self.config.flow_matching_n_timesteps,
                    cfg=self.config.flow_matching_cfg,
                    rescale_cfg=self.config.flow_matching_rescale_cfg,
                )
            
            # Vocode mel to wav
            predicted_audio = self.vocoder_decode_func(predicted_mel.transpose(1, 2))
            audio_np = predicted_audio.squeeze().cpu().numpy()
            
            return audio_np
            
        except Exception as e:
            logger.error(f"Error decoding audio tokens with flow matching: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        input_ids1: torch.Tensor,
        audio_input: Optional[torch.Tensor] = None,
        framerate: Optional[float] = None,
        input_framerate: Optional[float] = None,
        force_text_ids: Optional[torch.Tensor] = None,
        force_audio_ids: Optional[torch.Tensor] = None,
        force_length_ids: Optional[torch.Tensor] = None,
        output_text_only=False,
        flow_matching_prompt_audio_path: Optional[str] = None,
    ) -> Dict:
        """Generate interleaved text and audio response.
        
        Args:
            input_ids: First part of tokenized prompt
            input_ids1: Second part of tokenized prompt (assistant prefix)
            audio_input: Optional audio input tensor
            framerate: Output frame rate for audio generation (0.8-1.0)
            input_framerate: Input frame rate for encoding audio (0.8-1.0). If None, uses config.input_framerate
            force_text_ids: Optional tensor of token IDs to force for text generation
            output_text_only: If True, generate only text tokens (no audio)
        """
        framerate = framerate or self.config.default_framerate
        # framerate = max(self.config.framerate_min, min(self.config.framerate_max, framerate))
        
        # If input_framerate is not provided at call time, fall back to the
        # configured default (config.input_framerate) instead of a hardcoded 1.0
        input_framerate = (
            input_framerate
            if input_framerate is not None
            else self.config.input_framerate
        )

        input_ids = input_ids.to(self.device)
        input_ids1 = input_ids1.to(self.device)
        # Prompt embeddings
        inputs_embeds = self.model.model.embed_tokens(input_ids)
        inputs_embeds1 = self.model.model.embed_tokens(input_ids1)

        # If there's audio input, encode it and insert embeddings between <|begin_of_audio|> and <|end_of_audio|>
        avg_input_framerate = 0.0
        if audio_input is not None:
            codec_output = self._encode_audio(audio_input, input_framerate)
            use_qwen3 = getattr(self.model.config, "use_qwen3_feature", False)
            use_whisper = getattr(self.model.config, "use_whisper_fetaure", False)
            use_qwen25o = getattr(self.model.config, "use_qwen25o_feature", False)
            if self.model.config.use_sensevoice_feature or use_qwen3 or use_qwen25o or use_whisper:
                # Handle flexible framerate return: dict with semantic_features + token_lengths
                if isinstance(codec_output, dict):
                    semantic_features = codec_output["semantic_features"]  # [T, H]
                    token_lengths = codec_output["token_lengths"]  # [T]
                    base_rate = 25.0 if use_qwen25o else 12.5
                    avg_input_framerate = base_rate * token_lengths.shape[-1] / token_lengths.sum().item()
                    print(f'input_framerate: {input_framerate}, avg_input_framerate: {avg_input_framerate}')
                else:
                    semantic_features = codec_output  # [T, H]
                    token_lengths = None
                audio_embeds = self.model.audio_embed_transform(semantic_features.to(self.device))  # [T, H]
                # Add length embeddings when flexible framerate is enabled (matches training).
                # For Qwen we intentionally do NOT add length embeddings here.

                # TODO remove this
                # if (
                #     getattr(self.model.config, 'enable_flexible_framerate', False)
                #     and token_lengths is not None
                #     and self.model.config.use_sensevoice_feature
                # ):
                #     token_lengths = token_lengths.to(self.device)
                #     # Clamp to valid embedding indices (0 to max_length_classes-1)
                #     token_lengths = (token_lengths.clamp(1, self.model.config.max_length_classes) - 1).long()
                #     length_embeds = self.model.talker_model.length_embedding(token_lengths.unsqueeze(0))
                #     audio_embeds = audio_embeds + length_embeds.squeeze(0)
            else:
                audio_semantic_codes = codec_output["semantic_codes"]
                if audio_semantic_codes.dim() == 3:
                    audio_semantic_codes = audio_semantic_codes.squeeze(1)  # [B, T]
                audio_semantic_codes = audio_semantic_codes.squeeze(0)  # [T]

                # Use the model's embedding path (keeps parity with training)
                audio_embeds = self.model.audio_embed_tokens(audio_semantic_codes.to(self.device), dtype=inputs_embeds.dtype)  # [T, H]

            # Apply input merging transformer if available
            _input_merging_transformer = getattr(self.model, "input_merging_transformer", None)
            if _input_merging_transformer is not None and not isinstance(_input_merging_transformer, torch.nn.Identity):
                use_input_merging_v2 = (
                    getattr(self.model.config, "use_input_merging_transformer", False)
                    and getattr(self.model.config, "use_input_merging_transformer_v2", False)
                    and getattr(self.model.config, "input_merging_transformer_num_layers", 0) > 0
                )
                if (
                    use_input_merging_v2
                    and isinstance(codec_output, dict)
                    and codec_output.get("alignment_matrix", None) is not None
                ):
                    alignment_matrix = codec_output["alignment_matrix"].to(self.device)
                    num_segments_per_item = codec_output["num_segments_per_item"].to(self.device)
                    audio_embeds = _input_merging_transformer(
                        audio_embeds.unsqueeze(0),
                        alignment_matrix,
                        num_segments_per_item,
                    ).squeeze(0)  # [G, H]
                else:
                    audio_embeds = _input_merging_transformer(audio_embeds.unsqueeze(0)).squeeze(0)  # [T, H]

            # Replace audio boundary token embeddings with learnable embeddings if enabled
            if getattr(self.model.config, "use_learnable_audio_boundary", False) and hasattr(self.model, "audio_start_embedding"):
                inputs_embeds = inputs_embeds.clone()
                inputs_embeds[0, -1] = self.model.audio_start_embedding.to(inputs_embeds.dtype)
                inputs_embeds1 = inputs_embeds1.clone()
                inputs_embeds1[0, 0] = self.model.audio_end_embedding.to(inputs_embeds1.dtype)

            inputs_embeds = torch.cat([inputs_embeds, audio_embeds.unsqueeze(0), inputs_embeds1], dim=1)
        else:
            # Even for text-only user input, we still need to append the assistant prefix (prompt1)
            inputs_embeds = torch.cat([inputs_embeds, inputs_embeds1], dim=1)

        # If stage 1.5 learnable prefix is enabled, inference must prepend it too.
        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            device=inputs_embeds.device,
            dtype=torch.long,
        )
        if getattr(self.model.config, "use_learnable_prefix", False):
            inputs_embeds, attention_mask = self.model._prepend_prefix_embeddings(
                inputs_embeds,
                attention_mask=attention_mask,
            )

        result = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id, # <endoftext> or <|im_end|> self.tokenizer('<|im_end|>').input_ids[0],
            audio_start_id=self.AUD_START_ID,
            audio_end_id=self.AUD_END_ID,
            framerate=framerate,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            do_sample=True,
            length_temperature=self.config.length_temperature,
            length_top_k=self.config.length_top_k,
            length_top_p=self.config.length_top_p,
            repetition_penalty=1.5,
            return_dict_in_generate=True,
            tokenizer=self.tokenizer,
            force_text_ids=force_text_ids,
            force_audio_ids=force_audio_ids,
            force_length_ids=force_length_ids,
            output_text_only=output_text_only,
        )
        
        # Extract results
        generated_ids = result["generated_ids"][0]  # [num_generated]
        length_ids = result["length_ids"] if result['length_ids'] is not None else None
        text_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"DEBUG [inference_flexislm.generate]: generated_ids shape={generated_ids.shape}, text_output='{text_output}', output_text_only={output_text_only}")
        if not text_output:
            print("DEBUG [inference_flexislm.generate]: text_output is empty!")
            
        audio_chunks = result['audio_ids']
        # Decode audio if requested
        audio_output = None
        if self.config.decode_audio:
            if result.get('acoustic_codes') is not None:
                with torch.no_grad():
                    audio_output = self.model.flexicodec_dict['model'].decode_from_latent(
                        result['acoustic_codes'].unsqueeze(0).transpose(1,2).float(),
                        result['length_ids'].unsqueeze(0),
                    )
            else:
                # When using forced speech prompt, use that prompt for flow matching; otherwise use config default
                fm_prompt_path = flow_matching_prompt_audio_path or self.config.flow_matching_prompt_audio_path
                fm_prompt_audio = self.prompt_audio_cache if fm_prompt_path == self.config.flow_matching_prompt_audio_path else None
                audio_output = self._decode_audio_tokens(
                    audio_chunks, 
                    length_ids, 
                    prompt_audio=fm_prompt_audio,
                    prompt_audio_path=fm_prompt_path,
                    framerate=framerate
                )
        
        return {
            "text": text_output,
            "audio": audio_output,
            "audio_tokens": audio_chunks,
            "length_ids": length_ids,
            "framerate": framerate,
            "audio_ids": result.get("audio_ids"),
            "avg_input_framerate": avg_input_framerate,
        }
    
    @torch.no_grad()
    def generate_from_text(
        self,
        text_input: str,
        history: str = "",
        framerate: Optional[float] = None,
        input_framerate: Optional[float] = None,
        force_text_ids: Optional[torch.Tensor] = None,
        force_audio_ids: Optional[torch.Tensor] = None,
        force_length_ids: Optional[torch.Tensor] = None,
        output_text_only: bool = False,
        use_system_prompt: Optional[bool] = None,
        flow_matching_prompt_audio_path: Optional[str] = None,
    ) -> Dict:
        """
        Generate response from text input.
        
        Args:
            text_input: Text input
            history: Conversation history
            framerate: Frame rate for output audio
            input_framerate: Frame rate for encoding input audio (not used for text-only input)
            force_text_ids: Optional tensor of token IDs to force for text generation [L]
            use_system_prompt: If True, prepend TTS system prompt. If None, uses global USE_SYSTEM_PROMPT.
            flow_matching_prompt_audio_path: When using forced speech prompt, use this path for flow matching decoder.
            
        Returns:
            Dict with generated text, audio_ids, and updated history
        """
        # Determine if assistant should have audio based on output_text_only flag
        assistant_has_audio = not output_text_only
        
        # Build prompt
        time1 = time.time()
        prompt, prompt1 = self._build_prompt(
            user_content=text_input,
            history=history,
            has_audio=False,
            use_system_prompt=use_system_prompt,
            assistant_has_audio=assistant_has_audio,
        )
        logger.info(f"generate_from_text called. {prompt = }{prompt1 = }")
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(self.device)
        inputs1 = self.tokenizer(
            prompt1,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids1 = inputs1["input_ids"].to(self.device)
        # Generate (no audio input)
        result = self.generate(
            input_ids,
            input_ids1,
            audio_input=None,
            framerate=framerate,
            input_framerate=input_framerate,
            force_text_ids=force_text_ids,
            force_audio_ids=force_audio_ids,
            force_length_ids=force_length_ids,
            output_text_only=output_text_only,
            flow_matching_prompt_audio_path=flow_matching_prompt_audio_path,
        )

        # Update history
        new_history = history + f"<|im_start|>user\n{text_input}<|im_end|>\n"
        new_history += f"<|im_start|>assistant\n{result['text']}<|im_end|>\n"
        result['history'] = new_history
        result['start_time'] = time1
        return result
    
    @torch.no_grad()
    def generate_from_audio(
        self,
        audio_path: str,
        text_query: str = "",
        history: str = "",
        framerate: Optional[float] = None,
        input_framerate: Optional[float] = None,
        output_text_only: bool = False,
    ) -> Dict:
        """
        Generate response from audio input.
        
        Args:
            audio_path: Path to input audio file
            text_query: Text query to accompany the audio
            history: Conversation history
            framerate: Frame rate for output audio
            input_framerate: Frame rate for encoding input audio (0.8-1.0). If None, uses config.input_framerate
            
        Returns:
            Dict with generated text, audio, and updated history
        """
        # if framerate == 1.0:
        #     # override framerate to 0.99
        #     framerate = 0.99
        # Load audio
        audio_tensor = self._load_audio(audio_path)
        
        # Determine if assistant should have audio based on output_text_only flag
        assistant_has_audio = not output_text_only
        
        # Build prompt (no system prompt for audio input - only TTS uses it unless dynamic is enabled)
        time1 = time.time()
        prompt, prompt1 = self._build_prompt(
            user_content=text_query,
            history=history,
            has_audio=True,
            assistant_has_audio=assistant_has_audio,
        )
        logger.info(f"generate_from_audio called. {text_query = } {prompt = }{prompt1 = }")
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(self.device)
        inputs1 = self.tokenizer(
            prompt1,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids1 = inputs1["input_ids"].to(self.device)
        # Generate
        result = self.generate(
            input_ids, 
            input_ids1, 
            audio_input=audio_tensor, 
            framerate=framerate,
            input_framerate=input_framerate,
            output_text_only=output_text_only,
        )
        
        print(f"DEBUG [inference_flexislm.generate_from_audio]: result text='{result.get('text')}'")
        if not result.get('text'):
            print("DEBUG [inference_flexislm.generate_from_audio]: result text is empty!")
            logger.warning("generate_from_audio returned empty text output.")
            
        # Update history
        new_history = history + f"<|im_start|>user\n{text_query}\n{AUD_START_TOKEN}{AUD_END_TOKEN}<|im_end|>\n"
        new_history += f"<|im_start|>assistant\n{result['text']}<|im_end|>\n"
        result['history'] = new_history
        result['start_time'] = time1
        return result
    
    @torch.no_grad()
    def generate_tts(
        self,
        sentence: str,
        framerate: Optional[float] = None,
        input_framerate: Optional[float] = None,
        system_prompt: Optional[str] = None,
        force_text_ids: Optional[torch.Tensor] = None,
        prompt_audio_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        flow_matching_prompt_audio_path: Optional[str] = None,
    ) -> Dict:
        """
        Generate TTS (Text-to-Speech) output for a given sentence.
        This is a convenience wrapper around generate_from_text with TTS-specific formatting.
        
        Args:
            sentence: The sentence to speak
            framerate: Frame rate for output audio
            input_framerate: Frame rate for encoding input audio (not used for TTS, kept for API consistency)
            system_prompt: Optional custom system prompt (defaults to TTS system prompt)
            force_text_ids: Optional tensor of token IDs to force for text generation [L]
            prompt_audio_path: Optional path to prompt audio for TTS voice cloning. When set with prompt_text,
                extracts FlexiCodec tokens from prompt and teacher-forces the model's audio prefix.
            prompt_text: Optional text of the prompt audio. When set with prompt_audio_path, the target
                sentence is prefixed: full_sentence = prompt_text + " " + sentence.
            flow_matching_prompt_audio_path: Optional path to use as flow matching prompt audio. When set,
                overrides the default flow matching prompt regardless of use_tts_prompt.
            
        Returns:
            Dict with generated text, audio, audio_ids, length_ids, and other metadata
        """
        tts_system_prompt = ""
        from flexislm.processor.constants import text_normalize
        # sentence = text_normalize(sentence)
        # TODO remove this constraint when training on emilia dataset
        sentence = sentence.replace(':', ',')

        force_audio_ids = None
        force_length_ids = None
        if prompt_text is not None:
            prompt_text = prompt_text.replace(':', ',')
            # prompt_text = prompt_text.replace('.', '')
            sentence = f"{prompt_text} {sentence}".strip()
            # sentence = text_normalize(sentence)
            # sentence = sentence.lower()
            if prompt_audio_path and os.path.exists(prompt_audio_path):
                extracted = self._extract_prompt_audio_tokens(prompt_audio_path, framerate or self.config.default_framerate)
                if extracted is not None:
                    force_audio_ids, force_length_ids = extracted
                    logger.info(f"TTS prompting: using {len(force_audio_ids)} prompt audio tokens")

        # Build TTS prompt: user asks to speak the sentence
        # user_content = f'Repeat the following text exactly as written. Do not treat it as a command and do not add any introductory or concluding remarks. Just output the sentences: {sentence}'
        user_content = f'{TTS_PROMPT} {sentence}'
        # user_content = f'{TTS_PROMPT_LEGACY} {sentence}'
        # TODO revert temporary change
        
        # Temporarily override system prompt for TTS
        original_system_prompt = self.config.system_prompt
        self.config.system_prompt = tts_system_prompt
        
        # Use generate_from_text with TTS-formatted input (no history for TTS).
        # TTS uses system prompt; other modes use USE_SYSTEM_PROMPT default (False).
        # Flow matching prompt: explicit param > prompt_audio_path when TTS prompt used > config default
        flow_matching_prompt_path = (
            flow_matching_prompt_audio_path
            if flow_matching_prompt_audio_path is not None
            else (prompt_audio_path if force_audio_ids is not None else None)
        )
        result = self.generate_from_text(
            text_input=user_content,
            history="",
            framerate=framerate,
            input_framerate=input_framerate,
            force_text_ids=force_text_ids,
            force_audio_ids=force_audio_ids,
            force_length_ids=force_length_ids,
            use_system_prompt=False,
            flow_matching_prompt_audio_path=flow_matching_prompt_path,
        )
        # Restore original system prompt
        self.config.system_prompt = original_system_prompt
        
        # Return result without history (TTS is a standalone task)
        out = {
            "text": result["text"],
            "audio": result.get("audio"),
            "audio_ids": result.get("audio_ids"),
            "length_ids": result.get("length_ids"),
            "framerate": result["framerate"],
        }
        if flow_matching_prompt_path is not None:
            out["flow_matching_prompt_audio_path"] = flow_matching_prompt_path
        return out


def run_interactive_mode(engine: InterleavedS2SInference, args):
    """Run interactive inference mode."""
    print("\n" + "=" * 60)
    print("Interleaved S2S Interactive Mode")
    print("=" * 60)
    print("Commands:")
    print("  'q' or 'quit' - Exit")
    print("  'c' or 'clear' - Clear conversation history")
    print("  'h' or 'history' - Show conversation history")
    print("  't:<text>' - Text input mode (e.g., 't:Hello!')")
    print("  'a:<path>' - Audio input mode (e.g., 'a:audio/example.wav')")
    print("  'tts:<sentence>' - TTS mode (e.g., 'tts:Hello world')")
    print("  'f:<rate>' - Set output frame rate (e.g., 'f:0.91')")
    print("  'if:<rate>' - Set input frame rate (e.g., 'if:0.85')")
    print("  'f:show' - Show current framerates")
    print("  's:interleaved' or 's:text_only' - Switch system prompt type")
    print("  's:show' - Show current system prompt")
    print("=" * 60 + "\n")
    
    history = ""
    conversation_count = 1
    chat_count = 1
    current_framerate = args.framerate
    
    # System prompt type tracking
    current_system_prompt_type = args.system_prompt_type
    print(f"Initial system prompt type: {current_system_prompt_type}")
    print(f"System prompt: {engine.config.system_prompt}\n")
    
    # Create output directory
    output_dir = args.output_dir or "outputs/interleaved_interactive"
    os.makedirs(output_dir, exist_ok=True)
    conversation_dir = os.path.join(output_dir, f"conversation_{conversation_count}")
    os.makedirs(conversation_dir, exist_ok=True)
    
    # Optional debug audio file, provided by CLI.
    debug_audio_path = args.debug_audio_path

    def decode_audio(result):
        nonlocal chat_count

        audio_ids = result.get("audio_ids")
        if audio_ids is None:
            return

        if torch.is_tensor(audio_ids):
            audio_ids = audio_ids.to(engine.device)
        else:
            audio_ids = torch.as_tensor(audio_ids, device=engine.device)
        if audio_ids.dim() == 0:
            audio_ids = audio_ids.unsqueeze(0)

        length_ids = result.get("length_ids")
        if length_ids is not None:
            if torch.is_tensor(length_ids):
                length_ids = length_ids.to(engine.device)
            else:
                length_ids = torch.as_tensor(length_ids, device=engine.device)
            if length_ids.dim() == 0:
                length_ids = length_ids.unsqueeze(0)

        output_wav_path = None
        
        # Use flow matching decoder if enabled, otherwise use FlexiCodec
        if engine.config.use_flow_matching_decoder:
            # Use fixed prompt audio path for flow matching (cached)
            # Decode using flow matching decoder
            audio_np = engine._decode_audio_tokens(
                audio_tokens=audio_ids,
                length_ids=length_ids,
                prompt_audio=engine.prompt_audio_cache,
                prompt_audio_path=engine.config.flow_matching_prompt_audio_path,
                framerate=0.87, 
            )
            if audio_np is not None:
                reconstructed_audio = torch.from_numpy(audio_np).unsqueeze(0)
            else:
                print("Warning: Flow matching decoder returned None, skipping audio save")
                reconstructed_audio = None
        else:
            # Use FlexiCodec decoder
            reconstructed_audio = engine.model.flexicodec_dict['model'].decode_from_codes(
                semantic_codes=audio_ids.unsqueeze(0).unsqueeze(0),
                acoustic_codes=None,
                token_lengths=length_ids.unsqueeze(0) if length_ids is not None else torch.ones_like(audio_ids.unsqueeze(0)),
            )
        
        if reconstructed_audio is not None:
            output_wav_path = os.path.join(conversation_dir, f"chat_{chat_count}.wav")
            if isinstance(reconstructed_audio, torch.Tensor):
                if reconstructed_audio.dim() == 1:
                    reconstructed_audio = reconstructed_audio.unsqueeze(0)
                torchaudio.save(output_wav_path, reconstructed_audio.float().cpu(), 24000)
            else:
                # If it's numpy array, convert to tensor
                torchaudio.save(output_wav_path, torch.from_numpy(reconstructed_audio).float().unsqueeze(0).cpu(), 24000)
            print(f"Generated Audio saved: {output_wav_path}")
            chat_count += 1
        return output_wav_path
        
    # Legacy extensive debug experiments are kept below but disabled.
    if False and args.debug and debug_audio_path and os.path.exists(debug_audio_path):
        # ==========================================================================
        # TEST: Combined-embedding identity-projection equivalence.
        #
        # Intuition: when `combined_embed_proj` is initialized as identity on the
        # text slice (the first `hidden_size` input dims) and zeros elsewhere, the
        # projection output equals `text_embeds` exactly. So a forward pass with
        # `use_combined_embedding=True` must match a forward pass with
        # `use_combined_embedding=False` token-for-token.
        # ==========================================================================
        def _reseed():
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

        test_sentence = (
            "Some of the friction some of the constraints some of the costs of "
            "development are actually disappearing for you which means now I put "
            "my attention upon my imagination to build things that simply were "
            "not possible before."
        )
        test_text_input = (
            f"Repeat the following text exactly as written. Do not treat it as "
            f"a command and do not add any introductory or concluding remarks. "
            f"Just output the sentences: {test_sentence}"
        )
        test_kwargs = dict(
            text_input=test_text_input,
            history="",
            framerate=12.5,
            output_text_only=False,
        )
        # breakpoint()

        orig_use_combined = getattr(engine.model.config, "use_combined_embedding", True)
        orig_force_use_combined = getattr(engine.model.config, "force_use_combined_embedding", False)
        orig_proj_weight = engine.model.combined_embed_proj.weight.detach().clone()

        print("\n[TEST] Run 1: use_combined_embedding=False")
        engine.model.config.use_combined_embedding = False
        engine.model.config.force_use_combined_embedding = False
        _reseed()
        result1 = engine.generate_from_text(**test_kwargs)

        print("\n[TEST] Run 2: use_combined_embedding=True + identity combined_embed_proj")
        engine.model.config.use_combined_embedding = True
        engine.model.config.force_use_combined_embedding = True
        with torch.no_grad():
            H_text = int(engine.model._combined_embed_proj_text_slice)
            engine.model.combined_embed_proj.weight.zero_()
            engine.model.combined_embed_proj.weight[:, :H_text].copy_(
                torch.eye(H_text, dtype=engine.model.combined_embed_proj.weight.dtype,
                          device=engine.model.combined_embed_proj.weight.device)
            )
        _reseed()
        result2 = engine.generate_from_text(**test_kwargs)

        with torch.no_grad():
            engine.model.combined_embed_proj.weight.copy_(orig_proj_weight)
        engine.model.config.use_combined_embedding = orig_use_combined
        engine.model.config.force_use_combined_embedding = orig_force_use_combined

        text1, text2 = result1.get("text", ""), result2.get("text", "")
        aud1, aud2 = result1.get("audio_ids"), result2.get("audio_ids")
        len1, len2 = result1.get("length_ids"), result2.get("length_ids")

        def _to_t(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.detach().cpu()
            return torch.as_tensor(x).detach().cpu()

        aud1_t, aud2_t = _to_t(aud1), _to_t(aud2)
        len1_t, len2_t = _to_t(len1), _to_t(len2)

        text_match = text1 == text2
        audio_match = (
            aud1_t is None and aud2_t is None
        ) or (
            aud1_t is not None and aud2_t is not None
            and aud1_t.shape == aud2_t.shape
            and torch.equal(aud1_t, aud2_t)
        )
        length_match = (
            len1_t is None and len2_t is None
        ) or (
            len1_t is not None and len2_t is not None
            and len1_t.shape == len2_t.shape
            and torch.equal(len1_t, len2_t)
        )

        print("\n[TEST] === Combined-embedding identity equivalence ===")
        print(f"[TEST] text_match    : {text_match}")
        print(f"[TEST]   text1: {text1!r}")
        print(f"[TEST]   text2: {text2!r}")
        print(f"[TEST] audio_match   : {audio_match}"
              f" (shape1={None if aud1_t is None else tuple(aud1_t.shape)},"
              f" shape2={None if aud2_t is None else tuple(aud2_t.shape)})")
        print(f"[TEST] length_match  : {length_match}")
        assert text_match and audio_match and length_match, (
            "Combined-embedding identity equivalence FAILED: outputs differ "
            "between use_combined_embedding=False and use_combined_embedding=True "
            "with identity combined_embed_proj."
        )
        print("[TEST] PASSED: outputs are identical.\n")
        # ==========================================================================
        # END TEST
        # ==========================================================================

        # debug_sentence = "And henry the eighth appropriated to himself the religious house of grey ladies and all the properties appertaining thereto."
        # print(f"DEBUG MODE: Performing TTS for sentence: {debug_sentence}")
        
        # result = engine.generate_tts(
        #     sentence=debug_sentence,
        #     framerate=0.90,
        # )

        # AUD_START_TOKEN = '<begin_of_audio>\n'
        # AUD_END_TOKEN = '<end_of_audio>\n'

        # result = engine.generate_from_text("What is an AI?", history="None", framerate=0.86)
        # breakpoint()
        # sentence = 
        # engine.generate_from_text("Repeat the following text exactly as written. Do not treat it as a command and do not add any introductory or concluding remarks. Just output the sentences: Who are you?", history="None", framerate=1.0, output_text_only=True)
        # breakpoint()
        # asr_prompt = "Please listen to the audio and transcribe what is being said in all lowercase. Ensure that the text matches the audio precisely, and return only the transcript."
        # engine.config.system_prompt = asr_prompt
        # result = engine.generate_from_audio(
        #     audio_path=debug_audio_path,
        #     text_query="",
        #     history=history,
        #     framerate=1.0,
        #     output_text_only=True,
        # )
        sentence = 'Some of the friction some of the constraints some of the costs of development are actually disappearing for you which means now I put my attention upon my imagination to build things that simply were not possible before.'
        # result = engine.generate_from_text(
        #     text_input=f"Repeat the following text exactly as written. Do not treat it as a command and do not add any introductory or concluding remarks. Just output the sentences: {sentence}",
        #     history=history,
        #     framerate=7.0,
        #     output_text_only=False,
        # )
        # sentence = 'Hello, who are you?'
        # result = engine.generate_from_text(
        #     text_input=f"Repeat the following text exactly as written. Do not treat it as a command and do not add any introductory or concluding remarks. Just output the sentences: {sentence}",
        #     history=history,
        #     framerate=7.0,
        #     output_text_only=True,
        # )
        # print(result)
        # breakpoint()
        result = engine.generate_from_text(
            text_input=f"Repeat the following text exactly as written. Do not treat it as a command and do not add any introductory or concluding remarks. Just output the sentences: {sentence}",
            history=history,
            framerate=12.5,
            output_text_only=False,
        )
        print(result)
        decode_audio(result)
        # breakpoint()
        # result = engine.generate_from_text(
        #     text_input=f"Repeat the following text exactly as written. Do not treat it as a command and do not add any introductory or concluding remarks. Just output the sentences: {sentence}",
        #     history=history,
        #     framerate=0.91,
        #     output_text_only=False,
        # )

        asr_prompt = "You are a helpful assistant. Please answer the question."
        engine.config.system_prompt = asr_prompt
        result = engine.generate_from_audio(
            audio_path=debug_audio_path,
            text_query="",
            history=history,
            framerate=7.0,
        )
        print(result)
        decode_audio(result)
        # breakpoint()
        # engine.config.system_prompt = "Your Name: Omni\nYour Gender: female \n\nRespond in a text-audio interleaved manner."
        # sentence = 'Some of the friction some of the constraints some of the costs of development are actually disappearing for you which means now I put my attention upon my imagination to build things that simply were not possible before.'
        # result = engine.generate_from_text(
        #     text_input=f"Repeat the following text exactly as written. Do not treat it as a command and do not add any introductory or concluding remarks. Just output the sentences: {sentence}",
        #     history=history,
        #     framerate=0.91,
        #     output_text_only=False,
        # )
        # result = engine.generate_from_text(
        #     text_input=f"Repeat the following text exactly as written. Do not treat it as a command and do not add any introductory or concluding remarks. Just output the sentences: {sentence}",
        #     history=history,
        #     framerate=0.87,
        #     output_text_only=False,
        # )
        # breakpoint()
        # print(f"DEBUG MODE: Processing audio file: {debug_audio_path}")
        # engine.config.system_prompt = "Your Name: Omni\nYour Gender: female \n\nRespond in a text-audio interleaved manner."
        # result = engine.generate_from_audio(
        #     audio_path=debug_audio_path,
        #     text_query="",
        #     history=history,
        #     framerate=0.87,
        # )
        # breakpoint()

        # engine.config.system_prompt = "Your Name: Omni\nYour Gender: female \n\nRespond in a text-only manner."
        # result = engine.generate_from_audio(
        #     audio_path=debug_audio_path,
        #     text_query="Please listen to the audio and transcribe what is being said. Ensure that the text matches the audio precisely, and return only the transcript.",
        #     history=history,
        #     framerate=current_framerate,
        # )
        # print(f"\nGenerated Text: {result['text']}")
        # decode_audio(result)
        
        print("\nDebug mode completed. Entering interactive mode...\n")
    
    if args.debug:
        print("\n[DEBUG MODE] Running minimal API examples...")
        debug_sentence = (
            "And henry the eighth appropriated to himself the religious house of "
            "grey ladies and all the properties appertaining thereto."
        )

        # 1) TTS
        print(f"[DEBUG][TTS] sentence: {debug_sentence}")
        try:
            tts_result = engine.generate_tts(
                sentence=debug_sentence,
                framerate=current_framerate,
            )
            print(f"[DEBUG][TTS] text: {tts_result.get('text', '')}")
            decode_audio(tts_result)
        except Exception as e:
            print(f"[DEBUG][TTS] failed: {e}")

        # 2) T2S
        print("[DEBUG][T2S] text -> speech")
        try:
            t2s_result = engine.generate_from_text(
                text_input=f"Please read the following text: {debug_sentence}",
                history="",
                framerate=current_framerate,
                output_text_only=False,
            )
            print(f"[DEBUG][T2S] text: {t2s_result.get('text', '')}")
            decode_audio(t2s_result)
        except Exception as e:
            print(f"[DEBUG][T2S] failed: {e}")

        # 3) S2S and 4) S2T (require debug audio path)
        if debug_audio_path and os.path.exists(debug_audio_path):
            print(f"[DEBUG][S2S/S2T] audio: {debug_audio_path}")
            try:
                s2s_result = engine.generate_from_audio(
                    audio_path=debug_audio_path,
                    text_query="Please respond naturally to the audio in speech.",
                    history="",
                    framerate=current_framerate,
                    output_text_only=False,
                )
                print(f"[DEBUG][S2S] text: {s2s_result.get('text', '')}")
                decode_audio(s2s_result)
            except Exception as e:
                print(f"[DEBUG][S2S] failed: {e}")

            try:
                s2t_result = engine.generate_from_audio(
                    audio_path=debug_audio_path,
                    text_query="Please transcribe the speech in the audio.",
                    history="",
                    framerate=current_framerate,
                    output_text_only=True,
                )
                print(f"[DEBUG][S2T] text: {s2t_result.get('text', '')}")
            except Exception as e:
                print(f"[DEBUG][S2T] failed: {e}")
        else:
            print("[DEBUG][S2S/S2T] skipped (missing --debug_audio_path).")

        # 5) T2T
        print("[DEBUG][T2T] text -> text")
        try:
            t2t_result = engine.generate_from_text(
                text_input="What is dynamic frame rate in speech modeling? Answer in one sentence.",
                history="",
                framerate=current_framerate,
                output_text_only=True,
            )
            print(f"[DEBUG][T2T] text: {t2t_result.get('text', '')}")
        except Exception as e:
            print(f"[DEBUG][T2T] failed: {e}")

        print("\n[DEBUG MODE] Completed. Entering interactive mode...\n")

    while True:
        try:
            user_input = input("\nInput: ").strip()
            
            if user_input.lower() in ['q', 'quit']:
                print("Exiting...")
                break
            
            elif user_input.lower() in ['c', 'clear']:
                history = ""
                conversation_count += 1
                chat_count = 1
                conversation_dir = os.path.join(output_dir, f"conversation_{conversation_count}")
                os.makedirs(conversation_dir, exist_ok=True)
                print("History cleared. Starting new conversation.")
                continue
            
            elif user_input.lower() in ['h', 'history']:
                print(f"\nConversation History:\n{history if history else '(empty)'}")
                continue
            
            elif user_input.lower().startswith('f:'):
                framerate_cmd = user_input[2:].strip().lower()
                if framerate_cmd == 'show':
                    print(f"\nCurrent framerates:")
                    print(f"  Output framerate: {current_framerate}")
                    print(f"  Input framerate: {engine.config.input_framerate}")
                    print(f"  Range: [{engine.config.framerate_min}, {engine.config.framerate_max}]")
                else:
                    try:
                        new_rate = float(framerate_cmd)
                        if engine.config.framerate_min <= new_rate <= engine.config.framerate_max:
                            current_framerate = new_rate
                            print(f"Output frame rate set to: {current_framerate}")
                        else:
                            print(f"Invalid frame rate. Range: [{engine.config.framerate_min}, {engine.config.framerate_max}]")
                    except ValueError:
                        print("Invalid frame rate format. Use 'f:<number>' or 'f:show'")
                continue
            
            elif user_input.lower().startswith('if:'):
                try:
                    new_rate = float(user_input[3:])
                    if engine.config.framerate_min <= new_rate <= engine.config.framerate_max:
                        engine.config.input_framerate = new_rate
                        print(f"Input frame rate set to: {engine.config.input_framerate}")
                    else:
                        print(f"Invalid frame rate. Range: [{engine.config.framerate_min}, {engine.config.framerate_max}]")
                except ValueError:
                    print("Invalid frame rate format. Use 'if:<number>'")
                continue
            
            elif user_input.lower().startswith('s:'):
                sys_cmd = user_input[2:].strip().lower()
                if sys_cmd == 'show':
                    print(f"\nCurrent system prompt type: {current_system_prompt_type}")
                    print(f"System prompt: {engine.config.system_prompt}")
                elif sys_cmd == 'interleaved':
                    engine.config.system_prompt = ""
                    current_system_prompt_type = 'interleaved'
                    print(f"System prompt switched to: interleaved")
                    print(f"New system prompt: {engine.config.system_prompt}")
                elif sys_cmd == 'text_only':
                    engine.config.system_prompt = ""
                    current_system_prompt_type = 'text_only'
                    print(f"System prompt switched to: text_only")
                    print(f"New system prompt: {engine.config.system_prompt}")
                else:
                    print("Invalid system prompt command. Use 's:interleaved', 's:text_only', or 's:show'")
                continue
            
            # Determine input type
            if user_input.lower().startswith('t:'):
                # Text input
                text_input = user_input[2:].strip()
                if not text_input:
                    print("Empty text input. Please try again.")
                    continue
                
                print(f"Processing text input: {text_input}")
                result = engine.generate_from_text(
                    text_input=text_input,
                    history=history,
                    framerate=current_framerate,
                )
                
            elif user_input.lower().startswith('a:'):
                # Audio input
                audio_path = user_input[2:].strip()
                if not os.path.exists(audio_path):
                    print(f"Audio file not found: {audio_path}")
                    continue
                
                print(f"Processing audio input: {audio_path}")
                if current_system_prompt_type == 'interleaved':
                    query = ''
                else:
                    query = "Please listen to the audio and transcribe what is being said. Ensure that the text matches the audio precisely, and return only the transcript."
                result = engine.generate_from_audio(
                    audio_path=audio_path,
                    text_query=query,
                    history=history,
                    framerate=current_framerate,
                )
                
            elif user_input.lower().startswith('tts:'):
                # TTS input
                sentence = user_input[4:].strip()
                if not sentence:
                    print("Empty sentence. Please provide a sentence to speak.")
                    continue
                
                print(f"Processing TTS request: {sentence}")
                result = engine.generate_tts(
                    sentence=sentence,
                    framerate=current_framerate,
                )
                
                # For TTS, we don't update history (it's a standalone task)
                # Print output
                print(f"\nGenerated Text: {result['text']}")
                
                # Save audio if available
                if result.get("audio") is not None and AUDIO_AVAILABLE:
                    audio_np = result["audio"]
                    if isinstance(audio_np, np.ndarray):
                        output_wav_path = os.path.join(conversation_dir, f"tts_{chat_count}.wav")
                        torchaudio.save(output_wav_path, torch.from_numpy(audio_np).float().unsqueeze(0).cpu(), args.output_sample_rate)
                        print(f"Generated Audio saved: {output_wav_path}")
                        chat_count += 1
                elif result.get("audio_ids") is not None and AUDIO_AVAILABLE:
                    # Decode audio from tokens
                    audio_ids = torch.tensor(result["audio_ids"], device=engine.device)
                    length_ids = torch.tensor(result["length_ids"], device=engine.device) if result.get("length_ids") is not None else None
                    
                    fm_prompt_path = result.get("flow_matching_prompt_audio_path") or engine.config.flow_matching_prompt_audio_path
                    fm_prompt_audio = engine.prompt_audio_cache if fm_prompt_path == engine.config.flow_matching_prompt_audio_path else None
                    if engine.config.use_flow_matching_decoder:
                        audio_np = engine._decode_audio_tokens(
                            audio_tokens=audio_ids,
                            length_ids=length_ids,
                            prompt_audio=fm_prompt_audio,
                            prompt_audio_path=fm_prompt_path,
                            framerate=result.get("framerate"),
                        )
                    else:
                        audio_np = engine._decode_audio_tokens_flexicodec(
                            audio_tokens=audio_ids,
                            length_ids=length_ids,
                        )
                    
                    if audio_np is not None:
                        output_wav_path = os.path.join(conversation_dir, f"tts_{chat_count}.wav")
                        torchaudio.save(output_wav_path, torch.from_numpy(audio_np).float().unsqueeze(0).cpu(), args.output_sample_rate)
                        print(f"Generated Audio saved: {output_wav_path}")
                        chat_count += 1
                else:
                    print("(No audio generated)")
                
                continue
                
            else:
                # Default to text input
                text_input = user_input
                if not text_input:
                    print("Empty input. Use 't:<text>' for text or 'a:<path>' for audio.")
                    continue
                
                print(f"Processing as text input: {text_input}")
                result = engine.generate_from_text(
                    text_input=text_input,
                    history=history,
                    framerate=current_framerate,
                )
            
            # Update history
            if "history" in result:
                history = result["history"]
            
            # Print output
            print(f"\nGenerated Text: {result['text']}")
            decode_audio(result)
                
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()


def run_batch_mode(engine: InterleavedS2SInference, args):
    """Run batch inference mode from JSONL file."""
    logger.info(f"Running batch inference from: {args.input_file}")
    
    output_dir = args.output_dir or "outputs/interleaved_batch"
    os.makedirs(output_dir, exist_ok=True)
    
    output_jsonl_path = os.path.join(output_dir, "output_results.jsonl")
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    with open(output_jsonl_path, 'w', encoding='utf-8') as out_f:
        for line in tqdm(lines, desc="Batch Inference"):
            try:
                data = json.loads(line.strip())
                conv_id = data.get("id", "unknown")
                
                # Check if this is a TTS task (has "messages" field)
                if "messages" in data:
                    # TTS task format: {"messages": [...], "audios": [...]}
                    messages = data.get("messages", [])
                    
                    # Extract sentence from user message
                    sentence = None
                    system_prompt = None
                    for msg in messages:
                        if msg.get("role") == "system":
                            system_prompt = msg.get("content", "")
                        elif msg.get("role") == "user":
                            content = msg.get("content", "")
                            # Extract sentence from "please speak the following sentence \"<sentence>\""
                            if 'please speak the following sentence' in content.lower():
                                match = re.search(r'"([^"]+)"', content)
                                if match:
                                    sentence = match.group(1)
                    
                    if sentence:
                        logger.info(f"[ID {conv_id}] TTS task: Generating speech for sentence: {sentence[:50]}...")
                        result = engine.generate_tts(
                            sentence=sentence,
                            framerate=args.framerate,
                            system_prompt=system_prompt,
                        )
                        
                        # Store output
                        data["output_text"] = result["text"]
                        data["output_audio_ids"] = result.get("audio_ids")
                        
                        # Save audio if available
                        if result.get("audio") is not None and AUDIO_AVAILABLE:
                            audio_np = result["audio"]
                            if isinstance(audio_np, np.ndarray):
                                output_wav_path = os.path.join(output_dir, f"{conv_id}_tts.wav")
                                torchaudio.save(output_wav_path, torch.from_numpy(audio_np).float().unsqueeze(0).cpu(), args.output_sample_rate)
                                data["output_wav"] = output_wav_path
                                logger.info(f"[ID {conv_id}] TTS audio saved: {output_wav_path}")
                        elif result.get("audio_ids") is not None and AUDIO_AVAILABLE:
                            # Decode audio from tokens
                            audio_ids = torch.tensor(result["audio_ids"], device=engine.device)
                            length_ids = torch.tensor(result["length_ids"], device=engine.device) if result.get("length_ids") is not None else None
                            
                            fm_prompt_path = result.get("flow_matching_prompt_audio_path") or engine.config.flow_matching_prompt_audio_path
                            fm_prompt_audio = engine.prompt_audio_cache if fm_prompt_path == engine.config.flow_matching_prompt_audio_path else None
                            if engine.config.use_flow_matching_decoder:
                                audio_np = engine._decode_audio_tokens(
                                    audio_tokens=audio_ids,
                                    length_ids=length_ids,
                                    prompt_audio=fm_prompt_audio,
                                    prompt_audio_path=fm_prompt_path,
                                    framerate=result.get("framerate"),
                                )
                            else:
                                audio_np = engine._decode_audio_tokens_flexicodec(
                                    audio_tokens=audio_ids,
                                    length_ids=length_ids,
                                )
                            
                            if audio_np is not None:
                                output_wav_path = os.path.join(output_dir, f"{conv_id}_tts.wav")
                                torchaudio.save(output_wav_path, torch.from_numpy(audio_np).float().unsqueeze(0).cpu(), args.output_sample_rate)
                                data["output_wav"] = output_wav_path
                                logger.info(f"[ID {conv_id}] TTS audio saved: {output_wav_path}")
                        
                        logger.info(f"[ID {conv_id}] TTS Text: {result['text'][:100]}...")
                    else:
                        logger.warning(f"[ID {conv_id}] TTS task but could not extract sentence from messages")
                
                else:
                    # Original dialogue format
                    # Create output directory for this conversation
                    conversation_dir = os.path.join(output_dir, f"{conv_id}")
                    os.makedirs(conversation_dir, exist_ok=True)
                    
                    history = ""
                    dialogue = data.get("dialogue", [])
                    
                    for round_idx, round_item in enumerate(dialogue):
                        source_wav = round_item.get("source_wav", None)
                        source_text = round_item.get("source_text", "")
                        
                        # Generate response
                        if source_wav and os.path.exists(source_wav):
                            result = engine.generate_from_audio(
                                audio_path=source_wav,
                                text_query=source_text or "Please respond to the audio.",
                                history=history,
                                framerate=args.framerate,
                            )
                        else:
                            result = engine.generate_from_text(
                                text_input=source_text,
                                history=history,
                                framerate=args.framerate,
                            )
                        
                        # Update history
                        history = result["history"]
                        
                        # Store output text
                        round_item["output_text"] = result["text"]
                        
                        # Save audio
                        if result['audio_ids'] is not None and AUDIO_AVAILABLE:
                            reconstructed_audio = engine.model.flexicodec_dict['model'].decode_from_codes(
                                semantic_codes=result['audio_ids'].unsqueeze(0).unsqueeze(0),
                                acoustic_codes=None,
                                token_lengths=torch.ones_like(result['audio_ids'].unsqueeze(0)),
                            )
                            output_wav_path = os.path.join(conversation_dir, f"chat_{round_idx + 1}.wav")
                            torchaudio.save(output_wav_path, reconstructed_audio.float().squeeze(1).cpu(), 16000)
                            round_item["output_wav"] = output_wav_path
                            logger.info(f"[ID {conv_id}, Round {round_idx + 1}] Audio saved: {output_wav_path}")
                        
                        logger.info(f"[ID {conv_id}, Round {round_idx + 1}] Text: {result['text'][:100]}...")
                
                # Write result
                out_f.write(json.dumps(data, ensure_ascii=False) + "\n")
                
            except Exception as e:
                logger.error(f"Error processing line: {e}")
                import traceback
                traceback.print_exc()
    
    logger.info(f"Batch inference completed. Results saved to: {output_jsonl_path}")


@logger.catch(onerror=lambda _: sys.exit(1))
def main():
    parser = argparse.ArgumentParser(description="Interleaved S2S Inference")

    parser.add_argument("-m", "--model_path", type=str, default=MODEL_PATH,
                        help="Path to checkpoint directory (base or LoRA-finetuned ParallelS2S model)")
    parser.add_argument("--flexicodec_ckpt", type=str, 
                        default=None,
                        help="Path to FlexiCodec checkpoint")
    parser.add_argument("--flexicodec_config", type=str,
                        default=None,
                        help="Path to FlexiCodec config")
    parser.add_argument("--sensevoice_path", type=str,
                        default=None,
                        help="Path to SenseVoice model")
    
    # Flow matching decoder arguments
    parser.add_argument("-d", "--use_flow_matching_decoder", action="store_true", default=False,
                        help="Use flow matching decoder instead of FlexiCodec decode_from_codes")
    parser.add_argument("--flow_matching_ckpt", type=str, default=None,
                        help="Path to flow matching model checkpoint")
    parser.add_argument("--flow_matching_vocoder", type=str, default=None,
                        help="Path to flow matching vocoder checkpoint")
    parser.add_argument("--flow_matching_prompt_audio", type=str,
                        default=None,
                        help="Path to prompt audio file for flow matching decoder")
    parser.add_argument("--flow_matching_n_timesteps", type=int, default=20,
                        help="Number of diffusion timesteps for flow matching")
    parser.add_argument("--flow_matching_cfg", type=float, default=2.0,
                        help="Classifier-free guidance scale for flow matching")
    parser.add_argument("--flow_matching_rescale_cfg", type=float, default=0.75,
                        help="Rescaling factor for CFG in flow matching")
    
    # Model configuration
    parser.add_argument("--audio_vocab_size", type=int, default=32768,
                        help="Audio vocabulary size")
    parser.add_argument("--max_length_classes", type=int, default=32,
                        help="Number of length classes")
    # Note: use_omni_token is read from the checkpoint config.json (ParallelS2SConfig), not CLI.
    # Note: use_sensevoice_feature is now read from model config, not from command line
    
    # Frame rate control
    parser.add_argument("--framerate", type=float, default=7.0,
                        help="Default frame rate for output audio generation (0.8-1.0)")
    parser.add_argument("--input_framerate", type=float, default=1.0,
                        help="Input framerate: <=1.0 = merging_threshold, >1.0 = target_rate in Hz (e.g. 6.0)")
    parser.add_argument("--input_base_rate", type=float, default=12.5,
                        help="Base frame rate in Hz for target-rate merging (when input_framerate > 1.0)")
    parser.add_argument("--no_dynamic_merging", action="store_true",
                        help="Use uniform merging instead of dynamic threshold search when target_rate is set")
    parser.add_argument("--framerate_min", type=float, default=1.0,
                        help="Minimum frame rate")
    parser.add_argument("--framerate_max", type=float, default=100.0,
                        help="Maximum frame rate")
    parser.add_argument("--enable_flexible_framerate", action="store_true",
                        help="Enable flexible frame rate with SenseVoice feature merging")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=600,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--do_sample", action="store_true", default=True,
                        help="Use sampling instead of greedy decoding")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="Repetition penalty")
    
    # Length prediction
    parser.add_argument("--length_temperature", type=float, default=1.0,
                        help="Temperature for length sampling")
    parser.add_argument("--length_top_k", type=int, default=1,
                        help="Top-k for length sampling")
    
    # I/O arguments
    parser.add_argument("--input_file", type=str, default=None,
                        help="Input JSONL file for batch mode")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--output_sample_rate", type=int, default=24000,
                        help="Output audio sample rate")
    parser.add_argument("--no_audio", action="store_true",
                        help="Disable audio decoding")
    
    # System prompt arguments
    parser.add_argument("-s", "--system_prompt_type", type=str, default="interleaved",
                        choices=["interleaved", "text_only"],
                        help="System prompt type: 'interleaved' for audio-text interleaved response, 'text_only' for text-only response")
    parser.add_argument("--custom_system_prompt", type=str, default=None,
                        help="Custom system prompt (overrides system_prompt_type)")
    
    # Runtime arguments
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["auto", "bfloat16", "float16", "float32"],
                        help="Torch dtype")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Run minimal debug examples (TTS, T2S, S2S, S2T, T2T) before interactive mode")
    parser.add_argument("--debug_audio_path", type=str, default=None,
                        help="Optional audio path used by S2S/S2T examples in --debug mode")
    
    args = parser.parse_args()
    if not args.model_path:
        parser.error("--model_path is required.")
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Determine system prompt
    if args.custom_system_prompt:
        system_prompt = args.custom_system_prompt
    elif args.system_prompt_type == "text_only":
        system_prompt = ""
    else:  # interleaved
        system_prompt = ""
    
    logger.info(f"Using system prompt type: {args.system_prompt_type}")
    logger.info(f"System prompt: {system_prompt}")
    
    # Create config
    config = InterleavedInferenceConfig(
        model_path=args.model_path,
        flexicodec_ckpt_path=args.flexicodec_ckpt,
        flexicodec_config_path=args.flexicodec_config,
        sensevoice_path=args.sensevoice_path,
        audio_vocab_size=args.audio_vocab_size,
        max_length_classes=args.max_length_classes,
        # use_omni_token is set from checkpoint after model load
        # use_sensevoice_feature will be read from model config during model loading
        framerate_min=args.framerate_min,
        framerate_max=args.framerate_max,
        default_framerate=args.framerate,
        input_framerate=args.input_framerate,
        input_base_rate=args.input_base_rate,
        dynamic_merging=not args.no_dynamic_merging,
        enable_flexible_framerate=args.enable_flexible_framerate,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        length_temperature=args.length_temperature,
        length_top_k=args.length_top_k,
        output_sample_rate=args.output_sample_rate,
        decode_audio=not args.no_audio,
        torch_dtype=args.torch_dtype,
        system_prompt=system_prompt,
        use_flow_matching_decoder=args.use_flow_matching_decoder,
        flow_matching_ckpt_path=args.flow_matching_ckpt,
        flow_matching_vocoder_path=args.flow_matching_vocoder,
        flow_matching_prompt_audio_path=args.flow_matching_prompt_audio,
        flow_matching_n_timesteps=args.flow_matching_n_timesteps,
        flow_matching_cfg=args.flow_matching_cfg,
        flow_matching_rescale_cfg=args.flow_matching_rescale_cfg,
    )
    
    try:
        # Initialize inference engine
        engine = InterleavedS2SInference(config, device=args.device)
        
        # Run appropriate mode
        if args.input_file:
            run_batch_mode(engine, args)
        else:
            run_interactive_mode(engine, args)
            
    except Exception as e:
        logger.error(f"Failed to initialize InterleavedS2S inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
