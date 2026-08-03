# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from typing import Optional, Tuple, Union, List


class MultimodalQwen2Config(Qwen2Config):
    """
    Extend Qwen2Config to support multimodal configuration, compatible with glm4voice token design.
    """
    def __init__(
        self,
        adaptor_input_dim: int = 5120,  # Audio embedding dimension; defaults to hidden_size.
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.adaptor_input_dim = adaptor_input_dim        
        # Add model type identifier.
        self.model_type = "multimodal_qwen2"
