# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""
Interleaved SpeechLM with FlexiCodec tokens.

This model implements the **interleaving scheme** from your diagram:
- A single token sequence containing text, control tokens (e.g. `audio_start_kHz`, `audio_end`),
  and audio RVQ tokens.
- For audio RVQ tokens, the encoder is the **sum of embeddings**:
  `emb(token_id) + emb(len_prev)` where `len_prev` is the previous token's
  length attribute from FlexiCodec.
- The backbone is a Qwen2-style causal LM with optional continuous-audio
  frontends such as SenseVoice, Qwen3-ASR, Qwen2.5-Omni Encoder,
  or Whisper-large-v3.

Training objectives:
- Standard causal LM loss over the joint text+audio vocabulary (like a normal LM).
- Optional auxiliary loss to predict the length attribute for each token.

The data pipeline is expected to provide:
- `input_ids`: interleaved text+audio token ids.
- `length_labels`: discretised length attribute for each token (or -100 where undefined).
  The model internally constructs the *input* length ids as a 1-token-right-shift
  of `length_labels` so that at step *t* it conditions on the length of token *t-1*,
  matching the diagram.
"""
import contextlib
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM, AutoModelForCausalLM, WhisperModel
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.trainer_pt_utils import LabelSmoother
from typing import Optional, Tuple, Union, List, Dict
import random
import math
import deepspeed
import torch.distributed as dist
import os
from src.models.configuration_flexislm import MultimodalQwen2Config
from src.models.generation_alignment import DelayedAudioLengthBuffer
from accelerate import init_empty_weights
from flexicodec.infer import prepare_model, encode_flexicodec
import flexicodec.model_blocks.mimi.transformer as Stransformer
import flexicodec.model_blocks.mimi.transformer_windowed as Stransformer_windowed
import loguru
logger = loguru.logger
IGNORE_TOKEN_ID = -100
# Talker vocab: first 3 tokens are special (AUD_START, AUD_END, AUD_TAG), then audio codes


def _resolve_text_alignment_pad_loss_weight(config) -> float:
    """CE weight for text–speech alignment padding (extended text to match speech length).

    If ``config.text_alignment_pad_loss_weight`` is set, that value is used. Otherwise:
    ``0.0`` when ``freeze_talker`` (no pad CE), else ``0.1``.
    """
    v = getattr(config, "text_alignment_pad_loss_weight", None)
    if v is not None:
        return float(v)
    return 0.0 if getattr(config, "freeze_talker", False) else 0.1


def _qwen25o_output_dim_from_config(config_path: str) -> int:
    """Read the retained Qwen2.5-Omni audio projection output dimension."""
    if not config_path or not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"qwen25o_encoder_config_path not found: {config_path}"
        )
    import json

    with open(config_path, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    if "thinker_config" in config:
        config = config["thinker_config"]
    if "audio_config" in config:
        config = config["audio_config"]
    output_dim = config.get("output_dim")
    if not isinstance(output_dim, int) or output_dim <= 0:
        raise ValueError(
            "Qwen2.5-Omni audio config requires a positive integer output_dim: "
            f"{config_path}"
        )
    return output_dim


# Talker vocab: first 3 tokens are special (AUD_START, AUD_END, AUD_TAG), then audio codes
AUDIO_TOKEN_OFFSET = 3  # offset for audio codes in talker vocab (indices 0,1,2 = special tokens)
def _patch_qwen3_audio_encoder_forward(encoder):
    """Patch Qwen3ASRAudioEncoder.forward so chunking is along time axis (not mel).
    See AmphionASR/src/qwen3_aut/qwen3_encoder_adapter.py for rationale.
    Also adds forward_cnn_only and forward_transformer_only for batched transformer encoding.
    """
    import types
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import _get_feat_extract_output_lengths

    def _forward_cnn_only_single(self, input_features, feature_lens):
        """Run only the CNN + positional embedding + pack for a single sample.
        input_features: (H, T) mel, feature_lens: (1,) with value T.
        Returns (hidden_states (S, D), cu_seqlens (1+num_windows,)).
        """
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
        chunk_lengths = torch.tensor(
            [self.n_window * 2] * int(chunk_num.sum().item()),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
            batch_first=True,
        )
        padded_feature = padded_feature.unsqueeze(1)
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (int(cnn_len) // window_aftercnn)
            remainder = int(cnn_len) % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)
        return hidden_states, cu_seqlens

    def _forward_transformer_only(self, hidden_states, cu_seqlens):
        """Run only the encoder layers + ln_post + proj (proj may be Identity) on packed batch."""
        attention_mask = self._prepare_attention_mask(hidden_states, cu_seqlens)
        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs[0]
        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states

    orig_forward = encoder.forward

    def _patched_forward(
        self,
        input_features,
        feature_lens=None,
        *args,
        **kwargs,
    ):
        if feature_lens is None:
            if input_features.ndim == 2:
                feature_lens = torch.tensor(
                    [input_features.shape[0]], device=input_features.device, dtype=torch.long
                )
            elif input_features.ndim == 3:
                feature_lens = torch.tensor(
                    [input_features.shape[1]], device=input_features.device, dtype=torch.long
                )
            else:
                raise ValueError(f"Unexpected input_features shape: {tuple(input_features.shape)}")
        feature_lens = feature_lens.to(dtype=torch.long)
        if input_features.ndim != 2:
            return orig_forward(input_features, feature_lens=feature_lens, *args, **kwargs)
        t = int(feature_lens[0].item()) if feature_lens.numel() > 0 else input_features.shape[0]
        if input_features.shape[0] == t:
            input_features_ft = input_features.transpose(0, 1).contiguous()
        else:
            input_features_ft = input_features.contiguous()
        return orig_forward(input_features_ft, feature_lens=feature_lens, *args, **kwargs)

    encoder.forward = types.MethodType(_patched_forward, encoder)
    encoder.forward_cnn_only = types.MethodType(_forward_cnn_only_single, encoder)
    encoder.forward_transformer_only = types.MethodType(_forward_transformer_only, encoder)
    return encoder


num_params = lambda x: f"{sum(p.numel() for p in x.parameters()):,}"


def get_rank():
    """Get the current process rank in distributed training, or 0 if not distributed."""
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def get_world_size():
    """Get the world size in distributed training, or 1 if not distributed."""
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def choose_rank_shifted_option(options):
    """Sample one shared option index, then rotate it by rank.

    This keeps per-step randomness while spreading different ranks across the
    available options instead of letting every rank pick the same entry.
    Collisions are still unavoidable when ``world_size > len(options)``.
    """
    if not options:
        raise ValueError("options must be non-empty")

    base_index = random.randrange(len(options))

    return options[(base_index + get_rank()) % len(options)]


def debug_print(message, rank=None, force=False):
    """Print debug message with rank information.
    
    Args:
        message: Message to print
        rank: Optional rank to include (if None, will get current rank)
        force: If True, print even if not rank 0 (useful for debugging all ranks)
    """
    return
    if rank is None:
        rank = get_rank()
    world_size = get_world_size()
    
    if force or rank == 0:
        logger.info(f"[Rank {rank}/{world_size}] {message}")
    elif rank < 4:  # Print for first 4 ranks to avoid too much output
        logger.info(f"[Rank {rank}/{world_size}] {message}")


class Mlp(nn.Module):
    """MLP module following the codebase pattern."""
    
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InterleavedMergingTransformer(nn.Module):
    """FlexiCodec-style merging transformer that operates on an interleaved
    sequence of pre-merge frames and per-group query tokens.

    Inspired by ``QueryTokenAggregator`` in
    ``zero_shot_tts_training/realtime_communication/taste_v2/modeling_flexicodec.py``.

    Differences vs. the v1 ``input_merging_transformer``:
      * v1 first calls ``aggregate_features`` (mean pool inside each group) to
        get a length-G sequence and then runs a transformer over those merged
        tokens only -- it never sees the original pre-merge frames.
      * v2 builds a length-(T+G) sequence where one query token is inserted
        right after each group's last frame.  A standard transformer attends
        over both pre-merge frames and query tokens, then we gather the
        outputs at the query positions as the merged representation.

    Args:
        d_model: hidden dimension (matches ``config.hidden_size``).
        num_heads / num_layers / dim_feedforward / context / causal: forwarded
            to ``ProjectedTransformer``.
        use_mean_pooling_init: if True, init each query as the mean of its
            group's pre-merge frames; otherwise use a single learnable token.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        context: int,
        causal: bool = False,
        use_mean_pooling_init: bool = True,
        input_dim: Optional[int] = None,
    ):
        super().__init__()
        # ``input_dim`` is the dimensionality of the pre-merge frames coming
        # from ``audio_embed_transform`` (i.e. ``config.hidden_size``). When it
        # differs from ``d_model`` the inner ``ProjectedTransformer`` will
        # down-project to ``d_model`` and project back to ``input_dim``, so the
        # merged output keeps the same dimensionality as the LM hidden state.
        if input_dim is None:
            input_dim = d_model
        self.d_model = d_model
        self.input_dim = input_dim
        self.use_mean_pooling_init = use_mean_pooling_init

        if not self.use_mean_pooling_init:
            self.query_token = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)

        _transformer_module = Stransformer_windowed if not causal else Stransformer
        self.transformer = _transformer_module.ProjectedTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            causal=causal,
            layer_scale=0.01,
            context=context,
            conv_layout=False,
            max_period=10000,
            gating='none',
            norm='layer_norm',
            positional_embedding='rope',
            dim_feedforward=dim_feedforward,
            input_dimension=input_dim,
            output_dimensions=[input_dim],
        )

    def forward(
        self,
        features: torch.Tensor,
        alignment_matrix: torch.Tensor,
        num_segments_per_item: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: pre-merge frame features, shape ``(B, T, D)``.
            alignment_matrix: binary group-membership matrix, shape
                ``(B, G, T)``. ``alignment_matrix[b, g, t] = 1`` iff frame
                ``t`` belongs to group ``g`` of batch item ``b``.
            num_segments_per_item: ``(B,)`` int tensor with the number of
                valid groups for each item.

        Returns:
            merged features of shape ``(B, G, D)``.
        """
        B, T, D = features.shape
        _B, G, _T = alignment_matrix.shape
        device = features.device

        # Trim time dimension to alignment matrix length if needed.
        if T > _T:
            features = features[:, :_T, :]
            T = _T
        assert T == _T, (
            f"InterleavedMergingTransformer: feature time {T} must match "
            f"alignment time {_T}"
        )

        # Group / frame validity masks
        group_mask = (
            torch.arange(G, device=device).unsqueeze(0)
            < num_segments_per_item.unsqueeze(1)
        )  # (B, G)

        align_long = alignment_matrix.to(torch.long)
        # For each (b, g), the index of its last frame; 0 for empty/padded groups.
        group_last_frame_indices = (
            align_long * torch.arange(T, device=device).view(1, 1, T)
        ).max(dim=2).values  # (B, G)

        valid_last_indices = group_last_frame_indices.masked_fill(~group_mask, -1)
        frame_lengths = valid_last_indices.max(dim=1).values + 1  # (B,)
        frame_mask = (
            torch.arange(T, device=device).unsqueeze(0) < frame_lengths.unsqueeze(1)
        )  # (B, T)

        # Destination index of each frame in the interleaved sequence:
        # original t shifted by the number of queries that precede it.
        last_indices_for_count = group_last_frame_indices.clone()
        last_indices_for_count = last_indices_for_count.masked_fill(
            ~group_mask, T + 1
        )  # push padded queries to the right
        num_queries_before = (
            last_indices_for_count.unsqueeze(2)
            < torch.arange(T, device=device).view(1, 1, T)
        ).sum(dim=1)  # (B, T)
        frame_dest = torch.arange(T, device=device).unsqueeze(0) + num_queries_before
        # Each query lands right after its group's last frame.
        query_dest = (
            group_last_frame_indices
            + torch.arange(G, device=device).unsqueeze(0)
            + 1
        )  # (B, G)

        # Build per-group queries (in (B, G, D)).
        if self.use_mean_pooling_init:
            align_float = alignment_matrix.to(features.dtype)
            summed = torch.einsum('bgt,btd->bgd', align_float, features)
            counts = align_float.sum(dim=2).clamp(min=1).unsqueeze(-1)
            queries = summed / counts
        else:
            queries = self.query_token.expand(B, G, D).to(features.dtype)

        # Concat then permute into the interleaved order.
        source_seq = torch.cat([features, queries], dim=1)  # (B, T+G, D)
        dest_indices = torch.cat([frame_dest, query_dest], dim=1)  # (B, T+G)
        source_mask = torch.cat([frame_mask, group_mask], dim=1)  # (B, T+G)

        max_len = T + G
        dest_indices_masked = dest_indices.masked_fill(~source_mask, max_len)
        perm = dest_indices_masked.argsort(dim=1)  # (B, T+G)
        perm_expanded = perm.unsqueeze(-1).expand(-1, -1, D)
        interleaved = torch.gather(source_seq, 1, perm_expanded)  # (B, T+G, D)

        transformer_out = self.transformer(interleaved)  # (B, T+G, D)

        # Recover query positions in the interleaved sequence and gather them.
        inverse_perm = perm.argsort(dim=1)
        query_pos_in_interleaved = inverse_perm[:, T:]  # (B, G)
        query_pos_expanded = query_pos_in_interleaved.unsqueeze(-1).expand(-1, -1, D)
        merged = torch.gather(transformer_out, 1, query_pos_expanded)  # (B, G, D)

        merged = merged.masked_fill(~group_mask.unsqueeze(-1), 0.0)
        return merged


class ParallelS2SConfig(MultimodalQwen2Config):
    """Config for parallel text+audio SpeechLM.

    Compared to :class:`MultimodalQwen2Config` this adds:
    - ``audio_vocab_size``: size of RVQ token vocab (per-codebook).
    - ``padded_audio_vocab_size``: actual size used when expanding LM vocab.
    - ``max_length_classes``: number of discrete length classes for the
      FlexiCodec ``token_lengths`` attribute.
    - ``talker_*``: configuration for the talker transformer.
    - ``talker_pretrained_model_path``: optional path to a pretrained model (e.g. Qwen2.5-0.5B) to
      initialise the talker backbone. When set, ``lm_head`` and ``embed_tokens`` are replaced with
      randomly initialised layers sized for the talker audio vocab, and ``talker_hidden_size`` /
      ``talker_num_layers`` / ``talker_num_attention_heads`` / ``talker_intermediate_size`` are
      overridden from the pretrained config. Disabled by default (``None``).
    - ``speech_delay_tokens``: number of null tokens to delay speech generation.
    - ``use_chained_architecture``: enable the chained 0.5B->3B->0.5B architecture.
    - ``chained_3b_model_path``: path to the frozen 3B textual LLM.
    - ``chained_adaptor_hidden_size``: hidden size for the adaptors.
    - ``use_attention_gating``: use attention-based gating to replace FiLM in chained mode.
    - ``enable_flexible_framerate``: enable flexible frame rate with SenseVoice feature merging.
    - ``uniform_merging``: use uniform input feature merging with a random 4-12 Hz target rate.
    - ``output_uniform_merging``: ablation; force FlexiCodec output-side (assistant) merging to be
      UNIFORM with a random 4-12 Hz target rate, instead of similarity-based merging.
    - ``max_tokens_per_group``: maximum tokens per group when using flexible frame rate.
    - ``per_sample_frame_rate_embed``: when enabled, use per-sample frame rate (from code_lens/feature_lens*15, range [0, 15] Hz) instead of unified merge threshold embed.
    - ``freeze_llm``: if True, during training only optimize text loss (speech and length losses ignored).
    - ``only_train_llm``: if True, freeze talker/adaptor and only train main LLM; loss = text_loss.
    - ``freeze_talker``: if True, freeze talker modules and exclude talker loss (speech_loss, len_loss) from training; loss = text_loss only.
    - ``text_alignment_pad_loss_weight``: optional float; weight for CE on alignment-extended text pad tokens (see :meth:`_resolve_text_alignment_pad_loss_weight`). If omitted, defaults to 0 with ``freeze_talker`` else 0.1.
    - ``no_pad``: if True, do not pad text token IDs up to the speech length in the parallel sequence. The model
      accepts text/audio length mismatch; text loss is ignored on the extra tail and thinker hiddens are zeroed
      for those positions before building talker conditioning.
    - ``lora_rank`` / ``lora_alpha``: when training with PEFT LoRA, these mirror the adapter hyperparameters (persisted in ``config.json`` for inference and bookkeeping).
    - ``use_omni_token``: when True, training used Omni audio boundary tokens and system prompts (persisted in ``config.json``).
    """

    model_type = "parallel_s2s"

    def __init__(
        self,
        audio_vocab_size: int = 32768,
        text_vocab_size: int = 151936,
        padded_audio_vocab_size: int = 32768,
        max_length_classes: int = 32,
        framerate_min: float = 1.0,
        framerate_max: float = 1.0,
        framerate_token_id: Optional[int] = None,
        enable_flexible_framerate: bool = False,
        uniform_merging: bool = False,
        output_uniform_merging: bool = False,
        use_sinusoidal: bool = False,
        per_sample_frame_rate_embed: bool = False,
        max_tokens_per_group: Optional[int] = 8,
        training_framerate_options: Optional[Union[List[float], str]] = None,
        training_input_framerate_options: Optional[Union[List[float], str]] = None,
        codec_hidden_size=768,
        use_sensevoice_feature=False,
        use_qwen3_feature=False,
        use_whisper_fetaure=False,
        qwen3_encoder_path: Optional[str] = None,
        qwen3_encoder_config_path: Optional[str] = None,
        whisper_encoder_path: Optional[str] = None,
        use_qwen25o_feature=False,
        qwen25o_encoder_path: Optional[str] = None,
        qwen25o_encoder_config_path: Optional[str] = None,
        use_joint_text_audio_vocab=False,
        add_length_embeddings=False,
        use_mlp_for_audio_embed=False,
        audio_embed_mlp_hidden_ratio=4.0,
        audio_embed_mlp_dropout=0.0,
        audio_end_id=None,
        use_learnable_prefix=False,
        num_prefix_tokens=32,
        use_chained_architecture=False,
        chained_3b_model_path=None,
        chained_adaptor_hidden_size=512,
        use_attention_gating: bool = False,
        # Parallel architecture parameters
        talker_hidden_size: Optional[int] = None,
        talker_num_layers: int = 20,
        talker_num_attention_heads: int = 8,
        talker_intermediate_size: Optional[int] = None,
        talker_pretrained_model_path: Optional[str] = None,
        speech_delay_tokens: int = 5,
        talker_concat_lm_text_output: bool = False,
        use_concat_len_emb: bool = False,
        talker_embed_v2: bool = False,
        early_diverge_talker: bool = False,
        text_loss_weight: float = 1.0,
        length_loss_weight: float = 1.0,
        text_alignment_pad_loss_weight: Optional[float] = None,
        extend_lm_head: bool = False,
        no_pad: bool = False,
        freeze_llm: bool = False,
        only_train_llm: bool = False,
        freeze_talker: bool = False,
        finetune_speech_encoder: bool = False,
        lora_rank: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        use_omni_token: bool = False,
        AUD_START_TOKEN = None,
        AUD_END_TOKEN = None,
        AUD_TAG_TOKEN = None,
        # Input merging transformer (local windowed transformer after audio_embed_transform)
        use_input_merging_transformer: bool = False,
        input_merging_transformer_num_layers: int = 4,
        input_merging_transformer_d_model: int = 0,
        input_merging_transformer_num_heads: int = 8,
        input_merging_transformer_dim_feedforward: int = 2048,
        input_merging_transformer_context: int = 32,
        input_merging_transformer_causal: bool = False,
        # Input merging transformer v2 (FlexiCodec-style: the transformer processes an
        # interleaved sequence of pre-merge frames + per-group query tokens, so the
        # merged representation is produced by attention over the un-merged frames
        # instead of a separate aggregate_features + refine transformer pipeline).
        use_input_merging_transformer_v2: bool = False,
        # Learnable audio boundary embeddings (replace audio_start/end token IDs)
        use_learnable_audio_boundary: bool = True,
        # Ablation: replace the length prediction head with a second-audio-token
        # prediction head (group size 2). When enabled, ``audio_token_lengths``
        # supplied by the dataloader is reinterpreted as the second audio token
        # id (in audio-vocab space, 0-indexed) at each step instead of a length
        # class. Default disabled.
        predict_second_audio_token: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.early_diverge_talker = early_diverge_talker
        self.audio_vocab_size = audio_vocab_size
        self.padded_audio_vocab_size = padded_audio_vocab_size
        self.max_length_classes = max_length_classes
        self.text_vocab_size = text_vocab_size
        # Continuous frame-rate control range [framerate_min, framerate_max]
        self.framerate_min = framerate_min
        self.framerate_max = framerate_max
        self.codec_hidden_size = codec_hidden_size
        self.use_sensevoice_feature = use_sensevoice_feature
        self.use_qwen3_feature = use_qwen3_feature
        self.use_whisper_fetaure = use_whisper_fetaure
        self.use_whisper_feature = use_whisper_fetaure
        self.qwen3_encoder_path = qwen3_encoder_path
        self.qwen3_encoder_config_path = qwen3_encoder_config_path
        self.whisper_encoder_path = whisper_encoder_path
        self.use_qwen25o_feature = use_qwen25o_feature
        self.qwen25o_encoder_path = qwen25o_encoder_path
        self.qwen25o_encoder_config_path = qwen25o_encoder_config_path
        # Optional special token id whose hidden state will receive the
        # frame-rate embedding. If None, the first token is used.
        self.framerate_token_id = framerate_token_id
        self.use_joint_text_audio_vocab = use_joint_text_audio_vocab
        self.add_length_embeddings = add_length_embeddings
        self.use_mlp_for_audio_embed = use_mlp_for_audio_embed
        self.audio_embed_mlp_hidden_ratio = audio_embed_mlp_hidden_ratio
        self.audio_embed_mlp_dropout = audio_embed_mlp_dropout
        self.audio_end_id = audio_end_id
        # Learnable prefix tokens
        self.use_learnable_prefix = use_learnable_prefix
        self.num_prefix_tokens = num_prefix_tokens
        # Chained architecture parameters
        self.use_chained_architecture = use_chained_architecture
        self.chained_3b_model_path = chained_3b_model_path
        self.chained_adaptor_hidden_size = chained_adaptor_hidden_size
        self.use_attention_gating = use_attention_gating
        # Parallel architecture parameters (always enabled)
        self.talker_hidden_size = talker_hidden_size or kwargs.get('hidden_size', 896)
        self.talker_num_layers = talker_num_layers
        self.talker_num_attention_heads = talker_num_attention_heads
        self.talker_intermediate_size = talker_intermediate_size or (self.talker_hidden_size * 4)
        self.speech_delay_tokens = speech_delay_tokens
        self.talker_pretrained_model_path = talker_pretrained_model_path
        self.talker_concat_lm_text_output = talker_concat_lm_text_output
        self.use_concat_len_emb = use_concat_len_emb
        self.talker_embed_v2 = talker_embed_v2
        self.text_loss_weight = text_loss_weight
        self.length_loss_weight = length_loss_weight
        self.text_alignment_pad_loss_weight = text_alignment_pad_loss_weight
        self.extend_lm_head = bool(extend_lm_head)
        self.no_pad = bool(no_pad)
        # Combined embeddings: whether main LM receives text+audio+length embeddings
        self.use_combined_embedding = kwargs.get('use_combined_embedding', True)
        # When True, use combined embedding even if use_lora (overrides use_lora's text-only behavior)
        self.force_use_combined_embedding = kwargs.get('force_use_combined_embedding', False)

        self.AUD_START_TOKEN = AUD_START_TOKEN
        self.AUD_END_TOKEN = AUD_END_TOKEN
        self.AUD_TAG_TOKEN = AUD_TAG_TOKEN
        # Flexible frame rate parameters
        self.enable_flexible_framerate = enable_flexible_framerate or uniform_merging
        self.uniform_merging = uniform_merging
        self.output_uniform_merging = output_uniform_merging
        self.use_sinusoidal = use_sinusoidal
        self.per_sample_frame_rate_embed = per_sample_frame_rate_embed
        self.max_tokens_per_group = max_tokens_per_group
        # Training-time discrete framerate options (defaults match previous hardcoded values)
        if training_framerate_options is None:
            self.training_framerate_options = [0.87, 0.91, 1.0]
        elif isinstance(training_framerate_options, str):
            self.training_framerate_options = [float(x.strip()) for x in training_framerate_options.split(",")]
        else:
            self.training_framerate_options = list(training_framerate_options)
        if training_input_framerate_options is None:
            self.training_input_framerate_options = [0.87, 0.91, 1.0, 0.85]
        elif isinstance(training_input_framerate_options, str):
            self.training_input_framerate_options = [float(x.strip()) for x in training_input_framerate_options.split(",")]
        else:
            self.training_input_framerate_options = list(training_input_framerate_options)
        self.freeze_llm = freeze_llm
        self.only_train_llm = only_train_llm
        self.freeze_talker = freeze_talker
        self.finetune_speech_encoder = finetune_speech_encoder
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        # When True, training used <|audio_bos|>/<|audio_eos|> and Omni system prompts; persisted for inference.
        self.use_omni_token = use_omni_token
        # Input merging transformer parameters
        self.use_input_merging_transformer = use_input_merging_transformer
        self.input_merging_transformer_num_layers = input_merging_transformer_num_layers
        self.input_merging_transformer_d_model = input_merging_transformer_d_model
        self.input_merging_transformer_num_heads = input_merging_transformer_num_heads
        self.input_merging_transformer_dim_feedforward = input_merging_transformer_dim_feedforward
        self.input_merging_transformer_context = input_merging_transformer_context
        self.input_merging_transformer_causal = input_merging_transformer_causal
        # v2 (interleaved pre-merge frames + per-group query tokens)
        self.use_input_merging_transformer_v2 = use_input_merging_transformer_v2
        self.use_learnable_audio_boundary = use_learnable_audio_boundary
        self.predict_second_audio_token = bool(predict_second_audio_token)


@dataclass
class InterleavedS2SOutputWithPast(CausalLMOutputWithPast):
    length_logits: Optional[torch.FloatTensor] = None
    length_loss: Optional[torch.FloatTensor] = None
    text_loss: Optional[torch.FloatTensor] = None
    text_ce_loss: Optional[torch.FloatTensor] = None  # unweighted mean text CE (before text_loss_weight)
    text_token_loss: Optional[torch.FloatTensor] = None
    audio_token_loss: Optional[torch.FloatTensor] = None
    acoustic_loss: Optional[torch.FloatTensor] = None
    acoustic_ce_loss: Optional[torch.FloatTensor] = None  # unweighted mean acoustic CE
    acoustic_per_codebook_loss: Optional[torch.FloatTensor] = None  # [n_q_a] unweighted CE per codebook
    loss_text_only_data: Optional[torch.FloatTensor] = None
    loss_audio_dialog_data: Optional[torch.FloatTensor] = None

class InterleavedQwen2Model(Qwen2Model):
    """Thin Qwen2Model wrapper used by :class:`ParallelS2SForCausalLM`.

    We always drive this model with ``inputs_embeds`` that already include
    the RVQ + length embedding sum, so we do not override any behaviour
    besides calling ``post_init``.
    """

    def __init__(self, config: ParallelS2SConfig):
        super().__init__(config)
        self.config = config
        # Training-time option: freeze everything except talker; also disables
        # feeding speech-conditioned embeddings back into the main LLM input.
        self.post_init()
        # self.config.max_tokens_per_group = None

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        """Required by PEFT when wrapping this model with LoRA. ParallelS2SForCausalLM uses custom generate with inputs_embeds."""
        if past_key_values is not None and input_ids is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            **kwargs,
        }


class TalkerModel(nn.Module):
    """Talker transformer with projection layers for conditioning and embedding projection.

    Contains:
    - model: Qwen2ForCausalLM (separate mode) or Qwen2Model (joint mode) - the base transformer
    - lm_to_talker_proj: projects main LM hidden to talker hidden
    - talker_cond_proj: combines [base, fr, audio, length] conditioning
    - talker_embed_to_hidden: projects talker embeddings to main LM hidden (separate mode only)
    - talker_length_decoder: predicts length class from talker hidden state
    - length_embedding: embeds length classes (used for main LM + talker conditioning)
    """

    def __init__(self, config: ParallelS2SConfig):
        super().__init__()
        self.config = config
        rank = get_rank()
        talker_vocab_size = (AUDIO_TOKEN_OFFSET + config.padded_audio_vocab_size) if not config.use_joint_text_audio_vocab else 1
        if not config.use_joint_text_audio_vocab:
            if getattr(config, 'talker_pretrained_model_path', None):
                # Load pretrained model (e.g. Qwen2.5-0.5B) as talker backbone.
                # Replace lm_head and embed_tokens since our talker vocab size differs.
                pretrained = AutoModelForCausalLM.from_pretrained(config.talker_pretrained_model_path)
                pretrained_cfg = pretrained.config
                # Update config to match pretrained architecture
                config.talker_hidden_size = pretrained_cfg.hidden_size
                config.talker_num_layers = pretrained_cfg.num_hidden_layers
                config.talker_num_attention_heads = pretrained_cfg.num_attention_heads
                config.talker_intermediate_size = pretrained_cfg.intermediate_size
                # Replace vocab-dependent layers with randomly initialized ones
                pretrained.lm_head = nn.Linear(pretrained_cfg.hidden_size, talker_vocab_size, bias=False)
                pretrained.model.embed_tokens = nn.Embedding(talker_vocab_size, pretrained_cfg.hidden_size)
                self.model = pretrained
                logger.info(
                    f"Loaded pretrained talker from {config.talker_pretrained_model_path}, "
                    f"replaced lm_head and embed_tokens for vocab_size={talker_vocab_size}"
                )
            else:
                talker_config = Qwen2Config(
                    vocab_size=talker_vocab_size,
                    hidden_size=config.talker_hidden_size,
                    intermediate_size=config.talker_intermediate_size,
                    num_hidden_layers=config.talker_num_layers,
                    num_attention_heads=config.talker_num_attention_heads,
                    num_key_value_heads=config.talker_num_attention_heads,
                    max_position_embeddings=config.max_position_embeddings,
                    rope_theta=config.rope_theta if hasattr(config, 'rope_theta') else 10000.0,
                    attention_dropout=config.attention_dropout if hasattr(config, 'attention_dropout') else 0.0,
                    hidden_dropout=0.0,
                )
                self.model = Qwen2ForCausalLM(talker_config)
            self.model.config._attn_implementation = getattr(config, '_attn_implementation', 'sdpa')
            if getattr(config, 'talker_embed_v2', False):
                self.talker_embed_to_hidden = None
                debug_print(f"  - talker_embed_v2: skipping talker_embed_to_hidden (embeds stay at talker_hidden_size={config.talker_hidden_size})", rank=rank)
            elif config.talker_hidden_size != config.hidden_size:
                self.talker_embed_to_hidden = nn.Linear(config.talker_hidden_size, config.hidden_size)
                debug_print(f"  - Created talker_embed_to_hidden: {config.talker_hidden_size} -> {config.hidden_size}", rank=rank)
            else:
                self.talker_embed_to_hidden = None
        else:
            raise ValueError("use_joint_text_audio_vocab must be False for separate mode")

        if getattr(config, 'talker_embed_v2', False):
            self.lm_to_talker_proj = nn.Identity()
            debug_print(f"  - talker_embed_v2: lm_to_talker_proj set to Identity (no projection)", rank=rank)
        elif config.hidden_size != config.talker_hidden_size:
            self.lm_to_talker_proj = nn.Linear(config.hidden_size, config.talker_hidden_size)
            debug_print(f"  - Created LM->Talker projection: {config.hidden_size} -> {config.talker_hidden_size}", rank=rank)
        else:
            self.lm_to_talker_proj = nn.Identity()
        if getattr(config, 'talker_embed_v2', False):
            # base and (optional) text_context stay at hidden_size; audio, length, fr are talker_hidden_size
            num_lm_slots = 2 if getattr(config, "talker_concat_lm_text_output", False) else 1
            cond_in = config.hidden_size * num_lm_slots + config.talker_hidden_size * 3
        else:
            num_cond = 5 if getattr(config, "talker_concat_lm_text_output", False) else 4
            cond_in = config.talker_hidden_size * num_cond
        self.talker_cond_proj = nn.Linear(cond_in, config.talker_hidden_size)
        debug_print(
            f"  - Created talker_cond_proj: {cond_in} -> {config.talker_hidden_size} (talker_embed_v2={getattr(config, 'talker_embed_v2', False)})",
            rank=rank,
        )
        # Output dimensionality of the secondary head fed by talker_hidden:
        # by default it predicts a length class; in the second-audio-token
        # ablation it predicts another audio token (talker-vocab space).
        predict_second_audio_token = bool(getattr(config, "predict_second_audio_token", False))
        self._predict_second_audio_token = predict_second_audio_token
        secondary_out = (
            (AUDIO_TOKEN_OFFSET + config.padded_audio_vocab_size)
            if predict_second_audio_token
            else config.max_length_classes
        )
        if config.use_sinusoidal:
            hidden_features = config.talker_hidden_size * 4
            self.talker_length_decoder = Mlp(
                in_features=config.talker_hidden_size,
                hidden_features=hidden_features,
                out_features=secondary_out,
                act_layer=nn.GELU,
                drop=0.0,
            )
            debug_print(f"  - Created talker_length_decoder (MLP): {config.talker_hidden_size} -> {hidden_features} -> {secondary_out} (use_sinusoidal=True, predict_second_audio_token={predict_second_audio_token})", rank=rank)
        else:
            self.talker_length_decoder = nn.Linear(config.talker_hidden_size, secondary_out)
            debug_print(f"  - Created talker_length_decoder: {config.talker_hidden_size} -> {secondary_out} (predict_second_audio_token={predict_second_audio_token})", rank=rank)
        length_emb_dim = config.talker_hidden_size if getattr(config, 'talker_embed_v2', False) else config.hidden_size
        # When predicting a second audio token, the conditioning embedding
        # for the previous step's secondary token must span the full talker
        # vocab instead of the length-class table.
        secondary_emb_size = (
            (AUDIO_TOKEN_OFFSET + config.padded_audio_vocab_size)
            if predict_second_audio_token
            else config.max_length_classes
        )
        self.length_embedding = nn.Embedding(secondary_emb_size, length_emb_dim)
        debug_print(f"  - Created length_embedding: {secondary_emb_size} classes, {length_emb_dim} dims (talker_embed_v2={getattr(config, 'talker_embed_v2', False)}, predict_second_audio_token={predict_second_audio_token})", rank=rank)


class ParallelS2SForCausalLM(Qwen2ForCausalLM):
    """Interleaved SpeechLM (text + FlexiCodec RVQ tokens).

    Encoder:
        ``inputs_embeds = embed_tokens(input_ids) + length_embedding(length_ids)``
    where ``length_ids`` is (by default) the previous token's length class.

    Outputs:
        - Standard LM logits over a joint text+audio vocabulary
        - Optional length logits for predicting length classes
    """

    config_class = ParallelS2SConfig

    def audio_embed_tokens(self, audio_token_ids, dtype=torch.bfloat16):
        """
        Get audio token embeddings.
        Joint mode: uses model.embed_tokens (text+audio vocab).
        Separate mode: uses talker's embed_tokens (indices 0,1,2=special, 3+=audio codes).
        
        Args:
            audio_token_ids: Audio token IDs
                - [T] for single sequence, [B, T] for batch
                - Values in [0, audio_vocab_size-1]; negatives clamped to 0
        
        Returns:
            Audio embeddings [T, H] or [B, T, H]
        """
        if self.config.use_joint_text_audio_vocab:
            raise
        else:
            # Separate mode: use talker's embed_tokens. audio_token_ids are (token_id - text_vocab_size) in [-3,-2,-1,0..audio_vocab_size-1].
            # Map to talker vocab: talker_id = raw + 3 (0=AUD_START, 1=AUD_END, 2=AUD_TAG, 3..=audio codes)
            talker_ids = (audio_token_ids + AUDIO_TOKEN_OFFSET).to(torch.long)
            emb = self.talker_model.model.model.embed_tokens(talker_ids).to(dtype)
            if self.talker_model.talker_embed_to_hidden is not None:
                emb = self.talker_model.talker_embed_to_hidden(emb)
        return emb

    def __init__(self, config: ParallelS2SConfig):
        rank = get_rank()
        logger.info(f"Initializing ParallelS2SForCausalLM w/ {config = }")
        
        # We skip super().__init__(config) which maps to Qwen2ForCausalLM.__init__ 
        # because it creates a standard Qwen2Model and calls post_init().
        # Instead, we directly call Qwen2PreTrainedModel.__init__ to set self.config
        super(Qwen2ForCausalLM, self).__init__(config)
            
        self.config = config
        self.vocab_size = config.vocab_size
        # lm_head will be created later dynamically or as extended_text_lm_head,
        # but Qwen2ForCausalLM expects self.lm_head to exist if tie_word_embeddings is true.
        # We'll create a standard lm_head for compatibility, though it may be replaced.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Training-time option: freeze everything except talker; also disables
        # feeding speech-conditioned embeddings back into the main LLM input.

        # FlexiCodec model (optional)
        self._flexicodec_dict = None
        # Trainable deep copy of SenseVoice when finetune_speech_encoder (FlexiCodec.semantic_model stays frozen).
        self.sensevoice_finetune_copy = None
        self._qwen3_encoder = None
        self._whisper_encoder = None
        self._qwen25o_encoder = None
        self._qwen25o_encoder_dim = None

        with init_empty_weights():
            self.model = InterleavedQwen2Model(config)
        logger.info(f"  - Created InterleavedQwen2Model backbone")

        # Text and audio vocab sizes.
        self.text_vocab_size = config.text_vocab_size
        self.audio_vocab_size = config.padded_audio_vocab_size

        # Length prediction head (main LM; length_embedding is inside talker_model)
        self.max_length_classes = config.max_length_classes
        self.length_decoder = nn.Linear(config.hidden_size, config.max_length_classes)
        self.length_criterion = nn.CrossEntropyLoss(ignore_index=LabelSmoother.ignore_index)
        logger.info(f"  - Created length decoder: {config.hidden_size} -> {config.max_length_classes}")

        # When conditioning the main LM on text+audio+length+framerate, project the concatenation
        # back into the model hidden size (instead of summing the four embeddings).
        if getattr(config, 'talker_embed_v2', False):
            combined_in = config.hidden_size + config.talker_hidden_size * 3
        else:
            combined_in = config.hidden_size * 4
        self.combined_embed_proj = nn.Linear(combined_in, config.hidden_size, bias=False)
        # NOTE: identity-on-text-slice init is applied AFTER `self.post_init()` below,
        # because `post_init()` calls HF's `_init_weights` on every submodule and would
        # otherwise overwrite this layer with standard N(0, initializer_range).
        self._combined_embed_proj_text_slice = config.hidden_size
        logger.info(
            f"  - Created combined_embed_proj: {combined_in} -> {config.hidden_size} (talker_embed_v2={getattr(config, 'talker_embed_v2', False)}); will be initialized as identity on text_emb slice after post_init()",
        )
        self.extend_lm_head = bool(getattr(config, "extend_lm_head", False))
        self.extended_text_vocab_size = 1 if self.extend_lm_head else 0
        self.alignment_text_pad_token_id = self.text_vocab_size + 1 if self.extend_lm_head else 151643
        self.config.alignment_text_pad_token_id = self.alignment_text_pad_token_id
        if self.extend_lm_head:
            self.alignment_text_pad_embedding = nn.Embedding(1, config.hidden_size)
            self.extended_text_lm_head = nn.Linear(config.hidden_size, 1, bias=False)
            logger.info(
                f"  - Created extended text vocab: size=1, pad_token_id={self.alignment_text_pad_token_id}",
            )
        else:
            self.alignment_text_pad_embedding = None
            self.extended_text_lm_head = None
        
        # Talker transformer for parallel speech generation (always enabled)
        logger.info(f"Initializing talker transformer for parallel speech generation")
        logger.info(f"  - talker_hidden_size: {config.talker_hidden_size}")
        logger.info(f"  - talker_num_layers: {config.talker_num_layers}")
        logger.info(f"  - talker_num_attention_heads: {config.talker_num_attention_heads}")
        logger.info(f"  - speech_delay_tokens: {config.speech_delay_tokens}")
        
        # Create talker model (includes transformer + lm_to_talker_proj, talker_cond_proj, talker_embed_to_hidden)
        assert not config.use_joint_text_audio_vocab, "use_joint_text_audio_vocab must be False for separate mode"
        talker_vocab_size = (AUDIO_TOKEN_OFFSET + config.padded_audio_vocab_size) if not config.use_joint_text_audio_vocab else 1
        self.talker_model = TalkerModel(config)
        logger.info(f"  - Created TalkerModel with vocab {talker_vocab_size}, {config.talker_num_layers} layers")
        
        # Separate mode: talker has embed_tokens and lm_head; no separate speech_lm_head/audio_token_embedding
        self.talker_vocab_size = talker_vocab_size if not config.use_joint_text_audio_vocab else None
        # Learnable delay embeddings for speech delay
        self.speech_delay_embeddings = nn.Parameter(
            torch.randn(config.speech_delay_tokens, config.talker_hidden_size) * 0.02
        )
        logger.info(f"  - Created speech delay embeddings: {config.speech_delay_tokens} tokens x {config.talker_hidden_size} dims")
        
        # Learnable framerate embeddings (20 embeddings for 0.80, 0.81, ..., 0.99)
        # Only created when use_sinusoidal=False; otherwise use continuous sinusoidal embedding
        fr_emb_dim = config.talker_hidden_size if getattr(config, 'talker_embed_v2', False) else config.hidden_size
        if not config.use_sinusoidal:
            self.framerate_embeddings = nn.Embedding(21, fr_emb_dim)
            logger.info(f"  - Created framerate embeddings: 20 embeddings x {fr_emb_dim} dims (talker_embed_v2={getattr(config, 'talker_embed_v2', False)})")
        else:
            self.framerate_embeddings = None
            logger.info(f"  - Using sinusoidal framerate embedding (use_sinusoidal=True)")
        logger.info(f"Talker transformer initialization complete")
        # Standard (non-chained) architecture: create with 0.5B hidden size
        # When use_qwen3_feature, audio_embed_transform is created lazily with in_features=encoder_dim
        if getattr(config, "use_qwen3_feature", False):
            self.config.codec_hidden_size = 1024
        if getattr(config, "use_whisper_fetaure", False):
            self.config.codec_hidden_size = 1280
            self._whisper_encoder_dim = 1280
        if getattr(config, "use_qwen25o_feature", False):
            self.config.codec_hidden_size = _qwen25o_output_dim_from_config(
                config.qwen25o_encoder_config_path
            )
        if getattr(config, "use_qwen25o_feature", False):
            encoder_dim = self.config.codec_hidden_size
            if encoder_dim == config.hidden_size:
                self.audio_embed_transform = nn.Identity()
                logger.info(
                    f"  - Created Identity audio embed transform for Qwen2.5-Omni "
                    f"({encoder_dim} -> {config.hidden_size})"
                )
            elif config.use_mlp_for_audio_embed:
                hidden_features = int(config.hidden_size * config.audio_embed_mlp_hidden_ratio)
                self.audio_embed_transform = Mlp(
                    in_features=encoder_dim,
                    hidden_features=hidden_features,
                    out_features=config.hidden_size,
                    act_layer=nn.GELU,
                    drop=config.audio_embed_mlp_dropout,
                )
                logger.info(
                    f"  - Created MLP audio embed transform for Qwen2.5-Omni: "
                    f"{encoder_dim} -> {hidden_features} -> {config.hidden_size}"
                )
            else:
                self.audio_embed_transform = nn.Linear(
                    encoder_dim, config.hidden_size
                )
                logger.info(
                    f"  - Created Linear audio embed transform for Qwen2.5-Omni: "
                    f"{encoder_dim} -> {config.hidden_size}"
                )
        elif config.use_mlp_for_audio_embed:
            hidden_features = int(config.hidden_size * config.audio_embed_mlp_hidden_ratio)
            self.audio_embed_transform = Mlp(
                in_features=self.config.codec_hidden_size,
                hidden_features=hidden_features,
                out_features=config.hidden_size,
                act_layer=nn.GELU,
                drop=config.audio_embed_mlp_dropout,
            )
            logger.info(f"  - Created MLP audio embed transform: {config.codec_hidden_size} -> {hidden_features} -> {config.hidden_size}")
        else:
            self.audio_embed_transform = nn.Linear(config.codec_hidden_size, config.hidden_size)
            logger.info(f"  - Created Linear audio embed transform: {config.codec_hidden_size} -> {config.hidden_size}")
        # Optional local windowed merging transformer (applied after audio_embed_transform)
        if getattr(config, "use_input_merging_transformer", False):
            mt_num_layers = getattr(config, "input_merging_transformer_num_layers", 4)
            mt_num_heads = getattr(config, "input_merging_transformer_num_heads", 8)
            mt_dim_feedforward = getattr(config, "input_merging_transformer_dim_feedforward", 2048)
            mt_context = getattr(config, "input_merging_transformer_context", 32)
            mt_causal = getattr(config, "input_merging_transformer_causal", False)
            # By default the merging transformer's hidden size matches the LM
            # hidden size. Setting ``input_merging_transformer_d_model`` to a
            # smaller value (e.g. 768) shrinks the merging transformer's inner
            # dim while keeping the I/O dim equal to ``config.hidden_size``.
            mt_d_model_cfg = getattr(config, "input_merging_transformer_d_model", 0) or 0
            mt_input_dim = config.hidden_size
            mt_d_model = mt_d_model_cfg if mt_d_model_cfg > 0 else mt_input_dim
            mt_v2 = getattr(config, "use_input_merging_transformer_v2", False)
            if mt_num_layers == 0:
                self.input_merging_transformer = nn.Identity()
            elif mt_v2:
                # FlexiCodec-style: transformer attends over an interleaved
                # sequence of pre-merge frames and per-group query tokens.
                self.input_merging_transformer = InterleavedMergingTransformer(
                    d_model=mt_d_model,
                    num_heads=mt_num_heads,
                    num_layers=mt_num_layers,
                    dim_feedforward=mt_dim_feedforward,
                    context=mt_context,
                    causal=mt_causal,
                    use_mean_pooling_init=True,
                    input_dim=mt_input_dim,
                )
            else:
                _transformer_module = Stransformer_windowed if not mt_causal else Stransformer
                self.input_merging_transformer = _transformer_module.ProjectedTransformer(
                    d_model=mt_d_model,
                    num_heads=mt_num_heads,
                    num_layers=mt_num_layers,
                    causal=mt_causal,
                    layer_scale=0.01,
                    context=mt_context,
                    conv_layout=False,
                    max_period=10000,
                    gating='none',
                    norm='layer_norm',
                    positional_embedding='rope',
                    dim_feedforward=mt_dim_feedforward,
                    input_dimension=mt_input_dim,
                    output_dimensions=[mt_input_dim],
                )
            mt_params = sum(p.numel() for p in self.input_merging_transformer.parameters())
            logger.info(
                f"  - Created input merging transformer (v{'2' if mt_v2 else '1'}): "
                f"input_dim={mt_input_dim}, d_model={mt_d_model}, layers={mt_num_layers}, "
                f"heads={mt_num_heads}, ffn={mt_dim_feedforward}, context={mt_context}, "
                f"causal={mt_causal}, params={mt_params/1e6:.2f}M"
            )
        else:
            self.input_merging_transformer = nn.Identity()

        # Learnable audio boundary embeddings (replace audio_start/end token IDs with learned vectors)
        if getattr(config, "use_learnable_audio_boundary", False):
            self.audio_start_embedding = nn.Parameter(torch.randn(config.hidden_size) * 0.02)
            self.audio_end_embedding = nn.Parameter(torch.randn(config.hidden_size) * 0.02)
            logger.info(f"  - Created learnable audio boundary embeddings (dim={config.hidden_size})")

        self.post_init()

        # Re-apply identity-on-text init for `combined_embed_proj` *after* `post_init()`,
        # which otherwise overwrites it via HF's `_init_weights`. With this init, at
        # step 0 the main LM input equals `text_embeds` exactly, so the talker
        # conditioning matches the `use_combined_embedding=False` path and the
        # audio_token / length losses start from the same place as the no-combine setup.
        with torch.no_grad():
            H_text = int(self._combined_embed_proj_text_slice)
            self.combined_embed_proj.weight.zero_()
            self.combined_embed_proj.weight[:, :H_text].copy_(torch.eye(H_text))
        logger.info(
            f"  - Re-initialized combined_embed_proj as identity on text slice "
            f"(first {self._combined_embed_proj_text_slice} input dims), audio/length/framerate slices zeroed."
        )
        logger.info(f"ParallelS2SForCausalLM initialization complete")

    def _embed_text_tokens(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """Embed text ids, including the external alignment-pad token."""
        if bool(getattr(self.config, "no_pad", False)) or not self.extend_lm_head:
            return self.model.embed_tokens(token_ids)

        pad_mask = token_ids.eq(self.alignment_text_pad_token_id)
        safe_token_ids = token_ids.masked_fill(pad_mask, 0)
        embeds = self.model.embed_tokens(safe_token_ids)
        if not bool(pad_mask.any().item()):
            return embeds

        pad_embed = self.alignment_text_pad_embedding.weight[0].to(
            device=embeds.device, dtype=embeds.dtype
        )
        view_shape = [1] * pad_mask.dim() + [pad_embed.shape[0]]
        pad_embed = pad_embed.view(*view_shape)
        return torch.where(pad_mask.unsqueeze(-1), pad_embed, embeds)

    def _compute_text_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to text logits, including the external pad token."""
        base_logits = self.lm_head(hidden_states)
        if bool(getattr(self.config, "no_pad", False)) or not self.extend_lm_head:
            return base_logits

        gap_logits = torch.zeros(
            *base_logits.shape[:-1],
            1,
            device=base_logits.device,
            dtype=base_logits.dtype,
        )
        extra_pad_logits = self.extended_text_lm_head(hidden_states)
        return torch.cat([base_logits, gap_logits, extra_pad_logits], dim=-1)

    
    # ------------------------------------------------------------------
    # Lazy initialization of FlexiCodec model
    # ------------------------------------------------------------------
    @property
    def flexicodec_dict(self):
        """Lazily initialize FlexiCodec model when first accessed."""
        rank = get_rank()
        if self._flexicodec_dict is None:
            debug_print("Lazily initializing FlexiCodec model...", rank=rank)
            debug_print(f"  - Training mode: {self.training}", rank=rank)
            debug_print(f"  - Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}", rank=rank)
            
            # 1. Disable ZeRO-3 partitioning for this specific model
            # This ensures the model loads as a standard PyTorch model (replicated)
            try:
                ctx = deepspeed.zero.Init(enabled=False) if self.training else contextlib.nullcontext()
                # ctx = contextlib.nullcontext()
            except:
                ctx = contextlib.nullcontext()
            with ctx:
                debug_print("  - Loading FlexiCodec model from checkpoint...", rank=rank)
                ckpt_path = (
                    getattr(self.config, "flexicodec_ckpt_path", None)
                    or os.environ.get("FLEXICODEC_CKPT_PATH")
                )
                sensevoice_small_path = (
                    getattr(self.config, "sensevoice_small_path", None)
                    or os.environ.get("SENSEVOICE_SMALL_PATH")
                )
                config_path = (
                    getattr(self.config, "flexicodec_config_path", None)
                    or os.environ.get("FLEXICODEC_CONFIG_PATH")
                )
                missing = []
                if not ckpt_path:
                    missing.append("FLEXICODEC_CKPT_PATH")
                if not sensevoice_small_path:
                    missing.append("SENSEVOICE_SMALL_PATH")
                if not config_path:
                    missing.append("FLEXICODEC_CONFIG_PATH")
                if missing:
                    raise ValueError(
                        "Missing FlexiCodec paths. Set env vars or config fields: "
                        + ", ".join(missing)
                    )
                self._flexicodec_dict = prepare_model(
                    ckpt_path=ckpt_path,
                    sensevoice_small_path=sensevoice_small_path,
                    config_path=config_path,
                )

                debug_print("  - FlexiCodec model loaded successfully", rank=rank)
            
                # 2. Explicitly move the model to the current device
                # ZeRO-3 models often live on 'meta' device; you need this on the actual GPU.
                current_device = torch.cuda.current_device()
                debug_print(f"  - Moving FlexiCodec model to device: {current_device}", rank=rank)
                self._flexicodec_dict['model'].to(current_device)
                
                # 3. FlexiCodec + built-in SenseVoice always eval/frozen. Optional: deep copy SenseVoice for finetuning.
                codec_net = self._flexicodec_dict["model"]
                debug_print("  - Setting FlexiCodec model to eval mode and freezing parameters", rank=rank)
                codec_net.eval()
                for p in codec_net.parameters():
                    p.requires_grad = False
                finetune_sv = (
                    getattr(self.config, "finetune_speech_encoder", False)
                    and self.config.use_sensevoice_feature
                    and not getattr(self.config, "only_train_llm", False)
                )
                if finetune_sv and hasattr(codec_net, "semantic_model") and codec_net.semantic_model is not None:
                    import copy as _copy

                    ref_dtype = next(self.parameters()).dtype
                    if self.sensevoice_finetune_copy is None:
                        debug_print(
                            "  - finetune_speech_encoder: deep-copying SenseVoice to sensevoice_finetune_copy (FlexiCodec copy stays frozen)",
                            rank=rank,
                        )
                        try:
                            self.sensevoice_finetune_copy = _copy.deepcopy(codec_net.semantic_model)
                        except Exception as e:
                            raise RuntimeError(
                                "finetune_speech_encoder requires deepcopy(codec.semantic_model); failed. "
                                f"Try a different PyTorch / FunASR version. Original error: {e}"
                            ) from e
                    self.sensevoice_finetune_copy.train()
                    for p in self.sensevoice_finetune_copy.parameters():
                        p.requires_grad = True
                    self.sensevoice_finetune_copy.to(device=current_device, dtype=ref_dtype)
                else:
                    self.sensevoice_finetune_copy = None
                debug_print("  - FlexiCodec model initialization complete", rank=rank)
        else:
            debug_print("FlexiCodec model already initialized (reusing)", rank=rank, force=True)
                
        return self._flexicodec_dict

    # ------------------------------------------------------------------
    # Lazy initialization of Qwen3 audio encoder (optional replacement for SenseVoice)
    # ------------------------------------------------------------------
    @property
    def qwen3_encoder_and_projection(self):
        """Lazily load Qwen3 ASR encoder (without projection) and set audio_embed_transform to project encoder_dim -> hidden_size.

        The checkpoint is expected to be saved without proj1/proj2 (see AmphionASR README and
        local/check_encoder.py). We do not require manually editing the installed qwen_asr package:
        we load with strict=False on filtered state_dict and replace proj1/act/proj2 with Identity
        so forward returns pre-projection hidden states. The only projection is self.audio_embed_transform.
        """
        if self._qwen3_encoder is None:
            if not self.config.use_qwen3_feature:
                raise RuntimeError("qwen3_encoder_and_projection should not be accessed when use_qwen3_feature is False")
            path = self.config.qwen3_encoder_path
            config_path = self.config.qwen3_encoder_config_path
            if not path:
                raise ValueError(
                    "use_qwen3_feature=True requires qwen3_encoder_path. "
                    "Example: path/to/qwen3_encoder_dir or path/to/encoder.pth."
                )
            if os.path.isdir(path):
                import glob
                pth_files = glob.glob(os.path.join(path, "*.pth"))
                if not pth_files:
                    raise FileNotFoundError(
                        f"qwen3_encoder_path is a directory but no .pth file found: {path}"
                    )
                path = pth_files[0]
                if not config_path:
                    config_candidates = glob.glob(os.path.join(self.config.qwen3_encoder_path, "*config*.json"))
                    if not config_candidates:
                        raise FileNotFoundError(
                            f"qwen3_encoder_config_path not set and no *config*.json found in {self.config.qwen3_encoder_path}"
                        )
                    config_path = config_candidates[0]
            if not config_path:
                raise ValueError(
                    "use_qwen3_feature=True requires qwen3_encoder_config_path (path to encoder config .json)."
                )
            rank = get_rank()
            debug_print("Lazily initializing Qwen3 audio encoder...", rank=rank)
            import json as _json
            from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder

            saved_dict = torch.load(path, map_location="cpu")
            # Checkpoint is "without projection" (no proj1/proj2). Per AmphionASR README, the
            # upstream qwen_asr class still defines proj1/proj2 in __init__ and forward. We avoid
            # requiring the user to manually edit the package by: (1) loading only non-proj keys
            # with strict=False, (2) replacing proj1/act/proj2 with Identity so forward returns
            # pre-projection hidden states.
            filtered_dict = {k: v for k, v in saved_dict.items() if "proj1" not in k and "proj2" not in k}
            with open(config_path, "r") as _f:
                _cfg_dict = _json.load(_f)
            config = Qwen3ASRAudioEncoder.config_class(**_cfg_dict)
            encoder = Qwen3ASRAudioEncoder(config)
            missing, unexpected = encoder.load_state_dict(filtered_dict, strict=False)
            if missing:
                print(f"  - Qwen3 load missing keys (expected if checkpoint is without_proj): {missing}")
            if unexpected:
                print(f"  - Qwen3 load unexpected keys: {unexpected}")
            # Bypass projection so we get pre-projection hidden states (no manual edit of qwen_asr needed)
            if hasattr(encoder, "proj1"):
                encoder.proj1 = nn.Identity()
            if hasattr(encoder, "proj2"):
                encoder.proj2 = nn.Identity()
            if hasattr(encoder, "act"):
                encoder.act = nn.Identity()
            encoder = _patch_qwen3_audio_encoder_forward(encoder)
            # Pre-projection dim is d_model (transformer output); fallback to conv_out.out_features
            encoder_dim = getattr(encoder.config, "d_model", encoder.conv_out.out_features)
            self._qwen3_encoder_dim = encoder_dim
            try:
                ctx = deepspeed.zero.Init(enabled=False) if self.training else contextlib.nullcontext()
            except:
                ctx = contextlib.nullcontext()
            with ctx:
                device = torch.cuda.current_device()
                model_dtype = next(self.parameters()).dtype
                encoder = encoder.to(device=device).to(model_dtype)
                self.audio_embed_transform = self.audio_embed_transform.to(device)
            finetune_qwen = (
                getattr(self.config, "finetune_speech_encoder", False)
                and self.config.use_qwen3_feature
                and not getattr(self.config, "only_train_llm", False)
            )
            if finetune_qwen:
                encoder.train()
                for p in encoder.parameters():
                    p.requires_grad = True
            else:
                encoder.eval()
                for p in encoder.parameters():
                    p.requires_grad = False
            self._qwen3_encoder = encoder
        return self._qwen3_encoder

    def _get_qwen3_encoder_dim(self) -> int:
        """Return Qwen3 encoder output dim (only valid after encoder has been loaded)."""
        if getattr(self, "_qwen3_encoder_dim", None) is not None:
            return self._qwen3_encoder_dim
        if self._qwen3_encoder is not None:
            return getattr(self._qwen3_encoder.config, "d_model", self._qwen3_encoder.conv_out.out_features)
        raise RuntimeError("Qwen3 encoder not loaded yet")

    def _encode_user_audio_qwen3(
        self,
        user_audio_features: torch.Tensor,
        user_audio_features_lens: Optional[torch.Tensor],
        u_rows: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode user audio with Qwen3 encoder in batch. user_audio_features: [B, T, 128] (128 mel bins required).
        Returns u_codec [B, T, encoder_dim] (no projection; audio_embed_transform is applied in forward).
        """
        encoder = self.qwen3_encoder_and_projection
        feats_btf = user_audio_features[u_rows]
        feature_lens = user_audio_features_lens[u_rows] if user_audio_features_lens is not None else None
        if feature_lens is None:
            feature_lens = torch.full(
                (feats_btf.shape[0],),
                feats_btf.shape[1],
                device=feats_btf.device,
                dtype=torch.long,
            )
        B = feats_btf.shape[0]
        enc_dim = self._get_qwen3_encoder_dim()

        if B == 0:
            u_codec = torch.zeros((0, 0, enc_dim), dtype=dtype, device=device)
            u_codec_lens = torch.zeros(0, device=device, dtype=torch.long)
            return u_codec, u_codec_lens

        use_enc_grad = (
            getattr(self.config, "finetune_speech_encoder", False)
            and self.config.use_qwen3_feature
            and not getattr(self.config, "only_train_llm", False)
        )
        _grad_ctx = torch.enable_grad if use_enc_grad else torch.no_grad

        # CNN forward one-by-one, then transformer in batch
        # feats_btf: (B, T_max, F) with valid length per row in feature_lens
        with _grad_ctx():
            cnn_hidden_list = []
            cu_seqlens_list = []
            for b in range(B):
                T_b = int(feature_lens[b].item())
                # Single sample (F, T_b) for encoder
                single_feat = feats_btf[b, :T_b, :].transpose(0, 1).contiguous()
                fl_b = feature_lens[b : b + 1]
                # print(next(self._qwen3_encoder.parameters()).device)
                # print(single_feat.device)
                # print(fl_b.device)

                h_b, cu_b = encoder.forward_cnn_only(single_feat, fl_b)
                cnn_hidden_list.append(h_b)
                cu_seqlens_list.append(cu_b)

            # Merge packed hidden states and cumulative sequence lengths for batched transformer
            hidden_states = torch.cat(cnn_hidden_list, dim=0)
            merged_cu = [0]
            offset = 0
            for cu_b in cu_seqlens_list:
                merged_cu.extend((cu_b[1:] + offset).tolist())
                offset += int(cu_b[-1].item())
            merged_cu_seqlens = torch.tensor(merged_cu, device=device, dtype=torch.int32)
            out_packed = encoder.forward_transformer_only(hidden_states, merged_cu_seqlens)

            # Split by sample and pad to [B, max_T', enc_dim]
            per_sample_lengths = [int(cu[-1].item()) for cu in cu_seqlens_list]
            starts = [0]
            for L in per_sample_lengths[:-1]:
                starts.append(starts[-1] + L)
            max_T = max(per_sample_lengths)
            u_codec = torch.zeros((B, max_T, enc_dim), dtype=dtype, device=device)
            for b in range(B):
                s, L = starts[b], per_sample_lengths[b]
                u_codec[b, :L] = out_packed[s : s + L]
            u_codec_lens = torch.tensor(per_sample_lengths, device=device, dtype=torch.long)
        return u_codec, u_codec_lens

    @property
    def whisper_encoder_and_projection(self):
        """Lazily load Whisper-large-v3 encoder and reuse audio_embed_transform for projection."""
        if self._whisper_encoder is None:
            if not getattr(self.config, "use_whisper_fetaure", False):
                raise RuntimeError(
                    "whisper_encoder_and_projection should not be accessed when use_whisper_fetaure is False"
                )
            rank = get_rank()

            candidate_paths = []
            if getattr(self.config, "whisper_encoder_path", None):
                candidate_paths.append(self.config.whisper_encoder_path)
            if os.environ.get("WHISPER_ENCODER_PATH"):
                candidate_paths.append(os.environ["WHISPER_ENCODER_PATH"])
            candidate_paths.extend(
                [
                    "openai/whisper-large-v3",
                ]
            )
            resolved_path = None
            for candidate in candidate_paths:
                if candidate == "openai/whisper-large-v3" or os.path.exists(candidate):
                    resolved_path = candidate
                    break
            if resolved_path is None:
                raise FileNotFoundError(
                    "Unable to resolve a Whisper-large-v3 checkpoint path. "
                    "Set config.whisper_encoder_path or place the checkpoint in a known local path."
                )

            debug_print(f"Lazily initializing Whisper encoder from {resolved_path}...", rank=rank)
            whisper_model = WhisperModel.from_pretrained(resolved_path)
            encoder = whisper_model.encoder
            del whisper_model
            self._whisper_encoder_dim = getattr(encoder.config, "d_model", 1280)
            try:
                ctx = deepspeed.zero.Init(enabled=False) if self.training else contextlib.nullcontext()
            except Exception:
                ctx = contextlib.nullcontext()
            with ctx:
                device = torch.cuda.current_device()
                encoder = encoder.to(device=device)
                self.audio_embed_transform = self.audio_embed_transform.to(device)
            finetune_whisper = (
                getattr(self.config, "finetune_speech_encoder", False)
                and getattr(self.config, "use_whisper_fetaure", False)
                and not getattr(self.config, "only_train_llm", False)
            )
            if finetune_whisper:
                encoder.train()
                for p in encoder.parameters():
                    p.requires_grad = True
            else:
                encoder.eval()
                for p in encoder.parameters():
                    p.requires_grad = False
            self._whisper_encoder = encoder
        return self._whisper_encoder

    def _get_whisper_encoder_dim(self) -> int:
        """Return Whisper encoder output dim (only valid after encoder has been loaded)."""
        if getattr(self, "_whisper_encoder_dim", None) is not None:
            return self._whisper_encoder_dim
        if self._whisper_encoder is not None:
            return getattr(self._whisper_encoder.config, "d_model", 1280)
        raise RuntimeError("Whisper encoder not loaded yet")

    def _encode_user_audio_whisper(
        self,
        user_audio_features: torch.Tensor,
        user_audio_features_lens: Optional[torch.Tensor],
        u_rows: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode user audio with Whisper-large-v3 in batch. user_audio_features: [B, T, 128]."""
        encoder = self.whisper_encoder_and_projection
        feats_btf = user_audio_features[u_rows]
        feature_lens = user_audio_features_lens[u_rows] if user_audio_features_lens is not None else None
        if feature_lens is None:
            feature_lens = torch.full(
                (feats_btf.shape[0],),
                feats_btf.shape[1],
                device=feats_btf.device,
                dtype=torch.long,
            )
        if feats_btf.shape[0] == 0:
            u_codec = torch.zeros((0, 0, self._get_whisper_encoder_dim()), dtype=dtype, device=device)
            u_codec_lens = torch.zeros(0, device=device, dtype=torch.long)
            return u_codec, u_codec_lens

        use_enc_grad = (
            getattr(self.config, "finetune_speech_encoder", False)
            and getattr(self.config, "use_whisper_fetaure", False)
            and not getattr(self.config, "only_train_llm", False)
        )
        _grad_ctx = torch.enable_grad if use_enc_grad else torch.no_grad
        with _grad_ctx():
            outputs = encoder(
                input_features=feats_btf.transpose(1, 2).contiguous().to(device=device, dtype=torch.float32),
                return_dict=True,
            )
            u_codec = outputs.last_hidden_state.to(dtype=dtype)
            u_codec_lens = ((feature_lens + 1) // 2).clamp(max=u_codec.shape[1])
        return u_codec, u_codec_lens

    # ------------------------------------------------------------------
    # Lazy initialization of Qwen2.5 Omni audio encoder
    # ------------------------------------------------------------------
    @property
    def qwen25o_encoder_and_projection(self):
        """Lazily load Qwen2.5-Omni audio encoder (KEEP native proj) and use
        encoder output directly.

        Notes:
        - Unlike Qwen3 path, we DO keep encoder.proj.
        - Therefore encoder output dim is config.output_dim, not d_model.
        - qwen25o_encoder_path is expected to be a sharded checkpoint directory.
        - qwen25o_encoder_config_path should point to either:
            (1) audio encoder config json, or
            (2) full omni config json containing "audio_config".
        """
        if self._qwen25o_encoder is None:
            if not self.config.use_qwen25o_feature:
                raise RuntimeError(
                    "qwen25o_encoder_and_projection should not be accessed when "
                    "use_qwen25o_feature is False"
                )

            path = self.config.qwen25o_encoder_path
            config_path = self.config.qwen25o_encoder_config_path

            if not path:
                raise ValueError(
                    "use_qwen25o_feature=True requires qwen25o_encoder_path "
                    "(path to the checkpoint directory)."
                )
            if not config_path:
                raise ValueError(
                    "use_qwen25o_feature=True requires qwen25o_encoder_config_path "
                    "(path to encoder config .json or full omni config json)."
                )

            import json as _json
            from safetensors.torch import load_file
            from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
                Qwen2_5OmniAudioEncoder,
            )
            from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
                Qwen2_5OmniAudioEncoderConfig,
            )

            if not os.path.isdir(path):
                raise FileNotFoundError(f"qwen25o_encoder_path not found: {path}")
            if not os.path.isfile(config_path):
                raise FileNotFoundError(f"qwen25o_encoder_config_path not found: {config_path}")

            rank = get_rank()
            debug_print("Lazily initializing Qwen2.5-Omni audio encoder...", rank=rank)

            with open(config_path, "r") as _f:
                _cfg_dict = _json.load(_f)

            # Support either:
            #   1) raw audio encoder config
            #   2) full omni config containing "audio_config"
            if "thinker_config" in _cfg_dict:
                _cfg_dict = _cfg_dict["thinker_config"] # get the thinker config if passed in offical ckpts config
            if "audio_config" in _cfg_dict: # get the audio config if passed in thinker config
                _cfg_dict = _cfg_dict["audio_config"]

            config = Qwen2_5OmniAudioEncoderConfig(**_cfg_dict)
            encoder = Qwen2_5OmniAudioEncoder(config)

            index_file = os.path.join(path, "model.safetensors.index.json")
            with open(index_file, "r") as f:
                index_data = _json.load(f)

            weight_map = index_data.get("weight_map", {})
            if not weight_map:
                raise ValueError(f"Checkpoint index has no weight_map entries: {index_file}")

            expected_keys = set(encoder.state_dict())
            checkpoint_keys = set(weight_map)
            longest_key_length = max(map(len, expected_keys))
            anchor_keys = {
                key for key in expected_keys if len(key) == longest_key_length
            }
            candidate_prefixes = {
                checkpoint_key[:-len(anchor_key)]
                for checkpoint_key in checkpoint_keys
                for anchor_key in anchor_keys
                if checkpoint_key.endswith(anchor_key)
            }
            matching_prefixes = []
            for prefix in candidate_prefixes:
                remapped_keys = {
                    key[len(prefix):]
                    for key in checkpoint_keys
                    if key.startswith(prefix)
                }
                if remapped_keys == expected_keys:
                    matching_prefixes.append(prefix)

            if len(matching_prefixes) != 1:
                raise ValueError(
                    "Could not uniquely infer the Qwen2.5-Omni audio encoder "
                    "weight prefix from the checkpoint index: "
                    f"found {len(matching_prefixes)} complete matches "
                    f"({sorted(matching_prefixes)!r})."
                )
            weight_prefix = matching_prefixes[0]

            # Load on demand: only open shards containing audio encoder weights.
            required_shards = {
                shard for key, shard in weight_map.items()
                if key.startswith(weight_prefix)
            }
            remapped_dict = {}
            for shard in sorted(required_shards):
                shard_path = os.path.join(path, shard)
                part = load_file(shard_path, device="cpu")
                for key, value in part.items():
                    if key.startswith(weight_prefix):
                        remapped_key = key[len(weight_prefix):]
                        remapped_dict[remapped_key] = value

            # IMPORTANT: keep proj weights, so do NOT filter out "proj.*"
            missing, unexpected = encoder.load_state_dict(remapped_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    "Qwen2.5-Omni audio encoder weights did not load completely: "
                    f"missing keys={missing}, unexpected keys={unexpected}."
                )

            # Since we KEEP proj, output dim is output_dim
            encoder_dim = getattr(encoder.config, "output_dim", None)
            if encoder_dim is None:
                raise RuntimeError("Failed to infer Qwen2.5-Omni encoder dim from config.output_dim")
            self._qwen25o_encoder_dim = encoder_dim
            if encoder_dim != self.config.codec_hidden_size:
                raise RuntimeError(
                    "Qwen2.5-Omni encoder output_dim changed after model "
                    f"initialization: expected {self.config.codec_hidden_size}, "
                    f"loaded {encoder_dim}"
                )

            try:
                ctx = deepspeed.zero.Init(enabled=False) if self.training else contextlib.nullcontext()
            except Exception:
                ctx = contextlib.nullcontext()

            with ctx:
                device = torch.cuda.current_device()
                model_dtype = next(self.parameters()).dtype
                
                # Check if modules are on meta device before calling .to()
                def safe_to_device(module, device, dtype):
                    if any(p.device.type == 'meta' for p in module.parameters()):
                        module.to_empty(device=device)
                    else:
                        module = module.to(device=device)
                    return module.to(dtype)

                encoder = safe_to_device(encoder, device, model_dtype)
                self.audio_embed_transform = safe_to_device(self.audio_embed_transform, device, model_dtype)

            finetune_qwen = (
                getattr(self.config, "finetune_speech_encoder", False)
                and self.config.use_qwen25o_feature
                and not getattr(self.config, "only_train_llm", False)
            )
            if finetune_qwen:
                encoder.train()
                for p in encoder.parameters():
                    p.requires_grad = True
            else:
                encoder.eval()
                for p in encoder.parameters():
                    p.requires_grad = False

            self._qwen25o_encoder = encoder

        return self._qwen25o_encoder

    def _get_qwen25o_encoder_dim(self) -> int:
        """Return Qwen2.5-Omni encoder output dim (only valid after encoder has been loaded)."""
        if getattr(self, "_qwen25o_encoder_dim", None) is not None:
            return self._qwen25o_encoder_dim
        if self._qwen25o_encoder is not None:
            return getattr(self._qwen25o_encoder.config, "output_dim", None)
        raise RuntimeError("Qwen2.5-Omni encoder not loaded yet")

    def _encode_user_audio_qwen25o(
        self,
        user_audio_features: torch.Tensor,
        user_audio_features_lens: Optional[torch.Tensor],
        u_rows: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode user audio with Qwen2.5-Omni encoder in batch.

        user_audio_features: [B_all, T, 128]
        returns:
            u_codec: [B, T', encoder_dim], where encoder_dim = output_dim
            u_codec_lens: [B]
        """
        encoder = self.qwen25o_encoder_and_projection
        feats_btf = user_audio_features[u_rows]
        feature_lens = user_audio_features_lens[u_rows] if user_audio_features_lens is not None else None

        if feature_lens is None:
            feature_lens = torch.full(
                (feats_btf.shape[0],),
                feats_btf.shape[1],
                device=feats_btf.device,
                dtype=torch.long,
            )

        B = feats_btf.shape[0]
        enc_dim = self._get_qwen25o_encoder_dim()

        if B == 0:
            u_codec = torch.zeros((0, 0, enc_dim), dtype=dtype, device=device)
            u_codec_lens = torch.zeros(0, device=device, dtype=torch.long)
            return u_codec, u_codec_lens

        use_enc_grad = (
            getattr(self.config, "finetune_speech_encoder", False)
            and self.config.use_qwen25o_feature
            and not getattr(self.config, "only_train_llm", False)
        )
        _grad_ctx = torch.enable_grad if use_enc_grad else torch.no_grad

        # Official encoder expects flattened features [F, sum_T]
        max_len = feats_btf.size(1)
        mask = torch.arange(max_len, device=feats_btf.device).unsqueeze(0) < feature_lens.unsqueeze(1)
        flat_features = feats_btf[mask] # [sum_T, F]
        input_features = flat_features.transpose(0, 1).contiguous() # [F, sum_T]

        def _get_qwen25o_output_lengths(input_lengths: torch.Tensor):
            """
            Match HF Qwen2.5-Omni audio path:
            conv2: kernel=3, stride=2, padding=1
            avg_pooler: kernel=2, stride=2
            conv1 keeps length unchanged.
            """
            # after conv2
            aftercnn_lens = torch.div(input_lengths - 1, 2, rounding_mode="floor") + 1
            # after avg_pooler
            output_lens = torch.div(aftercnn_lens - 2, 2, rounding_mode="floor") + 1
            output_lens = torch.clamp(output_lens, min=0)
            return aftercnn_lens, output_lens

        with _grad_ctx():
            aftercnn_lens, output_lens = _get_qwen25o_output_lengths(feature_lens)

            outputs = encoder(
                input_features=input_features, # expected: torch.Size([128, 674])
                feature_lens=feature_lens, # expected: tensor([674], device='cuda:0')
                aftercnn_lens=aftercnn_lens, # expected: tensor([337], device='cuda:0')
            )

            # Official forward returns BaseModelOutput(last_hidden_state=token_audio)
            # token_audio is already after proj, shape [sum(T'), output_dim]
            out_packed = outputs.last_hidden_state # expected: torch.Size([168, 3584])

            per_sample_lengths = output_lens.tolist()
            starts = [0]
            for L in per_sample_lengths[:-1]:
                starts.append(starts[-1] + L)

            max_T = max(per_sample_lengths) if per_sample_lengths else 0
            u_codec = torch.zeros((B, max_T, enc_dim), dtype=dtype, device=device)

            for b in range(B):
                s, L = starts[b], per_sample_lengths[b]
                if L > 0:
                    u_codec[b, :L] = out_packed[s : s + L].to(device=device, dtype=dtype)

            u_codec_lens = output_lens.to(device=device, dtype=torch.long)
        
        return u_codec, u_codec_lens

    def _finetune_sensevoice_semantic(self) -> bool:
        """Gradients through FlexiCodec SenseVoice (semantic_model) when finetuning the speech encoder."""
        return (
            getattr(self.config, "finetune_speech_encoder", False)
            and self.config.use_sensevoice_feature
            and not getattr(self.config, "only_train_llm", False)
        )

    # ------------------------------------------------------------------
    # Helper: SenseVoice feature merging for flexible frame rate
    # ------------------------------------------------------------------
    def _perform_similarity_alignment_vectorized(
        self,
        h_frames_v: torch.Tensor,
        x_lens=None,
        merging_threshold=None,
        target_rate: Optional[float] = None,
        base_rate: float = 12.5,
        dynamic_merging: bool = True,
    ):
        """
        Perform alignment for an entire batch in a fully vectorized manner.
        
        Modes:
        1. Target-rate + dynamic_merging=True: Find similarity threshold that yields
           n = (target_rate/base_rate)*T segments; merge by similarity (respects acoustic boundaries).
        2. Target-rate + dynamic_merging=False: Uniform merging to achieve n tokens (ignores similarity).
        3. Similarity-threshold mode (target_rate=None): Use merging_threshold directly.
        
        Args:
            h_frames_v: (B, T, D) hidden features
            x_lens: (B,) valid lengths per item
            merging_threshold: used when target_rate is None
            target_rate: target frame rate in Hz (e.g. 6.0)
            base_rate: base frame rate in Hz (default 12.5)
            dynamic_merging: if True and target_rate set, search for threshold to hit target; else uniform merge
                
        Returns:
            alignment_matrix, sim, num_segments_per_item, token_lengths
        """
        if merging_threshold is not None and merging_threshold > 1.0:
            target_rate = merging_threshold
            merging_threshold = None
            dynamic_merging = True

        B, T, D = h_frames_v.shape
        if T <= 1:
            sim = torch.zeros(B, max(0, T - 1), device=h_frames_v.device, dtype=h_frames_v.dtype)
            token_lengths = torch.ones(B, 1, device=h_frames_v.device, dtype=torch.long)
            return (
                torch.ones(B, 1, T, device=h_frames_v.device, dtype=h_frames_v.dtype),
                sim,
                torch.ones(B, device=h_frames_v.device, dtype=torch.long),
                token_lengths,
            )

        device = h_frames_v.device
        dtype = h_frames_v.dtype

        if x_lens is not None:
            valid_lens = x_lens.long()
            valid_frame_mask = torch.arange(T, device=device).unsqueeze(0) < x_lens.unsqueeze(1)
        else:
            valid_lens = torch.full((B,), T, device=device, dtype=torch.long)
            valid_frame_mask = torch.ones(B, T, device=device, dtype=torch.bool)

        ratio = target_rate / base_rate if target_rate is not None else None
        target_n_per_item = (
            (ratio * valid_lens.float()).round().long().clamp(min=1)
            if target_rate is not None
            else None
        )

        # --- Target-rate mode ---
        if target_rate is not None:
            if dynamic_merging:
                # Compute similarity between adjacent frames
                sim = F.cosine_similarity(h_frames_v[:, :-1, :], h_frames_v[:, 1:, :], dim=2)
                if x_lens is not None:
                    valid_transition_mask = valid_frame_mask[:, :-1] & valid_frame_mask[:, 1:]
                    sim = torch.where(valid_transition_mask, sim, torch.zeros_like(sim))

                # Directly select boundaries via top-k lowest similarity to hit target_n segments.
                # target_n segments require (target_n - 1) boundaries from (L - 1) transitions.
                # Pick the (target_n - 1) lowest-similarity transitions as boundaries.
                is_new_group_boundary = torch.zeros(B, T - 1, device=device, dtype=torch.bool)
                for b in range(B):
                    L_b = int(valid_lens[b].item())
                    target_n_b = int(target_n_per_item[b].item())
                    if L_b <= 1:
                        continue
                    n_transitions = L_b - 1
                    n_boundaries = min(target_n_b - 1, n_transitions)
                    if n_boundaries >= n_transitions:
                        is_new_group_boundary[b, :n_transitions] = True
                    elif n_boundaries <= 0:
                        pass  # everything merges into one segment
                    else:
                        sim_b = sim[b, :n_transitions]
                        boundary_similarities, boundary_indices = torch.topk(sim_b, n_boundaries, largest=False)
                        boundary_similarities_max = boundary_similarities.max()
                        is_new_group_boundary[b, boundary_indices] = True
            else:
                # Uniform merging: ignore similarity
                sim = torch.zeros(B, T - 1, device=device, dtype=dtype)
                max_segments = int(target_n_per_item.max().item())
                frame_to_segment_map = torch.zeros(B, T, device=device, dtype=torch.long)
                for b in range(B):
                    L_b = int(valid_lens[b].item())
                    n_b = int(target_n_per_item[b].item())
                    if L_b <= 0:
                        continue
                    t_vals = torch.arange(L_b, device=device, dtype=torch.float32)
                    group_ids = (t_vals * n_b / L_b).long().clamp(max=n_b - 1)
                    frame_to_segment_map[b, :L_b] = group_ids
                num_segments_per_item = target_n_per_item
                alignment_matrix = torch.zeros(B, max_segments, T, device=device, dtype=torch.bool)
                batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, T)
                frame_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
                alignment_matrix[batch_indices, frame_to_segment_map, frame_indices] = True
                token_lengths = alignment_matrix.sum(dim=2).long()
                return alignment_matrix.float(), sim, num_segments_per_item, token_lengths
        else:
            # --- Similarity-threshold mode (target_rate=None): use merging_threshold directly ---
            current_threshold = merging_threshold
            sim = F.cosine_similarity(h_frames_v[:, :-1, :], h_frames_v[:, 1:, :], dim=2)
            if x_lens is not None:
                valid_transition_mask = valid_frame_mask[:, :-1] & valid_frame_mask[:, 1:]
                sim = torch.where(valid_transition_mask, sim, torch.zeros_like(sim))
            is_new_group_boundary = sim <= current_threshold
        is_new_group_padded = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=device), is_new_group_boundary], dim=1
        )

        if self.config.max_tokens_per_group is not None:
            arange_t = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0)
            segment_start_markers = arange_t * is_new_group_padded.long()
            last_segment_start_indices = torch.cummax(segment_start_markers, dim=1).values
            frame_indices_in_segment = arange_t - last_segment_start_indices
            is_split_boundary = (frame_indices_in_segment % self.config.max_tokens_per_group) == 0
            frame_to_segment_map = torch.cumsum(is_split_boundary.long(), dim=1) - 1
        else:
            frame_to_segment_map = torch.cumsum(is_new_group_padded.long(), dim=1) - 1

        if x_lens is not None:
            num_segments_per_item = torch.zeros(B, device=device, dtype=torch.long)
            for b in range(B):
                valid_length = x_lens[b]
                num_segments_per_item[b] = frame_to_segment_map[b, valid_length - 1] + 1
        else:
            num_segments_per_item = frame_to_segment_map.max(dim=1).values + 1

        max_segments = int((frame_to_segment_map.max(dim=1).values + 1).max().item())

        # Build alignment matrices
        alignment_matrix = torch.zeros(B, max_segments, T, device=device, dtype=torch.bool)
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, T)
        frame_indices = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        alignment_matrix[batch_indices, frame_to_segment_map, frame_indices] = True
        token_lengths = alignment_matrix.sum(dim=2).long()
        return alignment_matrix.float(), sim, num_segments_per_item, token_lengths
    
    def aggregate_features(
        self,
        features: torch.Tensor,
        alignment_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate features using alignment matrix.
        
        Args:
            features: torch.Tensor, shape (batch_size, feat_len, feature_dim) or (batch_size, feature_dim, feat_len)
            alignment_matrix: torch.Tensor, shape (batch_size, num_groups, feat_len)
                
        Returns:
            torch.Tensor, shape (batch_size, num_groups, feature_dim)
        """
        # Handle both (B, T, D) and (B, D, T) input formats.
        # alignment_matrix is (B, num_groups, feat_len); feat_len must match features' time dim.
        # Output is always (B, num_groups, feature_dim).
        alignment_T = alignment_matrix.shape[-1]
        if features.shape[1] == alignment_T:
            # Features are (B, T, D)
            pass
        elif features.shape[2] == alignment_T:
            # Features are (B, D, T)
            features = features.transpose(1, 2)  # (B, T, D)
        else:
            raise ValueError(
                f"aggregate_features: features shape {features.shape} incompatible with "
                f"alignment shape {alignment_matrix.shape} (alignment last dim={alignment_T})"
            )

        # Ensure alignment matrix is float and on the correct device
        alignment_float = alignment_matrix.to(features.device, dtype=features.dtype)

        # Calculate the sum of features for each group via vectorized operation
        summed_features = torch.einsum('bgt,btd->bgd', alignment_float, features)

        # Calculate the number of frames assigned to each group
        group_frame_counts = alignment_float.sum(dim=2)  # (batch_size, num_groups)

        # To avoid division by zero, clamp counts to a minimum of 1
        group_frame_counts = group_frame_counts.clamp(min=1)

        # Reshape counts for broadcasting
        counts_reshaped = group_frame_counts.unsqueeze(-1)  # (batch_size, num_groups, 1)

        # Compute the average; output is always (B, num_groups, feature_dim)
        aggregated_features = summed_features / counts_reshaped

        return aggregated_features
    
    # ------------------------------------------------------------------
    # Helper: learnable frame-rate embedding
    # ------------------------------------------------------------------
    def _framerate_sinusoidal_embedding(
        self,
        framerate: float,
        device: torch.device,
        dtype: torch.dtype,
        framerate_min: Optional[float] = None,
        framerate_max: Optional[float] = None,
    ) -> torch.Tensor:
        """Map framerate to embedding.

        When use_sinusoidal=True: continuous scalar -> sinusoidal embedding (Transformer-style).
        When use_sinusoidal=False: maps to 20 learnable embeddings (0.80, 0.81, ..., 1.00).
        Optional framerate_min/framerate_max override config (e.g. 0, 15 for per-sample Hz range).
        """
        if self.config.use_sinusoidal:
            # Continuous scalar -> sinusoidal embedding in [framerate_min, framerate_max]
            cfg = self.config
            f_min = framerate_min if framerate_min is not None else cfg.framerate_min
            f_max = framerate_max if framerate_max is not None else cfg.framerate_max
            f = max(min(float(framerate), f_max), f_min)
            x = (f - f_min) / max(f_max - f_min, 1e-6)
            x_t = torch.tensor(x, device=device, dtype=dtype)

            dim = self.config.talker_hidden_size if getattr(self.config, 'talker_embed_v2', False) else self.config.hidden_size
            half_dim = dim // 2
            inv_freq = torch.exp(
                torch.arange(0, half_dim, device=device, dtype=dtype)
                * (-math.log(10000.0) / max(half_dim, 1))
            )
            angles = x_t * inv_freq
            sin = torch.sin(angles)
            cos = torch.cos(angles)

            emb = torch.zeros(dim, device=device, dtype=dtype)
            emb[0:half_dim] = sin
            emb[half_dim:2 * half_dim] = cos
            return emb.unsqueeze(0)  # [1, H]
        else:
            # Learnable: clamp to [0.80, 0.99] and round to nearest 0.01
            f = max(min(float(framerate), 0.99), 0.80)
            f_rounded = round(f * 100) / 100.0
            index = int((f_rounded - 0.80) * 100)
            index = max(0, min(20, index))
            emb = self.framerate_embeddings(torch.tensor(index, device=device))
            return emb.unsqueeze(0)  # [1, H]

    def _framerate_sinusoidal_embedding_batch(
        self,
        framerates: torch.Tensor,  # [B]
        device: torch.device,
        dtype: torch.dtype,
        framerate_min: float = 0.0,
        framerate_max: float = 15.0,
    ) -> torch.Tensor:
        """Map per-sample framerates [B] to embeddings [B, H] using sinusoidal encoding.
        Used when per_sample_frame_rate_embed=True with range [0, 15] Hz.
        """
        f = framerates.clamp(min=framerate_min, max=framerate_max)
        x = (f - framerate_min) / max(framerate_max - framerate_min, 1e-6)
        dim = self.config.talker_hidden_size if getattr(self.config, 'talker_embed_v2', False) else self.config.hidden_size
        half_dim = dim // 2
        inv_freq = torch.exp(
            torch.arange(0, half_dim, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(half_dim, 1))
        )
        angles = x.unsqueeze(-1) * inv_freq.unsqueeze(0)  # [B, half_dim]
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        emb = torch.zeros(framerates.shape[0], dim, device=device, dtype=dtype)
        emb[:, 0:half_dim] = sin
        emb[:, half_dim:2 * half_dim] = cos
        return emb  # [B, H]
    
    # ------------------------------------------------------------------
    # Helper: build parallel text and speech sequences
    # ------------------------------------------------------------------
    def _build_parallel_text_speech_sequences(
        self,
        text_token_ids: torch.LongTensor,
        audio_token_ids: torch.LongTensor,
        audio_token_lengths: torch.LongTensor,
        framerate: float,
        framerate_min: Optional[float] = None,
        framerate_max: Optional[float] = None,
    ) -> Tuple:
        """Build parallel text and speech sequences for training.
        
        This replaces the interleaving approach with parallel processing:
        - Text tokens are processed through the main LM
        - Speech tokens (delayed) are processed through talker
        - Framerate embedding is added to each speech token
        - If use_combined_embedding=True: Main LM receives text+audio+length embeddings
        - If use_combined_embedding=False: Main LM receives only text embeddings
        
        Args:
            text_token_ids: [T] text token IDs
            audio_token_ids: [A] audio token IDs (already shifted by text_vocab_size)
            audio_token_lengths: [A] length attribute for each audio token
            framerate: Frame rate value in [framerate_min, framerate_max] (required)
        
        Returns:
            text_embeds: [T, H] embeddings for main LM (text-only or combined)
            text_labels: [T] token IDs for text LM loss
            speech_embeds: [D+A, H_talker] speech embeddings with delay (D=speech_delay_tokens)
            speech_labels: [D+A] token IDs for speech loss (first D are ignore_index)
            speech_length_labels: [D+A] length labels for speech (shifted by 1)
            alignment_pad_len: number of tail text tokens added only to match speech length (for weighted CE).
        """
        is_audio_empty = audio_token_ids.numel() == 0
        device = text_token_ids.device
        dtype = self.model.embed_tokens.weight.dtype
        no_pad = bool(getattr(self.config, "no_pad", False))
        predict_second_audio = bool(getattr(self.config, "predict_second_audio_token", False))
        # In second-audio-token mode, ``audio_token_lengths`` carries the second
        # audio token id at each step (audio-vocab space, may be negative for
        # special tokens). Pad the delay/eos positions with the same special
        # ids used on the semantic side so the secondary head learns to emit
        # AUD_START/AUD_END at those positions.
        if predict_second_audio:
            secondary_pad_value = self.text_vocab_size - 2  # AUD_START in text-shifted audio space
            secondary_eos_value = self.text_vocab_size - 3  # AUD_END
            secondary_pad_audio = secondary_pad_value - self.text_vocab_size  # -2
            secondary_eos_audio = secondary_eos_value - self.text_vocab_size  # -3
        else:
            secondary_pad_audio = 0
            secondary_eos_audio = 0
        # In second-audio-token mode (group size 2), the dataloader supplies a
        # 12.5Hz audio token sequence with all-ones lengths. We split it into
        # two 6.25Hz streams: the primary stream (even indices) is predicted
        # by the main talker head and the secondary stream (odd indices) is
        # predicted by the (formerly length) head. Both have length A//2.
        if predict_second_audio and not is_audio_empty:
            orig_len = audio_token_ids.size(0)
            if orig_len % 2 != 0:
                audio_token_ids = audio_token_ids[:orig_len - 1]
                audio_token_lengths = audio_token_lengths[:orig_len - 1]
            # Build secondary stream from odd-indexed audio tokens.
            # ``audio_token_ids`` is in text-shifted space; convert back to
            # audio-vocab space (will be remapped to talker-vocab below).
            secondary_audio = audio_token_ids[1::2] - self.text_vocab_size
            audio_token_ids = audio_token_ids[0::2]
            audio_token_lengths = secondary_audio
        text_len = len(text_token_ids)
        # append eos speech
        audio_token_ids = torch.cat([audio_token_ids, torch.full((1,), self.text_vocab_size-3, device=device)])
        # prepend delay tokens to speech
        audio_token_ids = torch.cat([torch.full((self.config.speech_delay_tokens,), self.text_vocab_size-2, device=device), audio_token_ids])
        audio_len = len(audio_token_ids)
        # prepend delay and shift to speech length
        audio_token_lengths = torch.cat(
            [
                torch.full((self.config.speech_delay_tokens + 1,), secondary_pad_audio, device=device, dtype=audio_token_lengths.dtype),
                audio_token_lengths,
            ]
        )

        # Align time axis between text stream and speech stream.
        # Default behaviour: pad text token IDs up to speech length (for alignment CE).
        # no_pad=True: do NOT pad text token IDs; we will instead pad thinker *hidden states* (not inputs)
        # with zeros to match the talker/audio length right before talker conditioning.
        pad_len = audio_len - text_len
        if pad_len >= 0:
            if not no_pad:
                text_token_ids = torch.cat(
                    [
                        text_token_ids,
                        torch.full(
                            (pad_len,),
                            self.alignment_text_pad_token_id,
                            device=device,
                            dtype=text_token_ids.dtype,
                        ),
                    ]
                )
        else:
            # is_audio_empty = True # do not use the audio tokens
            # pad speech to text len
            audio_token_ids = torch.cat([audio_token_ids, torch.full((-pad_len,), self.text_vocab_size-3, device=device)])
            audio_token_lengths = torch.cat(
                [
                    audio_token_lengths,
                    torch.full((-pad_len,), secondary_eos_audio, device=device, dtype=audio_token_lengths.dtype),
                ]
            )
        alignment_pad_len = 0 if (no_pad and pad_len > 0) else max(0, int(pad_len))
        pad_w = _resolve_text_alignment_pad_loss_weight(self.config)
        # Text embeddings
        text_embeds = self._embed_text_tokens(text_token_ids)  # [T, H] (or [A, H] when padded)
        # Alignment padding: ignore in CE when weight is 0; otherwise keep real pad ids for weighted CE.
        # In no_pad mode, we do not create any alignment-pad targets here (thinker length stays text_len).
        if alignment_pad_len > 0 and pad_w <= 0.0:
            text_labels = text_token_ids.clone()
            text_labels[-alignment_pad_len:] = IGNORE_TOKEN_ID
        else:
            text_labels = text_token_ids

        # Speech sequence - with delay
        # Speech labels: audio-vocab ids (no text_vocab_size offset)
        speech_labels = audio_token_ids - self.text_vocab_size  # [A]
        
        # Embed audio tokens using audio embedding
        audio_embeds = self.audio_embed_tokens(speech_labels, dtype=dtype)  # [A, H]
            
        speech_labels = speech_labels + self.text_vocab_size  # [A]
        if predict_second_audio:
            # Second-audio-token ablation: ``audio_token_lengths`` carries
            # audio-vocab ids (negative values denote AUD_START/AUD_END).
            # Map to talker-vocab ids (+AUDIO_TOKEN_OFFSET) for both the
            # conditioning embedding and the CE target.
            secondary_talker_ids = (audio_token_lengths + AUDIO_TOKEN_OFFSET).to(torch.long)
            length_embeds = self.talker_model.length_embedding(secondary_talker_ids.clamp(min=0))
            audio_token_lengths = secondary_talker_ids
        else:
            length_embeds = self.talker_model.length_embedding(audio_token_lengths)
        if is_audio_empty:
            speech_labels = torch.full_like(audio_token_ids, IGNORE_TOKEN_ID)
            audio_token_lengths = torch.full_like(audio_token_lengths, IGNORE_TOKEN_ID)
            # DeepSpeed ZeRO-2 fix: multiply by 0.0 instead of using zeros_like to preserve the autograd graph
            audio_embeds = audio_embeds * 0.0
            length_embeds = length_embeds * 0.0

        # Optionally combine text, audio, length, and framerate embeddings for main LM
        use_text_only = (
            getattr(self.config, "only_train_talker", False)
            or getattr(self.config, "freeze_llm", False)
            or not getattr(self.config, "use_combined_embedding", True)
        ) and not getattr(self.config, "force_use_combined_embedding", False)
        fr_emb = self._framerate_sinusoidal_embedding(
            framerate, device, text_embeds.dtype,
            framerate_min=framerate_min, framerate_max=framerate_max,
        )
        main_len = int(text_embeds.shape[0])
        fr_emb_broadcast = fr_emb.expand(main_len, -1)  # [T_main, H]
        audio_embeds_main = audio_embeds[:main_len]
        length_embeds_main = length_embeds[:main_len]
        combined_cat = torch.cat([text_embeds, audio_embeds_main, length_embeds_main, fr_emb_broadcast], dim=-1)
        combined_embeds = self.combined_embed_proj(combined_cat)
        if use_text_only or is_audio_empty:
            # When only training talker or freeze_llm or use_lora (without force), do not feed audio/length embeddings into the main LLM input.
            # DeepSpeed ZeRO-2 fix: keep combined_embed_proj in the backward graph
            dummy_combined_embeds = combined_embeds.mean() * 0.0 if combined_embeds.numel() > 0 else combined_embeds.sum() * 0.0
            combined_embeds = text_embeds + dummy_combined_embeds
        assert text_labels.shape[-1] <= speech_labels.shape[-1]
        return (
            combined_embeds,
            text_labels,
            None,
            speech_labels,
            audio_token_lengths,
            audio_embeds,
            length_embeds,
            alignment_pad_len,
        )

    def _weighted_text_causal_lm_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        alignment_pad_mask: torch.Tensor,
        pad_weight: float,
        ignore_index: int,
        vocab_size: int,
    ) -> torch.Tensor:
        """Causal LM CE with extra weight on alignment-pad *target* positions (after shift)."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(logits.device)
        shift_pad = alignment_pad_mask[..., 1:].contiguous().to(device=logits.device, dtype=torch.bool)
        ce = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=ignore_index,
        )
        ce = ce.view_as(shift_labels).float()
        w = torch.where(
            shift_pad,
            ce.new_tensor(pad_weight),
            ce.new_tensor(1.0),
        )
        valid = shift_labels.ne(ignore_index)
        denom = (w * valid.float()).sum().clamp_min(1e-8)
        return (ce * w * valid.float()).sum() / denom

    # ------------------------------------------------------------------
    # Forward pass for grouped batch (user + assistant turns)
    # ------------------------------------------------------------------
    def _forward_grouped_batch(
        self,
        user_input_ids: torch.LongTensor,  # [B, L_user_max]
        user_attention_mask: Optional[torch.Tensor],  # [B, L_user_max]
        user_audio_tensors: Optional[torch.Tensor],  # [B, T_user_audio_max]
        user_audio_tensors_lens: Optional[torch.Tensor],  # [B] - actual lengths of audio tensors
        user_audio_features: Optional[torch.Tensor],  # [B, T_user_feat_max, D]
        user_audio_features_lens: Optional[torch.Tensor],  # [B] - actual lengths of audio features
        user_has_audio: Optional[torch.Tensor],  # [B] boolean
        assistant_input_ids: torch.LongTensor,  # [B, L_assistant_max]
        assistant_attention_mask: Optional[torch.Tensor],  # [B, L_assistant_max]
        assistant_audio_tensors: Optional[torch.Tensor],  # [B, T_assistant_audio_max]
        assistant_audio_tensors_lens: Optional[torch.Tensor],  # [B] - actual lengths of audio tensors
        assistant_audio_features: Optional[torch.Tensor],  # [B, T_assistant_feat_max, D]
        assistant_audio_features_lens: Optional[torch.Tensor],  # [B] - actual lengths of audio features
        assistant_has_audio: Optional[torch.Tensor],  # [B] boolean
        audio_start_id: int,
        audio_end_id: int,
        text_audio_interval_ratio: Optional[List[int]] = None,
        audio_length_shifted_by: int = 1,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, InterleavedS2SOutputWithPast]:
        """Batched forward for the grouped (user, assistant) turns.

        This mirrors the per-sample logic in `forward()`'s legacy path:
        - User turn (not predicted): insert encoded audio tokens before the first
          `audio_end_id` token if user has audio.
        - Assistant turn (predicted): if assistant has audio, interleave assistant
          text tokens and encoded audio tokens via `_interleave_text_audio_tokens`.

        Inputs originate from `src/train.py`'s `collate_fn_factory`:
        - user_* tensors are left padded
        - assistant_* tensors are right padded

        We remove per-turn padding, build per-sample sequences, then pad the final
        concatenated sequence across the batch.
        """
        from transformers.trainer_pt_utils import LabelSmoother
        from torch.nn.utils.rnn import pad_sequence
        # print(user_audio_features_lens, assistant_audio_features_lens)
        # print(user_input_ids.shape, assistant_input_ids.shape)
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        rank = get_rank()
        device = user_input_ids.device
        dtype = self.model.embed_tokens.weight.dtype
        batch_size = int(user_input_ids.shape[0])

        # When only_train_talker: retain text vocabulary weights in lm_head and embed_tokens by
        # restoring the pretrained slices before each forward (saved by training script).
        # if getattr(self, "only_train_talker", False):
        #     # LM head: first text_vocab_size - 3 output rows
        #     if not hasattr(self, "_pretrained_lm_head_text") or getattr(self, "_pretrained_lm_head_text", None) is None:
        #         self.register_buffer(
        #             "_pretrained_lm_head_text",
        #             self.lm_head.weight.data[: self.text_vocab_size - 3].clone(),
        #         )
        #     else:
        #         self.lm_head.weight.data[: self.text_vocab_size - 3].copy_(self._pretrained_lm_head_text)
            # Embed tokens: first text_vocab_size - 3 rows
            # if not hasattr(self, "_pretrained_embed_tokens_text") or getattr(self, "_pretrained_embed_tokens_text", None) is None:
            #     self.register_buffer(
            #         "_pretrained_embed_tokens_text",
            #         self.model.embed_tokens.weight.data[: self.text_vocab_size - 3].clone(),
            #     )
            # else:
            #     self.model.embed_tokens.weight.data[: self.text_vocab_size - 3].copy_(self._pretrained_embed_tokens_text)

        debug_print(f"_forward_grouped_batch called", rank=rank)
        debug_print(f"  - batch_size: {batch_size}", rank=rank)
        debug_print(f"  - device: {device}", rank=rank)
        debug_print(f"  - dtype: {dtype}", rank=rank)
        debug_print(f"  - user_input_ids shape: {user_input_ids.shape}", rank=rank)
        debug_print(f"  - assistant_input_ids shape: {assistant_input_ids.shape if assistant_input_ids is not None else None}", rank=rank)
        # One random framerate per batch, shifted by rank to reduce collisions.
        selected_framerate_input = choose_rank_shifted_option(
            self.config.training_input_framerate_options
        )
        use_uniform_input_merging = getattr(self.config, "uniform_merging", False)
        selected_input_target_rate = (
            round(random.uniform(4.0, 12.0), 2) if use_uniform_input_merging else None
        )
        selected_framerate = choose_rank_shifted_option(
            self.config.training_framerate_options
        )
        use_uniform_output_merging = getattr(self.config, "output_uniform_merging", False)
        # When output_uniform_merging is on we still want to honor user-provided
        # training_framerate_options (Hz, values > 1) as the uniform target rate;
        # only fall back to a random 4-12 Hz draw if the options resolve to a cosine
        # threshold (<= 1) or are not usable as a Hz value.
        if use_uniform_output_merging:
            if selected_framerate is not None and float(selected_framerate) > 1.0:
                selected_output_target_rate = float(selected_framerate)
            else:
                selected_output_target_rate = round(random.uniform(4.0, 12.0), 2)
        else:
            selected_output_target_rate = None
        # When using sinusoidal embedding during training, add random offset [0, 1] rounded to 3 decimals
        # if self.config.use_sinusoidal and self.training:
        # TODO see if this is necessary
        #     if float(selected_framerate) > 0:
        #         # offset = round(random.uniform(0.0, 0.5), 1)
        #         # offset = round(random.uniform(0.0, 0.5), 1)
        #     else:
        #         # offset = round(random.uniform(0.0, 0.009), 3)
        #         offset = 0
        #     selected_framerate = max(
        #         self.config.framerate_min,
        #         min(self.config.framerate_max, selected_framerate + offset),
        #     )
        debug_print(f"  - selected_framerate: {selected_framerate:.4f}", rank=rank)
        if selected_input_target_rate is not None:
            debug_print(
                f"  - selected_input_target_rate: {selected_input_target_rate:.2f} Hz",
                rank=rank,
            )
        if selected_output_target_rate is not None:
            debug_print(
                f"  - selected_output_target_rate: {selected_output_target_rate:.2f} Hz "
                f"(output_uniform_merging ablation)",
                rank=rank,
            )
        has_audio_dialog_data = bool(assistant_has_audio.any().item()) if assistant_has_audio is not None else False
        debug_print(f"  - has_audio_dialog_data: {has_audio_dialog_data}", rank=rank)

        # ------------------------------------------------------------------
        # Encode user/assistant audio in (sub)batches
        # ------------------------------------------------------------------
        debug_print(f"  - Starting audio encoding section", rank=rank)
        user_audio_row_to_subidx: Dict[int, int] = {}
        assistant_audio_row_to_subidx: Dict[int, int] = {}

        user_semantic_codes = None
        user_code_lens = None
        user_semantic_embeds = None
        user_token_lengths = None
        # v2 (interleaved) merging transformer state: when enabled we defer
        # aggregation until after audio_embed_transform so that the transformer
        # can attend over pre-merge frames and per-group query tokens jointly.
        u_alignment_v2 = None
        u_num_segments_v2 = None
        use_input_merging_v2 = (
            getattr(self.config, "use_input_merging_transformer", False)
            and getattr(self.config, "use_input_merging_transformer_v2", False)
            and getattr(self.config, "input_merging_transformer_num_layers", 0) > 0
        )
        dummy_encoder_loss = 0.0  # Added: collect dummy gradients for skipped encoder paths.

        assistant_semantic_codes = None
        assistant_token_lengths = None
        assistant_code_lens = None
        dummy_assistant_audio = 0.0
        if user_has_audio is not None and user_has_audio.any().item():
            u_rows = torch.nonzero(user_has_audio, as_tuple=True)[0]
            debug_print(f"    - Found {len(u_rows)} user samples with audio: {u_rows.tolist()}", rank=rank)
            for k, r in enumerate(u_rows.tolist()):
                user_audio_row_to_subidx[r] = k
            debug_print(
                f"    - use_sensevoice_feature: {self.config.use_sensevoice_feature}, "
                f"use_qwen3_feature: {self.config.use_qwen3_feature}, "
                f"use_whisper_fetaure: {getattr(self.config, 'use_whisper_fetaure', False)}, "
                f"use_qwen25o_feature: {self.config.use_qwen25o_feature}",
                rank=rank,
            )
            if self.config.use_qwen3_feature:
                debug_print(f"    - Encoding user audio with Qwen3 encoder", rank=rank)
                u_codec, u_codec_lens = self._encode_user_audio_qwen3(
                    user_audio_features,
                    user_audio_features_lens,
                    u_rows,
                    dtype,
                    device,
                )
                user_token_lengths = None
                if u_codec.shape[0] > 0:
                    assert u_codec.shape[2] == self._get_qwen3_encoder_dim(), f"u_codec shape: {u_codec.shape}"
            elif getattr(self.config, "use_whisper_fetaure", False):
                debug_print(f"    - Encoding user audio with Whisper encoder", rank=rank)
                u_codec, u_codec_lens = self._encode_user_audio_whisper(
                    user_audio_features,
                    user_audio_features_lens,
                    u_rows,
                    dtype,
                    device,
                )
                user_token_lengths = None
                if u_codec.shape[0] > 0:
                    assert u_codec.shape[2] == self._get_whisper_encoder_dim(), f"u_codec shape: {u_codec.shape}"
            elif self.config.use_qwen25o_feature:
                debug_print(f"    - Encoding user audio with Qwen2.5 Omni encoder", rank=rank)
                u_codec, u_codec_lens = self._encode_user_audio_qwen25o(
                    user_audio_features,
                    user_audio_features_lens,
                    u_rows,
                    dtype,
                    device,
                )
                user_token_lengths = None
                if u_codec.shape[0] > 0:
                    assert u_codec.shape[2] == self._get_qwen25o_encoder_dim(), f"u_codec shape: {u_codec.shape}"
            elif self.config.use_sensevoice_feature:
                debug_print(f"    - Encoding user audio with sensevoice features", rank=rank)
                u_codec = encode_flexicodec(
                    user_audio_tensors[u_rows],
                    self.flexicodec_dict,
                    audio_features=user_audio_features[u_rows],
                    audio_features_lens=(user_audio_features_lens[u_rows] if user_audio_features_lens is not None else None),
                    num_quantizers=1,
                    merging_threshold=1.0 if use_uniform_input_merging else selected_framerate_input,
                    return_semantic_feature=True,
                ).transpose(1,2).to(user_audio_tensors.dtype) # [B, T, H]
                    
                assert u_codec.shape[2] == self.config.codec_hidden_size, f"u_codec shape: {u_codec.shape}"
                u_codec_lens = user_audio_features_lens[u_rows] if user_audio_features_lens is not None else None
            # Apply flexible frame rate merging if enabled
            if self.config.enable_flexible_framerate:
                if use_uniform_input_merging:
                    debug_print(
                        f"    - Applying uniform input merging with target rate {selected_input_target_rate:.2f} Hz",
                        rank=rank,
                    )
                    u_alignment, sim, u_codec_lens, user_token_lengths = self._perform_similarity_alignment_vectorized(
                        u_codec,
                        u_codec_lens,
                        target_rate=selected_input_target_rate,
                        base_rate=25 if self.config.use_qwen25o_feature else 12.5,
                        dynamic_merging=False,
                    )
                else:
                    debug_print(
                        f"    - Applying flexible frame rate merging with threshold {selected_framerate_input:.2f}",
                        rank=rank,
                    )
                    u_alignment, sim, u_codec_lens, user_token_lengths = self._perform_similarity_alignment_vectorized(
                        u_codec, # [B, T, H]
                        u_codec_lens,
                        merging_threshold=selected_framerate_input,
                        base_rate=25 if self.config.use_qwen25o_feature else 12.5,
                    )
                if use_input_merging_v2:
                    # v2 path: keep pre-merge frames; defer aggregation to the
                    # interleaved merging transformer (run after audio_embed_transform).
                    u_alignment_v2 = u_alignment
                    u_num_segments_v2 = u_codec_lens  # num_segments_per_item from alignment fn
                    debug_print(
                        f"    - v2 merging transformer: deferring aggregation; "
                        f"alignment shape {u_alignment.shape}, max segments {int(u_codec_lens.max().item()) if u_codec_lens.numel() > 0 else 0}",
                        rank=rank,
                    )
                else:
                    u_codec_aggregated = self.aggregate_features(u_codec, u_alignment)

                    assert len(u_codec_aggregated.shape) == 3, f"Error: u_codec_aggregated shape: {u_codec_aggregated.shape}"
                    # assert (u_codec_aggregated.shape[2] == self.config.codec_hidden_size) or (u_codec_aggregated.shape[2] == self._get_qwen3_encoder_dim()), f"u_codec_aggregated shape: {u_codec_aggregated.shape}"
                    u_codec = u_codec_aggregated
            else:
                user_token_lengths = None
        else:
            if self.config.use_qwen3_feature:
                u_codec = torch.zeros((batch_size, 0, self._get_qwen3_encoder_dim()), dtype=dtype, device=device)
                use_enc_grad = getattr(self.config, "finetune_speech_encoder", False) and not getattr(self.config, "only_train_llm", False)
                if use_enc_grad:
                    # DeepSpeed ZeRO fix: run real forward pass to build identical autograd graph
                    # Use small random noise instead of zeros to avoid zero-variance LayerNorm NaNs
                    dummy_feats = torch.randn((1, 10, 128), device=device, dtype=dtype) * 1e-5
                    dummy_lens = torch.tensor([10], device=device, dtype=torch.long)
                    dummy_u_codec, _ = self._encode_user_audio_qwen3(
                        dummy_feats, dummy_lens, torch.tensor([0], device=device), dtype, device
                    )
                    dummy = dummy_u_codec.mean() * 0.0 if dummy_u_codec.numel() > 0 else dummy_u_codec.sum() * 0.0
                    u_codec = u_codec + dummy
                    dummy_encoder_loss = dummy_encoder_loss + dummy
            elif getattr(self.config, "use_whisper_fetaure", False):
                u_codec = torch.zeros((batch_size, 0, self._get_whisper_encoder_dim()), dtype=dtype, device=device)
                use_enc_grad = getattr(self.config, "finetune_speech_encoder", False) and not getattr(self.config, "only_train_llm", False)
                if use_enc_grad:
                    dummy_feats = torch.zeros((1, 10, 128), device=device, dtype=dtype)
                    dummy_lens = torch.tensor([10], device=device, dtype=torch.long)
                    dummy_u_codec, _ = self._encode_user_audio_whisper(
                        dummy_feats, dummy_lens, torch.tensor([0], device=device), dtype, device
                    )
                    dummy = dummy_u_codec.sum() * 0.0
                    u_codec = u_codec + dummy
                    dummy_encoder_loss = dummy_encoder_loss + dummy
            elif self.config.use_qwen25o_feature:
                u_codec = torch.zeros((batch_size, 0, self._get_qwen25o_encoder_dim()), dtype=dtype, device=device)
                use_enc_grad = getattr(self.config, "finetune_speech_encoder", False) and not getattr(self.config, "only_train_llm", False)
                if use_enc_grad:
                    # DeepSpeed ZeRO fix: run real forward pass to build identical autograd graph
                    # Use small random noise instead of zeros to avoid zero-variance LayerNorm NaNs
                    dummy_feats = torch.randn((1, 10, 128), device=device, dtype=dtype) * 1e-5
                    dummy_lens = torch.tensor([10], device=device, dtype=torch.long)
                    dummy_u_codec, _ = self._encode_user_audio_qwen25o(
                        dummy_feats, dummy_lens, torch.tensor([0], device=device), dtype, device
                    )
                    dummy = dummy_u_codec.mean() * 0.0 if dummy_u_codec.numel() > 0 else dummy_u_codec.sum() * 0.0
                    u_codec = u_codec + dummy
                    dummy_encoder_loss = dummy_encoder_loss + dummy
            else:
                u_codec = torch.zeros((batch_size, 0, self.config.codec_hidden_size), dtype=dtype, device=device)
                _sv_ft = self._finetune_sensevoice_semantic()
                if self.config.use_sensevoice_feature and _sv_ft:
                    # DeepSpeed ZeRO fix: run real forward pass to build identical autograd graph
                    dummy_audio = torch.zeros((1, 16000), device=device, dtype=torch.float)
                    dummy_u_codec = encode_flexicodec(
                        dummy_audio,
                        self.flexicodec_dict,
                        audio_features=None,
                        audio_features_lens=None,
                        num_quantizers=1,
                        merging_threshold=1.0,
                        return_semantic_feature=True,
                    )
                    dummy = dummy_u_codec.mean() * 0.0 if dummy_u_codec.numel() > 0 else dummy_u_codec.sum() * 0.0
                    u_codec = u_codec + dummy
                    dummy_encoder_loss = dummy_encoder_loss + dummy
        
        # Transform audio features to embedding space (audio_embed_transform: encoder_dim or codec_hidden_size -> hidden_size)
        user_semantic_embeds = self.audio_embed_transform(u_codec) # [B, T, H]
        
        # Apply optional local windowed merging transformer
        if use_input_merging_v2:
            # v2: interleave pre-merge frame embeddings with per-group query
            # tokens and let a transformer attend over both, then gather the
            # query positions as the merged representation.
            if u_alignment_v2 is not None and user_semantic_embeds.shape[1] > 0:
                user_semantic_embeds = self.input_merging_transformer(
                    user_semantic_embeds,
                    u_alignment_v2,
                    u_num_segments_v2,
                )  # [B, G, H]
            else:
                # Empty audio batch: run a tiny dummy forward through the v2
                # module so DDP still sees its parameters as used, then add a
                # zero-magnitude residual so gradients flow but values are
                # unchanged.
                _dev = user_semantic_embeds.device
                _dtype = user_semantic_embeds.dtype
                _H = self.config.hidden_size
                dummy_feats = torch.zeros((1, 1, _H), device=_dev, dtype=_dtype)
                dummy_align = torch.ones((1, 1, 1), device=_dev, dtype=_dtype)
                dummy_segs = torch.ones((1,), device=_dev, dtype=torch.long)
                dummy_out = self.input_merging_transformer(
                    dummy_feats, dummy_align, dummy_segs
                )
                user_semantic_embeds = user_semantic_embeds + dummy_out.sum() * 0.0
        else:
            user_semantic_embeds = self.input_merging_transformer(user_semantic_embeds) # [B, T, H]
        
        # # Add length embeddings if flexible framerate is enabled and we have token_lengths
        # if self.config.enable_flexible_framerate and user_token_lengths is not None and not self.config.use_qwen3_feature:
        #     # Get length embeddings for each merged token
        #     user_length_embeds = self.talker_model.length_embedding(user_token_lengths)  # [B, T, H]
        #     # Add length embeddings to audio embeddings
        #     user_semantic_embeds = user_semantic_embeds + user_length_embeds
        #     debug_print(f"    - Added length embeddings, shape: {user_length_embeds.shape}", rank=rank)
        
        debug_print(f"    - User semantic embeds shape: {user_semantic_embeds.shape}", rank=rank)
        

        debug_print(f"  - Checking assistant_has_audio: {assistant_has_audio is not None}", rank=rank)
        if assistant_has_audio is not None:
            debug_print(f"    - assistant_has_audio.any(): {assistant_has_audio.any().item()}", rank=rank)
            debug_print(f"    - assistant_has_audio values: {assistant_has_audio.tolist()}", rank=rank)
        
        if assistant_has_audio is not None and assistant_has_audio.any().item():
            debug_print(f"  - Processing assistant audio", rank=rank)
            if assistant_audio_tensors is None or assistant_audio_features is None:
                raise ValueError("assistant_audio_tensors and assistant_audio_features must be provided when assistant_has_audio is True")
            debug_print(f"    - assistant_audio_tensors shape: {assistant_audio_tensors.shape}", rank=rank)
            debug_print(f"    - assistant_audio_features shape: {assistant_audio_features.shape}", rank=rank)
            a_rows = torch.nonzero(assistant_has_audio, as_tuple=True)[0]
            debug_print(f"    - Found {len(a_rows)} assistant samples with audio: {a_rows.tolist()}", rank=rank)
            for k, r in enumerate(a_rows.tolist()):
                assistant_audio_row_to_subidx[r] = k

            debug_print(f"  - Encoding assistant audio for {len(a_rows)} samples", rank=rank)
            debug_print(f"    - assistant_audio_tensors[a_rows] shape: {assistant_audio_tensors[a_rows].shape}", rank=rank)
            debug_print(f"    - assistant_audio_features[a_rows] shape: {assistant_audio_features[a_rows].shape}", rank=rank)
            debug_print(f"    - Calling encode_flexicodec...", rank=rank)
            a_codec = encode_flexicodec(
                assistant_audio_tensors[a_rows],
                self.flexicodec_dict,
                audio_features=assistant_audio_features[a_rows],
                audio_features_lens=(assistant_audio_features_lens[a_rows] if assistant_audio_features_lens is not None else None),
                num_quantizers=1,
                merging_threshold=(
                    selected_output_target_rate
                    if use_uniform_output_merging
                    else selected_framerate
                ),
            )
            debug_print(f"  - Assistant audio encoding complete", rank=rank)
            debug_print(f"    - codec output keys: {a_codec.keys()}", rank=rank)
            debug_print(f"    - semantic_codes shape: {a_codec['semantic_codes'].shape}", rank=rank)
            debug_print(f"    - token_lengths shape: {a_codec['token_lengths'].shape}", rank=rank)
            assistant_semantic_codes = a_codec["semantic_codes"].squeeze(1).to(dtype=torch.long)  # [Na, T]
            assistant_token_lengths = a_codec["token_lengths"].to(dtype=torch.long)  # [Na, T]
            assistant_code_lens = a_codec.get("total_frames", a_codec.get("speech_token_len", None))  # [Na]
            # ------------------------------------------------------------------
            # Rank-0 only: print actual achieved output frame rate per sample.
            # Computed as code_lens / (audio_features_lens / 1.33) * 12.5  (Hz),
            # mirroring the per_sample_frame_rate_embed formula below.
            # ------------------------------------------------------------------
            if rank == 0 and assistant_code_lens is not None:
                try:
                    a_rows_list = a_rows.tolist()
                    actual_rates = []
                    for k, r in enumerate(a_rows_list):
                        code_len = float(assistant_code_lens[k].item())
                        if assistant_audio_features_lens is not None:
                            feat_len = float(assistant_audio_features_lens[r].item()) / 1.33
                            feat_len = max(feat_len, 1.0)
                            actual_rates.append(code_len / feat_len * 12.5)
                        else:
                            actual_rates.append(float("nan"))
                    requested = (
                        f"uniform target={selected_output_target_rate:.2f}Hz"
                        if use_uniform_output_merging
                        else f"selected_framerate={selected_framerate}"
                    )
                    rates_str = ", ".join(f"{r:.2f}" for r in actual_rates)
                    # print(
                    #     f"[rank0][output_framerate] requested={requested} "
                    #     f"| actual_rates_Hz=[{rates_str}] "
                    #     f"| code_lens={assistant_code_lens.tolist()}",
                    #     flush=True,
                    # )
                except Exception as _e:
                    print(f"[rank0][output_framerate] failed to compute: {_e}", flush=True)
            debug_print(f"    - assistant_semantic_codes shape: {assistant_semantic_codes.shape}", rank=rank)
            debug_print(f"    - assistant_token_lengths shape: {assistant_token_lengths.shape}", rank=rank)
            debug_print(f"    - assistant_code_lens: {assistant_code_lens}", rank=rank)
        else:
            debug_print(f"  - No assistant audio to process", rank=rank)
            # Ensure dummy_audio has non-zero variance to prevent LayerNorm nan
            dummy_audio = torch.randn((1, 16000), device=device, dtype=torch.float) * 1e-5
            dummy_a_codec = encode_flexicodec(
                dummy_audio,
                self.flexicodec_dict,
                audio_features=None,
                audio_features_lens=None,
                num_quantizers=1,
                merging_threshold=1.0,
            )

            _sc = dummy_a_codec["semantic_codes"].float()
            dummy = _sc.mean() * 0.0 if _sc.numel() > 0 else _sc.sum() * 0.0
            dummy_assistant_audio = dummy

        # Per-sample frame rate: (assistant_code_lens / assistant_audio_features_lens) * 15, range [0, 15] Hz
        per_sample_framerate = None
        if self.config.per_sample_frame_rate_embed and assistant_code_lens is not None and assistant_audio_features_lens is not None:
            per_sample_framerate = torch.zeros(batch_size, device=device, dtype=dtype)
            for i in range(batch_size):
                if assistant_has_audio is not None and bool(assistant_has_audio[i].item()):
                    sub_i = assistant_audio_row_to_subidx[i]
                    feat_len = (assistant_audio_features_lens[i].float() / 1.33).floor().clamp(min=1.0)
                    code_len = assistant_code_lens[sub_i].float()
                    # feat_len = a_codec['token_lengths'][i][:code_len].sum()
                    fr = (code_len / feat_len) * 12.5
                    fr = torch.nan_to_num(fr, nan=12.5)
                    per_sample_framerate[i] = fr.clamp(min=0.0, max=15.0)
                else:
                    per_sample_framerate[i] = 0.0
            debug_print(f"    - per_sample_framerate (0-15 Hz): {per_sample_framerate.tolist()}", rank=rank)
        # ------------------------------------------------------------------
        # Build per-sample (unpadded) sequences, then pad across batch
        # ------------------------------------------------------------------
        debug_print(f"  - Starting per-sample sequence building (parallel architecture)", rank=rank)
        no_pad = bool(getattr(self.config, "no_pad", False))
        all_inputs_embeds: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        all_length_labels: List[torch.Tensor] = []
        all_attention_masks: List[torch.Tensor] = []
        # Speech sequences (parallel architecture)
        all_speech_embeds: List[torch.Tensor] = []
        all_speech_labels: List[torch.Tensor] = []
        all_speech_length_labels: List[torch.Tensor] = []
        all_speech_attention_masks: List[torch.Tensor] = []
        # Extra conditioning for talker (aligned to full user+assistant sequence)
        all_talker_extra_conds: List[torch.Tensor] = []
        all_alignment_pad_masks: List[torch.Tensor] = []
        all_talker_attention_masks: List[torch.Tensor] = []
        for i in range(batch_size):
            debug_print(f"  - Processing sample {i}/{batch_size-1}", rank=rank)
            # ---- User turn (slice away left padding) ----
            debug_print(f"    - Building user turn for sample {i}", rank=rank)
            if user_attention_mask is not None:
                user_ids_i = user_input_ids[i][user_attention_mask[i]]
                debug_print(f"      - Using attention mask, user_ids_i length: {len(user_ids_i)}", rank=rank)
            else:
                user_ids_i = user_input_ids[i]
                debug_print(f"      - No attention mask, user_ids_i length: {len(user_ids_i)}", rank=rank)

            debug_print(f"      - Embedding user text tokens...", rank=rank)
            user_text_embeds = self._embed_text_tokens(user_ids_i)  # [Lu, H]
            debug_print(f"      - user_text_embeds shape: {user_text_embeds.shape}", rank=rank)
            
            if user_has_audio is not None and bool(user_has_audio[i].item()):
                debug_print(f"      - User has audio, processing...", rank=rank)
                if i not in user_audio_row_to_subidx:
                    raise RuntimeError("user_has_audio indicates audio, but no encoded audio found for this sample")
                sub_i = user_audio_row_to_subidx[i]
                debug_print(f"        - Using sub_i={sub_i} for user audio", rank=rank)

                if (
                    self.config.use_sensevoice_feature
                    or self.config.use_qwen3_feature
                    or getattr(self.config, "use_whisper_fetaure", False)
                    or self.config.use_qwen25o_feature
                ):
                    if self.config.use_qwen3_feature:
                        feature_source = "qwen3"
                    elif getattr(self.config, "use_whisper_fetaure", False):
                        feature_source = "whisper"
                    elif self.config.use_qwen25o_feature:
                        feature_source = "qwen25o"
                    else:
                        feature_source = "sensevoice"
                    
                    debug_print(f"        - Using {feature_source} features, getting embeds...", rank=rank)
                    user_audio_embeds = user_semantic_embeds[sub_i][:u_codec_lens[sub_i]] # [T,H]
                    debug_print(f"        - user_audio_embeds shape: {user_audio_embeds.shape}", rank=rank)
                else:
                    raise NotImplementedError

                debug_print(f"        - Finding audio_end_id={audio_end_id} in user_ids_i...", rank=rank)
                audio_end_pos = torch.where(user_ids_i == audio_end_id)[-1] # TODO support multi turn
                debug_print(f"        - Found audio_end_pos: {audio_end_pos.tolist()}", rank=rank)
                # assert audio_end_pos.numel() == 1, f"audio_end_id ({audio_end_id}) found multiple times in user_input_ids for sample {i}"
                if audio_end_pos.numel() == 0:
                    raise ValueError(f"audio_end_id ({audio_end_id}) not found in user_input_ids for sample {i}")
                audio_end_pos = int(audio_end_pos[0].item())
                debug_print(f"        - audio_end_pos: {audio_end_pos}", rank=rank)
                
                # Optionally replace audio_start/end token embeddings with learnable embeddings
                if getattr(self.config, "use_learnable_audio_boundary", False):
                    user_text_embeds = user_text_embeds.clone()
                    if audio_end_pos > 0:
                        user_text_embeds[audio_end_pos - 1] = self.audio_start_embedding.to(user_text_embeds.dtype)
                    user_text_embeds[audio_end_pos] = self.audio_end_embedding.to(user_text_embeds.dtype)
            else:
                audio_end_pos = 0
                user_audio_embeds = user_semantic_embeds[0][:0] # [T,H]
                
            user_inputs_embeds_i = torch.cat(
                [user_text_embeds[:audio_end_pos], user_audio_embeds, user_text_embeds[audio_end_pos:]],
                dim=0,
            )
            debug_print(f"        - user_inputs_embeds_i shape: {user_inputs_embeds_i.shape}", rank=rank)

            user_len_i = int(user_inputs_embeds_i.shape[0])
            debug_print(f"      - Creating user labels and masks, length={user_len_i}", rank=rank)
            user_labels_i = torch.full((user_len_i,), IGNORE_TOKEN_ID, device=device, dtype=torch.long)
            user_length_labels_i = torch.full((user_len_i,), IGNORE_TOKEN_ID, device=device, dtype=torch.long)
            user_attn_i = torch.ones((user_len_i,), device=device, dtype=torch.bool)

            # ---- Assistant turn (slice away right padding) ----
            debug_print(f"    - Building assistant turn for sample {i}", rank=rank)
            if assistant_attention_mask is not None:
                assistant_ids_i = assistant_input_ids[i][assistant_attention_mask[i]]
                debug_print(f"      - Using attention mask, assistant_ids_i length: {len(assistant_ids_i)}", rank=rank)
            else:
                assistant_ids_i = assistant_input_ids[i]
                debug_print(f"      - No attention mask, assistant_ids_i length: {len(assistant_ids_i)}", rank=rank)

            # Always call _interleave_text_audio_tokens to prevent DeepSpeed ZeRO-3 deadlock
            # When there's no audio, pass empty tensors to ensure all embedding layers are called
            debug_print(f"      - Checking if assistant has audio: {assistant_has_audio is not None}", rank=rank)
            if assistant_has_audio is not None:
                debug_print(f"        - assistant_has_audio[{i}]: {assistant_has_audio[i].item()}", rank=rank)
            
            if assistant_has_audio is not None and bool(assistant_has_audio[i].item()):
                debug_print(f"      - Assistant has audio, processing...", rank=rank)
                if i not in assistant_audio_row_to_subidx:
                    raise RuntimeError("assistant_has_audio indicates audio, but no encoded audio found for this sample")
                sub_i = assistant_audio_row_to_subidx[i]
                debug_print(f"        - Using sub_i={sub_i} for assistant audio", rank=rank)

                if assistant_code_lens is not None:
                    t_audio = int(assistant_code_lens[sub_i].item())
                    debug_print(f"        - Using code_lens, t_audio={t_audio}", rank=rank)
                else:
                    t_audio = int(assistant_semantic_codes.shape[1])
                    debug_print(f"        - No code_lens, using full sequence, t_audio={t_audio}", rank=rank)

                debug_print(f"        - Preparing audio_ids and audio_len_labels...", rank=rank)
                audio_ids = assistant_semantic_codes[sub_i, :t_audio] + self.text_vocab_size
                audio_len_labels = assistant_token_lengths[sub_i, :t_audio]
                debug_print(f"        - audio_ids shape: {audio_ids.shape}, audio_len_labels shape: {audio_len_labels.shape}", rank=rank)
            else:
                debug_print(f"      - Assistant has no audio, using empty tensors", rank=rank)
                # No audio: pass empty tensors to ensure embedding layers are still called
                audio_ids = torch.empty(0, device=device, dtype=torch.long)
                audio_len_labels = torch.empty(0, device=device, dtype=torch.long)
            
            # Process assistant turn with parallel architecture
            debug_print(f"  - Building parallel text and speech sequences for sample {i}", rank=rank)
            debug_print(f"    - text tokens: {len(assistant_ids_i)}, audio tokens: {len(audio_ids)}", rank=rank)
            framerate_for_sample = (
                per_sample_framerate[i].item()
                if per_sample_framerate is not None
                else selected_framerate
            )
            if per_sample_framerate is not None:
                framerate_for_sample = max(0.0, min(15.0, framerate_for_sample))
            (
                assistant_inputs_embeds_i,
                assistant_labels_i,
                _,
                speech_labels_i,
                speech_length_labels_i,
                assistant_audio_embeds_i,
                assistant_length_embeds_i,
                assistant_alignment_pad_len,
            ) = self._build_parallel_text_speech_sequences(
                text_token_ids=assistant_ids_i,
                audio_token_ids=audio_ids,
                audio_token_lengths=audio_len_labels,
                framerate=framerate_for_sample,
                framerate_min=0.0 if per_sample_framerate is not None else None,
                framerate_max=15.0 if per_sample_framerate is not None else None,
            )
            
            assistant_len_i = int(assistant_inputs_embeds_i.shape[0])
            assistant_attn_i = torch.ones((assistant_len_i,), device=device, dtype=torch.bool)

            # ---- Concatenate user + assistant ----
            debug_print(f"    - Concatenating user and assistant sequences for sample {i}...", rank=rank)
            inputs_embeds_i = torch.cat([user_inputs_embeds_i, assistant_inputs_embeds_i], dim=0)
            del assistant_inputs_embeds_i
            labels_i = torch.cat([user_labels_i, assistant_labels_i.to(device=device)], dim=0)
            del assistant_labels_i
            align_pad_mask_i = torch.zeros(
                (labels_i.shape[0],), device=device, dtype=torch.bool
            )
            if assistant_alignment_pad_len > 0:
                o = int(assistant_ids_i.shape[0])
                start = user_len_i + o
                align_pad_mask_i[start : start + int(assistant_alignment_pad_len)] = True
            
            length_labels_i = torch.cat([user_length_labels_i, speech_length_labels_i.to(device=device)], dim=0)
            del speech_length_labels_i
            speech_labels_i = torch.cat([user_length_labels_i, speech_labels_i.to(device=device)], dim=0) # IGNORE + speech labels
            attn_i = torch.cat([user_attn_i, assistant_attn_i], dim=0)
            del assistant_attn_i, user_attn_i

            # Build talker extra conditioning aligned to the full sequence:
            # only assistant segment has (audio_embeds, length_embeds) as separate conditions; user segment is zeros.
            # When use_concat_len_emb: replace delay token embeddings (first D positions) with length embeddings
            if getattr(self.config, "use_concat_len_emb", False):
                D = self.config.speech_delay_tokens
                if assistant_audio_embeds_i.shape[0] >= D:
                    assistant_audio_embeds_i = assistant_audio_embeds_i.clone()
                    assistant_audio_embeds_i[:D] = assistant_length_embeds_i[:D]
            # Concatenate [audio, length] along feature dim so both get projected separately in talker_cond_proj
            assistant_audio_length_cat = torch.cat([assistant_audio_embeds_i, assistant_length_embeds_i], dim=-1)  # [L_asst, 2*H_emb]
            emb_dim = assistant_audio_length_cat.shape[-1]
            talker_extra_cond_i = torch.cat(
                [
                    torch.zeros((user_len_i, emb_dim), device=device, dtype=dtype),
                    assistant_audio_length_cat,
                ],
                dim=0,
            )
            del assistant_audio_embeds_i, assistant_length_embeds_i

            debug_print(f"    - Combined sequence length for sample {i}: {inputs_embeds_i.shape[0]}", rank=rank)

            all_inputs_embeds.append(inputs_embeds_i)
            all_labels.append(labels_i)
            all_length_labels.append(length_labels_i)
            all_speech_labels.append(speech_labels_i)
            all_attention_masks.append(attn_i)
            all_talker_extra_conds.append(talker_extra_cond_i)
            all_alignment_pad_masks.append(align_pad_mask_i)
            all_talker_attention_masks.append(torch.ones((talker_extra_cond_i.shape[0],), device=device, dtype=torch.bool))
            debug_print(f"  - Sample {i} processing complete", rank=rank)
        # Pack (FA2) or pad across batch
        use_packing = batch_size > 1 and getattr(self.config, '_attn_implementation', '') == 'flash_attention_2'
        # use_packing = False
        main_sample_lengths = [t.shape[0] for t in all_inputs_embeds]
        talker_sample_lengths = [t.shape[0] for t in all_talker_extra_conds]
        talker_attention_mask = None
        talker_position_ids = None
        if use_packing:
            sample_lengths = main_sample_lengths
            inputs_embeds = torch.cat(all_inputs_embeds, dim=0).unsqueeze(0)
            labels = torch.cat(all_labels, dim=0).unsqueeze(0)
            length_labels = torch.cat(all_length_labels, dim=0).unsqueeze(0)
            speech_labels = torch.cat(all_speech_labels, dim=0).unsqueeze(0)
            alignment_pad_mask = torch.cat(all_alignment_pad_masks, dim=0).unsqueeze(0)
            talker_extra_conds = torch.cat(all_talker_extra_conds, dim=0).unsqueeze(0).to(inputs_embeds.dtype)
            attention_mask = None
            position_ids = torch.cat(
                [torch.arange(sl, device=device) for sl in sample_lengths]
            ).unsqueeze(0)
            if no_pad:
                talker_attention_mask = None
                talker_position_ids = torch.cat(
                    [torch.arange(sl, device=device) for sl in talker_sample_lengths]
                ).unsqueeze(0)
            debug_print(
                f"  - Packing {batch_size} samples: lengths={sample_lengths}, "
                f"total={sum(sample_lengths)}", rank=rank,
            )
        else:
            inputs_embeds = pad_sequence(all_inputs_embeds, batch_first=True, padding_value=0.0)
            labels = pad_sequence(all_labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
            length_labels = pad_sequence(all_length_labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
            attention_mask = pad_sequence(all_attention_masks, batch_first=True, padding_value=False).to(inputs_embeds.dtype)
            speech_labels = pad_sequence(all_speech_labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
            alignment_pad_mask = pad_sequence(all_alignment_pad_masks, batch_first=True, padding_value=False)
            talker_extra_conds = pad_sequence(all_talker_extra_conds, batch_first=True, padding_value=0.0).to(inputs_embeds.dtype)
            position_ids = None
            sample_lengths = main_sample_lengths
            if no_pad:
                talker_attention_mask = pad_sequence(all_talker_attention_masks, batch_first=True, padding_value=False).to(inputs_embeds.dtype)
                talker_position_ids = None
        # Compute speech logits via talker (parallel mode)
        # speech_length_labels = pad_sequence(all_speech_length_labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
        # speech_attention_mask = pad_sequence(all_speech_attention_masks, batch_first=True, padding_value=False).to(speech_embeds.dtype)

        # Use chained architecture if enabled
        debug_print(f"  - Using standard forward pass", rank=rank)
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        debug_print(f"  - Model forward pass complete", rank=rank)
        hidden_states = outputs.hidden_states[-1]  # [B, L, H]
        debug_print(f"  - hidden_states shape: {hidden_states.shape}", rank=rank)

        # Compute text logits from main LM
        debug_print(f"  - Computing text logits...", rank=rank)
        text_logits = self._compute_text_logits(hidden_states)  # [B, L, V_text_ext]
        debug_print(f"    - text_logits shape: {text_logits.shape}", rank=rank)
        logits = text_logits  # Text-only logits
        
        # Always run talker forward to ensure DeepSpeed ZeRO-3 parameter sync
        # across all ranks.  When no speech data exists, speech_labels are all
        # IGNORE_TOKEN_ID so the loss contribution is zero.
        has_speech_tokens = bool(speech_labels.max().item() > 0)

        # Build talker inputs: use -6th layer when early_diverge_talker else last layer
        if getattr(self.config, "early_diverge_talker", False) and outputs.hidden_states is not None:
            talker_hidden_src = outputs.hidden_states[-6]  # [B, L, H]
        else:
            talker_hidden_src = hidden_states

        # no_pad mode: do NOT forward audio-only tail through thinker; instead pad thinker hidden states with zeros
        # to match the talker/audio length right before talker conditioning.
        if no_pad:
            if use_packing:
                src_splits = torch.split(talker_hidden_src.squeeze(0), sample_lengths, dim=0)
                padded_parts = []
                for src_i, tl in zip(src_splits, talker_sample_lengths):
                    tl = int(tl)
                    ml = int(src_i.shape[0])
                    take = min(ml, tl)
                    z = torch.zeros((tl, src_i.shape[-1]), device=src_i.device, dtype=src_i.dtype)
                    if take > 0:
                        z[:take] = src_i[:take]
                    padded_parts.append(z)
                talker_hidden_src = torch.cat(padded_parts, dim=0).unsqueeze(0)
                # For talker-side packing, reuse sample_lengths variable downstream (e.g. fr_emb broadcast)
                sample_lengths = talker_sample_lengths
            else:
                max_talker_len = int(max(talker_sample_lengths) if len(talker_sample_lengths) > 0 else 0)
                padded = torch.zeros(
                    (batch_size, max_talker_len, talker_hidden_src.shape[-1]),
                    device=talker_hidden_src.device,
                    dtype=talker_hidden_src.dtype,
                )
                for bi, (ml, tl) in enumerate(zip(main_sample_lengths, talker_sample_lengths)):
                    tl = int(tl)
                    take = min(int(ml), tl)
                    if take > 0:
                        padded[bi, :take] = talker_hidden_src[bi, :take]
                talker_hidden_src = padded
        talker_base = self.talker_model.lm_to_talker_proj(talker_hidden_src) if self.talker_model.lm_to_talker_proj is not None else talker_hidden_src

        # Use projection (instead of addition) for conditioning the talker.
        # Concatenate [base, fr_emb, extra] and project to talker_hidden_size.
        if per_sample_framerate is not None:
            fr_emb = self._framerate_sinusoidal_embedding_batch(
                per_sample_framerate, device, dtype,
                framerate_min=0.0, framerate_max=15.0,
            )  # [B, H]
        else:
            fr_emb = self._framerate_sinusoidal_embedding(selected_framerate, device, dtype) if selected_framerate is not None else None
        if getattr(self.config, 'talker_embed_v2', False):
            fr_emb_talker = fr_emb if fr_emb is not None else torch.zeros(1, self.config.talker_hidden_size, device=device, dtype=dtype)
        else:
            fr_emb_talker = (
                self.talker_model.lm_to_talker_proj(fr_emb) if fr_emb is not None and self.talker_model.lm_to_talker_proj is not None
                else fr_emb if fr_emb is not None else torch.zeros(1, self.config.talker_hidden_size, device=device, dtype=dtype)
            )
        if fr_emb_talker.dim() == 2:
            if use_packing and fr_emb_talker.shape[0] == batch_size:
                fr_parts = [fr_emb_talker[i:i+1].expand(sl, -1) for i, sl in enumerate(sample_lengths)]
                fr_emb_talker = torch.cat(fr_parts, dim=0).unsqueeze(0)
            elif fr_emb_talker.shape[0] == batch_size:
                fr_emb_talker = fr_emb_talker.unsqueeze(1).expand(-1, talker_base.shape[1], -1)
            else:
                fr_emb_talker = fr_emb_talker.unsqueeze(0).expand(talker_base.shape[0], talker_base.shape[1], -1)
        # Project audio and length embeddings separately (along with other conditions)
        if talker_extra_conds is not None:
            if getattr(self.config, 'talker_embed_v2', False):
                H_emb = self.config.talker_hidden_size
                audio_extra = talker_extra_conds[..., :H_emb]
                length_extra = talker_extra_conds[..., H_emb:]
                audio_talker = audio_extra
                length_talker = length_extra
            else:
                H = self.config.hidden_size
                audio_extra = talker_extra_conds[..., :H]
                length_extra = talker_extra_conds[..., H:]
                audio_talker = (
                    self.talker_model.lm_to_talker_proj(audio_extra) if self.talker_model.lm_to_talker_proj is not None else audio_extra
                )
                length_talker = (
                    self.talker_model.lm_to_talker_proj(length_extra) if self.talker_model.lm_to_talker_proj is not None else length_extra
                )
        else:
            if getattr(self.config, 'talker_embed_v2', False):
                audio_talker = torch.zeros(*talker_base.shape[:-1], self.config.talker_hidden_size, device=device, dtype=dtype)
                length_talker = torch.zeros(*talker_base.shape[:-1], self.config.talker_hidden_size, device=device, dtype=dtype)
            else:
                audio_talker = torch.zeros_like(talker_base)
                length_talker = torch.zeros_like(talker_base)
        # Optional: concat LM embedding output (per-position, not averaged) as extra conditioning
        if getattr(self.config, "talker_concat_lm_text_output", False):
            lm_first_layer = outputs.hidden_states[0]  # [B, L, H] - embedding layer output
            if no_pad:
                if use_packing:
                    lm_splits = torch.split(lm_first_layer.squeeze(0), main_sample_lengths, dim=0)
                    padded_parts = []
                    for src_i, tl in zip(lm_splits, talker_sample_lengths):
                        tl = int(tl)
                        ml = int(src_i.shape[0])
                        take = min(ml, tl)
                        z = torch.zeros((tl, src_i.shape[-1]), device=src_i.device, dtype=src_i.dtype)
                        if take > 0:
                            z[:take] = src_i[:take]
                        padded_parts.append(z)
                    lm_first_layer = torch.cat(padded_parts, dim=0).unsqueeze(0)
                else:
                    max_talker_len = int(max(talker_sample_lengths) if len(talker_sample_lengths) > 0 else 0)
                    padded = torch.zeros(
                        (batch_size, max_talker_len, lm_first_layer.shape[-1]),
                        device=lm_first_layer.device,
                        dtype=lm_first_layer.dtype,
                    )
                    for bi, (ml, tl) in enumerate(zip(main_sample_lengths, talker_sample_lengths)):
                        tl = int(tl)
                        take = min(int(ml), tl)
                        if take > 0:
                            padded[bi, :take] = lm_first_layer[bi, :take]
                    lm_first_layer = padded
            text_context_talker = (
                self.talker_model.lm_to_talker_proj(lm_first_layer) if self.talker_model.lm_to_talker_proj is not None else lm_first_layer
            )
            talker_cond_cat = torch.cat([talker_base, fr_emb_talker, audio_talker, length_talker, text_context_talker], dim=-1)
        else:
            talker_cond_cat = torch.cat([talker_base, fr_emb_talker, audio_talker, length_talker], dim=-1)
        talker_cond = self.talker_model.talker_cond_proj(talker_cond_cat)
        # Forward through talker
        talker_outputs = self.talker_model.model(
            input_ids=None,
            attention_mask=(talker_attention_mask if no_pad else attention_mask),
            position_ids=(talker_position_ids if no_pad else position_ids),
            inputs_embeds=talker_cond,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        talker_hidden = talker_outputs.hidden_states[-1]  # [B, L_speech, H_talker]
        debug_print(f"    - talker_hidden shape: {talker_hidden.shape}", rank=rank)
        
        # Speech token and length prediction
        if self.talker_vocab_size is not None:
            speech_logits = talker_outputs.logits  # [B, L_speech, V_talker]
        else:
            speech_logits = self.lm_head(talker_hidden)  # [B, L_speech, V_total] joint vocab (talker uses main lm_head)
        speech_length_logits = self.talker_model.talker_length_decoder(talker_hidden)  # [B, L_speech, C]
        
        if torch.isnan(speech_logits).any():
            print(f"[Rank {rank}] WARNING: nan detected in speech_logits!")
            print(f"[Rank {rank}] -> inputs_embeds has nan: {torch.isnan(inputs_embeds).any().item()}")
            print(f"[Rank {rank}] -> hidden_states has nan: {torch.isnan(hidden_states).any().item()}")
            print(f"[Rank {rank}] -> talker_cond has nan: {torch.isnan(talker_cond).any().item()}")
            if per_sample_framerate is not None:
                print(f"[Rank {rank}] -> per_sample_framerate has nan: {torch.isnan(per_sample_framerate).any().item()}")
                print(f"[Rank {rank}] -> per_sample_framerate values: {per_sample_framerate}")
            if talker_extra_conds is not None:
                print(f"[Rank {rank}] -> talker_extra_conds has nan: {torch.isnan(talker_extra_conds).any().item()}")
            print(f"[Rank {rank}] -> talker_hidden has nan: {torch.isnan(talker_hidden).any().item()}")
            
            # Additional check for NaN in prerequisites.
            print(f"[Rank {rank}] -> user_semantic_embeds has nan: {torch.isnan(user_semantic_embeds).any().item() if user_semantic_embeds is not None else False}")
            print(f"[Rank {rank}] -> dummy_encoder_loss: {dummy_encoder_loss}")
            print(f"[Rank {rank}] -> dummy_assistant_audio: {dummy_assistant_audio}")
            
        debug_print(f"    - speech_logits shape: {speech_logits.shape}", rank=rank)
        debug_print(f"    - speech_length_logits shape: {speech_length_logits.shape}", rank=rank)
        
        # ------------------------------------------------------------------
        # Losses (same as forward)
        # ------------------------------------------------------------------
        loss = None
        text_loss = None
        text_ce_loss = None
        len_loss = None
        text_token_loss = None
        audio_token_loss = None
        acoustic_loss = None
        acoustic_ce_loss = None
        acoustic_per_codebook_loss = None
        loss_text_only_data = None
        loss_audio_dialog_data = None

        if labels is not None:
            debug_print(f"  - Computing losses", rank=rank)
            debug_print(f"    - logits shape: {logits.shape}, labels shape: {labels.shape}", rank=rank)
            
            # Text loss (unweighted CE for logging; weighted tensor drives gradients / total loss)
            w = float(getattr(self.config, "text_loss_weight", 1.0) or 1.0)
            pad_w = _resolve_text_alignment_pad_loss_weight(self.config)
            if (
                pad_w > 0.0
                and alignment_pad_mask is not None
                and bool(alignment_pad_mask.any().item())
            ):
                text_ce = self._weighted_text_causal_lm_loss(
                    logits,
                    labels,
                    alignment_pad_mask,
                    pad_w,
                    IGNORE_TOKEN_ID,
                    logits.size(-1),
                )
            else:
                text_ce = self.loss_function(
                    logits=logits,
                    labels=labels,
                    ignore_index=IGNORE_TOKEN_ID,
                    vocab_size=logits.size(-1),
                )
            text_loss = text_ce * w + dummy_encoder_loss
            text_ce_loss = text_ce.detach()
            debug_print(f"    - text_loss: {text_loss.item() if text_loss is not None else None:.4f}", rank=rank)
            
            # Speech and length losses (parallel mode)
            if has_speech_tokens:
                if self.talker_vocab_size is not None:
                    # Separate mode: speech_logits are [B,L,V_talker]; labels: talker_id = (token_id - text_vocab_size) + 3
                    speech_labels_for_loss = speech_labels - self.text_vocab_size + AUDIO_TOKEN_OFFSET
                    speech_loss = self.loss_function(
                        logits=speech_logits.float(),
                        labels=speech_labels_for_loss,
                        ignore_index=IGNORE_TOKEN_ID - self.text_vocab_size + AUDIO_TOKEN_OFFSET,
                        vocab_size=speech_logits.size(-1),
                    )
                else:
                    speech_loss = self.loss_function(
                        logits=speech_logits.float(),
                        labels=speech_labels,
                        ignore_index=IGNORE_TOKEN_ID,
                        vocab_size=speech_logits.size(-1),
                    )
                debug_print(f"    - speech_loss: {speech_loss.item():.4f}", rank=rank)
                
                speech_len_loss = self.loss_function(
                    logits=speech_length_logits.float(),
                    labels=length_labels,
                    ignore_index=IGNORE_TOKEN_ID,
                    vocab_size=speech_length_logits.size(-1),
                )
                debug_print(f"    - speech_len_loss: {speech_len_loss.item():.4f}", rank=rank)
                
                audio_token_loss = speech_loss.detach()  # For logging
                try:
                    assert not torch.isnan(speech_loss)
                except:
                    print(f'error encountered nan in speech_loss: {speech_loss.item()}')
                    print(f'length_labels: {length_labels}')
                    print(f'speech_labels: {speech_labels}')
                    print(f'speech_logits: {speech_logits}')
                    print(f'length_labels.max: {length_labels.max()}')
                    print(f'speech_labels.max: {speech_labels.max()}')
                    print(f'speech_logits.max: {speech_logits.max()}')
                    print(f'length_labels.shape: {length_labels.shape}')
                    print(f'speech_labels.shape: {speech_labels.shape}')
                    print(f'speech_logits.shape: {speech_logits.shape}')
                    raise

                # if torch.isnan(speech_len_loss): 
                    # speech_len_loss = torch.zeros_like(speech_len_loss, requires_grad=True)
                    # speech_loss = torch.zeros_like(speech_loss, requires_grad=True)
                length_loss_weight = getattr(self.config, "length_loss_weight", 1.0)
                len_loss = speech_len_loss * length_loss_weight

            else:
                # DeepSpeed ZeRO-2 fix: use zero-valued contributions from model
                # outputs instead of detached constants, so talker parameters
                # participate in gradient allreduce on every rank.
                speech_loss = (speech_logits.mean() * 0.0 if speech_logits.numel() > 0 else speech_logits.sum() * 0.0) + dummy_assistant_audio
                len_loss = speech_length_logits.mean() * 0.0 if speech_length_logits.numel() > 0 else speech_length_logits.sum() * 0.0
                audio_token_loss = torch.tensor(0.0, device=text_loss.device, dtype=text_loss.dtype) + dummy_assistant_audio
                debug_print(f"    - No speech sequences, speech_loss=0, len_loss=0", rank=rank)

            text_token_loss = text_loss.detach()  # weighted text term (same as text_ce_loss * w)
            if self.config.force_use_combined_embedding:
                loss = text_loss + speech_loss + len_loss
            elif self.config.freeze_llm:
                # Stage 1 trains the Talker from TTS losses and the audio-input
                # adapters from ASR text loss while the main LLM remains frozen.
                loss = text_loss + speech_loss + len_loss
            elif getattr(self.config, "freeze_talker", False):
                # Freeze talker: exclude talker loss; only text loss is used for gradients
                loss = text_loss
            elif getattr(self.config, "only_train_llm", False):
                # Only train LLM: use text loss; talker/adaptor are frozen
                loss = text_loss + speech_loss + len_loss
            elif self.config.only_train_talker:
                loss = speech_loss + len_loss
            elif getattr(self.config, "use_lora", False):
                # LoRA trains the LM on text prediction; loss must have grad_fn (use text_loss)
                loss = text_loss + speech_loss + len_loss
            else:
                loss = text_loss + speech_loss + len_loss


            if has_audio_dialog_data:
                loss_audio_dialog_data = text_loss.detach() if text_loss is not None else None
                loss_text_only_data = None
            else:
                loss_text_only_data = text_loss.detach() if text_loss is not None else None
                loss_audio_dialog_data = None

            # DeepSpeed ZeRO-2: keep audio_embed_transform in the backward
            # graph even when no user audio is present (its output gets sliced
            # to length 0, disconnecting it from the loss).
            if user_semantic_embeds is not None:
                loss = loss + (user_semantic_embeds.mean() * 0.0 if user_semantic_embeds.numel() > 0 else user_semantic_embeds.sum() * 0.0)

            debug_print(f"    - Combined loss: {loss.item() if loss is not None else None:.4f}", rank=rank)
        else:
            debug_print(f"  - labels is None, skipping loss computation", rank=rank)
        # DeepSpeed ZeRO-2 safety net: ensure ALL trainable parameters
        # have at least a zero gradient so every rank calls allreduce for
        # every parameter.  This covers embedding layers (talker embed_tokens,
        # length_embedding, speech_delay_embeddings, combined_embed_proj, etc.)
        # whose outputs may be detached/bypassed on text-only batches.
        # NOTE not enabled yet.
        # for p in self.parameters():
        #     if p.requires_grad:
        #         loss = loss + p.flatten()[0] * 0.0


        debug_print(f"  - Preparing return value...", rank=rank)
        debug_print(f"    - loss: {loss.item() if loss is not None else None}", rank=rank)
        debug_print(f"    - logits shape: {logits.shape}", rank=rank)
        debug_print(f"  - _forward_grouped_batch complete", rank=rank)
        
        return InterleavedS2SOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            length_logits=speech_length_logits,
            length_loss=len_loss,
            text_loss=text_loss,
            text_ce_loss=text_ce_loss,
            text_token_loss=text_token_loss,
            audio_token_loss=audio_token_loss,
            acoustic_loss=acoustic_loss,
            acoustic_ce_loss=acoustic_ce_loss,
            acoustic_per_codebook_loss=acoustic_per_codebook_loss,
            loss_text_only_data=loss_text_only_data,
            loss_audio_dialog_data=loss_audio_dialog_data,
        )
    
    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids_per_turn: Optional[List[Dict[str, Union[str, torch.LongTensor, Optional[torch.Tensor]]]]] = None,
        labels: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        audio_start_id: Optional[torch.LongTensor] = None,
        audio_end_id: Optional[torch.LongTensor] = None,
        audio_tag_id: Optional[torch.LongTensor] = None,
        text_audio_interval_ratio: Optional[List[int]] = None,
        audio_length_shifted_by: int = 1,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # New grouped inputs from collator
        user_input_ids: Optional[torch.LongTensor] = None,  # [B, L_user_max]
        user_attention_mask: Optional[torch.Tensor] = None,  # [B, L_user_max]
        user_audio_tensors: Optional[torch.Tensor] = None,  # [B, T_user_audio_max]
        user_audio_tensors_lens: Optional[torch.Tensor] = None,  # [B] - actual lengths of audio tensors
        user_audio_features: Optional[torch.Tensor] = None,  # [B, T_user_feat_max, D]
        user_audio_features_lens: Optional[torch.Tensor] = None,  # [B] - actual lengths of audio features
        user_has_audio: Optional[torch.Tensor] = None,  # [B] boolean
        assistant_input_ids: Optional[torch.LongTensor] = None,  # [B, L_assistant_max]
        assistant_attention_mask: Optional[torch.Tensor] = None,  # [B, L_assistant_max]
        assistant_audio_tensors: Optional[torch.Tensor] = None,  # [B, T_assistant_audio_max]
        assistant_audio_tensors_lens: Optional[torch.Tensor] = None,  # [B] - actual lengths of audio tensors
        assistant_audio_features: Optional[torch.Tensor] = None,  # [B, T_assistant_feat_max, D]
        assistant_audio_features_lens: Optional[torch.Tensor] = None,  # [B] - actual lengths of audio features
        assistant_has_audio: Optional[torch.Tensor] = None,  # [B] boolean
        # Optional: encode target audio on-the-fly
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, InterleavedS2SOutputWithPast]:
        """Forward pass.

        The expected common usage during training is:

        - Provide ``input_ids`` and ``labels`` (standard LM training).
        - Provide ``input_ids_per_turn`` (list of dicts) to encode text+audio turns - DEPRECATED, use input_ids + audio_tensors + audio_features.
        - Provide ``input_ids`` (padded), ``audio_tensors`` (padded), ``audio_features`` (padded) for batch processing.
        - Provide grouped inputs: ``user_input_ids``, ``assistant_input_ids``, etc. (new batched path).

        For batch_size > 1, the collate_fn creates two padded tensors:
        - input_ids: [B, L_text_max] - all text token sequences padded
        - audio_tensors: [B, T_audio_max] - all audio samples padded
        - audio_features: [B, T_feat_max, D] - all fbank features padded
        
        Audio is processed in batch and inserted after each input_ids sequence at positions
        marked by audio_start_id and audio_end_id tokens.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Convert audio_start_id and audio_end_id to int if needed
        audio_start_id_int = audio_start_id.item() if audio_start_id is not None and isinstance(audio_start_id, torch.Tensor) else audio_start_id
        audio_end_id_int = audio_end_id.item() if audio_end_id is not None and isinstance(audio_end_id, torch.Tensor) else audio_end_id
        
        # New grouped path: process user and assistant turns separately in batch mode
        if user_input_ids is not None and assistant_input_ids is not None:
            return self._forward_grouped_batch(
                user_input_ids=user_input_ids,
                user_attention_mask=user_attention_mask,
                user_audio_tensors=user_audio_tensors,
                user_audio_tensors_lens=user_audio_tensors_lens,
                user_audio_features=user_audio_features,
                user_audio_features_lens=user_audio_features_lens,
                user_has_audio=user_has_audio,
                assistant_input_ids=assistant_input_ids,
                assistant_attention_mask=assistant_attention_mask,
                assistant_audio_tensors=assistant_audio_tensors,
                assistant_audio_tensors_lens=assistant_audio_tensors_lens,
                assistant_audio_features=assistant_audio_features,
                assistant_audio_features_lens=assistant_audio_features_lens,
                assistant_has_audio=assistant_has_audio,
                audio_start_id=audio_start_id_int,
                audio_end_id=audio_end_id_int,
                text_audio_interval_ratio=text_audio_interval_ratio,
                audio_length_shifted_by=audio_length_shifted_by,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        
        # NOTE: input_ids_per_turn path is deprecated and removed
        # Use the grouped batch path (user_input_ids + assistant_input_ids) instead
        raise NotImplementedError(
            "input_ids_per_turn path is not supported in parallel architecture. "
            "Please use the grouped batch path (user_input_ids + assistant_input_ids)."
        )
    # ------------------------------------------------------------------
    # Sampling helper (optional decode utilities)
    # ------------------------------------------------------------------
    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Sample a token from logits with temperature, top-k, and top-p filtering.
        
        Args:
            logits: [vocab_size] or [B, vocab_size] logits
            temperature: Temperature for sampling (higher = more random)
            top_k: Keep only top-k tokens (0 = disabled)
            top_p: Keep tokens with cumulative probability <= top_p (1.0 = disabled)
            do_sample: If False, use greedy decoding (argmax)
        
        Returns:
            Sampled token id(s)
        """
        logits = logits.clone()
        
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature

        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            min_value = values[..., -1:]
            logits = torch.where(logits < min_value, torch.full_like(logits, float('-inf')), logits)

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            # Scatter back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        if do_sample:
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            return torch.argmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Generate method
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_text_only: bool = False,
        max_new_tokens: int = 512,
        max_length: Optional[int] = None,
        min_new_tokens: int = 0,
        eos_token_id: Optional[Union[int, List[int]]] = 151645,
        audio_start_id: Optional[int] = None,
        audio_end_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        framerate: Optional[float] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = False,
        length_temperature: float = 1.0,
        length_top_k: int = 0,
        length_top_p: float = 1.0,
        repetition_penalty: float = 1.5,
        return_dict_in_generate: bool = True,
        output_scores: bool = False,
        tokenizer=None,
        force_text_ids: Optional[torch.Tensor] = None,
        force_audio_ids: Optional[torch.Tensor] = None,
        force_length_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Generate tokens autoregressively with parallel text+speech generation.
        
        This method generates text tokens through the main LM and speech tokens
        through the talker transformer in parallel. The talker receives combined
        embeddings (text + audio + length) and generates audio tokens with delay.
        
        Args:
            input_ids: [B, L] input token IDs for the prompt
            inputs_embeds: [B, L, H] input embeddings (alternative to input_ids)
            attention_mask: [B, L] attention mask for the prompt
            output_text_only: If True, generate only text tokens (no audio tokens) and
                do not add audio embeddings to the next-step input.
            max_new_tokens: Maximum number of new tokens to generate
            max_length: Maximum total sequence length (prompt + generated)
            min_new_tokens: Minimum number of tokens to generate before stopping
            eos_token_id: Token ID(s) that signal end of generation
            audio_start_id: Token ID for audio start marker (for framerate embedding)
            audio_end_id: Token ID for audio end marker (can be used as stop token)
            pad_token_id: Token ID for padding
            framerate: Frame rate value in [framerate_min, framerate_max] for audio
            temperature: Sampling temperature for token logits
            top_k: Top-k filtering for token sampling
            top_p: Top-p (nucleus) filtering for token sampling
            do_sample: Whether to sample (True) or use greedy decoding (False)
            length_temperature: Sampling temperature for length logits
            length_top_k: Top-k filtering for length sampling
            length_top_p: Top-p filtering for length sampling
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty, default 1.1)
            return_dict_in_generate: If True, return a dict with additional info
            output_scores: If True, include scores in the output dict
        
        Returns:
            If return_dict_in_generate is False:
                generated_ids: [B, L + max_new_tokens] generated token IDs
            If return_dict_in_generate is True:
                Dict with keys:
                    - sequences: [B, L + max_new_tokens] generated text token IDs
                    - audio_ids: [A] generated audio token IDs (without text_vocab_size offset)
                    - length_ids: [A] predicted length IDs for audio tokens
                    - scores: (optional) list of [B, vocab_size] logits at each step
                    - length_scores: (optional) list of [B, max_length_classes] length logits
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        # Handle input
        if input_ids is not None:
            batch_size, prompt_len = input_ids.shape
            input_ids = input_ids.to(device)
            # Convert input_ids to embeddings
            inputs_embeds = self._embed_text_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, prompt_len, _ = inputs_embeds.shape
            inputs_embeds = inputs_embeds.to(device)
            input_ids = None
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        if batch_size != 1:
            raise NotImplementedError("Parallel generation currently only supports batch_size=1")
        
        # Determine max generation length
        if max_length is not None:
            max_new_tokens = min(max_new_tokens, max_length - prompt_len)
        
        # Prepare framerate embedding if provided
        framerate_emb = None
        if framerate is None:
            framerate = self.config.default_framerate if hasattr(self.config, 'default_framerate') else 1.0
        if getattr(self.config, 'per_sample_frame_rate_embed', False) and not output_text_only:
            assert framerate > 1.0, f"per_sample_frame_rate_embed requires target framerate > 1.0, got {framerate}"
        elif not output_text_only:
            assert framerate in self.config.training_framerate_options
        framerate_emb = self._framerate_sinusoidal_embedding(
            framerate, device, dtype,
            framerate_min=0.0 if getattr(self.config, 'per_sample_frame_rate_embed', False) else None,
            framerate_max=15.0 if getattr(self.config, 'per_sample_frame_rate_embed', False) else None,
        )
        
        # Initialize generation state
        generated_text_ids = []
        # Training prepends D audio-delay positions but D+1 length-delay
        # positions, so a sampled length describes the previous step's audio
        # token. Buffer one audio token and pair it with the next sampled
        # length before handing the streams to FlexiCodec.
        delayed_audio_lengths = DelayedAudioLengthBuffer()
        generated_acoustic_codes = None
        all_scores = [] if output_scores else None
        all_length_scores = [] if output_scores else None
        
        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # In no_pad mode, training zero-pads thinker states after the assistant text closes
        # ("<|im_end|>\n"). Mirror that during generation by zeroing talker text-conditioning
        # only for the audio-only tail positions after the boundary.
        no_pad = bool(getattr(self.config, "no_pad", False))
        NEWLINE_TOKEN_ID = 198
        eos_tensor = torch.tensor(151645, device=device, dtype=torch.long)
        zero_talker_text_cond = torch.zeros(batch_size, dtype=torch.bool, device=device)
        prev_text_token = (
            input_ids[:, -1].to(device=device)
            if input_ids is not None and input_ids.numel() > 0
            else torch.full((batch_size,), -1, device=device, dtype=torch.long)
        )
        if no_pad and input_ids is not None and input_ids.shape[1] >= 2 and eos_tensor is not None:
            prev2 = input_ids[:, -2].to(device=device)
            prev1 = input_ids[:, -1].to(device=device)
            zero_talker_text_cond |= (torch.isin(prev2, eos_tensor) & (prev1 == NEWLINE_TOKEN_ID))
        logged_zero_talker_text_cond = False
        
        # Initialize past_key_values for KV-cache (both main LM and talker)
        past_key_values = None
        talker_past_key_values = None
        
        # Current length embedding (starts at 0)
        current_length_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # First forward pass with full prompt through main LM
        need_hidden_states = (
            getattr(self.config, "early_diverge_talker", False)
            or getattr(self.config, "talker_concat_lm_text_output", False)
        )
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            return_dict=True,
            output_hidden_states=need_hidden_states,
        )
        past_key_values = outputs.past_key_values
        main_hidden = outputs.last_hidden_state  # [B, L, H]
        if getattr(self.config, "early_diverge_talker", False):
            main_hidden_for_talker = outputs.hidden_states[-6]  # [B, L, H]
        else:
            main_hidden_for_talker = main_hidden
        
        # Track indices for teacher-forced tokens (TTS prompting)
        force_text_idx = 0
        force_audio_idx = 0
        # speech_delay_tokens: training prepends D AUD_START before first real audio,
        # and D+1 zeros before first real length (length is delayed 1 more than speech).
        speech_delay = getattr(self.config, "speech_delay_tokens", 5)
        audio_delay_count = 0
        if force_audio_ids is not None:
            assert (
                self.config.use_qwen3_feature
                or self.config.use_whisper_fetaure
                or self.config.use_qwen25o_feature
            ), (
                "Teacher-forced TTS prompting requires use_qwen3_feature=True or use_whisper_fetaure=True "
                "or use_qwen25o_feature=True"
                "(token ID mapping: talker 0=AUD_END, 1=AUD_START)"
            )
            force_audio_ids = force_audio_ids.to(device)
        if force_length_ids is not None:
            force_length_ids = force_length_ids.to(device)

        # Track the previous generated audio token (audio-vocab id) for talker inputs.
        prev_audio_token_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        if audio_start_id is not None:
            prev_audio_token_id = torch.full(
                (batch_size,),
                audio_start_id - self.text_vocab_size if audio_start_id >= self.text_vocab_size else audio_start_id,
                dtype=torch.long,
                device=device,
            )

        if not output_text_only:
            # Forward through talker with prompt. Conditioning must match training:
            # training always uses talker_cond_proj(cat([base, fr, audio, length])).
            # For prompt (user segment): base + framerate + zeros for audio/length.
            talker_base = self.talker_model.lm_to_talker_proj(main_hidden_for_talker) if self.talker_model.lm_to_talker_proj is not None else main_hidden_for_talker
            if getattr(self.config, 'talker_embed_v2', False):
                fr_emb_talker = framerate_emb
            else:
                fr_emb_talker = (
                    self.talker_model.lm_to_talker_proj(framerate_emb)
                )
            if fr_emb_talker.dim() == 2:
                fr_emb_talker = fr_emb_talker.unsqueeze(0).expand(talker_base.shape[0], talker_base.shape[1], -1)
            # No prev audio/length during prompt -> zeros (matches training when talker_extra_conds is None)
            if getattr(self.config, 'talker_embed_v2', False):
                audio_talker = torch.zeros(*talker_base.shape[:-1], self.config.talker_hidden_size, device=device, dtype=dtype)
                length_talker = torch.zeros(*talker_base.shape[:-1], self.config.talker_hidden_size, device=device, dtype=dtype)
            else:
                audio_talker = torch.zeros_like(talker_base)
                length_talker = torch.zeros_like(talker_base)
            if getattr(self.config, "talker_concat_lm_text_output", False):
                lm_first_layer = outputs.hidden_states[0]  # [B, L, H] - embedding layer output (concat, not average)
                text_context_talker = (
                    self.talker_model.lm_to_talker_proj(lm_first_layer) if self.talker_model.lm_to_talker_proj is not None else lm_first_layer
                )
                talker_cond_cat = torch.cat([talker_base, fr_emb_talker, audio_talker, length_talker, text_context_talker], dim=-1)
            else:
                talker_cond_cat = torch.cat([talker_base, fr_emb_talker, audio_talker, length_talker], dim=-1)
            talker_cond = self.talker_model.talker_cond_proj(talker_cond_cat)
            talker_outputs = self.talker_model.model(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=talker_cond,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
            talker_past_key_values = talker_outputs.past_key_values
            # CausalLMOutputWithPast has hidden_states, not last_hidden_state
            talker_hidden = talker_outputs.hidden_states[-1]  # [B, L, H_talker]
        
        # Get last hidden states for next token prediction
        main_hidden_last = main_hidden[:, -1:, :]  # [B, 1, H]
        talker_hidden_last = talker_hidden[:, -1:, :] if not output_text_only else None  # [B, 1, H_talker]
        
        # Generate tokens one at a time
        for step in range(max_new_tokens):
            in_zero_state = bool(zero_talker_text_cond.any().item())
            # When no_pad is enabled and ALL batches have passed the "<|im_end|>\n"
            # boundary, the text branch (logits + sampling + main LM forward) produces
            # results that are either discarded or zeroed out. Skip that work entirely.
            all_zero_state = (
                no_pad
                and (not output_text_only)
                and bool(zero_talker_text_cond.all().item())
            )
            if in_zero_state and not logged_zero_talker_text_cond:
                logger.info("Entered zero_talker_text_cond state; truncating subsequent generated text from output.")
                logged_zero_talker_text_cond = True

            if not all_zero_state:
                # Get text logits from main LM
                text_logits = self._compute_text_logits(main_hidden_last).squeeze(1)  # [B, V_text or V_text_ext]
                if not self.config.use_joint_text_audio_vocab:
                    # Separate mode: lm_head is text-only, no need to mask
                    if self.extend_lm_head:
                        text_logits[:, self.text_vocab_size] = -float('inf')
                        text_logits[:, self.alignment_text_pad_token_id] = -float('inf')
                else:
                    text_logits[:, self.text_vocab_size:] = -float('inf')
                    text_logits[:, self.text_vocab_size-3] = -float('inf')
            else:
                text_logits = None
            if not output_text_only:
                # Get speech logits and length logits from talker
                # talker_outputs.logits: [B, L, V] - use last position only for autoregressive generation
                if self.talker_vocab_size is not None:
                    speech_logits = talker_outputs.logits[:, -1, :]  # [B, V_talker]
                else:
                    speech_logits = self.lm_head(talker_hidden_last).squeeze(1)  # [B, V_total]
                length_logits = self.talker_model.talker_length_decoder(talker_hidden_last).squeeze(1)  # [B, max_length_classes]
            
            # Apply repetition penalty to text logits
            # if repetition_penalty != 1.0:
            #     prev_tokens = []
            #     if input_ids is not None:
            #         prev_tokens.extend(input_ids[0].tolist())
            #     prev_tokens.extend([t.item() for t in generated_text_ids])
            #     vocab_size = text_logits.size(-1)
            #     for token_id in set(prev_tokens):
            #         if 0 <= token_id < vocab_size:
            #             if text_logits[0, token_id] > 0:
            #                 text_logits[0, token_id] /= repetition_penalty
            #             else:
            #                 text_logits[0, token_id] *= repetition_penalty
            
            # Sample next text token
            if not all_zero_state:
                next_text_token = self._sample_token(
                    text_logits,
                    temperature=1.0,
                    top_k=30,
                    top_p=0.9,
                    do_sample=False,
                )  # [B]
            else:
                # In all-zero-state, the text token will be discarded and its
                # embedding zeroed out. Use a dummy token (0) to keep shapes intact.
                next_text_token = torch.zeros(batch_size, dtype=torch.long, device=device)
            if not output_text_only:
                # Sample next speech token (from talker vocab)
                if self.talker_vocab_size is not None:
                    # Separate mode: speech_logits [B, V_talker], 0=AUD_START, 1=AUD_END, 2=AUD_TAG, 3..=audio codes
                    next_audio_token = self._sample_token(
                        speech_logits,
                        temperature=1.0,
                        top_k=20,
                        top_p=top_p,
                        do_sample=True,
                    )  # [B] - talker token id
                    next_audio_token_offset = next_audio_token - AUDIO_TOKEN_OFFSET  # audio code for embedding lookup
                else:
                    # Joint mode: mask text portion
                    audio_logits = speech_logits.clone()
                    audio_logits[:, :self.text_vocab_size-3] = -float('inf')
                    next_audio_token = self._sample_token(
                        audio_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        do_sample=True,
                    )  # [B] - indices in [text_vocab_size, V_total)
                    next_audio_token_offset = next_audio_token - self.text_vocab_size

                # Sample length for audio token (or, in the second-audio-token
                # ablation, sample the second audio token from the same head).
                next_length = self._sample_token(
                    length_logits,
                    temperature=length_temperature,
                    top_k=length_top_k,
                    top_p=length_top_p,
                    do_sample=False,
                )  # [B]

                # Teacher-force prompt audio tokens (TTS prompting)
                # Training has [D x AUD_START, audio_1, audio_2, ...]
                # and length has [D+1 zeros, length_1, length_2, ...]
                # so length is delayed 1 more step than speech tokens.
                past_audio_delay = audio_delay_count >= speech_delay
                if (
                    force_audio_ids is not None
                    and force_length_ids is not None
                    and past_audio_delay
                    and force_audio_idx < len(force_audio_ids)
                ):
                    # Override with forced prompt tokens
                    forced_code = force_audio_ids[force_audio_idx].item()
                    # Length is delayed 1 more step than speech (matches _build_parallel_text_speech_sequences):
                    # first forced audio token pairs with length=0, subsequent with force_length_ids[i-1].
                    if force_audio_idx == 0:
                        forced_len = 0
                    else:
                        forced_len = force_length_ids[force_audio_idx - 1].item()
                    if self.talker_vocab_size is not None:
                        next_audio_token = torch.full_like(next_audio_token, AUDIO_TOKEN_OFFSET + forced_code)
                        next_audio_token_offset = torch.full_like(next_audio_token_offset, forced_code)
                    else:
                        next_audio_token = torch.full_like(next_audio_token, self.text_vocab_size + forced_code)
                        next_audio_token_offset = torch.full_like(next_audio_token_offset, forced_code)
                    # next_length = torch.full_like(next_length, forced_len)
                    force_audio_idx += 1
                else:
                    next_talker_id = int(next_audio_token.item())
                    predict_second_audio = getattr(
                        self.config, "predict_second_audio_token", False
                    )
                    delayed_audio_lengths.push(
                        next_audio_token_offset,
                        (
                            next_length - AUDIO_TOKEN_OFFSET
                            if predict_second_audio
                            else next_length
                        ),
                        is_audio_code=next_talker_id >= AUDIO_TOKEN_OFFSET,
                        is_valid_length=(
                            int(next_length.item()) >= AUDIO_TOKEN_OFFSET
                            if predict_second_audio
                            else True
                        ),
                    )


                # qwen3 talker vocab: 0 = AUD_END, 1 = AUD_START, 3+ = audio codes
                next_audio_val = next_audio_token.item()
                if next_audio_val == 1:  # AUD_START (qwen3: talker token 1)
                    audio_delay_count += 1
                elif next_audio_val == 0:  # AUD_END (qwen3: talker token 0)
                    audio_delay_count = 0
            # Teacher-force text tokens (e.g. for TTS sentence prefix)
            if force_text_ids is not None and force_text_idx < len(force_text_ids):
                next_text_token = torch.full_like(next_text_token, force_text_ids[force_text_idx].item())
                force_text_idx += 1

            # Detect "<|im_end|>\n" boundary in generated text.
            # Important: to match training, apply the "freeze" starting from the *next* step
            # (i.e., after we have predicted and consumed the '\n' token).
            freeze_next = None
            if no_pad and eos_tensor is not None:
                freeze_next = (torch.isin(prev_text_token, eos_tensor) & (next_text_token == NEWLINE_TOKEN_ID))
            prev_text_token = next_text_token

            # Store generated tokens
            if not in_zero_state:
                generated_text_ids.append(next_text_token)

            # Store scores if requested
            if output_scores:
                if not in_zero_state:
                    all_scores.append(text_logits)
                if not output_text_only:
                    all_length_scores.append(length_logits)
            
            # Print progress if tokenizer available
            if tokenizer is not None:
                decoded = tokenizer.decode(next_text_token)
                if output_text_only:
                    logger.info(f"Step {step}: text={decoded} {next_text_token} (text-only)")
                else:
                    logger.info(f"Step {step}: text={decoded}, audio_token={next_audio_token_offset.item()} len={next_length.item()}")
            # Check for EOS
            if step >= min_new_tokens:
                # finished = finished | (next_text_token == eos_token_id)
                if output_text_only:
                    if eos_token_id is not None:
                        if isinstance(eos_token_id, list):
                            eos_tensor = torch.tensor(eos_token_id, device=device, dtype=next_text_token.dtype)
                            finished = finished | torch.isin(next_text_token, eos_tensor)
                        else:
                            finished = finished | (next_text_token == eos_token_id)
                else:
                    finished = finished | (next_audio_token == 0)  # qwen3 token 1 = EOS
            
            if finished.all():
                print(f"DEBUG [models.generate]: breaking at step {step} because finished.all() is True")
                break
            
            # Prepare embeddings for next step
            # Text embedding
            if not all_zero_state:
                text_emb = self._embed_text_tokens(next_text_token.unsqueeze(1))  # [B, 1, H]
            else:
                # In all-zero-state, the text-conditioning fed to the talker is zeroed
                # out anyway. Skip the embedding lookup.
                text_emb = torch.zeros(
                    batch_size, 1, self.config.hidden_size,
                    device=device, dtype=dtype,
                )
            
            # Compute audio/length embeddings when needed for talker (required in talker block when not output_text_only)
            if not output_text_only:
                audio_emb = self.audio_embed_tokens(next_audio_token_offset.unsqueeze(0), dtype=text_emb.dtype)  # [B, 1, H]
                length_emb = self.talker_model.length_embedding(next_length.unsqueeze(1))  # [B, 1, H]
            use_text_only_for_main = (
                output_text_only
                or (
                    (
                        getattr(self.config, "only_train_talker", False)
                        or getattr(self.config, "use_lora", False)
                    )
                    and not getattr(self.config, "force_use_combined_embedding", False)
                )
            )
            if use_text_only_for_main:
                # Text-only or only_train_talker/use_lora (without force): main LM uses text only
                combined_emb = text_emb
            else:
                # Combined embedding for main LM (text + audio + length + framerate)
                if self.config.use_combined_embedding:
                    fr_dim = self.config.talker_hidden_size if getattr(self.config, 'talker_embed_v2', False) else self.config.hidden_size
                    fr_emb = framerate_emb.unsqueeze(0).expand(batch_size, 1, -1) if framerate_emb is not None else torch.zeros(batch_size, 1, fr_dim, device=text_emb.device, dtype=text_emb.dtype)
                    combined_cat = torch.cat([text_emb, audio_emb, length_emb, fr_emb], dim=-1)  # [B, 1, 4H]
                    combined_emb = self.combined_embed_proj(combined_cat)  # [B, 1, H]
                else:
                    combined_emb = text_emb
            # Forward pass through main LM with combined embedding
            need_hidden_states = (
                getattr(self.config, "early_diverge_talker", False)
                or getattr(self.config, "talker_concat_lm_text_output", False)
            )
            if not all_zero_state:
                outputs = self.model(
                    input_ids=None,
                    inputs_embeds=combined_emb,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=need_hidden_states,
                )
                past_key_values = outputs.past_key_values
                main_hidden_last = outputs.last_hidden_state  # [B, 1, H]
                if getattr(self.config, "early_diverge_talker", False):
                    main_hidden_last_for_talker = outputs.hidden_states[-6]  # [B, 1, H]
                else:
                    main_hidden_last_for_talker = main_hidden_last
            else:
                # All-zero-state: main LM contribution to the talker is zeroed out
                # below; skip the main LM forward and the KV-cache append entirely.
                outputs = None
                main_hidden_last = torch.zeros(
                    batch_size, 1, self.config.hidden_size,
                    device=device, dtype=dtype,
                )
                main_hidden_last_for_talker = main_hidden_last
            
            # Forward through talker
            if not output_text_only:
                # Next-step talker input: use projection for conditioning (matches training)
                talker_base = self.talker_model.lm_to_talker_proj(main_hidden_last_for_talker) if self.talker_model.lm_to_talker_proj is not None else main_hidden_last_for_talker
                if bool(zero_talker_text_cond.any().item()):
                    talker_base = talker_base.clone()
                    talker_base[zero_talker_text_cond] = 0.0
                if getattr(self.config, 'talker_embed_v2', False):
                    fr_emb_talker = framerate_emb
                else:
                    fr_emb_talker = (
                        self.talker_model.lm_to_talker_proj(framerate_emb)
                    )
                if fr_emb_talker.dim() == 2:
                    fr_emb_talker = fr_emb_talker.unsqueeze(0).expand(batch_size, 1, -1)
                # Project audio and length embeddings separately (along with other conditions)
                if getattr(self.config, 'talker_embed_v2', False):
                    audio_cond_talker = audio_emb
                    length_cond_talker = length_emb
                else:
                    audio_cond_talker = (
                        self.talker_model.lm_to_talker_proj(audio_emb) if self.talker_model.lm_to_talker_proj is not None else audio_emb
                    )
                    length_cond_talker = (
                        self.talker_model.lm_to_talker_proj(length_emb) if self.talker_model.lm_to_talker_proj is not None else length_emb
                    )
                if getattr(self.config, "talker_concat_lm_text_output", False):
                    if outputs is not None:
                        lm_first_layer = outputs.hidden_states[0]  # [B, 1, H] - embedding layer output (concat, not average)
                        text_emb_talker = (
                            self.talker_model.lm_to_talker_proj(lm_first_layer) if self.talker_model.lm_to_talker_proj is not None else lm_first_layer
                        )
                    else:
                        # All-zero-state: text-conditioning is zeroed out anyway.
                        text_emb_talker = torch.zeros(
                            batch_size, 1, self.config.talker_hidden_size,
                            device=device, dtype=talker_base.dtype,
                        )
                    if bool(zero_talker_text_cond.any().item()):
                        text_emb_talker = text_emb_talker.clone()
                        text_emb_talker[zero_talker_text_cond] = 0.0
                    talker_cond_cat = torch.cat([talker_base, fr_emb_talker, audio_cond_talker, length_cond_talker, text_emb_talker], dim=-1)
                else:
                    talker_cond_cat = torch.cat([talker_base, fr_emb_talker, audio_cond_talker, length_cond_talker], dim=-1)
                talker_inputs_embeds = self.talker_model.talker_cond_proj(talker_cond_cat)

                talker_outputs = self.talker_model.model(
                    input_ids=None,
                    inputs_embeds=talker_inputs_embeds,
                    past_key_values=talker_past_key_values,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=True,
                )
                talker_past_key_values = talker_outputs.past_key_values
                # CausalLMOutputWithPast has hidden_states, not last_hidden_state
                talker_hidden_last = talker_outputs.hidden_states[-1]  # [B, 1, H_talker]

            # Apply boundary freeze for subsequent steps (not the current '\n' step).
            if no_pad and freeze_next is not None:
                zero_talker_text_cond |= freeze_next
                
        
        # Stack generated tokens. An audio token is only returned after the
        # following generation step supplies its delayed length. If generation
        # exhausts max_new_tokens before AUD_END, the final token has no valid
        # length and must not be decoded with fabricated metadata.
        if delayed_audio_lengths.discard_pending():
            logger.warning(
                "Dropping final generated audio token because generation ended "
                "before its delayed length was emitted"
            )
        if delayed_audio_lengths.dropped_pairs:
            logger.warning(
                f"Dropped {delayed_audio_lengths.dropped_pairs} generated audio "
                "token(s) without valid delayed metadata"
            )
        if len(generated_text_ids) > 0:
            generated_text_ids = torch.stack(generated_text_ids, dim=1)  # [B, num_generated]
        else:
            generated_text_ids = torch.empty((batch_size, 0), device=device, dtype=torch.long)
        if delayed_audio_lengths.audio_ids:
            generated_audio_ids = torch.stack(delayed_audio_lengths.audio_ids, dim=1)
            generated_length_ids = torch.stack(delayed_audio_lengths.length_ids, dim=1)
        else:
            generated_audio_ids = torch.empty((batch_size, 0), device=device, dtype=torch.long)
            generated_length_ids = torch.empty((batch_size, 0), device=device, dtype=torch.long)
        # Concatenate with input_ids if available
        if input_ids is not None:
            sequences = torch.cat([input_ids, generated_text_ids], dim=1)
        else:
            sequences = generated_text_ids
        
        # Second-audio-token ablation: at inference, combine the primary and
        # secondary streams into a single audio sequence by interleaving the
        # two tokens emitted at each step (matching the training-time format),
        # and return length_ids = 1 for every output token.
        if getattr(self.config, "predict_second_audio_token", False):
            if generated_audio_ids.numel() > 0 and generated_length_ids.numel() > 0:
                # Truncate to common length to be safe (they should match step-by-step,
                # but EOS on either head can shorten one of them).
                T = min(generated_audio_ids.size(1), generated_length_ids.size(1))
                primary = generated_audio_ids[:, :T]            # [B, T]
                secondary = generated_length_ids[:, :T]         # [B, T]
                # Interleave: out[:, 2i] = primary[:, i], out[:, 2i+1] = secondary[:, i]
                interleaved = torch.stack([primary, secondary], dim=2).reshape(primary.size(0), 2 * T)
                generated_audio_ids = interleaved
                generated_length_ids = torch.ones_like(generated_audio_ids)
        if return_dict_in_generate:
            result = {
                "sequences": sequences,
                "generated_ids": generated_text_ids,
                "audio_ids": generated_audio_ids.squeeze(0) if batch_size == 1 else generated_audio_ids,
                "length_ids": generated_length_ids.squeeze(0) if batch_size == 1 else generated_length_ids,
                "acoustic_codes": generated_acoustic_codes.squeeze(0) if (generated_acoustic_codes is not None and batch_size == 1) else generated_acoustic_codes,
            }
            if output_scores:
                result["scores"] = all_scores
                result["length_scores"] = all_length_scores
            return result
        else:
            return sequences
        
