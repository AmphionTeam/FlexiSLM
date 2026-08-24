# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from typing import Optional


import transformers


from transformers.utils.versions import require_version
import logging
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default='Qwen/Qwen2.5-7B-Instruct',
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default='qwen2',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    use_joint_text_audio_vocab: bool = field(
        default=False,
        metadata={"help": "Whether to use joint text and audio vocab"},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default='Qwen/Qwen2.5-0.5B-Instruct', metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default='Qwen/Qwen2.5-0.5B-Instruct', metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default='cache/',
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    add_length_embeddings: bool = field(default=True, metadata={"help": ""})

    no_pad: bool = field(
        default=False,
        metadata={"help": "Use the no-padding text/audio alignment path."},
    )

    attn_implementation: Optional[str] = field(default="flash_attention_2", metadata={"help": ""})

    audio_tokenizer_path: str = field(default="THUDM/glm-4-voice-tokenizer", metadata={"help": ""})
    audio_tokenizer_type: str = field(default="glm4voice", metadata={"help": ""})
    text_audio_interval_ratio: list[int] = field(default=None, metadata={"help": ""})
    audio_model_freeze: bool = field(default=False, metadata={"help": ""})
    adaptor_input_dim: int = field(default=5120, metadata={"help": "Input dimension of the audio adaptor"})

    vision_model_name_or_path: str = field(default=None, metadata={"help": ""})
    vision_model_type: Optional[str] = field(default=None, metadata={"help": ""})
    vision_model_freeze: bool = field(default=False, metadata={"help": ""})

    language_model_freeze: bool = field(default=False, metadata={"help": ""})
    freeze_llm: bool = field(default=False, metadata={"help": "Freeze the main LLM backbone and text LM head."})
    freeze_adaptor: bool = field(default=False, metadata={"help": "Freeze adaptor and only train aligner components (length_decoder, length_embedding)"})
    only_train_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated top-level model module/parameter names. When set, freeze every parameter "
                "and then unfreeze only the listed components. The virtual component 'llm_lora' selects "
                "only PEFT adapter parameters inside the wrapped LLM without unfreezing the backbone."
            )
        },
    )
    only_train_talker: bool = field(
        default=False,
        metadata={
            "help": (
                "Freeze all components except the speech talker branch. "
                "When enabled, training_interleave will freeze the main LLM/audio adaptor/etc and only unfreeze "
                "the talker modules (and any required heads)."
            )
        },
    )
    only_train_llm: bool = field(
        default=False,
        metadata={
            "help": (
                "Freeze all components except the main LLM path. With LoRA, only LLM LoRA adapters are "
                "trainable. The lm_head, audio_embed_transform, and input_merging_transformer remain trainable; "
                "talker and length components are frozen."
            )
        },
    )
    early_diverge_talker: bool = field(
        default=False,
        metadata={
            "help": (
                "When enabled, the talker's input hidden state is the -6th layer of the LLM instead of the last layer."
            )
        },
    )
    freeze_talker: bool = field(
        default=False,
        metadata={
            "help": (
                "When enabled, freeze the talker modules and exclude talker loss (speech_loss, len_loss) from training. "
                "Only text loss is used. Useful for ASR-only or LLM-only fine-tuning."
            )
        },
    )
    # Talker transformer architecture parameters (parallel S2S model)
    talker_hidden_size: Optional[int] = field(
        default=None,
        metadata={"help": "Hidden size for talker transformer. Default: same as LLM hidden_size (e.g. 896 for 0.5B)"}
    )
    talker_num_layers: int = field(
        default=20,
        metadata={"help": "Number of transformer layers in the talker module"}
    )
    talker_num_attention_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads in the talker transformer"}
    )
    talker_intermediate_size: Optional[int] = field(
        default=None,
        metadata={"help": "Intermediate size for talker FFN. Default: talker_hidden_size * 4"}
    )
    talker_pretrained_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Enable pretrained talker initialization: set to a Qwen2.5-0.5B checkpoint path (local directory "
                'or Hugging Face hub id, e.g. "Qwen/Qwen2.5-0.5B"). '
                "All transformer blocks (attention, MLP, layer norms) load from the checkpoint. "
                "embed_tokens and lm_head are reinitialized for the talker vocabulary because they are vocab-dependent."
            )
        },
    )
    talker_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to a FlexiSLM checkpoint directory whose complete talker_model state initializes the Talker. "
                "The configured Talker architecture must exactly match the checkpoint."
            )
        },
    )
    speech_delay_tokens: int = field(
        default=5,
        metadata={"help": "Number of null tokens to delay speech generation"}
    )
    talker_concat_lm_text_output: bool = field(
        default=True,
        metadata={
            "help": (
                "When enabled, concatenate the LM's embedding output (hidden_states[0]) into the talker conditioning "
                "(per-position, not averaged). Gives the talker additional context about the LM representation."
            )
        },
    )
    use_concat_len_emb: bool = field(
        default=False,
        metadata={
            "help": (
                "When enabled, replace delay tokens (AUD_START_TOKEN) with length embeddings when passed to the talker."
            )
        },
    )
    talker_embed_v2: bool = field(
        default=False,
        metadata={
            "help": (
                "When enabled, talker embeddings (audio + length) stay at talker_hidden_size "
                "instead of being projected to the LLM hidden_size. Removes talker_embed_to_hidden "
                "and skips lm_to_talker_proj for audio/length conditions."
            )
        },
    )
    framerate_min: float = field(default=0.0, metadata={"help": "Minimum frame rate"})
    framerate_max: float = field(default=1.0, metadata={"help": "Maximum frame rate"})
    enable_flexible_framerate: bool = field(
        default=False,
        metadata={"help": "Enable flexible frame rate with SenseVoice feature merging (similarity-based)"}
    )
    uniform_merging: bool = field(
        default=False,
        metadata={
            "help": (
                "Use uniform input feature merging with a random target frame rate between 4 and 12 Hz. "
                "Supported for both SenseVoice and Qwen-ASR features."
            )
        }
    )
    output_uniform_merging: bool = field(
        default=False,
        metadata={
            "help": (
                "Ablation: force FlexiCodec output-side (assistant) merging to be UNIFORM "
                "with a random target frame rate between 4 and 12 Hz, instead of similarity-based "
                "merging. Affects the FlexiCodec encode call for assistant audio."
            )
        }
    )
    use_input_merging_transformer: bool = field(
        default=False,
        metadata={"help": "Add a local windowed transformer right after input merging (audio_embed_transform)."}
    )
    input_merging_transformer_num_layers: int = field(
        default=4,
        metadata={"help": "Number of layers in the input merging transformer."}
    )
    input_merging_transformer_d_model: int = field(
        default=768,
        metadata={"help": "Hidden size (d_model) of the input merging transformer. If <=0, defaults to config.hidden_size (backward compatible). When smaller, input/output projections bridge config.hidden_size <-> d_model."}
    )
    input_merging_transformer_num_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads in the input merging transformer."}
    )
    input_merging_transformer_dim_feedforward: int = field(
        default=2048,
        metadata={"help": "Feed-forward dimension in the input merging transformer."}
    )
    input_merging_transformer_context: int = field(
        default=32,
        metadata={"help": "Local window size (context frames) for the input merging transformer."}
    )
    input_merging_transformer_causal: bool = field(
        default=False,
        metadata={"help": "Whether the input merging transformer uses causal masking."}
    )
    use_input_merging_transformer_v2: bool = field(
        default=False,
        metadata={"help": "Use FlexiCodec-style v2 input merging transformer: process an interleaved sequence of pre-merge frames + per-group query tokens (instead of v1's aggregate-then-refine). Requires use_input_merging_transformer=True and enable_flexible_framerate=True."}
    )
    use_learnable_audio_boundary: bool = field(
        default=True,
        metadata={"help": "Use learnable embeddings for audio start/end instead of token ID embeddings."}
    )
    use_sinusoidal: bool = field(
        default=False,
        metadata={"help": "Use continuous sinusoidal framerate embedding instead of learnable discrete embeddings"}
    )
    per_sample_frame_rate_embed: bool = field(
        default=False,
        metadata={
            "help": (
                "When enabled, use per-sample frame rate embedding (code_lens/feature_lens*15, range 0-15 Hz) "
                "instead of unified merge threshold embed."
            )
        },
    )
    max_tokens_per_group: Optional[int] = field(
        default=16,
        metadata={"help": "Maximum number of consecutive frames that can be merged into a single group"}
    )
    text_loss_weight: float = field(
        default=2.0,
        metadata={"help": "Weight (width) multiplier for text loss in combined loss. Default 2.0."}
    )
    length_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight multiplier for length loss in parallel models. Default 1.0."}
    )
    text_alignment_pad_loss_weight: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Multiplier for text CE on alignment-extended pad tokens (text padded to speech length). "
                "If unset: 0 when freeze_talker, else 0.1. Set to 0 to ignore those targets; use a small value "
                "(e.g. 0.1) to keep weak supervision."
            )
        },
    )

    vision_projector_type: str = field(default="mlp", metadata={"help": ""})
    vision_projector_pre_norm: bool = field(default=False, metadata={"help": ""})
    vision_downsample_ratio: float = field(default=0.5, metadata={"help": ""})

    image_size: int = field(default=448, metadata={"help": ""})
    image_token_length: int = field(default=1025, metadata={"help": ""})
    max_num_frame: int = field(default=16, metadata={"help": ""})
    max_fps: int = field(default=1, metadata={"help": ""})
    min_patch_grid: int = field(default=1, metadata={"help": ""})
    max_patch_grid: int = field(default=12, metadata={"help": ""})
    vision_process_type: str = field(default="dynamic", metadata={"help": ""})
    vision_normalize_type: str = field(default="imagenet", metadata={"help": ""})

    model_max_length: int = field(default=1024, metadata={"help": ""})
    
    # LoRA-related parameters (defaults aligned with AmphionASR: r=64, lora_alpha=16, lora_dropout=0.05)
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA for fine-tuning"})
    use_combined_embedding: bool = field(
        default=True,
        metadata={"help": "Whether to use combined embedding (text+audio+length) for the main LM input"}
    )
    force_use_combined_embedding: bool = field(
        default=False,
        metadata={"help": "When True, use combined embedding (text+audio+length) even if use_lora; overrides use_lora's text-only behavior"}
    )
    lora_rank: int = field(default=32, metadata={"help": "LoRA rank (AmphionASR: 64)"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha parameter (AmphionASR: 16)"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout rate (AmphionASR: 0.05)"})
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj",
        metadata={"help": "Target modules for LoRA, comma separated (AmphionASR same set)"}
    )
    lora_modules_to_save: Optional[str] = field(
        default=None,
        metadata={"help": "Modules to save during LoRA training, comma separated. VQAdaptor will be automatically added."}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "LoRA bias type: none, all, or lora_only",
        "choices": ["none", "all", "lora_only"]
        },
        
    )
    lora_task_type: str = field(
        default="CAUSAL_LM",
        metadata={"help": "LoRA task type",
        "choices": ["CAUSAL_LM", "SEQ_2_SEQ_LM", "TOKEN_CLASSIFICATION", "QUESTION_ANSWERING"]
        }
    )
    
    # Parameters for continuing training from an existing LoRA checkpoint
    lora_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to existing LoRA checkpoint directory to continue training from"}
    )
    convert_from_lora: bool = field(
        default=False,
        metadata={"help": "When True, load LoRA checkpoint, merge adapter into base model, then do full-parameter training"}
    )
    convert_from_lora_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to LoRA checkpoint to convert from (when convert_from_lora=True). If not set, uses resume_from_checkpoint."}
    )
    use_new_lora_config: bool = field(
        default=False,
        metadata={"help": "Whether to add a new LoRA adapter for a different task while keeping the existing one"}
    )
    use_sensevoice_feature: bool = field(
        default=False,
        metadata={"help": "Whether to use SenseVoice feature for audio encoding"}
    )
    use_qwen3_feature: bool = field(
        default=False,
        metadata={"help": "Whether to use Qwen3 ASR encoder (without projection) for user audio encoding instead of SenseVoice"}
    )
    use_whisper_fetaure: bool = field(
        default=False,
        metadata={"help": "Whether to use Whisper-large-v3 encoder features for user audio encoding instead of SenseVoice"}
    )
    use_qwen25omni_feature: bool = field(
        default=False,
        metadata={"help": "Whether to use Qwen2.5 Omni encoder (with projection) for user audio encoding"}
    )
    use_omni_token: bool = field(
        default=True,
        metadata={
            "help": (
                "If True, interleaved dataset preprocessing uses <|audio_bos|> / <|audio_eos|> (AUD_START_TOKEN_OMNI / "
                "AUD_END_TOKEN_OMNI) instead of the default audio boundary tokens."
            )
        },
    )
    qwen3_encoder_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to Qwen3 ASR encoder checkpoint .pth (without projection). Required when use_qwen3_feature=True."}
    )
    qwen3_encoder_config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to Qwen3 ASR encoder config .json. Required when use_qwen3_feature=True."}
    )
    whisper_encoder_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional path to a local Whisper-large-v3 checkpoint directory. If unset, openai/whisper-large-v3 is used."
        },
    )
    qwen25omni_encoder_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to Qwen2.5-Omni audio encoder checkpoint directory. Required when use_qwen25omni_feature=True."}
    )
    qwen25omni_encoder_config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to Qwen2.5-Omni audio encoder config .json. Required when use_qwen25omni_feature=True."}
    )
    thinker_concat_user_speech: bool = field(
        default=False,
        metadata={"help": "If True (v5), thinker input applies concat+proj conditioning to user-speech positions as well."}
    )
    assistant_text_start_delay_tokens: int = field(
        default=-1,
        metadata={"help": "Delay tokens before assistant text starts. -1 means reuse speech_delay_tokens."}
    )
    finetune_speech_encoder: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, train a deep copy of SenseVoice (sensevoice_finetune_copy) when use_sensevoice_feature — "
                "FlexiCodec's built-in SenseVoice stays frozen. If use_qwen3_feature or use_whisper_fetaure or use_qwen25omni_feature, "
                "trains the selected speech encoder. "
                "Ignored when only_train_llm=True. Default False (encoders frozen)."
            )
        },
    )
    use_mlp_for_audio_embed: bool = field(
        default=False,
        metadata={"help": "Whether to use MLP instead of Linear layer for audio embedding transform"}
    )
    audio_embed_mlp_hidden_ratio: float = field(
        default=4.0,
        metadata={"help": "Hidden layer size multiplier for audio embed MLP (hidden_size * ratio)"}
    )
    audio_embed_mlp_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for audio embed MLP"}
    )
    # S2S (Speech-to-Speech) model arguments
    s2s_mode: bool = field(
        default=False,
        metadata={"help": "Enable S2S mode with FlexiCodec audio output"}
    )
    flexicodec_config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to FlexiCodec config YAML file"}
    )
    flexicodec_ckpt_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to FlexiCodec checkpoint file"}
    )
    sensevoice_small_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the local SenseVoiceSmall checkpoint used by FlexiCodec"}
    )
    flow_matching_decoder_ckpt_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the FlexiCodec flow-matching decoder checkpoint"}
    )
    flow_matching_vocoder_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the Vocos checkpoint used by the flow-matching decoder"}
    )
    audio_vocab_size: int = field(
        default=32768,
        metadata={"help": "Audio vocabulary size for S2S model"}
    )
    duration_classes: int = field(
        default=21,
        metadata={"help": "Number of duration prediction classes (0-20)"}
    )
    framerate_options: Optional[str] = field(
        default="0.87,0.91,1.0",
        metadata={"help": "Comma-separated frame rate options for FlexiCodec"}
    )
    training_framerate_options: Optional[str] = field(
        default="0.87,0.91",
        metadata={"help": "Comma-separated framerate options for training (assistant/target audio). Default: 0.87,0.91"}
    )
    training_input_framerate_options: Optional[str] = field(
        default="0.87,0.91,1.0,0.85",
        metadata={"help": "Comma-separated framerate options for training (user/input audio). Default: 0.87,0.91,1.0,0.85"}
    )
    default_framerate: float = field(
        default=1.0,
        metadata={"help": "Default frame rate for inference"}
    )
    use_dummy_target_audio: bool = field(
        default=False,
        metadata={"help": "Use dummy audio as placeholder for target audio in S2S training"}
    )
    dummy_audio_duration_sec: float = field(
        default=1.0,
        metadata={"help": "Duration of dummy audio in seconds"}
    )
    target_audio_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing target audio files for S2S training"}
    )
    
    # Learnable prefix tokens for stage 1.5
    use_learnable_prefix: bool = field(
        default=False,
        metadata={"help": "Use learnable prefix tokens to teach interleaved pattern"}
    )
    num_prefix_tokens: int = field(
        default=32,
        metadata={"help": "Number of learnable prefix tokens"}
    )
    
    predict_second_audio_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Ablation: replace the talker length-prediction head with a second-audio-token "
                "prediction head (group size 2). When enabled, the dataloader's "
                "``audio_token_lengths`` field is reinterpreted as the second audio token id "
                "(audio-vocab space, 0-indexed) at each step, and the secondary CE is "
                "computed over the talker audio vocabulary instead of length classes."
            )
        },
    )

    # Chained architecture parameters
    use_chained_architecture: bool = field(
        default=False,
        metadata={"help": "Enable chained 0.5B -> 3B -> 0.5B architecture with frozen 3B LLM"}
    )
    use_attention_gating: bool = field(
        default=False,
        metadata={"help": "Use attention-based gating to replace FiLM in chained architecture"}
    )
    chained_3b_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to frozen 3B textual LLM (e.g., Qwen2.5-3B-Instruct)"}
    )
    chained_adaptor_hidden_size: int = field(
        default=512,
        metadata={"help": "Hidden size for MLP adaptors in chained architecture"}
    )

    A: int = field(default=0, metadata={"help": ""})
    B: str = field(default=None, metadata={"help": ""})
    C: bool = field(default=False, metadata={"help": ""})

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to the data used for training.
    """

    dataset_name: Optional[str] = field(
        default="dataset/sts_finetune_stage1.yaml", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training datasets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    create_attention_mask: bool = field(default=False, metadata={"help": "create_attention_mask"})
    create_attention_mask_2d: bool = field(default=False, metadata={"help": "create_attention_mask_2d"})
    reset_position_ids: bool = field(default=False, metadata={"help": ""})
    reset_attention_mask: bool = field(default=False, metadata={"help": ""})
    cross_dataset_joint: bool = field(default=False, metadata={"help": ""})

    # dataset_joint: bool = field(default=False, metadata={"help": ""})
    # variable_length: bool = field(default=True, metadata={"help": ""})
    dataset_joint: bool = field(default=True, metadata={"help": ""})
    variable_length: bool = field(default=False, metadata={"help": ""})

    disable_text_normalize: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, skip processor text_normalize_llm in interleaved dataset preprocess "
                "(Qwen2Dataset / interleaved.py); text is tokenized as stored in the data."
            )
        },
    )

    D: int = field(default=0, metadata={"help": ""})
    E: str = field(default=None, metadata={"help": ""})
    F: bool = field(default=False, metadata={"help": ""})

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None:
            raise ValueError("Need either a dataset name or a training file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    vision_model_lr_mult: float = field(default=1.0, metadata={"help": ""})
    vision_model_lr_decay_rate: float = field(default=1.0, metadata={"help": ""})

    mtp_model_lr_mult: float = field(default=1.0, metadata={"help": ""})
    talker_learning_rate: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional learning rate for the whole talker stack. When set, parameters whose "
                "names contain 'talker' or 'input_merging_transformer' use this LR, including "
                "the talker transformer and the v1/v2 input merging transformer."
            )
        },
    )
    checkpoint_load_mode: str = field(
        default="weights_only",
        metadata={
            "help": (
                "How to load resume_from_checkpoint: 'resume' restores the complete Trainer "
                "state (model, optimizer, scheduler, global step, RNG, and dataloader cursor); "
                "'weights_only' loads only model weights and starts a new training state."
            )
        },
    )
    combine_proj_learning_rate: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional learning rate for parameters whose names contain "
                "'combined_embed_proj' (the projection from concatenated text/audio/"
                "length/framerate embeddings back to hidden size). If unset, these "
                "parameters use the base learning_rate."
            )
        },
    )
    output_dir: str = field(
        default="outputs/", 
        metadata={"help": "Directory to save model checkpoints and logs."}
    )
    max_tokens: int = field(default=2048, metadata={"help": ""})
    overwrite_output_dir: bool = field(default=True, metadata={"help": ""})
    group_by_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Group samples of roughly the same length when batching (uses Qwen2Dataset.lengths via ATrainer). "
                "Override with --group_by_length False if needed."
            )
        },
    )
    even_batches: bool = field(default=False, metadata={"help": ""})
    max_tokens_per_batch: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Max total tokens per batch. When set, enables dynamic batching + sequence packing: "
                "short-audio batches get more samples, long-audio batches get fewer. "
                "Overrides dataset YAML batching.max_cost for native WebDataset streams. "
                "per_device_train_batch_size likewise overrides dataset YAML batching.max_samples."
            )
        },
    )
    webdataset_steps_per_epoch: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optimizer steps that define one epoch for native WebDataset streams. "
                "Overrides dataset YAML sampling.steps_per_epoch so each recipe can match "
                "its GPU count and per_device_train_batch_size. Required for distributed "
                "training when the dataset YAML leaves steps_per_epoch unset."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.checkpoint_load_mode = str(self.checkpoint_load_mode).strip().lower()
        if self.checkpoint_load_mode not in {"resume", "weights_only"}:
            raise ValueError(
                "checkpoint_load_mode must be 'resume' or 'weights_only', got "
                f"{self.checkpoint_load_mode!r}"
            )

