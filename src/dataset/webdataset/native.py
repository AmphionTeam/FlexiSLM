"""FlexiSLM model-specific construction for the native streaming backend."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping

import yaml

from src.dataset.interleaved_processor import FlexiSampleProcessor, WorkerContext
from src.processor.constants import DEFAULT_TTS_SYSTEM_PROMPT, T2T_TTS_SYSTEM_PROMPT_OMNI

from .bucketing import BatchLimits
from .layouts import AdapterContext
from .manifest import load_shard_manifest
from .pipeline import (
    FlexiWebDataset,
    StreamingConfig,
    StreamingSourceConfig,
    decode_audio_asset,
)
from .shard_cache import ShardCacheConfig
from .shard_source import select_shards_by_ratio
from .types import SHARED_AUDIO_TASKS


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerContextFactory:
    """Pickle-safe lazy worker resource constructor for spawn/forkserver."""

    use_128_features: bool
    num_mel_bins: int

    def __call__(self) -> WorkerContext:
        # Keep heavyweight feature extractor imports and construction inside the
        # DataLoader worker while leaving the factory itself spawn-pickleable.
        from flexicodec.feature_extractors import FBankGen
        from src.dataset.interleaved import Qwen3FbankExtractor

        if self.use_128_features or self.num_mel_bins == 128:
            user_extractor = Qwen3FbankExtractor()
            assistant_extractor = FBankGen(sr=16000)
        else:
            user_extractor = FBankGen(sr=16000)
            assistant_extractor = None
        return WorkerContext(
            audio_decoder=decode_audio_asset,
            user_fbank_extractor=user_extractor,
            assistant_fbank_extractor=assistant_extractor,
        )


def load_dataset_config(path: str) -> Mapping[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        text = os.path.expandvars(handle.read())
    return yaml.safe_load(text) or {}


def is_webdataset_stream_config(path: str) -> bool:
    cfg = load_dataset_config(path)
    if cfg.get("dataset_backend") == "webdataset_stream":
        return True
    sources = cfg.get("dataset", {})
    return bool(sources) and all(
        source.get("data_format") == "webdataset_stream"
        for source in sources.values()
    )


def _normalize_split(split: str) -> str:
    if split in {"train"}:
        return "train"
    if split in {"validation", "eval"}:
        return "validation"
    raise ValueError(f"unsupported dataset split {split!r}")


def _split_sources(cfg: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    split = _normalize_split(split)
    if split == "train":
        return cfg.get("dataset") or {}
    return cfg.get("validation") or cfg.get("eval") or {}


def _runtime_for_split(cfg: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    """Return stream runtime settings for train or a finite validation pass."""
    runtime = cfg.get("webdataset_runtime") or {}
    if _normalize_split(split) == "train":
        return runtime

    overrides = runtime.get("validation") or runtime.get("eval") or {}
    train_sampling = runtime.get("sampling") or {}
    sampling = {
        "mode": "finite_padded",
        "shuffle": False,
        "seed": train_sampling.get("seed", runtime.get("seed", 42)),
    }
    sampling.update(overrides.get("sampling") or {})
    if "steps_per_epoch" not in (overrides.get("sampling") or {}):
        sampling.pop("steps_per_epoch", None)
    shuffle = {
        "max_samples": 1,
        "initial_samples": 0,
        "max_bytes": None,
    }
    shuffle.update(overrides.get("shuffle") or {})
    batching = dict(runtime.get("batching") or {})
    batching.update(overrides.get("batching") or {})
    if "num_batches" not in (overrides.get("batching") or {}):
        batching.pop("num_batches", None)
    batching["drop_last"] = bool(batching.get("drop_last", False))
    return {
        "sampling": sampling,
        "shuffle": shuffle,
        "bucketing": overrides.get("bucketing", runtime.get("bucketing", {})),
        "batching": batching,
        "errors": overrides.get("errors", runtime.get("errors", {})),
        "cache": overrides.get("cache", runtime.get("cache", {})),
    }


def build_qwen2_webdataset(
    cfg_path,
    tokenizer,
    *,
    training_args=None,
    seed=42,
    use_qwen3_feature=None,
    use_whisper_fetaure=None,
    use_qwen25o_feature=None,
    use_omni_token=None,
    disable_text_normalize_llm=None,
    split: str = "train",
    **_unused,
):
    """Build a distributed dynamic stream from the FlexiSLM dataset YAML shape."""
    cfg = load_dataset_config(cfg_path)
    split = _normalize_split(split)
    sources = _split_sources(cfg, split)
    if split == "validation" and not sources:
        return None
    global_stream = cfg.get("dataset_backend") == "webdataset_stream"
    stream_sources = [
        (name, source)
        for name, source in sources.items()
        if global_stream or source.get("data_format") == "webdataset_stream"
    ]
    if not stream_sources or len(stream_sources) != len(sources):
        raise ValueError(
            "webdataset_stream cannot be mixed with non-streaming dataset sources"
        )
    runtime = _runtime_for_split(cfg, split)
    shuffle_cfg = runtime.get("shuffle", {})
    batch_cfg = runtime.get("batching", {})
    configured_shuffle_max_bytes = shuffle_cfg.get("max_bytes", 2 * 1024**3)
    shuffle_max_bytes = (
        None if configured_shuffle_max_bytes is None else int(configured_shuffle_max_bytes)
    )

    sampling_cfg = runtime.get("sampling", {})
    sampling_seed = int(sampling_cfg.get("seed", runtime.get("seed", seed)))
    bucketing_cfg = runtime.get("bucketing", {})
    errors_cfg = runtime.get("errors", {})
    cache_cfg = runtime.get("cache", {})
    configured_num_batches = batch_cfg.get(
        "num_batches", sampling_cfg.get("steps_per_epoch")
    )
    train_max_cost = (
        getattr(training_args, "max_tokens_per_batch", None)
        if training_args is not None
        else None
    )
    train_max_samples = (
        getattr(training_args, "per_device_train_batch_size", None)
        if training_args is not None
        else None
    )
    dataset_max_cost = batch_cfg.get("max_cost")
    dataset_max_samples = batch_cfg.get("max_samples")
    if train_max_cost is not None:
        configured_max_cost = train_max_cost
        if dataset_max_cost is not None and float(dataset_max_cost) != float(train_max_cost):
            logger.info(
                "Using training max_tokens_per_batch=%s instead of dataset batching.max_cost=%s",
                train_max_cost,
                dataset_max_cost,
            )
    else:
        configured_max_cost = dataset_max_cost
    if train_max_samples is not None:
        configured_max_samples = train_max_samples
        if (
            dataset_max_samples is not None
            and int(dataset_max_samples) != int(train_max_samples)
        ):
            logger.info(
                "Using training per_device_train_batch_size=%s instead of dataset batching.max_samples=%s",
                train_max_samples,
                dataset_max_samples,
            )
    else:
        configured_max_samples = dataset_max_samples
        if configured_max_samples is None and configured_max_cost is None:
            configured_max_samples = batch_cfg.get("fixed_batch_size", 1)
    batch_size = int(
        train_max_samples
        if train_max_samples is not None
        else batch_cfg.get("fixed_batch_size", configured_max_samples or 1)
    )
    max_samples = (
        int(configured_max_samples) if configured_max_samples is not None else None
    )

    # Expand brace patterns before rank/worker assignment so every process sees
    # the same concrete shard list. Mixing weights are source-level, independent
    # of how many shards each source contains.
    source_configs = []
    adapter_contexts = {}
    try:
        import webdataset as wds
    except ImportError:
        wds = None
    for source_name, source in stream_sources:
        web_cfg = source.get("webdataset", {})
        shards = source.get("data_paths", [])
        manifest = web_cfg.get("manifest", source.get("shard_manifest"))
        if manifest and shards:
            raise ValueError(
                f"streaming source {source_name!r} cannot configure both "
                "data_paths and a shard manifest"
            )
        if manifest:
            expanded_shards = list(load_shard_manifest(str(manifest)))
        else:
            if isinstance(shards, str):
                shards = [shards]
            expanded_shards = []
            for value in shards:
                if wds is None:
                    expanded_shards.append(value)
                else:
                    expanded_shards.extend(wds.shardlists.expand_urls(value))
        ratio = source.get("ratio", 1.0)
        shard_count = len(tuple(dict.fromkeys(expanded_shards)))
        selected_shards = select_shards_by_ratio(
            expanded_shards,
            ratio=ratio,
            seed=sampling_seed,
            source_name=source_name,
        )
        normalized_ratio = float(ratio)
        logger.info(
            "WebDataset %s source %s: ratio=%s selected %d shard slots from %d unique shards with seed=%d",
            split,
            source_name,
            normalized_ratio,
            len(selected_shards),
            shard_count,
            sampling_seed,
        )
        source_configs.append(
            StreamingSourceConfig(
                name=source_name,
                shards=selected_shards,
                layout=web_cfg.get("layout", "auto"),
                weight=float(source.get("weight", web_cfg.get("weight", 1.0))),
                ratio=normalized_ratio,
            )
        )
        adapter_contexts[source_name] = AdapterContext(
            source_name=source_name,
            tasks=tuple(web_cfg.get("tasks", SHARED_AUDIO_TASKS)),
            task_policy=web_cfg.get("task_policy", "all"),
            task_weights=web_cfg.get(
                "task_weights", {task: 1.0 for task in SHARED_AUDIO_TASKS}
            ),
            seed=sampling_seed,
            duplicate_member_policy=web_cfg.get("duplicate_member_policy", "error"),
            audio_extension_preference=tuple(
                web_cfg.get("audio_extensions", ("wav", "flac", "mp3", "m4a", "ogg"))
            ),
            physical_sample_atomic=bool(web_cfg.get("physical_sample_atomic", False)),
            asr_prompt=web_cfg.get("asr_prompt", "Transcribe the following audio:"),
            tts_prompt_template=web_cfg.get(
                "tts_prompt_template", "Read the following text out loud: {text}"
            ),
        )

    shard_cache = None
    if cache_cfg and bool(cache_cfg.get("enabled", True)):
        shard_cache = ShardCacheConfig(
            directory=str(cache_cfg.get("directory", "")),
            max_bytes=int(cache_cfg.get("max_bytes", 0)),
            cache_local_files=bool(cache_cfg.get("cache_local_files", True)),
            copy_chunk_bytes=int(cache_cfg.get("copy_chunk_bytes", 8 * 1024**2)),
        )

    shuffle_max_samples = int(shuffle_cfg.get("max_samples", 4096))
    shuffle_initial_samples = int(shuffle_cfg.get("initial_samples", 1024))
    if shuffle_initial_samples <= 0:
        # ``initial_samples: 0`` means fill the configured sample window, not
        # "skip sample-level shuffling".
        shuffle_initial_samples = shuffle_max_samples
    logger.info(
        "WebDataset %s sample-level shuffle: max_samples=%d initial_samples=%d "
        "max_bytes=%s shard_shuffle=%s",
        split,
        shuffle_max_samples,
        shuffle_initial_samples,
        shuffle_max_bytes,
        bool(sampling_cfg.get("shuffle", True)),
    )

    stream_config = StreamingConfig(
        sources=tuple(source_configs),
        source_name=(source_configs[0].name if len(source_configs) == 1 else "mixed"),
        batch_size=batch_size,
        shuffle_max_samples=shuffle_max_samples,
        shuffle_initial_samples=shuffle_initial_samples,
        shuffle_max_bytes=shuffle_max_bytes,
        seed=sampling_seed,
        drop_last=bool(batch_cfg.get("drop_last", False)),
        num_batches=(int(configured_num_batches) if configured_num_batches is not None else None),
        sampling_mode=sampling_cfg.get("mode", "finite_exact"),
        shard_shuffle=bool(sampling_cfg.get("shuffle", True)),
        batch_limits=BatchLimits(
            max_cost=(float(configured_max_cost) if configured_max_cost is not None else None),
            max_samples=max_samples,
            min_samples=int(batch_cfg.get("min_samples", 1)),
            pool_samples=int(bucketing_cfg.get("pool_samples", 2048)),
            pool_bytes=int(bucketing_cfg.get("pool_bytes", 1024**3)),
            chunk_size=int(bucketing_cfg.get("chunk_size", 128)),
            oversized_sample=batch_cfg.get("oversized_sample", "drop"),
        ),
        max_consecutive_errors=int(errors_cfg.get("max_consecutive_errors", 100)),
        quarantine_path=errors_cfg.get("quarantine_path"),
        shard_cache=shard_cache,
        for_evaluation=(split == "validation"),
    )

    use_omni = bool(use_omni_token if use_omni_token is not None else cfg.get("use_omni_token", False))
    processor = FlexiSampleProcessor(
        tokenizer,
        default_system_message=(
            T2T_TTS_SYSTEM_PROMPT_OMNI if use_omni else DEFAULT_TTS_SYSTEM_PROMPT
        ),
        disable_text_normalize_llm=bool(
            disable_text_normalize_llm
            if disable_text_normalize_llm is not None
            else cfg.get("disable_text_normalize", False)
        ),
        use_omni_token=use_omni,
    )

    use_128 = bool(use_qwen3_feature or use_whisper_fetaure or use_qwen25o_feature)
    num_mel_bins = int(cfg.get("num_mel_bins", 128 if use_128 else 80))

    return FlexiWebDataset(
        stream_config,
        processor,
        WorkerContextFactory(use_128, num_mel_bins),
        adapter_contexts=adapter_contexts,
    )
