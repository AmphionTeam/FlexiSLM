"""Native WebDataset streaming with distributed dynamic batching."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence, Union

import torch

from src.dataset.interleaved_processor import FlexiSampleProcessor, WorkerContext

from .bucketing import (
    BatchLimits,
    DynamicBatchIterator,
    pool_sort_samples,
    projected_padding_cost,
)
from .layouts import AdapterContext, PhysicalLayoutRegistry
from .observability import QuarantineWriter, SharedStreamMetrics, StreamErrorReporter
from .shard_cache import NodeLocalShardCache, ShardCacheConfig
from .shard_source import (
    SharedEpoch,
    ShardRef,
    ShardSource,
    assigned_shards,
    assigned_source_shards,
    current_topology,
    validate_shard_ratio,
)
from .shuffle import byte_bounded_shuffle

logger = logging.getLogger(__name__)

_BATCH_COST_KEY = "_batch_cost"


class BatchCostCollateFn:
    """Picklable collate wrapper that promotes stream padding cost to ``batch_cost``."""

    def __init__(self, collate_fn: Callable):
        self.collate_fn = collate_fn

    def __call__(self, batch):
        cost = None
        if isinstance(batch, list):
            for sample in batch:
                if not isinstance(sample, dict) or _BATCH_COST_KEY not in sample:
                    continue
                if cost is None:
                    cost = sample[_BATCH_COST_KEY]
                sample.pop(_BATCH_COST_KEY, None)
        collated = self.collate_fn(batch)
        if cost is not None and isinstance(collated, dict):
            value = torch.as_tensor(cost, dtype=torch.float32)
            collated["batch_cost"] = value.reshape(())
        return collated


def collate_with_batch_cost(collate_fn: Callable):
    """Preserve the WebDataset padding cost through the collator as ``batch_cost``."""
    return BatchCostCollateFn(collate_fn)


class AudioDecodeError(RuntimeError):
    """Compressed audio bytes could not be decoded by the configured backend."""

    def __init__(self, member_key: str, codec: str, detail: str):
        self.member_key = member_key
        self.codec = codec
        super().__init__(
            f"failed to decode {member_key} as {codec} with SoundFile: {detail}"
        )


@dataclass(frozen=True)
class StreamingSourceConfig:
    name: str
    shards: Sequence[str]
    layout: str = "auto"
    weight: float = 1.0
    ratio: float = 1.0

    def __post_init__(self):
        if not self.name:
            raise ValueError("streaming source name must not be empty")
        if not self.shards:
            raise ValueError(f"streaming source {self.name!r} requires at least one shard")
        if self.weight <= 0:
            raise ValueError(f"streaming source {self.name!r} weight must be positive")
        validate_shard_ratio(self.name, self.ratio)


@dataclass(frozen=True)
class StreamingConfig:
    shards: Sequence[str] = ()
    layout: str = "auto"
    batch_size: int = 1
    shuffle_max_samples: int = 4096
    shuffle_initial_samples: int = 1024
    shuffle_max_bytes: Optional[int] = 2 * 1024**3
    seed: int = 0
    drop_last: bool = False
    source_name: str = "webdataset"
    num_batches: Optional[int] = None
    sampling_mode: str = "finite_exact"
    shard_shuffle: bool = True
    batch_limits: Optional[BatchLimits] = None
    max_consecutive_errors: int = 100
    quarantine_path: Optional[str] = None
    shard_cache: Optional[ShardCacheConfig] = None
    sources: Sequence[StreamingSourceConfig] = ()
    for_evaluation: bool = False

    def __post_init__(self):
        if not self.shards and not self.sources:
            raise ValueError("native WebDataset requires at least one shard URL")
        if self.shards and self.sources:
            raise ValueError("configure either shards or sources, not both")
        source_names = [source.name for source in self.sources]
        if len(source_names) != len(set(source_names)):
            raise ValueError("streaming source names must be unique")
        if (
            self.sampling_mode != "resampled"
            and self.sources
            and any(source.weight != self.sources[0].weight for source in self.sources[1:])
        ):
            raise ValueError("non-uniform source weights require sampling.mode=resampled")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.shuffle_max_samples <= 0:
            raise ValueError("shuffle max_samples must be positive")
        if self.shuffle_max_bytes is not None and self.shuffle_max_bytes <= 0:
            raise ValueError("shuffle max_bytes must be positive when set")
        if not 0 <= self.shuffle_initial_samples <= self.shuffle_max_samples:
            raise ValueError("shuffle_initial_samples must not exceed shuffle_max_samples")
        if self.num_batches is not None and self.num_batches <= 0:
            raise ValueError("num_batches must be positive when configured")
        if self.sampling_mode not in {"finite_exact", "finite_padded", "resampled"}:
            raise ValueError("sampling_mode must be finite_exact, finite_padded, or resampled")
        if self.sampling_mode == "resampled" and self.num_batches is None:
            raise ValueError("resampled mode requires steps_per_epoch or num_batches")
        if self.max_consecutive_errors <= 0:
            raise ValueError("max_consecutive_errors must be positive")

    @property
    def effective_batch_limits(self) -> BatchLimits:
        return self.batch_limits or BatchLimits(max_samples=self.batch_size)


def decode_audio_asset(asset):
    """Decode an ``AudioAsset`` from compressed in-tar bytes with SoundFile."""
    import io
    import soundfile as sf

    try:
        waveform, sample_rate = sf.read(
            io.BytesIO(asset.data), dtype="float32", always_2d=True
        )
    except Exception as exc:
        raise AudioDecodeError(asset.member_key, asset.codec, str(exc)) from exc
    return torch.from_numpy(waveform.T.copy()), sample_rate


def _warn_and_continue(error: Exception) -> bool:
    logger.warning("WebDataset tar read error; skipping entry: %s", error)
    return True


def sequential_physical_samples(
    shards: Iterable[Union[str, ShardRef]],
    *,
    shard_cache: Optional[NodeLocalShardCache] = None,
) -> Iterator[Mapping[str, Any]]:
    """Open assigned shards one at a time and yield grouped physical samples."""
    try:
        import webdataset as wds
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "webdataset_stream requires the pinned 'webdataset' dependency"
        ) from exc

    from webdataset.tariterators import group_by_keys, tar_file_expander

    for shard in shards:
        source_name = shard.source_name if isinstance(shard, ShardRef) else None
        url = shard.url if isinstance(shard, ShardRef) else shard
        # Keep the original URL in sample metadata even when bytes are read
        # through a node-local cached path, preserving stable logical UIDs.
        path_context = (
            shard_cache.open_path(url) if shard_cache is not None else nullcontext(url)
        )
        with path_context as open_url:
            # tarfile_to_samples does not explicitly close the stream returned by
            # gopen in WebDataset 1.0.2. Own it here so early epoch termination and
            # repeated/resampled reads cannot leak one descriptor per shard.
            stream = wds.gopen(open_url)
            expanded = tar_file_expander(
                [{"url": url, "stream": stream}], handler=_warn_and_continue
            )
            grouped = group_by_keys(expanded, handler=_warn_and_continue)
            try:
                for sample in grouped:
                    if source_name is not None:
                        sample["__source__"] = source_name
                    yield sample
            finally:
                grouped.close()
                expanded.close()
                stream.close()


class FlexiWebDataset(torch.utils.data.IterableDataset):
    """Rank/worker-sharded stream that emits complete dynamically sized batches."""

    is_native_webdataset = True

    def __init__(
        self,
        config: StreamingConfig,
        processor: FlexiSampleProcessor,
        worker_context_factory: Callable[[], WorkerContext],
        *,
        adapter_context: Optional[AdapterContext] = None,
        adapter_contexts: Optional[Mapping[str, AdapterContext]] = None,
        registry: Optional[PhysicalLayoutRegistry] = None,
        physical_samples_factory: Optional[Callable[[], Iterable[Mapping[str, Any]]]] = None,
        shared_epoch: Optional[SharedEpoch] = None,
        metrics: Optional[SharedStreamMetrics] = None,
    ):
        super().__init__()
        topology = current_topology()
        if (
            topology.world_size > 1
            and config.sampling_mode != "resampled"
            and not config.for_evaluation
        ):
            raise ValueError(
                "distributed native WebDataset training requires "
                "sampling.mode=resampled with a fixed number of batches"
            )
        self.config = config
        self.processor = processor
        self.worker_context_factory = worker_context_factory
        self.adapter_context = adapter_context or AdapterContext(
            source_name=config.source_name, seed=config.seed
        )
        self.adapter_contexts = dict(adapter_contexts or {})
        configured_names = {source.name for source in config.sources}
        if self.adapter_contexts.keys() != configured_names and config.sources:
            raise ValueError(
                "adapter_contexts must contain exactly the configured source names"
            )
        self.registry = registry or PhysicalLayoutRegistry()
        self.physical_samples_factory = physical_samples_factory
        self.shared_epoch = shared_epoch or SharedEpoch()
        self.metrics = metrics or SharedStreamMetrics()
        self._quarantine = None

    @property
    def epoch(self) -> int:
        return self.shared_epoch.get()

    @property
    def num_batches(self) -> Optional[int]:
        return self.config.num_batches

    def __len__(self) -> int:
        if self.num_batches is None:
            raise TypeError(
                "stream length is unknown; set sampling.steps_per_epoch or train with max_steps"
            )
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.shared_epoch.set(epoch)

    def state_signature(self) -> str:
        """Fingerprint stream settings that affect deterministic replay."""

        def context_state(context: AdapterContext) -> dict[str, Any]:
            return {
                "source_name": context.source_name,
                "tasks": list(context.tasks),
                "task_policy": context.task_policy,
                "task_weights": dict(context.task_weights),
                "seed": context.seed,
                "duplicate_member_policy": context.duplicate_member_policy,
                "audio_extension_preference": list(
                    context.audio_extension_preference
                ),
                "physical_sample_atomic": context.physical_sample_atomic,
                "text_keys": list(context.text_keys),
                "asr_prompt": context.asr_prompt,
                "tts_prompt_template": context.tts_prompt_template,
            }

        payload = {
            # Bump when deterministic stream ordering changes. Rolling bucketing
            # is intentionally incompatible with cursors from the drain/refill
            # implementation.
            "pipeline_version": 2,
            "shards": list(self.config.shards),
            "sources": [asdict(source) for source in self.config.sources],
            "layout": self.config.layout,
            "batch_size": self.config.batch_size,
            "shuffle_max_samples": self.config.shuffle_max_samples,
            "shuffle_initial_samples": self.config.shuffle_initial_samples,
            "shuffle_max_bytes": self.config.shuffle_max_bytes,
            "seed": self.config.seed,
            "drop_last": self.config.drop_last,
            "num_batches": self.config.num_batches,
            "sampling_mode": self.config.sampling_mode,
            "shard_shuffle": self.config.shard_shuffle,
            "batch_limits": asdict(self.config.effective_batch_limits),
            "adapter_context": context_state(self.adapter_context),
            "adapter_contexts": {
                name: context_state(context)
                for name, context in sorted(self.adapter_contexts.items())
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def metrics_snapshot(self) -> dict[str, float]:
        """Return process-safe cumulative stream metrics for logging."""
        return self.metrics.snapshot()

    def _configure_observability(self, topology) -> None:
        self._quarantine = QuarantineWriter(
            self.config.quarantine_path, topology, self.epoch
        )
        self.adapter_context.error_reporter = StreamErrorReporter(
            self.metrics, self._quarantine, self.adapter_context.source_name
        )
        for context in self.adapter_contexts.values():
            context.error_reporter = StreamErrorReporter(
                self.metrics, self._quarantine, context.source_name
            )

    def _record_error(
        self, error, *, stage, physical=None, logical=None, source_name=None
    ) -> None:
        self.metrics.record_error(error, stage=stage)
        if self._quarantine is not None:
            self._quarantine.record(
                error,
                stage=stage,
                physical=physical,
                logical=logical,
                source_name=source_name,
            )

    def _worker_batch_limit(self) -> Optional[int]:
        if self.num_batches is None:
            return None
        topology = current_topology()
        quotient, remainder = divmod(self.num_batches, topology.num_workers)
        return quotient + int(topology.worker_id < remainder)

    def _physical_samples(self):
        if self.physical_samples_factory is not None:
            return iter(self.physical_samples_factory())
        topology = current_topology()
        mode = self.config.sampling_mode
        if self.config.sources:
            sources = tuple(
                ShardSource(source.name, source.shards, source.weight)
                for source in self.config.sources
            )
            shards = assigned_source_shards(
                sources,
                mode=mode,
                seed=self.config.seed,
                epoch=self.epoch,
                topology=topology,
                shuffle=self.config.shard_shuffle,
            )
        else:
            shards = assigned_shards(
                self.config.shards,
                mode=mode,
                seed=self.config.seed,
                epoch=self.epoch,
                topology=topology,
                shuffle=self.config.shard_shuffle,
            )
        shard_cache = None
        if self.config.shard_cache is not None:
            shard_cache = NodeLocalShardCache(
                self.config.shard_cache, self.metrics.increment
            )
        return sequential_physical_samples(shards, shard_cache=shard_cache)

    @staticmethod
    def _invalid_metadata_duration(logical):
        """Return the first invalid metadata duration without decoding audio."""
        for binding in logical.bindings:
            duration = logical.assets[binding.asset_id].duration
            if duration is None:
                continue
            maximum = 30.0 if binding.role == "user" else 60.0
            if not math.isfinite(duration) or duration < 1.5 or duration > maximum:
                return binding.role, duration, maximum
        return None

    def _logical_samples(self):
        # AdapterContext is process-local after DataLoader worker creation.
        self.adapter_context.epoch = self.epoch
        for context in self.adapter_contexts.values():
            context.epoch = self.epoch
        layouts = {source.name: source.layout for source in self.config.sources}
        for physical in self._physical_samples():
            self.metrics.increment("physical_samples_seen")
            source_name = physical.get("__source__")
            context = self.adapter_contexts.get(source_name, self.adapter_context)
            expected_layout = layouts.get(source_name, self.config.layout)
            try:
                adapter = self.registry.resolve(physical, expected_layout)
                self.metrics.record_layout(adapter.name)
                for logical in adapter.expand(physical, context):
                    invalid_duration = self._invalid_metadata_duration(logical)
                    if invalid_duration is not None:
                        role, duration, maximum = invalid_duration
                        self.metrics.record_duration_filter(role)
                        logger.debug(
                            "Filtering logical sample %s from duration metadata: "
                            "%s audio %.2fs is outside [1.5, %.0f]",
                            logical.uid,
                            role,
                            duration,
                            maximum,
                        )
                        continue
                    self.metrics.record_logical(logical.task)
                    yield logical
            except Exception as exc:
                self._record_error(
                    exc,
                    stage="layout",
                    physical=physical,
                    source_name=context.source_name,
                )
                logger.warning(
                    "Skipping invalid physical sample %s::%s: %s",
                    physical.get("__url__", "<unknown>"),
                    physical.get("__key__", "<unknown>"),
                    exc,
                )

    def _prepared_samples(self):
        topology = current_topology()
        stream_seed = self.config.seed + 1_000_003 * self.epoch + 97 * topology.consumer_id
        logical = byte_bounded_shuffle(
            self._logical_samples(),
            max_samples=self.config.shuffle_max_samples,
            initial_samples=self.config.shuffle_initial_samples,
            max_bytes=self.config.shuffle_max_bytes,
            rng=random.Random(stream_seed),
            on_buffer_change=self.metrics.record_shuffle_buffer,
        )
        for sample in logical:
            try:
                yield self.processor.prepare(sample)
            except Exception as exc:
                self._record_error(exc, stage="prepare", logical=sample)
                logger.warning("Skipping logical sample %s during prepare: %s", sample.uid, exc)

    def _ordered_samples(self):
        limits = self.config.effective_batch_limits
        topology = current_topology()
        rng = random.Random(
            self.config.seed + 2_000_003 * self.epoch + 193 * topology.consumer_id
        )
        return pool_sort_samples(
            self._prepared_samples(),
            pool_samples=limits.pool_samples,
            pool_bytes=limits.pool_bytes,
            chunk_size=limits.chunk_size,
            rng=rng,
            on_pool_drain=self.metrics.record_bucket_pool,
        )

    def __iter__(self):
        topology = current_topology()
        self._configure_observability(topology)
        worker = self.worker_context_factory()
        limits = self.config.effective_batch_limits
        ordered = iter(self._ordered_samples())
        candidates = DynamicBatchIterator(
            ordered,
            limits,
            on_sample_dropped=lambda _: self.metrics.record_filtered("oversized"),
        )
        emitted = 0
        batch_limit = self._worker_batch_limit()

        while batch_limit is None or emitted < batch_limit:
            started = time.perf_counter()
            try:
                candidate = next(candidates)
            except StopIteration:
                if batch_limit is not None:
                    raise RuntimeError(
                        "WebDataset stream ended after "
                        f"{emitted}/{batch_limit} batches for this worker"
                    )
                return
            materialized, accepted_lengths = self._materialize_with_refill(
                candidate, candidates, worker, limits
            )
            if not materialized:
                continue
            required = (
                limits.max_samples
                if self.config.drop_last and limits.max_cost is None
                else limits.min_samples
            )
            if self.config.drop_last and len(materialized) < required:
                self.metrics.record_filtered("drop_last", len(materialized))
                continue
            self.metrics.record_batch(
                accepted_lengths, time.perf_counter() - started
            )
            if materialized:
                materialized[0][_BATCH_COST_KEY] = projected_padding_cost(
                    accepted_lengths
                )
            yield materialized
            emitted += 1

    def _materialize_with_refill(self, batch, replacements, worker, limits):
        output = []
        accepted_lengths = []
        target_size = len(batch)
        pending = iter(batch)
        errors = 0
        rejected_replacements = 0

        while len(output) < target_size:
            try:
                prepared = next(pending)
            except StopIteration:
                try:
                    prepared = replacements.take_sample()
                except StopIteration:
                    break
                if limits.max_cost is not None and projected_padding_cost(
                    accepted_lengths + [prepared.lengths]
                ) > limits.max_cost:
                    replacements.put_back(prepared)
                    rejected_replacements += 1
                    break
            try:
                output.append(self.processor.materialize(prepared, worker))
                accepted_lengths.append(prepared.lengths)
                errors = 0
                rejected_replacements = 0
            except Exception as exc:
                errors += 1
                self._record_error(
                    exc, stage="materialize", logical=prepared.canonical
                )
                logger.warning(
                    "Skipping logical sample %s during audio materialization: %s",
                    prepared.canonical.uid,
                    exc,
                )
                if errors >= self.config.max_consecutive_errors:
                    raise RuntimeError(
                        "maximum consecutive WebDataset materialization errors exceeded"
                    ) from exc
        return output, accepted_lengths

    def build_loader(
        self,
        *,
        collate_fn,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
    ):
        kwargs = dict(
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_with_batch_cost(collate_fn),
        )
        if num_workers > 0:
            kwargs["persistent_workers"] = persistent_workers
            if prefetch_factor is not None:
                kwargs["prefetch_factor"] = prefetch_factor
        try:
            import webdataset as wds
        except ImportError as exc:
            raise RuntimeError("webdataset_stream requires 'webdataset'") from exc
        loader = wds.WebLoader(self, **kwargs)
        return StatefulWebLoader(loader, self, num_workers=num_workers)


class StatefulWebLoader:
    """Track delivered batches and deterministically replay to a checkpoint.

    State is recorded in the main training process, rather than in workers, so
    DataLoader prefetch does not advance the checkpoint cursor. Restoring is
    intentionally O(batches already consumed in the epoch): the stream is
    recreated from its deterministic epoch seeds and replayed without exposing
    skipped batches to the trainer.
    """

    STATE_VERSION = 1

    def __init__(self, loader, dataset: FlexiWebDataset, *, num_workers: int):
        self.loader = loader
        self.dataset = dataset
        self.num_workers = int(num_workers)
        self._epoch = dataset.epoch
        self._batches_yielded = 0
        self._resume_batches: Optional[int] = None
        self._resume_epoch: Optional[int] = None

    def __len__(self):
        return len(self.loader)

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        preserve_resume = (
            self._resume_batches is not None and self._resume_epoch == epoch
        )
        self.dataset.set_epoch(epoch)
        self._epoch = epoch
        if not preserve_resume:
            self._batches_yielded = 0
            self._resume_batches = None
            self._resume_epoch = None

    def __iter__(self):
        skip = self._resume_batches or 0
        self._resume_batches = None
        self._resume_epoch = None
        if skip:
            # Trainer loads its checkpoint RNG before entering this iterator when
            # native state restoration disables Trainer's own data skipping.
            # Replay must not consume that model RNG state.
            python_rng = random.getstate()
            torch_rng = torch.random.get_rng_state()
            cuda_rng = (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() and torch.cuda.is_initialized()
                else None
            )
            try:
                iterator = iter(self.loader)
                for index in range(skip):
                    try:
                        next(iterator)
                    except StopIteration as exc:
                        raise RuntimeError(
                            "native WebDataset checkpoint cursor exceeds the replayed "
                            f"epoch ({skip} requested, {index} available)"
                        ) from exc
            finally:
                random.setstate(python_rng)
                torch.random.set_rng_state(torch_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
        else:
            iterator = iter(self.loader)
        while True:
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            metrics = getattr(self.dataset, "metrics", None)
            if metrics is not None:
                metrics.record_main_loader_wait(time.perf_counter() - started)
            self._batches_yielded += 1
            yield batch

    def state_dict(self) -> dict[str, Any]:
        topology = current_topology()
        return {
            "version": self.STATE_VERSION,
            "epoch": self._epoch,
            "batches_yielded": self._batches_yielded,
            "rank": topology.rank,
            "world_size": topology.world_size,
            "num_workers": self.num_workers,
            "stream_signature": self.dataset.state_signature(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        topology = current_topology()
        expected = {
            "version": self.STATE_VERSION,
            "rank": topology.rank,
            "world_size": topology.world_size,
            "num_workers": self.num_workers,
            "stream_signature": self.dataset.state_signature(),
        }
        mismatches = {
            name: (state.get(name), value)
            for name, value in expected.items()
            if state.get(name) != value
        }
        if mismatches:
            detail = ", ".join(
                f"{name}={actual!r} (expected {wanted!r})"
                for name, (actual, wanted) in mismatches.items()
            )
            raise ValueError(f"incompatible native WebDataset checkpoint: {detail}")
        epoch = int(state.get("epoch", 0))
        batches = int(state.get("batches_yielded", 0))
        if epoch < 0 or batches < 0:
            raise ValueError("native WebDataset checkpoint cursor must be non-negative")
        self.dataset.set_epoch(epoch)
        self._epoch = epoch
        self._batches_yielded = batches
        self._resume_batches = batches
        self._resume_epoch = epoch
