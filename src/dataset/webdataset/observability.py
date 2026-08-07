"""Process-safe metrics and quarantine output for native WebDataset streams."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .bucketing import projected_padding_cost
from .shard_source import StreamTopology

_COUNTERS = (
    "physical_samples_seen",
    "logical_samples_emitted",
    "logical_samples_emitted_s2s",
    "logical_samples_emitted_asr",
    "logical_samples_emitted_tts",
    "layout_matches_s2s_pair",
    "layout_matches_shared_audio_tasks",
    "unknown_layouts",
    "ambiguous_layouts",
    "malformed_json",
    "missing_members",
    "duplicate_members",
    "prepare_failures",
    "audio_decode_failures",
    "audio_decode_failures_wav",
    "audio_decode_failures_flac",
    "audio_decode_failures_mp3",
    "audio_decode_failures_m4a",
    "audio_decode_failures_ogg",
    "audio_decode_failures_other",
    "samples_filtered",
    "samples_filtered_layout",
    "samples_filtered_prepare",
    "samples_filtered_materialize",
    "samples_filtered_drop_last",
    "samples_filtered_oversized",
    "batches_emitted",
    "batch_size_sum",
    "batch_cost_sum",
    "unpadded_cost_sum",
    "data_wait_time_sum",
    "shuffle_buffer_samples_peak",
    "shuffle_buffer_bytes_peak",
    "bucket_pools_drained",
    "bucket_fill_ratio_sum",
    "shard_cache_hits",
    "shard_cache_misses",
    "shard_cache_bytes_written",
    "shard_cache_evictions",
)
_INDEX = {name: index for index, name in enumerate(_COUNTERS)}
_ERROR_COUNTERS = {
    "UnknownLayoutError": "unknown_layouts",
    "AmbiguousLayoutError": "ambiguous_layouts",
    "MalformedJsonError": "malformed_json",
    "MissingMemberError": "missing_members",
    "DuplicateMemberError": "duplicate_members",
}


class SharedStreamMetrics:
    """Small fixed metric set shared by the main process and loader workers."""

    def __init__(self):
        self._values = mp.Array("d", len(_COUNTERS), lock=True)

    def increment(self, name: str, value: float = 1.0) -> None:
        try:
            index = _INDEX[name]
        except KeyError as exc:
            raise KeyError(f"unknown WebDataset metric {name!r}") from exc
        with self._values.get_lock():
            self._values[index] += float(value)

    def set_max(self, name: str, value: float) -> None:
        """Atomically retain the largest value observed across all workers."""
        try:
            index = _INDEX[name]
        except KeyError as exc:
            raise KeyError(f"unknown WebDataset metric {name!r}") from exc
        with self._values.get_lock():
            self._values[index] = max(self._values[index], float(value))

    def record_shuffle_buffer(self, samples: int, retained_bytes: int) -> None:
        """Record process-wide shuffle high-water marks."""
        with self._values.get_lock():
            sample_index = _INDEX["shuffle_buffer_samples_peak"]
            byte_index = _INDEX["shuffle_buffer_bytes_peak"]
            self._values[sample_index] = max(
                self._values[sample_index], float(samples)
            )
            self._values[byte_index] = max(
                self._values[byte_index], float(retained_bytes)
            )

    def record_bucket_pool(self, fill_ratio: float) -> None:
        """Record one bounded bucketing pool drain and its capacity usage."""
        ratio = min(1.0, max(0.0, float(fill_ratio)))
        with self._values.get_lock():
            self._values[_INDEX["bucket_pools_drained"]] += 1.0
            self._values[_INDEX["bucket_fill_ratio_sum"]] += ratio

    def record_filtered(self, reason: str, value: float = 1.0) -> None:
        """Count filtered samples with a bounded, process-safe reason label."""
        counter = f"samples_filtered_{reason}"
        if counter not in _INDEX:
            raise KeyError(f"unknown WebDataset filter reason {reason!r}")
        with self._values.get_lock():
            self._values[_INDEX["samples_filtered"]] += float(value)
            self._values[_INDEX[counter]] += float(value)

    def record_error(self, error: Exception, *, stage: str) -> None:
        counter = _ERROR_COUNTERS.get(type(error).__name__)
        if counter is not None:
            self.increment(counter)
        elif stage == "prepare":
            self.increment("prepare_failures")
        elif stage == "materialize":
            self.increment("audio_decode_failures")
            codec = str(getattr(error, "codec", "other")).lower()
            codec_counter = f"audio_decode_failures_{codec}"
            if codec_counter not in _INDEX:
                codec_counter = "audio_decode_failures_other"
            self.increment(codec_counter)
        self.record_filtered(stage)

    def record_layout(self, layout: str) -> None:
        layout_counter = f"layout_matches_{layout}"
        if layout_counter in _INDEX:
            self.increment(layout_counter)

    def record_logical(self, task: str) -> None:
        self.increment("logical_samples_emitted")
        task_counter = f"logical_samples_emitted_{task}"
        if task_counter in _INDEX:
            self.increment(task_counter)

    def record_batch(self, lengths: Sequence[Any], data_wait_time: float) -> None:
        if not lengths:
            return
        padded = projected_padding_cost(lengths)
        unpadded = float(sum(item.total for item in lengths))
        with self._values.get_lock():
            self._values[_INDEX["batches_emitted"]] += 1.0
            self._values[_INDEX["batch_size_sum"]] += len(lengths)
            self._values[_INDEX["batch_cost_sum"]] += padded
            self._values[_INDEX["unpadded_cost_sum"]] += unpadded
            self._values[_INDEX["data_wait_time_sum"]] += max(0.0, data_wait_time)

    def snapshot(self) -> dict[str, float]:
        with self._values.get_lock():
            raw = {name: float(self._values[index]) for name, index in _INDEX.items()}
        batches = raw["batches_emitted"]
        padded = raw["batch_cost_sum"]
        raw["batch_size"] = raw["batch_size_sum"] / batches if batches else 0.0
        raw["batch_cost"] = padded / batches if batches else 0.0
        raw["padding_ratio"] = 1.0 - raw["unpadded_cost_sum"] / padded if padded else 0.0
        raw["data_wait_time"] = raw["data_wait_time_sum"] / batches if batches else 0.0
        pools = raw["bucket_pools_drained"]
        raw["bucket_fill_ratio"] = (
            raw["bucket_fill_ratio_sum"] / pools if pools else 0.0
        )
        return raw


class StreamErrorReporter:
    """Bridge task-isolated adapter failures to stream observability."""

    def __init__(
        self,
        metrics: SharedStreamMetrics,
        quarantine: "QuarantineWriter",
        source_name: Optional[str] = None,
    ):
        self.metrics = metrics
        self.quarantine = quarantine
        self.source_name = source_name

    def record_task(self, sample: Mapping[str, Any], task: str, error: Exception) -> None:
        self.metrics.record_error(error, stage="layout")
        self.quarantine.record(
            error,
            stage="layout",
            physical=sample,
            source_name=self.source_name,
            logical_task=task,
        )


class QuarantineWriter:
    """Append structured sample failures without retaining raw payload bytes."""

    def __init__(self, path: Optional[str], topology: StreamTopology, epoch: int):
        self.path = self._consumer_path(path, topology) if path else None
        self.topology = topology
        self.epoch = int(epoch)

    @staticmethod
    def _consumer_path(path: str, topology: StreamTopology) -> str:
        expanded = os.path.expandvars(os.path.expanduser(path))
        if topology.num_consumers == 1:
            return expanded
        target = Path(expanded)
        suffix = target.suffix or ".jsonl"
        stem = target.name[: -len(target.suffix)] if target.suffix else target.name
        name = f"{stem}.rank{topology.rank}.worker{topology.worker_id}{suffix}"
        return str(target.with_name(name))

    def record(
        self,
        error: Exception,
        *,
        stage: str,
        physical: Optional[Mapping[str, Any]] = None,
        logical: Any = None,
        source_name: Optional[str] = None,
        logical_task: Optional[str] = None,
        member_key: Optional[str] = None,
    ) -> None:
        if self.path is None:
            return
        canonical = getattr(logical, "canonical", logical)
        source = source_name or getattr(canonical, "source_name", None)
        shard = getattr(canonical, "shard_url", None)
        key = getattr(canonical, "physical_key", None)
        task = logical_task or getattr(canonical, "task", None)
        uid = getattr(canonical, "uid", None)
        if physical is not None:
            source = source or physical.get("__source__")
            shard = shard or physical.get("__url__")
            key = key or physical.get("__key__")
        record = {
            "source": source,
            "shard": shard,
            "key": key,
            "uid": uid,
            "task": task,
            "member": member_key or getattr(error, "member_key", None),
            "codec": getattr(error, "codec", None),
            "stage": stage,
            "error": type(error).__name__,
            "detail": str(error)[:4096],
            "epoch": self.epoch,
            "rank": self.topology.rank,
            "worker": self.topology.worker_id,
        }
        target = Path(self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(descriptor, line)
        finally:
            os.close(descriptor)
