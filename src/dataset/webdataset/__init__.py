"""Canonical types and physical layout adapters for native WebDataset input."""

from .layouts import (
    AdapterContext,
    AmbiguousLayoutError,
    AudioBindingMismatchError,
    DuplicateMemberError,
    ListErrorReporter,
    MalformedJsonError,
    MetadataDecodeError,
    MissingMemberError,
    PhysicalLayoutRegistry,
    S2SPairAdapter,
    SharedAudioTasksAdapter,
    UnknownLayoutError,
)
from .bucketing import (
    BatchLimits,
    DynamicBatchIterator,
    dynamic_batches,
    pool_sort_samples,
    projected_padding_cost,
)
from .shard_source import SharedEpoch, StreamTopology, assigned_shards
from .shuffle import byte_bounded_shuffle, sample_nbytes
from .types import AudioAsset, AudioBinding, CanonicalSample, LengthVector

__all__ = [
    "AdapterContext",
    "AmbiguousLayoutError",
    "AudioAsset",
    "AudioBinding",
    "AudioBindingMismatchError",
    "CanonicalSample",
    "DuplicateMemberError",
    "LengthVector",
    "ListErrorReporter",
    "MalformedJsonError",
    "MetadataDecodeError",
    "MissingMemberError",
    "PhysicalLayoutRegistry",
    "S2SPairAdapter",
    "SharedAudioTasksAdapter",
    "UnknownLayoutError",
    "byte_bounded_shuffle",
    "sample_nbytes",
    "BatchLimits",
    "DynamicBatchIterator",
    "dynamic_batches",
    "pool_sort_samples",
    "projected_padding_cost",
    "SharedEpoch",
    "StreamTopology",
    "assigned_shards",
]
