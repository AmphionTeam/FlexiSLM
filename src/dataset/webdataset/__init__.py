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
]
