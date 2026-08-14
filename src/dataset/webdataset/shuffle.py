"""Bounded-memory streaming shuffle helpers."""

from __future__ import annotations

import random
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional


def sample_nbytes(sample: Mapping[str, Any]) -> int:
    """Return the compressed payload size retained by a physical/logical sample."""
    assets = getattr(sample, "assets", None)
    if assets is not None:
        # Shared ASR/TTS assets are intentionally counted once per logical sample.
        return sum(len(asset.data) for asset in assets.values())
    return sum(len(value) for value in sample.values() if isinstance(value, bytes))


def byte_bounded_shuffle(
    source: Iterable[Any],
    *,
    max_samples: int,
    max_bytes: Optional[int] = None,
    initial_samples: int = 0,
    rng: random.Random | None = None,
    on_buffer_change: Optional[Callable[[int, int], None]] = None,
) -> Iterator[Any]:
    """Shuffle a stream while bounding retained sample count and optional bytes.

    Once either bound is reached, a random buffered item is emitted before the
    next input item is retained. The final buffer is drained in random order.
    ``initial_samples`` controls startup latency and must not exceed the sample
    bound; byte pressure may start output earlier. ``max_bytes=None`` disables
    the storage cap so only ``max_samples`` limits the window.
    """
    if max_samples <= 0:
        raise ValueError("shuffle max_samples must be positive")
    if max_bytes is not None and max_bytes <= 0:
        raise ValueError("shuffle max_bytes must be positive when set")
    if initial_samples < 0 or initial_samples > max_samples:
        raise ValueError("initial_samples must be in [0, max_samples]")

    rng = rng or random.Random()
    buffer = []
    sizes = []
    buffered_bytes = 0

    def observe_buffer() -> None:
        if on_buffer_change is not None:
            on_buffer_change(len(buffer), buffered_bytes)

    def pop_random():
        nonlocal buffered_bytes
        index = rng.randrange(len(buffer))
        item = buffer[index]
        buffered_bytes -= sizes[index]
        buffer[index] = buffer[-1]
        sizes[index] = sizes[-1]
        buffer.pop()
        sizes.pop()
        return item

    for item in source:
        size = sample_nbytes(item)
        while buffer and (
            len(buffer) >= max_samples
            or (max_bytes is not None and buffered_bytes + size > max_bytes)
        ):
            yield pop_random()
        if max_bytes is not None and size > max_bytes:
            # The source already materialized this one item; do not retain it in
            # addition to the bounded shuffle buffer.
            yield item
            continue
        buffer.append(item)
        sizes.append(size)
        buffered_bytes += size
        observe_buffer()
        if initial_samples and len(buffer) >= initial_samples:
            yield pop_random()

    while buffer:
        yield pop_random()
