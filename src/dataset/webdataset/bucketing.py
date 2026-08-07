"""Bounded streaming length bucketing and dynamic batch construction."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .shuffle import sample_nbytes
from .types import LengthVector


@dataclass(frozen=True)
class BatchLimits:
    max_cost: Optional[float] = None
    max_samples: int = 1
    min_samples: int = 1
    pool_samples: int = 2048
    pool_bytes: int = 1024**3
    chunk_size: int = 128
    oversized_sample: str = "drop"

    def __post_init__(self):
        if self.max_cost is not None and self.max_cost <= 0:
            raise ValueError("max_cost must be positive")
        if self.max_samples <= 0 or not 1 <= self.min_samples <= self.max_samples:
            raise ValueError("batch sample limits are invalid")
        if self.pool_samples <= 0 or self.pool_bytes <= 0 or self.chunk_size <= 0:
            raise ValueError("bucketing bounds must be positive")
        if self.oversized_sample not in {"drop", "single"}:
            raise ValueError("oversized_sample must be 'drop' or 'single'")


def projected_padding_cost(lengths: Iterable[LengthVector]) -> float:
    """Estimate the collator's independently padded text/audio token cost."""
    values = list(lengths)
    if not values:
        return 0.0
    count = len(values)
    return float(
        count
        * (
            max(item.user_text_tokens for item in values)
            + max(item.assistant_text_tokens for item in values)
            + max(item.user_audio_tokens for item in values)
            + max(item.assistant_audio_tokens for item in values)
            + max(item.codec_tokens for item in values)
        )
    )


def _prepared_nbytes(sample: Any) -> int:
    canonical = getattr(sample, "canonical", sample)
    return sample_nbytes(canonical)


def pool_sort_samples(
    source: Iterable[Any],
    *,
    pool_samples: int,
    pool_bytes: int,
    chunk_size: int,
    rng: random.Random,
    on_pool_drain: Optional[Callable[[float], None]] = None,
) -> Iterator[Any]:
    """Sort bounded pools by length, then shuffle similarly sized chunks."""
    pool = []
    retained_bytes = 0

    def drain():
        if on_pool_drain is not None:
            fill_ratio = max(
                len(pool) / pool_samples,
                retained_bytes / pool_bytes,
            )
            on_pool_drain(min(1.0, fill_ratio))
        pool.sort(key=lambda item: item.lengths.total)
        chunks = [pool[index : index + chunk_size] for index in range(0, len(pool), chunk_size)]
        rng.shuffle(chunks)
        for chunk in chunks:
            # Avoid a fixed order among samples with equal/near-equal lengths.
            rng.shuffle(chunk)
            yield from chunk

    for sample in source:
        size = _prepared_nbytes(sample)
        if pool and (len(pool) >= pool_samples or retained_bytes + size > pool_bytes):
            yield from drain()
            pool = []
            retained_bytes = 0
        pool.append(sample)
        retained_bytes += size
    if pool:
        yield from drain()


class DynamicBatchIterator(Iterator[list[Any]]):
    """Greedy batch iterator whose next sample is available for refill."""

    def __init__(
        self,
        source: Iterable[Any],
        limits: BatchLimits,
        on_sample_dropped: Optional[Callable[[Any], None]] = None,
    ):
        self.source = iter(source)
        self.limits = limits
        self.on_sample_dropped = on_sample_dropped
        self.pending = None

    def take_sample(self):
        """Take the boundary sample first, then continue the ordered source."""
        if self.pending is not None:
            sample, self.pending = self.pending, None
            return sample
        return next(self.source)

    def put_back(self, sample: Any) -> None:
        if self.pending is not None:
            raise RuntimeError("dynamic batch iterator already has a pending sample")
        self.pending = sample

    def __iter__(self):
        return self

    def __next__(self) -> list[Any]:
        batch: list[Any] = []
        while True:
            try:
                sample = self.take_sample()
            except StopIteration:
                if batch:
                    return batch
                raise

            single_cost = projected_padding_cost([sample.lengths])
            if self.limits.max_cost is not None and single_cost > self.limits.max_cost:
                if self.limits.oversized_sample == "single":
                    if batch:
                        self.pending = sample
                        return batch
                    return [sample]
                if self.on_sample_dropped is not None:
                    self.on_sample_dropped(sample)
                continue

            candidate = batch + [sample]
            exceeds = len(candidate) > self.limits.max_samples or (
                self.limits.max_cost is not None
                and projected_padding_cost(item.lengths for item in candidate)
                > self.limits.max_cost
            )
            if exceeds and batch:
                self.pending = sample
                return batch
            batch = candidate


def dynamic_batches(
    source: Iterable[Any],
    limits: BatchLimits,
    on_sample_dropped: Optional[Callable[[Any], None]] = None,
) -> Iterator[list[Any]]:
    """Greedily form batches constrained by projected padding cost and size."""
    yield from DynamicBatchIterator(source, limits, on_sample_dropped)
