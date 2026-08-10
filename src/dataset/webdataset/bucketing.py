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
    max_samples: Optional[int] = 1
    min_samples: int = 1
    pool_samples: int = 2048
    pool_bytes: int = 1024**3
    chunk_size: int = 128
    oversized_sample: str = "drop"

    def __post_init__(self):
        if self.max_cost is not None and self.max_cost <= 0:
            raise ValueError("max_cost must be positive")
        if self.max_samples is not None and (
            self.max_samples <= 0 or not 1 <= self.min_samples <= self.max_samples
        ):
            raise ValueError("batch sample limits are invalid")
        if self.max_samples is None and self.max_cost is None:
            raise ValueError("at least one batch limit must be configured")
        if self.min_samples <= 0:
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
    """Continuously refill a bounded pool and emit similar-length chunks."""
    source = iter(source)
    pool = []
    ready = []
    pending = None
    retained_bytes = 0
    source_exhausted = False

    def observe_pool() -> None:
        if on_pool_drain is None:
            return
        fill_ratio = max(
            (len(pool) + len(ready)) / pool_samples,
            retained_bytes / pool_bytes,
        )
        on_pool_drain(min(1.0, fill_ratio))

    def take_source():
        nonlocal pending, source_exhausted
        if pending is not None:
            sample, pending = pending, None
            return sample
        if source_exhausted:
            return None
        try:
            return next(source)
        except StopIteration:
            source_exhausted = True
            return None

    def refill_one() -> None:
        nonlocal pending, retained_bytes
        sample = take_source()
        if sample is None:
            return
        size = _prepared_nbytes(sample)
        retained_samples = len(pool) + len(ready)
        if retained_samples and (
            retained_samples >= pool_samples or retained_bytes + size > pool_bytes
        ):
            pending = sample
            return
        # A single item larger than the byte limit is allowed through without
        # retaining another item beside it, matching byte_bounded_shuffle.
        pool.append(sample)
        retained_bytes += size

    while len(pool) + len(ready) < pool_samples and not source_exhausted:
        before = len(pool)
        refill_one()
        if pending is not None or len(pool) == before:
            break

    while pool or ready or pending is not None:
        if not ready:
            if not pool:
                refill_one()
                if not pool:
                    break
            observe_pool()
            pool.sort(key=lambda item: item.lengths.total)
            chunk_count = (len(pool) + chunk_size - 1) // chunk_size
            chunk_index = rng.randrange(chunk_count)
            start = chunk_index * chunk_size
            ready = pool[start : start + chunk_size]
            del pool[start : start + chunk_size]
            # Avoid a fixed order among samples with equal/near-equal lengths.
            rng.shuffle(ready)

        sample = ready.pop()
        retained_bytes -= _prepared_nbytes(sample)
        yield sample

        # This code runs when the consumer requests its next sample, spreading
        # one upstream read/prepare operation across each delivered item.
        refill_one()


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
            exceeds = (
                self.limits.max_samples is not None
                and len(candidate) > self.limits.max_samples
            ) or (
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
