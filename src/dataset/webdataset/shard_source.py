"""Deterministic epoch and shard assignment for streaming WebDataset."""

from __future__ import annotations

import hashlib
import itertools
import math
import multiprocessing as mp
import numbers
import os
import random
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch


def validate_shard_ratio(source_name: str, ratio: object) -> float:
    """Validate and normalize one source's shard retention ratio."""
    normalized_ratio = None
    if not isinstance(ratio, bool) and isinstance(ratio, numbers.Real):
        try:
            normalized_ratio = float(ratio)
        except (OverflowError, TypeError, ValueError):
            pass
    if (
        normalized_ratio is None
        or not math.isfinite(normalized_ratio)
        or not 0 < normalized_ratio <= 1
    ):
        raise ValueError(
            f"WebDataset source {source_name!r} has invalid ratio {ratio!r}; "
            "expected a finite number in the range 0 < ratio <= 1"
        )
    return normalized_ratio


def select_shards_by_ratio(
    shards: Sequence[str],
    *,
    ratio: object,
    seed: int,
    source_name: str,
) -> tuple[str, ...]:
    """Select a stable, seed-based shard subset while preserving input order."""
    normalized_ratio = validate_shard_ratio(source_name, ratio)
    # Treat each URL as one candidate if data paths overlap after expansion.
    candidates = tuple(dict.fromkeys(shards))
    if normalized_ratio == 1.0 or not candidates:
        return candidates

    selected_count = max(1, math.floor(len(candidates) * normalized_ratio))
    # Derive a source-specific seed without Python's process-randomized hash().
    digest = hashlib.sha256(
        f"{int(seed)}\0{source_name}".encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    selected_indices = sorted(rng.sample(range(len(candidates)), selected_count))
    return tuple(candidates[index] for index in selected_indices)


class SharedEpoch:
    """A process-shared epoch counter visible to persistent workers."""

    def __init__(self, epoch: int = 0):
        self._value = mp.Value("q", int(epoch))

    def set(self, epoch: int) -> None:
        with self._value.get_lock():
            self._value.value = int(epoch)

    def get(self) -> int:
        return int(self._value.value)


@dataclass(frozen=True)
class ShardSource:
    """A named collection of shards participating in source-level mixing."""

    name: str
    shards: Sequence[str]
    weight: float = 1.0


@dataclass(frozen=True)
class ShardRef:
    """A shard URL tagged with the source whose layout should interpret it."""

    url: str
    source_name: str


@dataclass(frozen=True)
class StreamTopology:
    rank: int = 0
    world_size: int = 1
    worker_id: int = 0
    num_workers: int = 1

    @property
    def consumer_id(self) -> int:
        return self.rank * self.num_workers + self.worker_id

    @property
    def num_consumers(self) -> int:
        return self.world_size * self.num_workers


def current_topology() -> StreamTopology:
    """Return the data-parallel rank and current DataLoader worker identity."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    worker = torch.utils.data.get_worker_info()
    return StreamTopology(
        rank=rank,
        world_size=world_size,
        worker_id=worker.id if worker is not None else 0,
        num_workers=worker.num_workers if worker is not None else 1,
    )


def assigned_shards(
    shards: Sequence[str],
    *,
    mode: str,
    seed: int,
    epoch: int,
    topology: StreamTopology,
    shuffle: bool = True,
) -> Iterator[str]:
    """Yield shards assigned to one rank/worker before any tar is opened.

    ``finite_exact`` never duplicates a shard. ``finite_padded`` repeats the
    shuffled prefix so every consumer receives the same shard count.
    ``resampled`` independently samples an unbounded deterministic stream for
    each consumer.
    """
    if not shards:
        return
    if mode not in {"finite_exact", "finite_padded", "resampled"}:
        raise ValueError(f"unsupported WebDataset sampling mode: {mode!r}")

    if mode == "resampled":
        rng = random.Random(seed + 1_000_003 * epoch + 97 * topology.consumer_id)
        while True:
            yield shards[rng.randrange(len(shards))]
        return

    order = list(shards)
    if shuffle:
        random.Random(seed + epoch).shuffle(order)
    consumers = topology.num_consumers
    if mode == "finite_padded":
        target = ((len(order) + consumers - 1) // consumers) * consumers
        order = list(itertools.islice(itertools.cycle(order), target))
    yield from order[topology.consumer_id :: consumers]


def assigned_source_shards(
    sources: Sequence[ShardSource],
    *,
    mode: str,
    seed: int,
    epoch: int,
    topology: StreamTopology,
    shuffle: bool = True,
) -> Iterator[ShardRef]:
    """Assign named shards, sampling sources by weight in resampled mode.

    Source weights apply at source level rather than shard level, so adding
    shards to a source does not silently change its sampling ratio. Finite
    modes consume every shard and therefore require uniform weights.
    """
    if not sources:
        return
    names = [source.name for source in sources]
    if len(names) != len(set(names)):
        raise ValueError("WebDataset source names must be unique")
    if any(not source.name or not source.shards for source in sources):
        raise ValueError("each WebDataset source requires a name and at least one shard")
    if any(source.weight <= 0 for source in sources):
        raise ValueError("WebDataset source weights must be positive")

    if mode != "resampled":
        if any(source.weight != sources[0].weight for source in sources[1:]):
            raise ValueError("non-uniform source weights require sampling.mode=resampled")
        refs = [
            ShardRef(url, source.name)
            for source in sources
            for url in source.shards
        ]
        yield from assigned_shards(
            refs,
            mode=mode,
            seed=seed,
            epoch=epoch,
            topology=topology,
            shuffle=shuffle,
        )
        return

    rng = random.Random(seed + 1_000_003 * epoch + 97 * topology.consumer_id)
    cumulative = []
    total = 0.0
    for source in sources:
        total += float(source.weight)
        cumulative.append(total)
    while True:
        value = rng.random() * total
        source_index = next(
            index for index, threshold in enumerate(cumulative) if value < threshold
        )
        source = sources[source_index]
        yield ShardRef(
            source.shards[rng.randrange(len(source.shards))], source.name
        )
