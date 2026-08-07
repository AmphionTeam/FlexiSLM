"""Deterministic epoch and shard assignment for streaming WebDataset."""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import random
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch


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
