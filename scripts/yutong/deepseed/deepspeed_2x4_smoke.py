#!/usr/bin/env python3
"""Validate a two-node, four-GPU-per-node DeepSpeed launch with NCCL."""

from __future__ import annotations

import datetime as dt
import os
import socket
import sys
import time
from collections import defaultdict

import torch
import torch.distributed as dist


EXPECTED_WORLD_SIZE = 8
EXPECTED_NODES = 2
EXPECTED_LOCAL_WORLD_SIZE = 4


def required_int_env(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"DeepSpeed launcher did not set {name}")
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from exc


def main() -> int:
    rank = required_int_env("RANK")
    world_size = required_int_env("WORLD_SIZE")
    local_rank = required_int_env("LOCAL_RANK")
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", EXPECTED_LOCAL_WORLD_SIZE))

    if world_size != EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            f"expected WORLD_SIZE={EXPECTED_WORLD_SIZE} (2x4), got {world_size}"
        )
    if local_world_size != EXPECTED_LOCAL_WORLD_SIZE:
        raise RuntimeError(
            f"expected LOCAL_WORLD_SIZE={EXPECTED_LOCAL_WORLD_SIZE}, got {local_world_size}"
        )
    if not 0 <= local_rank < EXPECTED_LOCAL_WORLD_SIZE:
        raise RuntimeError(f"LOCAL_RANK must be in [0, 3], got {local_rank}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in a DeepSpeed worker")
    if torch.cuda.device_count() < EXPECTED_LOCAL_WORLD_SIZE:
        raise RuntimeError(
            f"expected at least 4 visible GPUs per node, got {torch.cuda.device_count()}"
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=dt.timedelta(seconds=180),
    )

    try:
        # Verify that every one of the 8 ranks participates in an NCCL collective.
        checksum = torch.tensor(float(rank + 1), device="cuda")
        dist.all_reduce(checksum, op=dist.ReduceOp.SUM)
        expected_checksum = EXPECTED_WORLD_SIZE * (EXPECTED_WORLD_SIZE + 1) / 2
        if checksum.item() != expected_checksum:
            raise RuntimeError(
                f"NCCL all-reduce returned {checksum.item()}, expected {expected_checksum}"
            )

        # Exercise a nontrivial payload and report a rough collective duration.
        payload_mib = int(os.environ.get("SMOKE_ALLREDUCE_MIB", "32"))
        if payload_mib <= 0:
            raise RuntimeError("SMOKE_ALLREDUCE_MIB must be positive")
        elements = payload_mib * 1024 * 1024 // torch.tensor([], dtype=torch.float32).element_size()
        payload = torch.ones(elements, dtype=torch.float32, device="cuda")
        dist.all_reduce(payload)
        torch.cuda.synchronize()
        started = time.perf_counter()
        dist.all_reduce(payload)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        metadata = {
            "rank": rank,
            "local_rank": local_rank,
            "hostname": socket.gethostname(),
            "device": torch.cuda.get_device_name(local_rank),
            "allreduce_seconds": elapsed,
        }
        gathered = [None] * world_size
        dist.all_gather_object(gathered, metadata)

        validation_error = ""
        if rank == 0:
            by_host: dict[str, list[dict]] = defaultdict(list)
            for item in gathered:
                by_host[item["hostname"]].append(item)

            errors = []
            ranks = sorted(item["rank"] for item in gathered)
            if ranks != list(range(EXPECTED_WORLD_SIZE)):
                errors.append(f"global ranks are incomplete or duplicated: {ranks}")
            if len(by_host) != EXPECTED_NODES:
                errors.append(
                    f"expected {EXPECTED_NODES} hosts, observed {len(by_host)}: "
                    f"{sorted(by_host)}"
                )
            for hostname, items in sorted(by_host.items()):
                local_ranks = sorted(item["local_rank"] for item in items)
                if local_ranks != list(range(EXPECTED_LOCAL_WORLD_SIZE)):
                    errors.append(
                        f"host {hostname} has local ranks {local_ranks}, expected 0..7"
                    )
                max_seconds = max(item["allreduce_seconds"] for item in items)
                print(
                    f"host={hostname} ranks={sorted(item['rank'] for item in items)} "
                    f"local_ranks={local_ranks} max_{payload_mib}MiB_allreduce={max_seconds:.3f}s",
                    flush=True,
                )
            validation_error = "; ".join(errors)

        result = [validation_error]
        dist.broadcast_object_list(result, src=0)
        if result[0]:
            raise RuntimeError(result[0])

        dist.barrier()
        if rank == 0:
            print(
                "PASS: DeepSpeed launched 2 nodes x 4 GPUs and all 8 ranks "
                "completed NCCL collectives.",
                flush=True,
            )
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        rank = os.environ.get("RANK", "unknown")
        print(f"FAIL rank={rank}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
