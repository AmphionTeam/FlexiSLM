"""Atomic node-local cache for assigned WebDataset shards."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import unquote, urlparse


@dataclass(frozen=True)
class ShardCacheConfig:
    directory: str
    max_bytes: int
    cache_local_files: bool = True
    copy_chunk_bytes: int = 8 * 1024**2

    def __post_init__(self):
        if not self.directory:
            raise ValueError("node-local shard cache requires a directory")
        if self.max_bytes <= 0:
            raise ValueError("node-local shard cache max_bytes must be positive")
        if self.copy_chunk_bytes <= 0:
            raise ValueError("node-local shard cache copy_chunk_bytes must be positive")


class NodeLocalShardCache:
    """Copy shards once per node using file locks and atomic replacement.

    Cached filenames include a hash of the original URL. The tar reader still
    receives the original URL as sample metadata, keeping logical UIDs stable.
    """

    def __init__(
        self,
        config: ShardCacheConfig,
        event_callback: Optional[Callable[[str, float], None]] = None,
    ):
        self.config = config
        self.directory = Path(
            os.path.expandvars(os.path.expanduser(config.directory))
        ).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._event_callback = event_callback

    def _event(self, name: str, value: float = 1.0) -> None:
        if self._event_callback is not None:
            self._event_callback(name, value)

    @staticmethod
    def _local_path(url: str) -> Optional[Path]:
        parsed = urlparse(url)
        if parsed.scheme == "file":
            return Path(unquote(parsed.path))
        if not parsed.scheme:
            return Path(url)
        return None

    def _target(self, url: str) -> Path:
        parsed = urlparse(url)
        basename = Path(unquote(parsed.path)).name or "shard.tar"
        basename = re.sub(r"[^A-Za-z0-9._-]+", "_", basename)[:160]
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        return self.directory / f"{digest}--{basename}"

    @contextmanager
    def open_path(self, url: str):
        """Yield a local path while preventing eviction of the active shard."""
        local_path = self._local_path(url)
        if local_path is not None and not self.config.cache_local_files:
            yield url
            return
        target = self._target(url)
        if local_path is not None:
            try:
                if local_path.resolve() == target:
                    yield str(target)
                    return
            except FileNotFoundError:
                pass

        lock_path = target.with_name(target.name + ".lock")
        with lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            self._ensure_cached(url, local_path, target)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
            try:
                yield str(target)
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def resolve(self, url: str) -> str:
        """Populate the cache and return its path without an active-use lock."""
        with self.open_path(url) as path:
            return path

    def _ensure_cached(
        self, url: str, local_path: Optional[Path], target: Path
    ) -> None:
        if target.is_file():
            os.utime(target, None)
            self._event("shard_cache_hits")
            return

        temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
        try:
            size = self._copy(url, local_path, temporary)
            os.replace(temporary, target)
            self._event("shard_cache_misses")
            self._event("shard_cache_bytes_written", size)
            self._evict(exclude=target)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _copy(self, url: str, local_path: Optional[Path], target: Path) -> int:
        if local_path is not None:
            source = local_path.open("rb")
        else:
            try:
                import webdataset as wds
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("shard cache requires 'webdataset'") from exc
            source = wds.gopen(url)

        try:
            with target.open("wb") as output:
                shutil.copyfileobj(
                    source, output, length=self.config.copy_chunk_bytes
                )
                output.flush()
                os.fsync(output.fileno())
            return target.stat().st_size
        finally:
            source.close()

    def _evict(self, *, exclude: Path) -> None:
        global_lock = self.directory / ".eviction.lock"
        with global_lock.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            entries = []
            total = 0
            for path in self.directory.iterdir():
                if (
                    not path.is_file()
                    or path.name.endswith(".lock")
                    or not re.match(r"^[0-9a-f]{24}--", path.name)
                ):
                    continue
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                total += stat.st_size
                if path != exclude:
                    entries.append((stat.st_mtime_ns, path, stat.st_size))
            for _, path, size in sorted(entries):
                if total <= self.config.max_bytes:
                    break
                entry_lock = path.with_name(path.name + ".lock")
                with entry_lock.open("a+b") as entry_handle:
                    try:
                        fcntl.flock(
                            entry_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                        )
                    except BlockingIOError:
                        continue
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        continue
                total -= size
                self._event("shard_cache_evictions")
