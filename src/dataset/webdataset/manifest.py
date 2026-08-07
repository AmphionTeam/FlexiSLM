"""Validated shard manifest loading for native WebDataset sources."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse


def _resolve_url(value: str, manifest_path: Path) -> str:
    value = os.path.expandvars(os.path.expanduser(value.strip()))
    if not value:
        raise ValueError("shard manifest URL must not be empty")
    parsed = urlparse(value)
    if parsed.scheme or os.path.isabs(value):
        return value
    return str((manifest_path.parent / value).resolve())


def _entry_url(entry: Any, *, line: int) -> str:
    if isinstance(entry, str):
        return entry
    if not isinstance(entry, Mapping):
        raise ValueError(f"shard manifest entry {line} must be a string or object")
    if entry.get("disabled", False):
        return ""
    value = entry.get("url", entry.get("path"))
    if not isinstance(value, str):
        raise ValueError(f"shard manifest entry {line} requires string 'url' or 'path'")
    return value


def _json_entries(payload: Any) -> Iterable[Any]:
    if isinstance(payload, Mapping):
        payload = payload.get("shards")
    if not isinstance(payload, list):
        raise ValueError("JSON shard manifest root must be a list or {'shards': [...]} object")
    return payload


def load_shard_manifest(path: str) -> tuple[str, ...]:
    """Load concrete shard URLs from JSON, JSONL, or newline text.

    JSON/JSONL entries may be URL strings or objects containing ``url``/``path``.
    Object entries with ``disabled: true`` are ignored. Relative local paths are
    resolved against the manifest directory, and duplicate URLs are rejected.
    """
    expanded = Path(os.path.expandvars(os.path.expanduser(path))).resolve()
    if not expanded.is_file():
        raise FileNotFoundError(f"WebDataset shard manifest not found: {expanded}")

    suffix = expanded.suffix.lower()
    text = expanded.read_text(encoding="utf-8")
    if suffix == ".json":
        entries = list(_json_entries(json.loads(text)))
    else:
        entries = []
        for line_number, raw in enumerate(text.splitlines(), 1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if suffix == ".jsonl":
                try:
                    entries.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in shard manifest {expanded}:{line_number}: {exc}"
                    ) from exc
            else:
                entries.append(stripped)

    urls = []
    seen = set()
    for line_number, entry in enumerate(entries, 1):
        value = _entry_url(entry, line=line_number)
        if not value:
            continue
        url = _resolve_url(value, expanded)
        if url in seen:
            raise ValueError(f"duplicate shard URL in manifest {expanded}: {url}")
        seen.add(url)
        urls.append(url)
    if not urls:
        raise ValueError(f"WebDataset shard manifest is empty: {expanded}")
    return tuple(urls)
