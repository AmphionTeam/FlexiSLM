#!/usr/bin/env python3
"""Audit native WebDataset shards without decoding audio payloads."""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping, Optional, Sequence

# Allow direct execution from the repository root or from local/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.webdataset.layouts import (  # noqa: E402
    AdapterContext,
    AmbiguousLayoutError,
    DuplicateMemberError,
    ListErrorReporter,
    S2SPairAdapter,
    SharedAudioTasksAdapter,
    UnknownLayoutError,
)
from src.dataset.webdataset.manifest import load_shard_manifest  # noqa: E402
from src.dataset.webdataset.types import SHARED_AUDIO_TASKS  # noqa: E402

_BASE_PLUS_EXT = re.compile(r"^((?:.*/|)[^.]+)[.]([^/]*)$")
_AUDIO_MEMBER = re.compile(r"^(question|response|audio)\.(wav|flac|mp3|m4a|ogg)$", re.I)


class NonContiguousKeyError(ValueError):
    """A physical key reappeared after another key had started."""


class InvalidMemberNameError(ValueError):
    """A regular tar member cannot be grouped by WebDataset."""


class EmptyMemberError(ValueError):
    """A required payload is empty."""


_LAYOUTS = {
    "s2s_pair": S2SPairAdapter(),
    "shared_audio_tasks": SharedAudioTasksAdapter(),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate tar grouping, native layouts, metadata, and audio bindings."
    )
    parser.add_argument("shards", nargs="*", help="Tar paths or brace-expanded patterns")
    parser.add_argument("--manifest", help="JSON, JSONL, or text shard manifest")
    parser.add_argument(
        "--layout",
        choices=("auto", "s2s_pair", "shared_audio_tasks"),
        default="auto",
    )
    parser.add_argument("--source-name", default="inspection")
    parser.add_argument(
        "--tasks", nargs="+", choices=SHARED_AUDIO_TASKS, default=SHARED_AUDIO_TASKS
    )
    parser.add_argument("--max-error-examples", type=int, default=100)
    parser.add_argument(
        "--output", help="Write the JSON report to this path instead of stdout"
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Exit successfully even when validation errors are found",
    )
    args = parser.parse_args(argv)
    if bool(args.manifest) == bool(args.shards):
        parser.error("configure exactly one of positional shards or --manifest")
    if args.max_error_examples < 0:
        parser.error("--max-error-examples must be non-negative")
    return args


def _expand_shards(values: Sequence[str]) -> tuple[str, ...]:
    try:
        from braceexpand import braceexpand
    except ImportError:
        braceexpand = None
    result = []
    for value in values:
        expanded = list(braceexpand(value)) if braceexpand is not None else [value]
        result.extend(expanded)
    if not result:
        raise ValueError("at least one shard is required")
    if len(result) != len(set(result)):
        raise ValueError("duplicate shard path after brace expansion")
    return tuple(result)


@contextlib.contextmanager
def _open_shard(url: str) -> Iterator[BinaryIO]:
    if "://" not in url or url.startswith("file://"):
        path = url[7:] if url.startswith("file://") else url
        with open(path, "rb") as stream:
            yield stream
        return
    import webdataset as wds

    stream = wds.gopen(url)
    try:
        yield stream
    finally:
        stream.close()


def _split_member(name: str) -> tuple[Optional[str], Optional[str]]:
    match = _BASE_PLUS_EXT.match(name)
    if match is None:
        return None, None
    return match.group(1), match.group(2)


class Inspector:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.counts: Counter[str] = Counter()
        self.layouts: Counter[str] = Counter()
        self.tasks: Counter[str] = Counter()
        self.codecs: Counter[str] = Counter()
        self.error_types: Counter[str] = Counter()
        self.error_examples: list[dict[str, Any]] = []
        self.shard_reports: list[dict[str, Any]] = []

    def error(
        self,
        error: Exception,
        *,
        shard: str,
        key: Optional[str] = None,
        member: Optional[str] = None,
    ) -> None:
        name = type(error).__name__
        self.error_types[name] += 1
        self.counts["errors"] += 1
        if len(self.error_examples) < self.args.max_error_examples:
            self.error_examples.append(
                {
                    "shard": shard,
                    "key": key,
                    "member": member,
                    "error": name,
                    "detail": str(error)[:4096],
                }
            )

    def _adapter(self, sample: Mapping[str, Any]):
        if self.args.layout != "auto":
            return _LAYOUTS[self.args.layout]
        keys = set(sample)
        s2s_clue = "json" in keys or any(
            key.startswith(("question.", "response.")) for key in keys
        )
        shared_clue = bool(
            {f"{task}.json" for task in SHARED_AUDIO_TASKS} & keys
        ) or any(key.startswith("audio.") for key in keys)
        if s2s_clue and shared_clue:
            raise AmbiguousLayoutError(
                "members contain both s2s_pair and shared_audio_tasks clues"
            )
        if s2s_clue:
            return _LAYOUTS["s2s_pair"]
        if shared_clue:
            return _LAYOUTS["shared_audio_tasks"]
        raise UnknownLayoutError("no native physical layout clues found")

    def inspect_sample(
        self, sample: dict[str, Any], duplicate_suffixes: Sequence[str]
    ) -> None:
        shard = str(sample["__url__"])
        key = str(sample["__key__"])
        self.counts["physical_samples"] += 1
        if duplicate_suffixes:
            self.error(
                DuplicateMemberError(
                    f"duplicate member suffixes within key: {sorted(set(duplicate_suffixes))}"
                ),
                shard=shard,
                key=key,
            )
            return
        try:
            adapter = self._adapter(sample)
            reporter = ListErrorReporter()
            context = AdapterContext(
                source_name=self.args.source_name,
                tasks=tuple(self.args.tasks),
                error_reporter=reporter,
            )
            self.layouts[adapter.name] += 1
            logical = list(adapter.expand(sample, context))
            for item in logical:
                self.tasks[item.task] += 1
                self.counts["logical_samples"] += 1
            for _, task, error in reporter.errors:
                self.error(error, shard=shard, key=key, member=f"{task}.json")
        except Exception as error:
            self.error(error, shard=shard, key=key)

    def inspect_shard(self, shard: str) -> None:
        before_samples = self.counts["physical_samples"]
        before_members = self.counts["members"]
        before_bytes = self.counts["member_bytes"]
        before_errors = self.counts["errors"]
        self.counts["shards"] += 1
        closed_keys: set[str] = set()
        current_key: Optional[str] = None
        sample: dict[str, Any] = {}
        duplicate_suffixes: list[str] = []

        def flush() -> None:
            nonlocal sample, duplicate_suffixes
            if sample:
                self.inspect_sample(sample, duplicate_suffixes)
            sample = {}
            duplicate_suffixes = []

        try:
            with _open_shard(shard) as stream, tarfile.open(
                fileobj=stream, mode="r|*"
            ) as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    self.counts["members"] += 1
                    self.counts["member_bytes"] += member.size
                    key, suffix = _split_member(member.name)
                    if key is None or suffix is None:
                        self.error(
                            InvalidMemberNameError(
                                "member name cannot be split into a WebDataset key and suffix"
                            ),
                            shard=shard,
                            member=member.name,
                        )
                        continue
                    if member.size == 0:
                        self.error(
                            EmptyMemberError("regular tar member has an empty payload"),
                            shard=shard,
                            key=key,
                            member=member.name,
                        )
                    if key != current_key:
                        if current_key is not None:
                            closed_keys.add(current_key)
                            flush()
                        if key in closed_keys:
                            self.error(
                                NonContiguousKeyError(
                                    "sample key is non-contiguous in tar stream"
                                ),
                                shard=shard,
                                key=key,
                                member=member.name,
                            )
                        current_key = key
                        sample = {"__key__": key, "__url__": shard}
                    if suffix in sample:
                        duplicate_suffixes.append(suffix)
                    match = _AUDIO_MEMBER.fullmatch(suffix)
                    if match:
                        self.codecs[match.group(2).lower()] += 1
                        sample[suffix] = b""
                    elif suffix.endswith("json"):
                        extracted = archive.extractfile(member)
                        sample[suffix] = extracted.read() if extracted is not None else b""
                    else:
                        sample[suffix] = b""
                flush()
        except Exception as error:
            self.error(error, shard=shard)
        self.shard_reports.append(
            {
                "shard": shard,
                "physical_samples": self.counts["physical_samples"] - before_samples,
                "members": self.counts["members"] - before_members,
                "member_bytes": self.counts["member_bytes"] - before_bytes,
                "errors": self.counts["errors"] - before_errors,
            }
        )

    def report(self) -> dict[str, Any]:
        summary = {
            name: self.counts[name]
            for name in (
                "shards",
                "members",
                "member_bytes",
                "physical_samples",
                "logical_samples",
                "errors",
            )
        }
        return {
            "ok": self.counts["errors"] == 0,
            "summary": summary,
            "layouts": dict(sorted(self.layouts.items())),
            "tasks": dict(sorted(self.tasks.items())),
            "audio_codecs": dict(sorted(self.codecs.items())),
            "error_types": dict(sorted(self.error_types.items())),
            "error_examples": self.error_examples,
            "shards": self.shard_reports,
        }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    shards = (
        load_shard_manifest(args.manifest)
        if args.manifest
        else _expand_shards(args.shards)
    )
    inspector = Inspector(args)
    for shard in shards:
        inspector.inspect_shard(shard)
    report = inspector.report()
    output = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0 if report["ok"] or args.allow_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
