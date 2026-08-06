#!/usr/bin/env python3
"""Precompute a FlexiSLM-compatible JSONL index from WebDataset tar shards.

Example:
    python local/precompute_webdataset_index.py \
        --input '/path/to/shards/*.tar' \
        --output /path/to/webdataset_index.jsonl \
        --task tts \
        --prompt-template 'Read the following text out loud: {text}' \
        --limit 1000 \
        --audio-variant nonoise
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
from typing import Any, Callable, Iterable


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


DEFAULT_TTS_PROMPT_TEMPLATE = "Read the following text out loud: {text}"
DEFAULT_ASR_PROMPT = "Transcribe the following audio:"
INDEX_COLUMNS = (
    "messages",
    "audios",
    "audio_durations",
    "webdataset_tar_path",
    "webdataset_audio_member",
    "webdataset_audio_variant",
)


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = _non_negative_int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _validate_prompt_template(value: str) -> str:
    try:
        value.format(text="example")
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "prompt template must be format-compatible with the {text} field"
        ) from exc
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan WebDataset tar shards and write a FlexiSLM JSONL index with "
            "messages, audios, optional audio_durations, and WebDataset debug fields."
        )
    )
    parser.add_argument(
        "--input",
        "--data-path",
        "--tar-path",
        dest="data_path",
        required=True,
        help="Input tar glob, tar directory, or single .tar file. Quote globs in the shell.",
    )
    parser.add_argument(
        "--output",
        "--output-jsonl",
        dest="output_jsonl",
        required=True,
        help="Output JSONL index path.",
    )
    parser.add_argument(
        "--task",
        "--webdataset-task",
        dest="task",
        default="tts",
        choices=("tts", "asr", "s2t", "speech_to_text"),
        help="Row construction mode for source JSON records without messages.",
    )
    parser.add_argument(
        "--text-key",
        default="text",
        help="Source JSON key used as text when constructing TTS/ASR rows.",
    )
    parser.add_argument(
        "--prompt-template",
        "--tts-prompt-template",
        dest="tts_prompt_template",
        default=DEFAULT_TTS_PROMPT_TEMPLATE,
        type=_validate_prompt_template,
        help="TTS user prompt template. Use {text} for the source text.",
    )
    parser.add_argument(
        "--asr-prompt",
        default=DEFAULT_ASR_PROMPT,
        help="ASR user prompt prepended before the <|audio|> tag.",
    )
    parser.add_argument(
        "--audio-variant",
        default="noisy",
        choices=("noisy", "nonoise"),
        help="Audio member variant to prefer when JSON records expose variants.",
    )
    parser.add_argument(
        "--limit",
        type=_non_negative_int,
        default=None,
        help="Stop after writing this many rows. Useful for smoke indexes.",
    )
    parser.add_argument(
        "--log-interval",
        type=_positive_int,
        default=1000,
        help="Log write progress every N rows.",
    )
    return parser.parse_args()


def _load_dataset_helpers() -> tuple[Callable[[str], list[str]], Callable[..., Iterable[dict[str, Any]]]]:
    from src.dataset.base import (  # pylint: disable=import-outside-toplevel
        _collect_webdataset_tar_paths,
        _iter_webdataset_rows,
    )

    return _collect_webdataset_tar_paths, _iter_webdataset_rows


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_audio_durations(row: dict[str, Any]) -> None:
    audios = row.get("audios")
    audio_count = len(audios) if isinstance(audios, list) and audios else 1

    if "audio_durations" in row and row["audio_durations"] is not None:
        durations = row["audio_durations"]
        if not isinstance(durations, list):
            durations = [durations]
        parsed = [_to_float(duration) for duration in durations]
        if len(parsed) == 1 and audio_count > 1:
            parsed = parsed * audio_count
        row["audio_durations"] = parsed
        return

    duration = _to_float(row.get("duration"))
    if duration is not None:
        row["audio_durations"] = [duration] * audio_count


def _build_index_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    _normalize_audio_durations(row)
    return {key: row[key] for key in INDEX_COLUMNS if key in row}


def _data_info_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "webdataset_task": args.task,
        "task": args.task,
        "text_key": args.text_key,
        "tts_prompt_template": args.tts_prompt_template,
        "asr_prompt": args.asr_prompt,
    }


def write_index(args: argparse.Namespace) -> int:
    collect_tar_paths, iter_webdataset_rows = _load_dataset_helpers()
    tar_paths = collect_tar_paths(args.data_path)
    if not tar_paths:
        raise FileNotFoundError(
            f"No .tar shards found for input {args.data_path!r}. "
            "Pass a tar glob, a directory containing .tar files, or a single .tar file."
        )

    output_jsonl = os.path.abspath(args.output_jsonl)
    output_dir = os.path.dirname(output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logging.info(
        "Scanning %d tar shard(s) from %s; writing index to %s",
        len(tar_paths),
        args.data_path,
        output_jsonl,
    )
    logging.info(
        "Options: task=%s, text_key=%s, audio_variant=%s, limit=%s",
        args.task,
        args.text_key,
        args.audio_variant,
        args.limit,
    )

    tmp_path = f"{output_jsonl}.tmp"
    count = 0
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            rows = iter_webdataset_rows(
                tar_paths,
                _data_info_from_args(args),
                args.audio_variant,
                max_rows=args.limit,
                log_label="PrecomputeWebDatasetIndex",
            )
            for row in rows:
                index_row = _build_index_row(row)
                f.write(json.dumps(index_row, ensure_ascii=False) + "\n")
                count += 1
                if count % args.log_interval == 0:
                    logging.info("Wrote %d rows to %s", count, output_jsonl)
        os.replace(tmp_path, output_jsonl)
    except Exception:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise

    if count == 0:
        logging.warning("No valid rows were written to %s", output_jsonl)
    else:
        logging.info("Finished writing %d rows to %s", count, output_jsonl)
    return count


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    write_index(args)


if __name__ == "__main__":
    main()
