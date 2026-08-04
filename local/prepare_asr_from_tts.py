#!/usr/bin/env python3
"""Convert duration-annotated TTS JSONL files into ASR JSONL files in parallel."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import tempfile
from pathlib import Path

SYSTEM_PROMPT = "Your Name: Omni\nYour Gender: female \n\nRespond in a text-only manner."
TTS_PROMPT_PREFIX = "Read the following text out loud: "
AUDIO_TAG = "<|audio|>"


def extract_transcript(sample: dict) -> str:
    messages = sample.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages is not a list")

    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str) and content.endswith(AUDIO_TAG):
                return content[: -len(AUDIO_TAG)]

    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.startswith(TTS_PROMPT_PREFIX):
                return content[len(TTS_PROMPT_PREFIX) :]

    raise ValueError("cannot extract transcript from TTS messages")


def convert_line(raw_line: bytes, input_path: Path, byte_offset: int) -> bytes:
    try:
        sample = json.loads(raw_line)
        transcript = extract_transcript(sample)
        sample["messages"] = [
            {"content": SYSTEM_PROMPT, "role": "system"},
            {"content": f"Transcribe the following audio:{AUDIO_TAG}", "role": "user"},
            {"content": transcript, "role": "assistant"},
        ]

        # Durations and audio-token counts do not change, but the text estimate does.
        total_chars = sum(len(message["content"]) for message in sample["messages"])
        audio_tokens = sample.get("audio_tokens") or []
        sample["num_tokens_est"] = total_chars // 4 + sum(
            token for token in audio_tokens if token is not None
        )
        return (json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    except Exception as exc:
        raise ValueError(f"{input_path}: invalid record near byte {byte_offset}: {exc}") from exc


def convert_shard(
    input_path: Path,
    shard_path: Path,
    start: int,
    end: int,
) -> int:
    count = 0
    with input_path.open("rb") as source, shard_path.open("wb") as destination:
        source.seek(start)
        if start > 0:
            source.seek(start - 1)
            if source.read(1) != b"\n":
                source.readline()

        while source.tell() < end:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            if line.strip():
                destination.write(convert_line(line, input_path, offset))
                count += 1
    return count


def convert_file(input_path: Path, output_path: Path, workers: int, overwrite: bool) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists (use --overwrite): {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_size = input_path.stat().st_size
    shard_count = min(workers, max(1, file_size))
    print(
        f"Converting {input_path} -> {output_path} "
        f"({file_size / 1024**3:.2f} GiB, {shard_count} threads)",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix=f".{output_path.name}.shards-", dir=output_path.parent) as temp_dir:
        temp_root = Path(temp_dir)
        jobs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=shard_count) as executor:
            for index in range(shard_count):
                start = file_size * index // shard_count
                end = file_size * (index + 1) // shard_count
                shard_path = temp_root / f"{index:05d}.jsonl"
                jobs.append(
                    (
                        shard_path,
                        executor.submit(convert_shard, input_path, shard_path, start, end),
                    )
                )

            counts = [future.result() for _, future in jobs]

        temporary_output = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
        try:
            with temporary_output.open("wb") as destination:
                for shard_path, _ in jobs:
                    with shard_path.open("rb") as source:
                        shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
            os.replace(temporary_output, output_path)
        finally:
            temporary_output.unlink(missing_ok=True)

    print(f"Wrote {sum(counts):,} records to {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path, help="TTS *.with_durations.jsonl files")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    for input_path in args.inputs:
        marker = "_tts.with_durations.jsonl"
        if not input_path.name.endswith(marker):
            raise SystemExit(f"input name must end with {marker}: {input_path}")
        output_path = input_path.with_name(input_path.name[: -len(marker)] + "_asr.with_durations.jsonl")
        convert_file(input_path, output_path, args.workers, args.overwrite)


if __name__ == "__main__":
    main()
