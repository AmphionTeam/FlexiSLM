#!/usr/bin/env python3
"""Convert split GigaSpeech metadata into duration-annotated TTS and ASR JSONL."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

AUDIO_TOKENS_PER_SECOND = 17
AUDIO_TAG = "<|audio|>"
SYSTEM_PROMPT = "Your Name: Omni\nYour Gender: female \n\nRespond in a text-only manner."
TTS_PROMPT_PREFIX = "Read the following text out loud: "


@dataclass(frozen=True)
class ShardResult:
    index: int
    count: int
    duration: float


def encode(sample: dict) -> bytes:
    return (json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def samples(audio_path: str, transcript: str, duration: float) -> tuple[dict, dict]:
    duration = round(duration, 3)
    audio_tokens = int(duration * AUDIO_TOKENS_PER_SECOND)

    tts_messages = [
        {"role": "user", "content": f"{TTS_PROMPT_PREFIX}{transcript}"},
        {"role": "assistant", "content": f"{transcript}{AUDIO_TAG}"},
    ]
    asr_messages = [
        {"content": SYSTEM_PROMPT, "role": "system"},
        {"content": f"Transcribe the following audio:{AUDIO_TAG}", "role": "user"},
        {"content": transcript, "role": "assistant"},
    ]

    common = {
        "audios": [audio_path],
        "audio_durations": [duration],
        "audio_tokens": [audio_tokens],
    }
    tts = {
        "messages": tts_messages,
        **common,
        "num_tokens_est": sum(len(m["content"]) for m in tts_messages) // 4 + audio_tokens,
    }
    asr = {
        "messages": asr_messages,
        **common,
        "num_tokens_est": sum(len(m["content"]) for m in asr_messages) // 4 + audio_tokens,
    }
    return tts, asr


def convert_shard(
    metadata_path: Path,
    audio_root: Path,
    tts_shard: Path,
    asr_shard: Path,
    index: int,
    start: int,
    end: int,
    check_audio: bool,
) -> ShardResult:
    count = 0
    total_duration = 0.0
    with metadata_path.open("rb") as source, tts_shard.open("wb") as tts_out, asr_shard.open("wb") as asr_out:
        source.seek(start)
        if start > 0:
            source.seek(start - 1)
            if source.read(1) != b"\n":
                source.readline()

        while source.tell() < end:
            offset = source.tell()
            raw = source.readline()
            if not raw:
                break
            if not raw.strip():
                continue
            try:
                line = raw.decode("utf-8").rstrip("\r\n")
                sample_id, relative_audio, duration_text, transcript = line.split("\t", 3)
                if sample_id == "ID" and relative_audio == "AUDIO":
                    continue
                duration = float(duration_text)
                if duration <= 0 or not transcript:
                    raise ValueError("duration must be positive and transcript non-empty")
                audio_path = audio_root / relative_audio
                if check_audio and not audio_path.is_file():
                    raise FileNotFoundError(audio_path)
                tts, asr = samples(str(audio_path), transcript, duration)
                tts_out.write(encode(tts))
                asr_out.write(encode(asr))
                count += 1
                total_duration += duration
            except Exception as exc:
                raise ValueError(f"{metadata_path}: invalid row near byte {offset}: {exc}") from exc
    return ShardResult(index=index, count=count, duration=total_duration)


def concatenate(shards: list[Path], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as output:
            for shard in shards:
                with shard.open("rb") as source:
                    shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("path/to/data/GigaSpeech/M"))
    parser.add_argument("--output-dir", type=Path, default=Path("path/to/data/processed"))
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-audio-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    metadata_paths = [args.dataset_root / "metadata_bk.tsv", args.dataset_root / "metadata.tsv"]
    for path in metadata_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tts_output = args.output_dir / "gigaspeech_m_tts.with_durations.jsonl"
    asr_output = args.output_dir / "gigaspeech_m_asr.with_durations.jsonl"
    for output in (tts_output, asr_output):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"output already exists (use --overwrite): {output}")

    print(f"Converting GigaSpeech M with {args.workers} threads", flush=True)
    print(f"Metadata: {', '.join(map(str, metadata_paths))}", flush=True)
    print(f"TTS output: {tts_output}", flush=True)
    print(f"ASR output: {asr_output}", flush=True)

    with tempfile.TemporaryDirectory(prefix=".gigaspeech_m.shards-", dir=args.output_dir) as temp_name:
        temp_root = Path(temp_name)
        all_tts_shards: list[Path] = []
        all_asr_shards: list[Path] = []
        total_count = 0
        total_duration = 0.0

        for metadata_index, metadata_path in enumerate(metadata_paths):
            size = metadata_path.stat().st_size
            shard_count = min(args.workers, max(1, size))
            jobs: dict[concurrent.futures.Future[ShardResult], int] = {}
            tts_shards: list[Path] = []
            asr_shards: list[Path] = []
            print(f"Processing {metadata_path} ({size / 1024**2:.1f} MiB, {shard_count} shards)", flush=True)

            with concurrent.futures.ThreadPoolExecutor(max_workers=shard_count) as executor:
                for index in range(shard_count):
                    tts_shard = temp_root / f"{metadata_index}-{index:05d}.tts.jsonl"
                    asr_shard = temp_root / f"{metadata_index}-{index:05d}.asr.jsonl"
                    tts_shards.append(tts_shard)
                    asr_shards.append(asr_shard)
                    future = executor.submit(
                        convert_shard,
                        metadata_path,
                        args.dataset_root,
                        tts_shard,
                        asr_shard,
                        index,
                        size * index // shard_count,
                        size * (index + 1) // shard_count,
                        not args.skip_audio_check,
                    )
                    jobs[future] = index

                completed = 0
                results: list[ShardResult] = []
                for future in concurrent.futures.as_completed(jobs):
                    results.append(future.result())
                    completed += 1
                    if completed % 25 == 0 or completed == shard_count:
                        print(f"  completed shards: {completed}/{shard_count}", flush=True)

            file_count = sum(result.count for result in results)
            file_duration = sum(result.duration for result in results)
            total_count += file_count
            total_duration += file_duration
            all_tts_shards.extend(tts_shards)
            all_asr_shards.extend(asr_shards)
            print(f"  records: {file_count:,}; duration: {file_duration / 3600:.2f} hours", flush=True)

        print("Combining TTS shards...", flush=True)
        concatenate(all_tts_shards, tts_output)
        print("Combining ASR shards...", flush=True)
        concatenate(all_asr_shards, asr_output)

    print(f"DONE records={total_count:,} duration_hours={total_duration / 3600:.2f}", flush=True)


if __name__ == "__main__":
    main()
