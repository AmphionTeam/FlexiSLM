#!/usr/bin/env python3
"""
Pre-compute audio durations (and derived token counts) and append them to JSONL
training data.

Adds three fields to each sample:

* ``audio_durations``  – list of floats (seconds per audio, None on failure),
                          parallel to the ``audios`` field.
* ``audio_tokens``     – list of ints  (token count per audio = int(dur * 17),
                          None where duration is None), parallel to ``audios``.
* ``num_tokens_est``   – single int: total estimated token count for the sample
                          = text_tokens_est + sum(audio_tokens).
                          text_tokens_est = total_content_chars // 4  (char-based
                          approximation matching the training pipeline heuristic).

Text-only samples (empty ``audios``) get empty lists for the first two fields
and a text-only estimate for ``num_tokens_est``.

Having ``num_tokens_est`` stored in the JSONL lets the dataset's ``lengths``
property do a single bulk column read with zero per-sample computation, which
is the fastest possible path for the TokenBudgetBatchSampler.

Usage
-----
# Single file:
python local/precompute_audio_durations.py \\
    --input path/to/VoiceAssistant-400K/data.jsonl \\
    --audio-root path/to/VoiceAssistant-400K/audio \\
    --workers 32

# Batch mode – process every data_paths entry in a YAML config:
python local/precompute_audio_durations.py \\
    --yaml path/to/dataset_train.yaml \\
    --workers 32

Output is written next to the input file:
    data.jsonl  →  data.with_durations.jsonl

If the output file already exists, that input is skipped unless ``--overwrite``
is provided.
"""

# Tokens-per-second conversion factor used throughout the training pipeline.
AUDIO_TOKENS_PER_SECOND: int = 17

# Default root for resolving *relative* audio paths when no ``audio_root``
# is supplied (via --audio-root or the dataset YAML).
DEFAULT_AUDIO_ROOT = None

import argparse
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple


def get_duration(path: str) -> Optional[float]:
    """Return audio duration in seconds, or None on failure."""
    import torchaudio
    try:
        info = torchaudio.info(path)
        return info.num_frames / info.sample_rate
    except Exception:
        import soundfile as sf

        info = sf.info(path)
        if info.samplerate and info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)


def resolve_path(audio_path: str, audio_root: Optional[str]) -> str:
    if os.path.isabs(audio_path):
        return audio_path
    # Relative path: prefer explicit ``audio_root`` (CLI / YAML), then optional
    # module default. If both are missing, keep the relative path unchanged.
    root = audio_root if audio_root else DEFAULT_AUDIO_ROOT
    return os.path.join(root, audio_path) if root else audio_path


def process_line(args: Tuple[str, Optional[str]]) -> Tuple[str, int, int]:
    """Process one JSONL line: add audio_durations, audio_tokens, num_tokens_est.

    Returns (new_json_line, n_ok, n_fail).
    """
    line, audio_root = args
    line = line.rstrip("\n")
    if not line:
        return "", 0, 0
    sample = json.loads(line)
    audios = sample.get("audios", [])

    # --- audio fields (parallel to ``audios``) ---
    durations: List[Optional[float]] = []
    tokens: List[Optional[int]] = []
    ok, fail = 0, 0
    for p in audios:
        if p is None:
            durations.append(None)
            tokens.append(None)
            fail += 1
            continue
        full = resolve_path(p, audio_root)
        dur = get_duration(full)
        if dur is not None:
            dur_rounded = round(dur, 3)
            durations.append(dur_rounded)
            tokens.append(int(dur_rounded * AUDIO_TOKENS_PER_SECOND))
            ok += 1
        else:
            durations.append(None)
            tokens.append(None)
            fail += 1

    sample["audio_durations"] = durations
    sample["audio_tokens"] = tokens

    # --- total token estimate (text + audio) ---
    # Text estimate: total characters across all message contents // 4.
    # This mirrors the heuristic used in Qwen2Dataset.lengths at training time.
    messages = sample.get("messages", [])
    # Some source files contain malformed messages with `content: null`.
    # Skip the whole sample so one bad record does not abort the worker pool.
    for m in messages:
        if isinstance(m, dict) and not isinstance(m.get("content", ""), str):
            return "", 0, 0
    total_chars = sum(len(m.get("content", "")) for m in messages if isinstance(m, dict))
    text_tokens_est = total_chars // 4
    audio_tokens_est = sum(t for t in tokens if t is not None)
    sample["num_tokens_est"] = text_tokens_est + audio_tokens_est

    return json.dumps(sample, ensure_ascii=False), ok, fail


def process_file(input_path: str, audio_root: Optional[str], workers: int, overwrite: bool = False):
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}.with_durations{ext}"

    if os.path.isfile(output_path) and not overwrite:
        print(f"[SKIP] {output_path} (already exists)")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    print(f"[{os.path.basename(input_path)}] {total} lines, {workers} workers")

    args_list = [(line, audio_root) for line in lines]
    results: List[str] = [""] * total
    total_ok, total_fail = 0, 0

    if workers <= 1:
        for idx, a in enumerate(args_list):
            out, ok, fail = process_line(a)
            results[idx] = out
            total_ok += ok
            total_fail += fail
            if (idx + 1) % 10000 == 0:
                print(f"  {idx + 1}/{total} ...")
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for idx, (out, ok, fail) in enumerate(
                pool.map(process_line, args_list, chunksize=200)
            ):
                results[idx] = out
                total_ok += ok
                total_fail += fail
                if (idx + 1) % 10000 == 0:
                    print(f"  {idx + 1}/{total} ...")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            if r:
                f.write(r + "\n")

    print(f"  → {output_path}")
    print(f"  audio files: {total_ok} ok, {total_fail} failed/missing")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, default=None,
                        help="Single JSONL file to process")
    parser.add_argument("--audio-root", type=str, default=None,
                        help="Root dir for resolving relative audio paths")
    parser.add_argument("--yaml", type=str, default=None,
                        help="Dataset YAML config – process all data_paths entries")
    parser.add_argument("--workers", type=int, default=32,
                        help="Number of parallel workers (default: 32)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate output even if the .with_durations file already exists")
    args = parser.parse_args()

    if args.input:
        process_file(args.input, args.audio_root, args.workers, overwrite=args.overwrite)
    elif args.yaml:
        import yaml
        with open(args.yaml, "r") as f:
            cfg = yaml.safe_load(f)
        global_audio_root = cfg.get("audio_root", None)
        for name, info in cfg.get("dataset", {}).items():
            for data_path in info.get("data_paths", []):
                if not os.path.isfile(data_path):
                    print(f"[SKIP] {data_path} (not found)")
                    continue
                if data_path.endswith(".with_durations.jsonl"):
                    print(f"[SKIP] {data_path} (already exists)")
                    continue
                root = info.get("audio_root", global_audio_root)
                if args.audio_root:
                    root = args.audio_root
                process_file(data_path, root, args.workers, overwrite=args.overwrite)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
