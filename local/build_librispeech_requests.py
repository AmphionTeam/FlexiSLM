#!/usr/bin/env python
"""Build FlexiSLM ASR requests for LibriSpeech test subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data/benchmarks"
DEFAULT_SUBSETS = ("test-clean", "test-other")
ASR_PROMPT = "Transcribe the following audio:"


def build_subset(data_root: Path, subset: str, output_path: Path) -> int:
    subset_dir = data_root / "LibriSpeech/librispeech/LibriSpeech" / subset
    if not subset_dir.is_dir():
        raise FileNotFoundError(f"LibriSpeech subset not found: {subset_dir}")

    transcripts: dict[str, str] = {}
    for transcript_path in sorted(subset_dir.rglob("*.trans.txt")):
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            sample_id, _, text = line.partition(" ")
            if sample_id and text:
                transcripts[sample_id] = text

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for index, audio_path in enumerate(sorted(subset_dir.rglob("*.flac"))):
            sample_id = audio_path.stem
            reference = transcripts.get(sample_id)
            if reference is None:
                raise ValueError(f"Missing transcript for {audio_path}")
            request = {
                "index": index,
                "task": "asr",
                "input": {
                    "audio_path": str(audio_path.resolve()),
                    "model_prompt": ASR_PROMPT,
                },
                "evaluation": {
                    "reference_text": reference,
                    "prediction_text": None,
                    "group": f"librispeech_{subset}",
                },
                "metadata": {"sample_id": sample_id},
            }
            output_file.write(json.dumps(request, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", default=list(DEFAULT_SUBSETS))
    args = parser.parse_args()

    unknown = set(args.subsets) - set(DEFAULT_SUBSETS)
    if unknown:
        raise ValueError(f"Unsupported LibriSpeech subsets: {', '.join(sorted(unknown))}")

    total = 0
    for subset in args.subsets:
        output_path = args.out_dir / f"{subset}_requests.jsonl"
        count = build_subset(args.data_root, subset, output_path)
        total += count
        print(f"{subset:12s} -> {output_path.resolve()} ({count} requests)")
    print(f"Total requests: {total}")


if __name__ == "__main__":
    main()
