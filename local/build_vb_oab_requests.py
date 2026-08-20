#!/usr/bin/env python
"""Build audio_qa / s2s inference request files from VoiceBench / OpenAudioBench.

One request file per subset (``{subset}_requests.jsonl``) so jobs can be
distributed across GPUs and resumed incrementally.

Request schema (unified trace, input side):
    index, task, input.{audio_path, model_prompt}, evaluation.{subset, answer,
    audio_content, instruction, instruction_kwargs, reference_text},
    metadata.{sample_id, transcribe_model_path, transcribe_device}
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/benchmarks"


VOICEBENCH_DATASETS = {
    "mmsu",
    "openbookqa",
    "sd-qa",
    "advbench",
    "alpacaeval_full",
    "commoneval",
    "ifeval",
}
OPEN_AUDIO_BENCH_DATASET = "OpenAudioBench"


def _audio_stem(audio_path: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(audio_path).stem).strip("_") or "0"


def build_requests(
    dataset_file: Path,
    *,
    task: str,
    out_file: Path,
    transcribe_model_path: str | None,
    transcribe_device: str | None,
    include_text_prompt: bool,
) -> dict[str, int]:
    """Convert one dataset jsonl into a request jsonl; return per-subset counts."""
    if not dataset_file.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    subset_counts: dict[str, int] = {}
    with dataset_file.open(encoding="utf-8") as src, out_file.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_number, line in enumerate(src, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {dataset_file} at line {line_number}: {error}"
                ) from error
            index = int(item["index"])
            subset = str(item.get("subset") or dataset_file.stem)
            audio_path = str(item["audio_path"])
            audio_content = str(item.get("audio_content") or "")
            answer = item.get("answer")
            instruction = item.get("instruction")
            instruction_kwargs = item.get("instruction_kwargs")

            request = {
                "index": index,
                "task": task,
                "input": {
                    "audio_path": audio_path,
                    "model_prompt": audio_content if include_text_prompt else None,
                },
                "evaluation": {
                    "subset": subset,
                    "answer": answer,
                    "audio_content": audio_content,
                    "instruction": instruction,
                    "instruction_kwargs": instruction_kwargs,
                    "reference_text": answer,
                },
                "metadata": {
                    "sample_id": _audio_stem(audio_path),
                    "transcribe_model_path": transcribe_model_path,
                    "transcribe_device": transcribe_device,
                },
            }
            dst.write(json.dumps(request, ensure_ascii=False) + "\n")
            subset_counts[subset] = subset_counts.get(subset, 0) + 1
    return subset_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        choices=["voicebench", "openaudiobench"],
        required=True,
        help="Benchmark family to build requests for.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Directory containing benchmark JSONL files "
            "(default: data/benchmarks in the repository)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write {subset}_requests.jsonl files into.",
    )
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=None,
        help="Subsets to build (default: all available).",
    )
    parser.add_argument(
        "--task",
        choices=["audio_qa", "s2s"],
        default="audio_qa",
        help="audio_qa: text answer only; s2s: audio output + whisper transcription.",
    )
    parser.add_argument(
        "--transcribe-model-path",
        default=None,
        help="Whisper model path used to transcribe s2s generated audio "
        "(required when --task s2s).",
    )
    parser.add_argument(
        "--transcribe-device",
        default=None,
        help="Device for the s2s transcription model (default: engine device).",
    )
    parser.add_argument(
        "--no-text-prompt",
        action="store_true",
        help="Leave input.model_prompt empty (audio-only prompting).",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.benchmark == "voicebench":
        datasets = sorted(VOICEBENCH_DATASETS)
        data_dir = args.data_root / "VoiceBench"
    else:
        datasets = [OPEN_AUDIO_BENCH_DATASET]
        data_dir = args.data_root / "OpenAudioBench"
    if args.subsets:
        unknown = set(args.subsets) - set(datasets)
        if unknown:
            raise ValueError(
                f"Unknown {args.benchmark} subsets: {', '.join(sorted(unknown))}; "
                f"available: {', '.join(datasets)}"
            )
        datasets = [name for name in datasets if name in set(args.subsets)]

    if args.task == "s2s" and not args.transcribe_model_path:
        raise ValueError("--transcribe-model-path is required when --task s2s")

    grand_total = 0
    for dataset_name in datasets:
        dataset_file = data_dir / f"{dataset_name}.jsonl"
        out_file = out_dir / f"{dataset_name}_requests.jsonl"
        counts = build_requests(
            dataset_file,
            task=args.task,
            out_file=out_file,
            transcribe_model_path=args.transcribe_model_path,
            transcribe_device=args.transcribe_device,
            include_text_prompt=not args.no_text_prompt,
        )
        total = sum(counts.values())
        grand_total += total
        detail = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"{dataset_name:20s} -> {out_file}  ({total} requests: {detail})")

    print(f"Total requests: {grand_total} -> {out_dir}")


if __name__ == "__main__":
    main()
