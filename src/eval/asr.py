"""ASR metrics for existing prediction traces."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def evaluate_wer(
    trace_file: Path,
    evalkit_path: Path,
    language: str,
    dump_details: bool = False,
) -> dict[str, Any]:
    """Compute WER with Kimi Audio Evalkit without running ASR inference."""
    if not evalkit_path.is_dir():
        raise FileNotFoundError(f"Evalkit directory does not exist: {evalkit_path}")
    if not trace_file.is_file():
        raise FileNotFoundError(f"Trace file does not exist: {trace_file}")

    sys.path.insert(0, str(evalkit_path))
    from almeval.datasets.ds_asr import ASRDataset

    class ExistingTraceASRDataset(ASRDataset):
        DATASET_NAME = "Existing-FlexiSLM-Trace"
        DATASET_SERIES = "Existing-FlexiSLM-Trace"
        LANG = language

        def __init__(self) -> None:
            # evaluate_asr_from_file only needs LANG. Do not load a benchmark
            # dataset or touch the audio paths stored in the trace.
            pass

    evaluator = ExistingTraceASRDataset()
    return evaluator.evaluate_asr_from_file(
        str(trace_file), dump_judge=dump_details
    )
