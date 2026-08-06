"""WER evaluation for unified FlexiSLM inference traces."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .text import load_english_normalizer


REQUIRED_TRACE_FIELDS = {"index", "task", "input", "output", "evaluation", "metadata"}
REQUIRED_EVALUATION_FIELDS = {"reference_text", "prediction_text", "group"}


def _load_trace_records(trace_file: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    with trace_file.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in trace {trace_file} at line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Trace {trace_file} line {line_number} must be a JSON object"
                )
            missing = REQUIRED_TRACE_FIELDS - record.keys()
            if missing:
                raise ValueError(
                    f"Trace {trace_file} line {line_number} is missing fields: "
                    f"{', '.join(sorted(missing))}"
                )

            index = record["index"]
            if not isinstance(index, int) or isinstance(index, bool):
                raise ValueError(
                    f"Trace {trace_file} line {line_number} has a non-integer index"
                )
            if index in seen_indices:
                raise ValueError(f"Trace {trace_file} contains duplicate index {index}")
            seen_indices.add(index)

            evaluation = record["evaluation"]
            if not isinstance(evaluation, dict):
                raise ValueError(
                    f"Trace {trace_file} line {line_number} evaluation must be an object"
                )
            missing = REQUIRED_EVALUATION_FIELDS - evaluation.keys()
            if missing:
                raise ValueError(
                    f"Trace {trace_file} line {line_number} evaluation is missing: "
                    f"{', '.join(sorted(missing))}"
                )
            for field in ("reference_text", "group"):
                if not isinstance(evaluation[field], str):
                    raise ValueError(
                        f"Trace {trace_file} line {line_number} evaluation.{field} "
                        "must be a string"
                    )
            prediction = evaluation["prediction_text"]
            if prediction is not None and not isinstance(prediction, str):
                raise ValueError(
                    f"Trace {trace_file} line {line_number} "
                    "evaluation.prediction_text must be a string or null"
                )
            records.append(record)

    if not records:
        raise ValueError(f"Trace file contains no records: {trace_file}")
    return records


def _has_asr_transcription(record: dict[str, Any]) -> bool:
    transcription = record["evaluation"]["prediction_text"]
    return (
        isinstance(transcription, str)
        and bool(transcription.strip())
        and transcription.strip().lower() != "null"
    )


def evaluate_wer(
    trace_file: Path,
    evalkit_path: Path,
    language: str,
    dump_details: bool = False,
) -> dict[str, Any]:
    """Compute English WER from completed ASR transcriptions."""
    if not evalkit_path.is_dir():
        raise FileNotFoundError(f"Evalkit directory does not exist: {evalkit_path}")
    if not trace_file.is_file():
        raise FileNotFoundError(f"Trace file does not exist: {trace_file}")

    records = _load_trace_records(trace_file)
    completed_records = [record for record in records if _has_asr_transcription(record)]
    if not completed_records:
        raise ValueError(
            f"Trace has no completed ASR transcriptions: {trace_file}. "
            "Run ASR and fill evaluation.prediction_text before evaluation."
        )

    if language != "en":
        raise ValueError("ASR WER currently supports English traces only")
    import editdistance

    normalizer = load_english_normalizer(evalkit_path)
    records_by_evaluation_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in completed_records:
        records_by_evaluation_group[str(record["evaluation"]["group"])].append(record)

    metrics: dict[str, Any] = {}
    word_error_counts_by_index: dict[int, int] = {}
    for evaluation_group, group_records in records_by_evaluation_group.items():
        reference_words = [
            normalizer(str(record["evaluation"]["reference_text"])).split()
            for record in group_records
        ]
        transcription_words = [
            normalizer(str(record["evaluation"]["prediction_text"])).split()
            for record in group_records
        ]
        word_error_counts = [
            int(editdistance.eval(reference, transcription))
            for reference, transcription in zip(reference_words, transcription_words)
        ]
        reference_word_count = sum(len(words) for words in reference_words)
        if not reference_word_count:
            raise ValueError(f"Evaluation group has no reference words: {evaluation_group}")
        word_error_rate = sum(word_error_counts) / reference_word_count
        metrics[evaluation_group] = {
            "word_error_rate_percent": round(word_error_rate * 100, 2),
            "evaluated_record_count": len(group_records),
            "reference_word_count": reference_word_count,
            "word_error_count": sum(word_error_counts),
        }
        for record, word_error_count in zip(group_records, word_error_counts):
            word_error_counts_by_index[int(record["index"])] = word_error_count

    if dump_details:
        detailed_records = []
        for record in records:
            detailed_record = dict(record)
            detailed_record["evaluation"] = dict(record["evaluation"])
            detailed_record["evaluation"]["word_error_count"] = (
                word_error_counts_by_index.get(int(record["index"]))
            )
            detailed_records.append(detailed_record)
        details_path = trace_file.with_name(f"{trace_file.stem}_wer_details.jsonl")
        with details_path.open("w", encoding="utf-8") as file:
            for record in detailed_records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return metrics
