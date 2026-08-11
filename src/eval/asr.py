"""Trace loading and validation for unified FlexiSLM inference traces.

WER computation lives in the pinned Kimi-Audio-Evalkit repository
(``FlexiSLM-WER.evaluate``); this module only loads and validates traces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_TRACE_FIELDS = {"index", "task", "input", "output", "evaluation", "metadata"}
REQUIRED_EVALUATION_FIELDS = {"reference_text", "prediction_text"}


def _load_trace_records(trace_file: Path) -> list[dict[str, Any]]:
    """Load and validate unified trace records.

    Each record must carry ``evaluation.reference_text`` and
    ``evaluation.prediction_text`` plus at least one of
    ``evaluation.subset`` / ``evaluation.group`` (Evalkit grouping key).
    ``reference_text`` may be null for Open-QA subsets; ``prediction_text``
    may be null until a transcription is attached.
    """
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
            if "subset" not in evaluation and "group" not in evaluation:
                raise ValueError(
                    f"Trace {trace_file} line {line_number} evaluation must contain "
                    "'subset' or 'group'"
                )
            for field in ("reference_text", "prediction_text"):
                value = evaluation[field]
                if value is not None and not isinstance(value, str):
                    raise ValueError(
                        f"Trace {trace_file} line {line_number} evaluation.{field} "
                        "must be a string or null"
                    )
            records.append(record)

    if not records:
        raise ValueError(f"Trace file contains no records: {trace_file}")
    return records
