"""Aggregate per-subset VoiceBench performances into a run summary."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def write_voicebench_summary(
    *,
    results: list[dict[str, Any]],
    data_manifest: dict[str, dict[str, int]],
    evalkit_commit: str | None,
    judge_model: str,
    data_root: Path,
    out_path: Path,
) -> Path:
    """Write ``voicebench_summary.json`` with model, pin, judge, data manifest
    and per-job performance snapshots."""
    previous: dict[str, Any] = {}
    if out_path.is_file():
        try:
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous = loaded
        except (OSError, json.JSONDecodeError):
            pass

    previous_jobs = previous.get("jobs", {})
    if not isinstance(previous_jobs, dict):
        previous_jobs = {}
    jobs = dict(previous_jobs)
    jobs.update(
        {
            str(result.get("job_name")): {
                "dataset": result.get("dataset"),
                "eval_method": result.get("eval_method"),
                "prediction_source": result.get("prediction_source"),
                "performance": result.get("performance"),
                "auxiliary_metrics": result.get("auxiliary_metrics"),
            }
            for result in results
        }
    )

    previous_manifest = previous.get("data_manifest", {})
    if not isinstance(previous_manifest, dict):
        previous_manifest = {}
    merged_manifest = {**previous_manifest, **data_manifest}
    model = results[0].get("model") if results else previous.get("model")
    summary = {
        "model": model,
        "evalkit_commit": evalkit_commit,
        "judge_model": judge_model,
        "data_root": str(data_root),
        "data_manifest": merged_manifest,
        "date": str(datetime.now()),
        "jobs": jobs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Wrote VoiceBench summary: {out_path}", flush=True)
    return out_path
