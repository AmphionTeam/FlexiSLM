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
    model = results[0].get("model") if results else None
    summary = {
        "model": model,
        "evalkit_commit": evalkit_commit,
        "judge_model": judge_model,
        "data_root": str(data_root),
        "data_manifest": data_manifest,
        "date": str(datetime.now()),
        "jobs": {
            str(result.get("job_name")): {
                "dataset": result.get("dataset"),
                "eval_method": result.get("eval_method"),
                "performance": result.get("performance"),
            }
            for result in results
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Wrote VoiceBench summary: {out_path}", flush=True)
    return out_path
