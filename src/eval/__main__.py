"""CLI entry point for evaluating existing FlexiSLM traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .asr import evaluate_wer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

    missing = {"evalkit_path", "model_name", "jobs"} - config.keys()
    if missing:
        raise ValueError(f"Missing configuration fields: {', '.join(sorted(missing))}")
    if not isinstance(config["jobs"], list) or not config["jobs"]:
        raise ValueError("'jobs' must be a non-empty list")
    return config


def run(config_path: Path) -> list[Path]:
    config = load_config(config_path)
    evalkit_path = resolve_path(config["evalkit_path"])
    written_results: list[Path] = []

    for index, job in enumerate(config["jobs"]):
        if not isinstance(job, dict):
            raise ValueError(f"jobs[{index}] must be a mapping")
        required = {
            "name",
            "trace_file",
            "result_path",
            "task",
            "language",
            "metric",
        }
        missing = required - job.keys()
        if missing:
            raise ValueError(
                f"jobs[{index}] is missing: {', '.join(sorted(missing))}"
            )
        if job["task"] != "asr" or job["metric"] != "wer":
            raise ValueError(
                f"Unsupported evaluation: {job['task']}/{job['metric']}"
            )

        trace_file = resolve_path(job["trace_file"])
        result_path = resolve_path(job["result_path"])
        print(f"Evaluating job '{job['name']}': {trace_file}")
        metric_result = evaluate_wer(
            trace_file=trace_file,
            evalkit_path=evalkit_path,
            language=str(job["language"]),
            dump_details=bool(config.get("dump_details", False)),
        )

        result = {
            "model_name": str(config["model_name"]),
            "job_name": str(job["name"]),
            "trace_file": str(trace_file),
            "task": str(job["task"]),
            "language": str(job["language"]),
            "metrics": {str(job["metric"]): metric_result},
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(f"Wrote evaluation result: {result_path}")
        written_results.append(result_path)

    return written_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate existing FlexiSLM prediction traces."
    )
    parser.add_argument("config", type=Path, help="Path to an evaluation YAML")
    args = parser.parse_args()
    run(args.config.expanduser().resolve())


if __name__ == "__main__":
    main()
