"""CLI entry point for evaluating unified FlexiSLM inference traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .asr import evaluate_wer
from .tts import evaluate_tts_wer


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


def run(config_path: Path, selected_jobs: set[str] | None = None) -> list[Path]:
    config = load_config(config_path)
    evalkit_path = resolve_path(config["evalkit_path"])
    written_results: list[Path] = []
    configured_names = {str(job.get("name")) for job in config["jobs"] if isinstance(job, dict)}
    if selected_jobs:
        unknown = selected_jobs - configured_names
        if unknown:
            raise ValueError(f"Unknown evaluation jobs: {', '.join(sorted(unknown))}")

    for index, job in enumerate(config["jobs"]):
        if not isinstance(job, dict):
            raise ValueError(f"jobs[{index}] must be a mapping")
        if selected_jobs and str(job.get("name")) not in selected_jobs:
            continue
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
        if job["metric"] != "wer" or job["task"] not in {"asr", "tts"}:
            raise ValueError(
                f"Unsupported evaluation: {job['task']}/{job['metric']}"
            )

        trace_file = resolve_path(job["trace_file"])
        result_path = resolve_path(job["result_path"])
        print(f"Evaluating job '{job['name']}': {trace_file}")
        if job["task"] == "asr":
            metric_result = evaluate_wer(
                trace_file=trace_file,
                evalkit_path=evalkit_path,
                language=str(job["language"]),
                dump_details=bool(config.get("dump_details", False)),
            )
        else:
            tts_required = {
                "asr_backend",
                "asr_model_path",
                "device",
                "batch_size",
                "details_path",
            }
            tts_missing = tts_required - job.keys()
            if tts_missing:
                raise ValueError(
                    f"jobs[{index}] is missing TTS fields: "
                    f"{', '.join(sorted(tts_missing))}"
                )
            model_path_value = str(job["asr_model_path"])
            model_path = resolve_path(model_path_value)
            metric_result = evaluate_tts_wer(
                trace_file=trace_file,
                evalkit_path=evalkit_path,
                language=str(job["language"]),
                backend=str(job["asr_backend"]),
                model_path=str(model_path),
                device=str(job["device"]),
                batch_size=int(job["batch_size"]),
                details_path=resolve_path(job["details_path"]),
                resume=bool(job.get("resume", True)),
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
        description="Evaluate unified FlexiSLM inference traces."
    )
    parser.add_argument("config", type=Path, help="Path to an evaluation YAML")
    parser.add_argument(
        "--job",
        action="append",
        default=[],
        help="Run only the named job; repeat to select multiple jobs.",
    )
    args = parser.parse_args()
    run(args.config.expanduser().resolve(), set(args.job) or None)


if __name__ == "__main__":
    main()
