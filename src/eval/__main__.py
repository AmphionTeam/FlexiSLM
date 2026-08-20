"""CLI entry point: evaluate unified FlexiSLM traces via Kimi-Audio-Evalkit.

All scoring happens inside the pinned evalkit repository (see
``docs/evalkit_pinning.md``); this module only dispatches jobs and archives
the raw judge artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from .evalkit_adapter import (
    DEFAULT_JUDGE_MODEL,
    WER_DATASET_NAME,
    build_evalkit_dataset_file,
    run_evalkit_evaluate,
    subset_row_counts,
    verify_evalkit_commit,
)
from .tts import transcribe_tts_trace
from .voicebench import write_voicebench_summary


PROJECT_ROOT = Path(__file__).resolve().parents[2]

_SUPPORTED_BENCHMARKS = {
    "voicebench",
    "openaudiobench",
    "librispeech",
    "librispeech_pc",
    "seedtts_eval",
}
_WER_BENCHMARKS = {"librispeech", "librispeech_pc", "seedtts_eval"}
_SUPPORTED_WER_TASKS = {"asr", "tts"}


def resolve_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

    missing = {"evalkit_path", "data_root", "model_name", "jobs"} - config.keys()
    if missing:
        raise ValueError(f"Missing configuration fields: {', '.join(sorted(missing))}")
    if not isinstance(config["jobs"], list) or not config["jobs"]:
        raise ValueError("'jobs' must be a non-empty list")
    return config


def _job_meta(config: dict[str, Any], job: dict[str, Any], evalkit_commit: str | None) -> dict[str, Any]:
    return {
        "job_name": str(job["name"]),
        "evalkit_commit": evalkit_commit,
        "evalkit_path": str(resolve_path(config["evalkit_path"])),
        "data_root": str(resolve_path(config["data_root"])),
        "judge_model": str(config.get("judge_model", DEFAULT_JUDGE_MODEL)),
    }


def _configure_run_logs(config: dict[str, Any], config_path: Path) -> Path:
    """Keep evalkit API logs in one explicit directory for this eval run."""
    configured_dir = config.get("log_dir")
    log_dir = (
        resolve_path(configured_dir)
        if configured_dir is not None
        else config_path.parent / "logs"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    api_log = (log_dir / "llm_api_logs.jsonl").resolve()
    # Always override evalkit's cwd-based fallback, which otherwise creates
    # llm_api_logs.jsonl in whichever directory launched the evaluator.
    os.environ["LLM_API_LOG_FILE"] = str(api_log)
    print(f"Evaluation logs: {log_dir.resolve()}", flush=True)
    return api_log


def _write_result(result: dict[str, Any], result_path: Path) -> Path:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Wrote evaluation result: {result_path}", flush=True)
    return result_path


def _run_evalkit_job(
    *,
    benchmark: str,
    job: dict[str, Any],
    config: dict[str, Any],
    evalkit_path: Path,
    data_root: Path,
    model_name: str,
    judge_model: str,
    evalkit_commit: str | None,
    transcribe_callback=None,
) -> dict[str, Any]:
    """Run one voicebench / openaudiobench / asr job through evalkit."""
    required = {"name", "trace_file", "result_path"}
    missing = required - job.keys()
    if missing:
        raise ValueError(f"jobs[] is missing: {', '.join(sorted(missing))}")
    trace_file = resolve_path(job["trace_file"])
    result_path = resolve_path(job["result_path"])

    dataset_name = str(job.get("dataset", WER_DATASET_NAME))
    method = job.get("method")
    if dataset_name == WER_DATASET_NAME and method in (None, "default"):
        method = "wer"
    out_dir = trace_file.parent / "evalkit"
    eval_file = build_evalkit_dataset_file(
        trace_file=trace_file,
        dataset_name=dataset_name,
        model_name=model_name,
        evalkit_path=evalkit_path,
        data_root=data_root,
        out_dir=out_dir,
        transcribe_callback=transcribe_callback,
        subsets=job.get("subsets"),
        trace_indices_only=bool(job.get("trace_indices_only", False)),
        prediction_source=str(job.get("prediction_source", "auto")),
        include_prediction_text=bool(job.get("include_prediction_text", True)),
    )
    result, archived = run_evalkit_evaluate(
        eval_file=eval_file,
        dataset_name=dataset_name,
        method=str(method) if method is not None else None,
        judge_model=judge_model,
        evalkit_path=evalkit_path,
        data_root=data_root,
        work_dir=result_path.parent,
        judge_api_base=config.get("judge_api_base"),
        judge_api_key=config.get("judge_api_key"),
        judge_api_key_env=config.get("judge_api_key_env"),
        prediction_source=str(job.get("prediction_source", "auto")),
    )
    auxiliary_method = job.get("auxiliary_method")
    auxiliary_metrics = None
    auxiliary_archived: list[Path] = []
    if auxiliary_method:
        auxiliary_result, auxiliary_archived = run_evalkit_evaluate(
            eval_file=eval_file,
            dataset_name=dataset_name,
            method=str(auxiliary_method),
            judge_model=judge_model,
            evalkit_path=evalkit_path,
            data_root=data_root,
            work_dir=result_path.parent,
            dump_judge=False,
            judge_api_base=config.get("judge_api_base"),
            judge_api_key=config.get("judge_api_key"),
            judge_api_key_env=config.get("judge_api_key_env"),
        )
        auxiliary_metrics = {
            "eval_method": auxiliary_result.get("eval_method"),
            "performance": auxiliary_result.get("performance"),
        }

    all_archived = list(dict.fromkeys([*archived, *auxiliary_archived]))
    result = {
        **result,
        **_job_meta(config, job, evalkit_commit),
        "benchmark": benchmark,
        "prediction_source": str(job.get("prediction_source", "auto")),
        "trace_file": str(trace_file),
        "eval_file": str(eval_file),
        "archived_files": [str(path) for path in all_archived],
    }
    if "task" in job:
        result["task"] = str(job["task"])
    if auxiliary_metrics is not None:
        result["auxiliary_metrics"] = auxiliary_metrics
    _write_result(result, result_path)
    return result


def _run_asr_job(
    *,
    benchmark: str,
    job: dict[str, Any],
    config: dict[str, Any],
    evalkit_path: Path,
    data_root: Path,
    model_name: str,
    judge_model: str,
    evalkit_commit: str | None,
) -> dict[str, Any]:
    return _run_evalkit_job(
        benchmark=benchmark,
        job=job,
        config=config,
        evalkit_path=evalkit_path,
        data_root=data_root,
        model_name=model_name,
        judge_model=judge_model,
        evalkit_commit=evalkit_commit,
    )


def _run_tts_job(
    *,
    benchmark: str,
    job: dict[str, Any],
    config: dict[str, Any],
    evalkit_path: Path,
    data_root: Path,
    model_name: str,
    judge_model: str,
    evalkit_commit: str | None,
) -> dict[str, Any]:
    tts_required = {
        "asr_backend",
        "asr_model_path",
        "device",
        "batch_size",
        "details_path",
    }
    tts_missing = tts_required - job.keys()
    if tts_missing:
        raise ValueError(f"jobs[] is missing TTS fields: {', '.join(sorted(tts_missing))}")

    trace_file = resolve_path(job["trace_file"])
    transcriptions = transcribe_tts_trace(
        trace_file=trace_file,
        backend=str(job["asr_backend"]),
        model_path=str(resolve_path(job["asr_model_path"])),
        device=str(job["device"]),
        batch_size=int(job["batch_size"]),
        details_path=resolve_path(job["details_path"]),
        resume=bool(job.get("resume", True)),
        language=str(job.get("language", "en")),
    )
    result = _run_evalkit_job(
        benchmark=benchmark,
        job=job,
        config=config,
        evalkit_path=evalkit_path,
        data_root=data_root,
        model_name=model_name,
        judge_model=judge_model,
        evalkit_commit=evalkit_commit,
        transcribe_callback=lambda records: {
            int(record["index"]): transcriptions[int(record["index"])]
            for record in records
            if int(record["index"]) in transcriptions
        },
    )
    result["asr_backend"] = str(job["asr_backend"])
    result["asr_model_path"] = str(resolve_path(job["asr_model_path"]))
    return result


def run(config_path: Path, selected_jobs: set[str] | None = None) -> list[Path]:
    config_path = config_path.expanduser().resolve()
    config = load_config(config_path)
    _configure_run_logs(config, config_path)
    evalkit_path = resolve_path(config["evalkit_path"])
    data_root = resolve_path(config["data_root"])
    model_name = str(config["model_name"])
    judge_model = str(config.get("judge_model", DEFAULT_JUDGE_MODEL))
    evalkit_commit = verify_evalkit_commit(
        evalkit_path, config.get("evalkit_commit")
    )

    written_results: list[Path] = []
    configured_names = {
        str(job.get("name")) for job in config["jobs"] if isinstance(job, dict)
    }
    if selected_jobs:
        unknown = selected_jobs - configured_names
        if unknown:
            raise ValueError(f"Unknown evaluation jobs: {', '.join(sorted(unknown))}")

    voicebench_results: list[dict[str, Any]] = []
    voicebench_data_manifest: dict[str, dict[str, int]] = {}
    configured_summary_path = config.get("voicebench_summary_path")
    voicebench_summary_path = (
        resolve_path(configured_summary_path)
        if configured_summary_path is not None
        else config_path.parent / "voicebench_summary.json"
    )

    for index, job in enumerate(config["jobs"]):
        if not isinstance(job, dict):
            raise ValueError(f"jobs[{index}] must be a mapping")
        if selected_jobs and str(job.get("name")) not in selected_jobs:
            continue

        benchmark = str(job.get("benchmark", ""))
        if benchmark not in _SUPPORTED_BENCHMARKS:
            raise ValueError(
                f"jobs[{index}] unsupported benchmark {benchmark!r}; "
                f"expected one of {sorted(_SUPPORTED_BENCHMARKS)}"
            )

        task = str(job.get("task", ""))
        if benchmark in _WER_BENCHMARKS and task not in _SUPPORTED_WER_TASKS:
            raise ValueError(
                f"jobs[{index}] benchmark {benchmark!r} requires task to be "
                f"one of {sorted(_SUPPORTED_WER_TASKS)}"
            )

        print(f"Evaluating job '{job.get('name')}' (benchmark={benchmark})", flush=True)
        if benchmark in _WER_BENCHMARKS and task == "tts":
            result = _run_tts_job(
                benchmark=benchmark, job=job, config=config,
                evalkit_path=evalkit_path, data_root=data_root,
                model_name=model_name, judge_model=judge_model,
                evalkit_commit=evalkit_commit,
            )
        elif benchmark in _WER_BENCHMARKS and task == "asr":
            result = _run_asr_job(
                benchmark=benchmark, job=job, config=config,
                evalkit_path=evalkit_path, data_root=data_root,
                model_name=model_name, judge_model=judge_model,
                evalkit_commit=evalkit_commit,
            )
        else:
            dataset_name = str(job.get("dataset"))
            result = _run_evalkit_job(
                benchmark=benchmark, job=job, config=config,
                evalkit_path=evalkit_path, data_root=data_root,
                model_name=model_name, judge_model=judge_model,
                evalkit_commit=evalkit_commit,
            )
            if benchmark == "voicebench":
                voicebench_results.append(result)
                voicebench_data_manifest[dataset_name] = subset_row_counts(
                    evalkit_path, data_root, dataset_name,
                    resolve_path(job["result_path"]).parent,
                )

        result_path = resolve_path(job["result_path"])
        if result_path not in written_results:
            written_results.append(result_path)

    if voicebench_results:
        summary_path = write_voicebench_summary(
            results=voicebench_results,
            data_manifest=voicebench_data_manifest,
            evalkit_commit=evalkit_commit,
            judge_model=judge_model,
            data_root=data_root,
            out_path=voicebench_summary_path,
        )
        written_results.append(summary_path)

    return written_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate unified FlexiSLM inference traces via evalkit."
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
