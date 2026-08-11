"""Adapter: unified FlexiSLM inference traces -> Kimi-Audio-Evalkit evaluation.

All scoring logic lives inside the pinned Kimi-Audio-Evalkit repository
(``almeval/datasets/*.evaluate``); this module only:

- injects the runtime dataset root via ``PROJECT_CONFIG_PATH``,
- converts traces to the ``{model}_{DATASET_NAME}.jsonl`` eval files evalkit
  consumes,
- runs ``dataset.evaluate()`` and archives the raw judge artifacts.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

from .asr import _load_trace_records


DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
WER_DATASET_NAME = "FlexiSLM-WER"


def verify_evalkit_commit(evalkit_path: Path, expected: str | None) -> str | None:
    """Return the current evalkit HEAD; warn when it differs from the pin."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(evalkit_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        head = proc.stdout.strip()
    except Exception:
        return None
    if expected and head != expected:
        print(
            f"WARNING: evalkit HEAD is {head}, but eval config pins {expected} "
            f"(see docs/evalkit_pinning.md)",
            flush=True,
        )
    return head


def _write_runtime_config(data_root: Path, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    config = {"DATASETS": {"dataset_root": str(data_root), "datasets": {}}}
    path = work_dir / "evalkit_runtime_config.yaml"
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    return path


def activate_evalkit(evalkit_path: Path, data_root: Path, work_dir: Path) -> Path:
    """Make ``almeval`` importable and point its ConfigManager at ``data_root``.

    Must run before any dataset instantiation (ConfigManager is a process
    singleton). Returns the runtime config path.
    """
    evalkit_path = evalkit_path.resolve()
    if str(evalkit_path) not in sys.path:
        sys.path.insert(0, str(evalkit_path))
    runtime_config = _write_runtime_config(data_root, work_dir)
    os.environ["PROJECT_CONFIG_PATH"] = str(runtime_config)
    try:
        from almeval.utils.config_manager import ConfigManager

        ConfigManager._instance = None
        ConfigManager._config = None
    except ImportError:
        pass
    return runtime_config


def _direct_text(record: dict[str, Any]) -> str:
    """Model's direct text output (evaluation.prediction_text or output.text)."""
    prediction_text = record["evaluation"].get("prediction_text")
    if isinstance(prediction_text, str) and prediction_text.strip().lower() != "null":
        return prediction_text
    output_text = record["output"].get("text")
    return str(output_text) if isinstance(output_text, str) else ""


def _full_prompt(trace: dict[str, Any], item: dict[str, Any]) -> str:
    """The complete prompt actually sent to the model (for ifeval / vb-qa).

    Priority: instruction (str) > audio_content > model_prompt > input text >
    dataset question. For audio-QA benchmarks the question/instruction text
    lives in ``audio_content`` (the spoken content), which vb-qa/ifeval need.
    """
    evaluation = trace["evaluation"]
    for key in ("instruction", "audio_content", "prompt"):
        value = evaluation.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key in ("model_prompt", "text"):
        value = trace["input"].get(key)
        if isinstance(value, str) and value.strip():
            return value
    question = item.get("question")
    return str(question) if isinstance(question, str) else ""


def _wer_row(
    record: dict[str, Any], transcriptions: dict[int, str]
) -> dict[str, Any]:
    """One eval row for the generic FlexiSLM-WER benchmark (trace-only mode)."""
    evaluation = record["evaluation"]
    reference = evaluation.get("reference_text") or evaluation.get("answer")
    prediction_text = _direct_text(record)
    prediction = (
        transcriptions.get(int(record["index"])) or prediction_text or "null"
    )
    subset = (
        evaluation.get("subset")
        or evaluation.get("group")
        or WER_DATASET_NAME
    )
    row: dict[str, Any] = {
        "index": int(record["index"]),
        "subset": str(subset),
        "answer": str(reference or ""),
        "prediction": prediction,
        "prediction_text": prediction_text or None,
    }
    output_audio = record["output"].get("audio_path")
    if isinstance(output_audio, str) and output_audio:
        row["audio_path"] = output_audio
    return row


def build_evalkit_dataset_file(
    trace_file: Path,
    dataset_name: str,
    model_name: str,
    evalkit_path: Path,
    data_root: Path,
    out_dir: Path,
    transcribe_callback: Callable[
        [list[dict[str, Any]]], dict[int, str]
    ] | None = None,
) -> Path:
    """Build ``{model}_{DATASET_NAME}.jsonl`` for evalkit from a unified trace.

    For official datasets (VoiceBench / OpenAudioBench) the eval file rows come
    from evalkit's own ``dataset.data`` (field consistency guaranteed), overlaid
    with prediction / prompt / prediction_text from the trace by ``index``.

    For ``FlexiSLM-WER`` (ASR / TTS WER) there is no standalone dataset file:
    rows are built directly from trace records (subset=group, answer=reference).
    ``transcribe_callback`` maps trace index -> transcription of generated
    audio (used by TTS / s2s); without it the direct text output is used.
    """
    activate_evalkit(evalkit_path, data_root, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_file = out_dir / f"{model_name}_{dataset_name}.jsonl"

    records = _load_trace_records(trace_file)
    transcriptions: dict[int, str] = {}
    if transcribe_callback is not None:
        transcriptions = dict(transcribe_callback(records))

    if dataset_name == WER_DATASET_NAME:
        rows = [_wer_row(record, transcriptions) for record in records]
    else:
        from almeval.datasets import build_dataset

        dataset = build_dataset(dataset_name)
        by_index = {int(record["index"]): record for record in records}
        rows = []
        for item in dataset.data:
            index = int(item["index"])
            subset = str(item.get("subset") or dataset_name)
            row = dict(item)
            row["subset"] = subset
            trace = by_index.get(index)
            if trace is None:
                row["prediction"] = "null"
                row["prompt"] = str(item.get("question") or "")
                row["prediction_text"] = None
            else:
                prediction_text = _direct_text(trace)
                row["prediction"] = (
                    transcriptions.get(index) or prediction_text or "null"
                )
                row["prediction_text"] = prediction_text or None
                row["prompt"] = _full_prompt(trace, item)
            rows.append(row)

    with eval_file.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"Built evalkit eval file: {eval_file} ({len(rows)} rows)",
        flush=True,
    )
    return eval_file


def subset_row_counts(
    evalkit_path: Path,
    data_root: Path,
    dataset_name: str,
    work_dir: Path,
) -> dict[str, int]:
    """Per-subset row counts of the official dataset (for the run manifest)."""
    from collections import Counter

    activate_evalkit(evalkit_path, data_root, work_dir)
    from almeval.datasets import build_dataset

    dataset = build_dataset(dataset_name)
    return dict(
        Counter(str(item.get("subset") or dataset_name) for item in dataset.data)
    )


def archive_judge_artifacts(eval_file: Path, work_dir: Path) -> list[Path]:
    """Enumerate judge artifacts produced next to ``eval_file`` and copy the
    global LLM API log into ``work_dir``. Returns the archived file paths."""
    artifacts: list[Path] = []
    stem = eval_file.stem
    for pattern in (
        f"{stem}_judge*.jsonl",
        f"{stem}_wer_details.jsonl",
        f"{stem}_high_wer_summary*.json",
    ):
        artifacts.extend(sorted(eval_file.parent.glob(pattern)))
    llm_log = os.environ.get("LLM_API_LOG_FILE")
    if llm_log and Path(llm_log).is_file():
        dest = work_dir / "llm_api_logs.jsonl"
        try:
            shutil.copy2(llm_log, dest)
            artifacts.append(dest)
        except OSError as error:
            print(f"WARNING: could not archive LLM API log: {error}", flush=True)
    return [path for path in artifacts if path.is_file()]


def _configure_judge_env(
    judge_api_base: str | None,
    judge_api_key: str | None,
    judge_api_key_env: str | None,
) -> None:
    """Route the LLM judge to the configured model endpoint.

    evalkit's OpenAIWrapper reads OPENAI_API_BASE / OPENAI_API_KEY from the
    environment; this injects the eval-YAML judge settings so third-party
    OpenAI-compatible endpoints (e.g. DeepSeek) work without code changes.
    The key itself is referenced by env-var name (judge_api_key_env) to avoid
    plaintext keys in YAML; an inline judge_api_key is allowed as a fallback.
    """
    if judge_api_base:
        os.environ["OPENAI_API_BASE"] = judge_api_base
    if judge_api_key:
        os.environ["OPENAI_API_KEY"] = judge_api_key
    elif judge_api_key_env:
        value = os.environ.get(judge_api_key_env)
        if value:
            os.environ["OPENAI_API_KEY"] = value


def run_evalkit_evaluate(
    *,
    eval_file: Path,
    dataset_name: str,
    method: str | None,
    judge_model: str | None,
    evalkit_path: Path,
    data_root: Path,
    work_dir: Path,
    dump_judge: bool = True,
    wer_high_threshold: float = 50.0,
    judge_api_base: str | None = None,
    judge_api_key: str | None = None,
    judge_api_key_env: str | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    """Run evalkit's ``dataset.evaluate()`` and archive raw judge artifacts.

    Returns ``(performance, archived_paths)`` where ``performance`` is the
    ``format_performance`` dict produced by evalkit.
    """
    activate_evalkit(evalkit_path, data_root, work_dir)
    _configure_judge_env(judge_api_base, judge_api_key, judge_api_key_env)
    from almeval.datasets import build_dataset

    effective_method = method
    if not effective_method or effective_method == "default":
        effective_method = judge_model or DEFAULT_JUDGE_MODEL

    dataset = build_dataset(dataset_name)
    if dataset_name == WER_DATASET_NAME:
        result = dataset.evaluate(
            str(eval_file),
            dump_judge=dump_judge,
            method=effective_method,
            wer_high_threshold=wer_high_threshold,
        )
    else:
        result = dataset.evaluate(
            str(eval_file), dump_judge=dump_judge, method=effective_method
        )
    archived = archive_judge_artifacts(eval_file, work_dir)
    return result, archived
