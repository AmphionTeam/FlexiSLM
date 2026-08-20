"""Adapter: unified FlexiSLM inference traces -> Kimi-Audio-Evalkit evaluation.

All scoring logic lives inside the pinned Kimi-Audio-Evalkit repository
(``almeval/datasets/*.evaluate``); this module only:

- injects the runtime dataset root via ``PROJECT_CONFIG_PATH``,
- converts traces to the ``{model}_{DATASET_NAME}.jsonl`` eval files evalkit
  consumes,
- runs ``dataset.evaluate()`` and archives the raw judge artifacts.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
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
    """Return the model's direct text output.

    ``evaluation.prediction_text`` is only a compatibility fallback. For S2S
    traces it contains the generated audio's Whisper transcription, while the
    model's actual direct text is stored in ``output.text``.
    """
    output_text = record["output"].get("text")
    if isinstance(output_text, str) and output_text.strip().lower() != "null":
        return output_text
    output_audio = record["output"].get("audio_path")
    if isinstance(output_audio, str) and output_audio:
        # S2S prediction_text is the generated audio's ASR transcription, not
        # a fallback for a missing direct-text response.
        return ""
    prediction_text = record["evaluation"].get("prediction_text")
    if isinstance(prediction_text, str) and prediction_text.strip().lower() != "null":
        return prediction_text
    return ""


def _audio_transcription(record: dict[str, Any]) -> str:
    """Return the ASR transcription attached to a generated-audio trace."""
    output_audio = record["output"].get("audio_path")
    prediction_text = record["evaluation"].get("prediction_text")
    if (
        isinstance(output_audio, str)
        and output_audio
        and isinstance(prediction_text, str)
        and prediction_text.strip().lower() != "null"
    ):
        return prediction_text
    return ""


def _select_prediction(
    record: dict[str, Any],
    index: int,
    transcriptions: dict[int, str],
    source: str,
) -> tuple[str, str]:
    """Return ``(primary_prediction, direct_text)`` for one trace."""
    direct_text = _direct_text(record)
    audio_asr = transcriptions.get(index) or _audio_transcription(record)
    if source == "direct_text":
        prediction = direct_text
    elif source == "audio_asr":
        prediction = audio_asr
    else:
        prediction = audio_asr or direct_text
    return prediction or "null", direct_text


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
        transcriptions.get(int(record["index"]))
        or _audio_transcription(record)
        or prediction_text
        or "null"
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
    subsets: list[str] | None = None,
    trace_indices_only: bool = False,
    prediction_source: str = "auto",
    include_prediction_text: bool = True,
) -> Path:
    """Build ``{model}_{DATASET_NAME}.jsonl`` for evalkit from a unified trace.

    For official datasets (VoiceBench / OpenAudioBench) the eval file rows come
    from evalkit's own ``dataset.data`` (field consistency guaranteed), overlaid
    with prediction / prompt / prediction_text from the trace by ``index``.
    When ``subsets`` is given, only dataset rows whose subset is listed are
    emitted (e.g. evaluate only llama_questions/web_questions/trivia_qa of
    OpenAudioBench without polluting scores with un-inferred subsets). When
    ``trace_indices_only`` is true, official dataset rows missing from the
    trace are omitted; this keeps limited smoke runs from judging ``null``
    predictions for the rest of the dataset.

    For ``FlexiSLM-WER`` (ASR / TTS WER) there is no standalone dataset file:
    rows are built directly from trace records (subset=group, answer=reference).
    ``transcribe_callback`` maps trace index -> transcription of generated
    audio (used by TTS / s2s); without it the direct text output is used.

    ``prediction_source`` can pin the primary evaluation channel to
    ``direct_text`` or ``audio_asr``. The default ``auto`` preserves the
    historical audio-ASR-first fallback. ``include_prediction_text=False``
    prevents evalkit's OpenQA evaluator from launching a second judge pass for
    the auxiliary direct-text column.
    """
    if prediction_source not in {"auto", "direct_text", "audio_asr"}:
        raise ValueError(f"Unsupported prediction_source: {prediction_source!r}")

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
        selected_subsets = set(subsets) if subsets is not None else None
        rows = []
        for item in dataset.data:
            index = int(item["index"])
            subset = str(item.get("subset") or dataset_name)
            if selected_subsets is not None and subset not in selected_subsets:
                continue
            if trace_indices_only and index not in by_index:
                continue
            row = dict(item)
            row["subset"] = subset
            trace = by_index.get(index)
            if trace is None:
                row["prediction"] = "null"
                row["prompt"] = str(item.get("question") or "")
                prediction_text = ""
            else:
                prediction, prediction_text = _select_prediction(
                    trace, index, transcriptions, prediction_source
                )
                row["prediction"] = prediction
                row["prompt"] = _full_prompt(trace, item)
            if include_prediction_text:
                row["prediction_text"] = prediction_text or None
            else:
                row.pop("prediction_text", None)
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
    """Return judge artifacts without copying the shared API log per job."""
    del work_dir  # Kept in the signature for compatibility with existing callers.
    artifacts: list[Path] = []
    stem = eval_file.stem
    for pattern in (
        f"{stem}_judge*.jsonl",
        f"{stem}_wer_details.jsonl",
        f"{stem}_high_wer_summary*.json",
    ):
        artifacts.extend(sorted(eval_file.parent.glob(pattern)))
    llm_log = os.environ.get("LLM_API_LOG_FILE")
    if llm_log:
        artifacts.append(Path(llm_log).expanduser().resolve())
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


def _openqa_score(value: Any) -> float | None:
    """Parse one OpenQA judge response and enforce the 1--5 scale."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    try:
        score = float(text)
    except ValueError:
        match = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", text)
        if match is None:
            return None
        score = float(match.group(1))
    return score if 1.0 <= score <= 5.0 else None


def _run_openqa_judge(
    judge_model: Any,
    prompt_template: str,
    predictions: list[str],
    questions: list[str],
    *,
    threads: int = 10,
    timeout: float = 180.0,
    max_tokens: int = 4096,
) -> list[tuple[int, str | None]]:
    """Run the evalkit OpenQA judge with room for hidden reasoning tokens.

    Some configured OpenAI-compatible reasoning models consume evalkit's
    default 1024-token response budget before emitting the requested score.
    The rubric and sampling settings remain unchanged; only the response
    budget and outer timeout are increased.
    """
    from almeval.judge_models import judge_response

    async def process_one(index: int, semaphore: asyncio.Semaphore):
        async with semaphore:
            prompt = prompt_template.format(
                question=questions[index].strip(),
                prediction=predictions[index],
            )
            try:
                response = await asyncio.wait_for(
                    judge_response(
                        prompt,
                        judge_model=judge_model,
                        temperature=0.5,
                        top_p=0.95,
                        max_tokens=max_tokens,
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                print(
                    f"IFEval OpenQA sample {index} timed out after {timeout}s.",
                    flush=True,
                )
                response = None
            except Exception as error:
                print(
                    f"IFEval OpenQA sample {index} failed: {error}",
                    flush=True,
                )
                response = None
            return index, response

    async def process_all():
        semaphore = asyncio.Semaphore(threads)
        return await asyncio.gather(
            *(process_one(index, semaphore) for index in range(len(predictions)))
        )

    return asyncio.run(process_all())


def _ifeval_openqa_cached_evaluate(
    *,
    dataset: Any,
    eval_file: Path,
    judge_model_name: str,
    prediction_source: str,
    dump_judge: bool,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Judge IFEval with VoiceBench's OpenQA rubric, reusing valid rows.

    Unlike evalkit's generic OpenQA path, this requires a valid 1--5 score for
    every sample before reporting an average. Existing judge JSONL rows are
    reused only when their index, instruction, and prediction still match.
    Failed/time-out rows are retried without re-judging successful samples.
    """
    from almeval.datasets.ds_openqa import OPEN_QA_PROMPT
    from almeval.judge_models import get_judge_model

    rows: list[dict[str, Any]] = []
    with eval_file.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"IFEval eval file contains no samples: {eval_file}")

    safe_model = judge_model_name.replace("/", "-")
    judge_file = eval_file.with_name(
        f"{eval_file.stem}_{safe_model}_judge.jsonl"
    )
    cached_by_index: dict[int, dict[str, Any]] = {}
    if judge_file.is_file():
        with judge_file.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                cached = json.loads(line)
                try:
                    cached_by_index[int(cached["index"])] = cached
                except (KeyError, TypeError, ValueError):
                    continue

    def question(row: dict[str, Any]) -> str:
        return str(row.get("audio_content") or row.get("prompt") or "")

    judged_rows: list[dict[str, Any]] = []
    scores: dict[int, float] = {}
    for row in rows:
        index = int(row["index"])
        cached = cached_by_index.get(index)
        if (
            cached is not None
            and str(cached.get("prediction")) == str(row.get("prediction"))
            and question(cached) == question(row)
        ):
            score = _openqa_score(cached.get("judge_result"))
            if score is not None:
                scores[index] = score
                judged_rows.append(cached)
                continue
        judged_rows.append({**row, "judge_result": None})

    judge_model = get_judge_model(judge_model_name)
    for attempt in range(1, max_attempts + 1):
        pending_positions = [
            position
            for position, row in enumerate(judged_rows)
            if int(row["index"]) not in scores
        ]
        if not pending_positions:
            break
        print(
            f"IFEval OpenQA judge attempt {attempt}/{max_attempts}: "
            f"{len(pending_positions)} uncached samples",
            flush=True,
        )
        pending_predictions = [
            str(judged_rows[position].get("prediction") or "null")
            for position in pending_positions
        ]
        pending_questions = [
            question(judged_rows[position]) for position in pending_positions
        ]
        results = _run_openqa_judge(
            judge_model,
            OPEN_QA_PROMPT,
            pending_predictions,
            pending_questions,
        )
        for pending_index, response in results:
            position = pending_positions[pending_index]
            row = judged_rows[position]
            row["judge_result"] = response
            score = _openqa_score(response)
            if score is not None:
                scores[int(row["index"])] = score

        if dump_judge:
            with judge_file.open("w", encoding="utf-8") as file:
                for row in judged_rows:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")

    missing = [int(row["index"]) for row in judged_rows if int(row["index"]) not in scores]
    if missing:
        raise RuntimeError(
            "IFEval OpenQA judge did not return valid 1--5 scores for all "
            f"samples after {max_attempts} attempts; missing indices: {missing}"
        )

    if dump_judge:
        with judge_file.open("w", encoding="utf-8") as file:
            for row in judged_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    average = round(sum(scores.values()) / len(rows), 2)
    task_result: dict[str, Any] = {
        "score": average,
        "total": len(rows),
        "invalid": 0,
    }
    if prediction_source == "direct_text":
        task_result["score_direct_text"] = average
    elif prediction_source == "audio_asr":
        task_result["score_audio_asr"] = average
    return dataset.format_performance(
        dataset.get_model_name(str(eval_file)),
        {"ifeval": task_result},
        eval_method=judge_model_name,
    )


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
    prediction_source: str = "auto",
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
    elif dataset_name == "ifeval" and effective_method != "vb-ifeval":
        result = _ifeval_openqa_cached_evaluate(
            dataset=dataset,
            eval_file=eval_file,
            judge_model_name=effective_method,
            prediction_source=prediction_source,
            dump_judge=dump_judge,
        )
    else:
        result = dataset.evaluate(
            str(eval_file), dump_judge=dump_judge, method=effective_method
        )
    archived = archive_judge_artifacts(eval_file, work_dir)
    return result, archived
