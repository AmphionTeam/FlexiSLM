#!/usr/bin/env python3
"""Build a legacy-aligned evaluation plan from a single YAML config.

The plan YAML declares which benchmarks/subsets/modes to evaluate. Modes:
  s2t  audio in  -> text out (audio-only input, matches legacy VoiceBench)
  s2s  audio in  -> audio out (whisper-transcribed for evaluation)
  asr  audio in  -> text out (WER, e.g. LibriSpeech test-clean/test-other)
  tts  text in   -> audio out (WER)

This script generates, per (benchmark, subset, mode) job:
  1. a request JSONL (audio-only input for s2t/s2s, ASR prompt for asr),
  2. an inference manifest YAML consumed by the wave launcher,
  3. an evaluation config YAML consumed by ``python -m src.eval``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

VOICEBENCH_SUBSETS = {
    "alpacaeval_full",
    "commoneval",
    "ifeval",
    "sd-qa",
    "advbench",
    "mmsu",
    "openbookqa",
}
OPEN_AUDIO_BENCH_SUBSETS = {
    "llama_questions",
    "web_questions",
    "trivia_qa",
    "alpaca_eval",
    "reasoning_qa",
}
LIBRISPEECH_SUBSETS = {"test-clean", "test-other"}

# VoiceBench per-subset primary evaluation methods (evalkit
# dataset.evaluate methods). IFEval uses the same OpenQA 1--5 LLM judge as
# AlpacaEval/CommonEval for the paper table; its canonical strict/loose rules
# are run separately as auxiliary metrics.
VOICEBENCH_EVAL_METHOD = {
    "alpacaeval_full": "default",
    "commoneval": "default",
    "ifeval": "default",
    "sd-qa": "vb-qa",
    "advbench": "vb-advbench",
    "mmsu": "vb-mcq",
    "openbookqa": "vb-mcq",
}

ASR_PROMPT = "Transcribe the following audio:"
# FlexiCodec decodes native 16 kHz samples; do not relabel them as 24 kHz.
DEFAULT_OUTPUT_SAMPLE_RATE = 16_000


def _audio_stem(audio_path: str) -> str:
    return (
        re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(audio_path).stem).strip("_") or "0"
    )


def _write_jsonl(items: list[dict], out_file: Path) -> int:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as dst:
        for item in items:
            dst.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(items)


def build_audio_qa_requests(
    items: list[dict],
    *,
    task: str,
    out_file: Path,
    transcribe_model_path: str | None,
    limit: int | None,
) -> int:
    """Convert benchmark items into s2t (audio_qa) / s2s requests.

    Input is audio-only: ``model_prompt`` stays None so the inference core
    falls back to an empty text query (matching the legacy audio-only setup).
    """
    requests = []
    for item in items:
        index = int(item["index"])
        audio_path = str(item["audio_path"])
        audio_content = str(item.get("audio_content") or "")
        answer = item.get("answer")
        request = {
            "index": index,
            "task": task,
            "input": {"audio_path": audio_path, "model_prompt": None},
            "evaluation": {
                "subset": str(item.get("subset") or out_file.stem),
                "answer": answer,
                "audio_content": audio_content,
                "instruction": item.get("instruction"),
                "instruction_kwargs": item.get("instruction_kwargs"),
                "reference_text": answer,
            },
            "metadata": {
                "sample_id": _audio_stem(audio_path),
                "transcribe_model_path": transcribe_model_path if task == "s2s" else None,
                "transcribe_device": None,
            },
        }
        requests.append(request)
        if limit is not None and len(requests) >= limit:
            break
    return _write_jsonl(requests, out_file)


def build_asr_requests(
    *,
    subset: str,
    data_root: Path,
    out_file: Path,
    limit: int | None,
) -> int:
    """Build LibriSpeech ASR requests (flac + trans.txt) for WER evaluation."""
    ls_root = (
        data_root / "LibriSpeech" / "librispeech" / "LibriSpeech" / subset
    )
    if not ls_root.is_dir():
        raise FileNotFoundError(f"LibriSpeech subset dir not found: {ls_root}")

    transcripts: dict[str, str] = {}
    for trans_file in sorted(ls_root.rglob("*.trans.txt")):
        for line in trans_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            key, _, text = line.partition(" ")
            transcripts[key] = text.strip()

    requests = []
    for index, flac in enumerate(sorted(ls_root.rglob("*.flac"))):
        key = flac.stem
        requests.append(
            {
                "index": index,
                "task": "asr",
                "input": {"audio_path": str(flac), "model_prompt": ASR_PROMPT},
                "evaluation": {
                    "reference_text": transcripts.get(key, ""),
                    "prediction_text": None,
                    "group": f"librispeech_{subset}",
                },
                "metadata": {"sample_id": key},
            }
        )
        if limit is not None and len(requests) >= limit:
            break
    return _write_jsonl(requests, out_file)


def load_benchmark_items(benchmark: str, subset: str, data_root: Path) -> list[dict]:
    if benchmark == "voicebench":
        source = data_root / "VoiceBench" / f"{subset}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(f"VoiceBench source not found: {source}")
        with source.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    if benchmark == "openaudiobench":
        source = data_root / "OpenAudioBench" / "OpenAudioBench.jsonl"
        if not source.is_file():
            raise FileNotFoundError(f"OpenAudioBench source not found: {source}")
        with source.open(encoding="utf-8") as fh:
            return [
                json.loads(line)
                for line in fh
                if line.strip() and json.loads(line).get("subset") == subset
            ]
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def job_name(benchmark: str, subset: str, mode: str) -> str:
    return f"{benchmark}_{subset}_{mode}"


def mode_task(mode: str) -> str:
    return {"s2t": "audio_qa", "s2s": "s2s", "asr": "asr", "tts": "tts"}[mode]


def build_requests(
    *,
    benchmark: str,
    subset: str,
    mode: str,
    data_root: Path,
    requests_root: Path,
    transcribe_model_path: str | None,
    limit: int | None,
) -> Path:
    out_file = requests_root / f"{job_name(benchmark, subset, mode)}_requests.jsonl"
    if mode in ("s2t", "s2s"):
        items = load_benchmark_items(benchmark, subset, data_root)
        build_audio_qa_requests(
            items,
            task=mode_task(mode),
            out_file=out_file,
            transcribe_model_path=transcribe_model_path,
            limit=limit,
        )
    elif mode == "asr":
        build_asr_requests(
            subset=subset,
            data_root=data_root,
            out_file=out_file,
            limit=limit,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return out_file


def build_infer_manifest(
    *,
    plan: dict,
    jobs: list[dict],
    requests_root: Path,
    output_root: Path,
    workers_per_device_override: int | None = None,
) -> Path:
    engine = plan["engine"]
    checkpoint = str(plan["checkpoint"])
    target_framerate = float(engine["inference"].get("target_framerate_hz", 8.0))
    workers_per_device = (
        workers_per_device_override
        if workers_per_device_override is not None
        else int(engine["runtime"].get("workers_per_device", 1))
    )
    manifest = {
        "x-engine": {"engine": {"config": engine["config"]}},
        "jobs": [],
    }
    for job in jobs:
        name = job["name"]
        trace_dir = output_root / name
        manifest["jobs"].append(
            {
                "name": name,
                "config": {
                    "engine": {"config": engine["config"]},
                    "input": {
                        "path": str(requests_root / f"{name}_requests.jsonl")
                    },
                    "output": {
                        "trace_path": str(trace_dir / "traces.jsonl"),
                        "error_path": str(trace_dir / "errors.jsonl"),
                    },
                    "inference": {
                        "checkpoint": checkpoint,
                        "target_framerate_hz": target_framerate,
                        "output_sample_rate": DEFAULT_OUTPUT_SAMPLE_RATE,
                    },
                    "runtime": {
                        "devices": ["cuda:0"],
                        "fail_fast": False,
                        "workers_per_device": workers_per_device,
                    },
                },
            }
        )
    manifest_path = output_root / "infer_manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False)
    return manifest_path


def build_eval_config(
    *,
    plan: dict,
    jobs: list[dict],
    output_root: Path,
    trace_indices_only: bool = False,
) -> Path:
    eval_cfg = plan["eval"]
    config = {
        "evalkit_path": str(eval_cfg["evalkit_path"]),
        "data_root": str(plan["data_root"]),
        "model_name": str(plan["model_name"]),
        "judge_model": str(eval_cfg["judge_model"]),
        "judge_api_base": eval_cfg.get("judge_api_base"),
        "judge_api_key_env": eval_cfg.get("judge_api_key_env"),
        "judge_api_key": eval_cfg.get("judge_api_key"),
        "evalkit_commit": eval_cfg.get("evalkit_commit"),
        "log_dir": str(output_root / "logs"),
        "voicebench_summary_path": str(output_root / "voicebench_summary.json"),
        "jobs": [],
    }
    for job in jobs:
        benchmark, subset, mode = job["benchmark"], job["subset"], job["mode"]
        name = job["name"]
        trace_file = output_root / name / "traces.jsonl"
        result_path = output_root / name / "performance.json"
        common_job = {
            "name": name,
            "trace_file": str(trace_file),
            "result_path": str(result_path),
        }
        if trace_indices_only:
            common_job["trace_indices_only"] = True
        if benchmark == "openaudiobench":
            config["jobs"].append(
                {
                    **common_job,
                    "benchmark": "openaudiobench",
                    "dataset": "OpenAudioBench",
                    "subsets": [subset],
                    "method": "default",
                }
            )
        elif benchmark == "voicebench":
            voicebench_job = {
                **common_job,
                "benchmark": "voicebench",
                "dataset": subset,
                "method": VOICEBENCH_EVAL_METHOD[subset],
            }
            if subset == "ifeval":
                # Be explicit about the paper-table channel. Do not ask
                # evalkit to judge prediction_text as a second channel: S2T's
                # primary prediction is direct text, while S2S's is the cached
                # generated-audio ASR transcription.
                voicebench_job.update(
                    {
                        "prediction_source": (
                            "direct_text" if mode == "s2t" else "audio_asr"
                        ),
                        "include_prediction_text": False,
                        "auxiliary_method": "vb-ifeval",
                    }
                )
            config["jobs"].append(voicebench_job)
        elif benchmark == "librispeech":
            config["jobs"].append(
                {
                    **common_job,
                    "benchmark": "asr",
                }
            )
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark}")
    config_path = output_root / "eval_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    return config_path


def load_plan(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as fh:
        plan = yaml.safe_load(fh)
    required = {
        "model_name",
        "checkpoint",
        "data_root",
        "requests_root",
        "output_root",
        "engine",
        "benchmarks",
    }
    missing = required - plan.keys()
    if missing:
        raise ValueError(f"Plan is missing fields: {', '.join(sorted(missing))}")
    return plan


def expand_jobs(plan: dict) -> list[dict]:
    jobs = []
    for benchmark, spec in plan["benchmarks"].items():
        subsets = spec.get("subsets", [])
        modes = spec.get("modes", [])
        for subset in subsets:
            for mode in modes:
                jobs.append(
                    {
                        "name": job_name(benchmark, subset, mode),
                        "benchmark": benchmark,
                        "subset": subset,
                        "mode": mode,
                    }
                )
    if not jobs:
        raise ValueError("Plan expands to zero jobs")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Path to the evaluation plan YAML.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of samples per request file (smoke testing).",
    )
    args = parser.parse_args()

    plan = load_plan(args.plan)
    plan["data_root"] = Path(plan["data_root"]).expanduser().resolve()
    plan["requests_root"] = Path(plan["requests_root"]).expanduser().resolve()
    plan["output_root"] = Path(plan["output_root"]).expanduser().resolve()
    jobs = expand_jobs(plan)

    total = 0
    for job in jobs:
        request_file = build_requests(
            benchmark=job["benchmark"],
            subset=job["subset"],
            mode=job["mode"],
            data_root=plan["data_root"],
            requests_root=plan["requests_root"],
            transcribe_model_path=plan.get("transcribe_model_path"),
            limit=args.limit,
        )
        count = sum(1 for _ in request_file.open(encoding="utf-8"))
        total += count
        print(f"{job['name']:60s} -> {request_file} ({count})")

    manifest_path = build_infer_manifest(
        plan=plan,
        jobs=jobs,
        requests_root=plan["requests_root"],
        output_root=plan["output_root"],
        workers_per_device_override=1 if args.limit is not None else None,
    )
    eval_config_path = build_eval_config(
        plan=plan,
        jobs=jobs,
        output_root=plan["output_root"],
        trace_indices_only=args.limit is not None,
    )
    print(f"\nTotal requests: {total}")
    print(f"Infer manifest: {manifest_path}")
    print(f"Eval config:    {eval_config_path}")


if __name__ == "__main__":
    main()
