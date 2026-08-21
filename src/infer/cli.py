"""YAML-driven CLI for unified FlexiSLM inference traces."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Tuple


from .core import DEFAULT_OUTPUT_SAMPLE_RATE, infer

# src/infer/cli.py -> repository root (matches training/inference path conventions).
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CliConfig:
    """Validated runtime configuration for the inference CLI."""

    engine_config: dict[str, Any]
    devices: tuple[str, ...]
    workers_per_device: int
    input_path: Path
    trace_path: Path
    audio_dir: Path
    checkpoint: Optional[str]
    target_framerate_hz: Optional[float]
    transcribe_model_path: Optional[str]
    output_sample_rate: int
    fail_fast: bool
    error_path: Path


_WORKER_ENGINE: Any = None
_WORKER_OPTIONS: dict[str, Any] = {}


def load_config(config_path: Path) -> CliConfig:
    """Load and validate one inference YAML configuration."""
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "PyYAML is required by the inference CLI; install project requirements"
        ) from error

    config_path = config_path.expanduser().resolve()
    with config_path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

    engine = _mapping(raw.get("engine"), "engine")
    engine_config = _mapping(engine.get("config"), "engine.config")
    if (
        not engine_config.get("model_path")
        and not engine_config.get("auto_download")
        and not engine_config.get("checkpoint")
    ):
        raise ValueError(
            "engine.config.model_path is required unless auto_download is true "
            "or checkpoint is set to stage2_7B / stage2_0.5B"
        )

    runtime = _mapping(raw.get("runtime", {}), "runtime")
    devices_value = runtime.get("devices", ["cuda:0"])
    if not isinstance(devices_value, list) or not devices_value:
        raise ValueError("runtime.devices must be a non-empty list")
    if any(not isinstance(device, str) or not device.strip() for device in devices_value):
        raise ValueError("runtime.devices entries must be non-empty strings")
    devices = tuple(device.strip() for device in devices_value)
    if len(set(devices)) != len(devices):
        raise ValueError("runtime.devices must not contain duplicates")
    workers_per_device = runtime.get("workers_per_device", 1)
    if (
        not isinstance(workers_per_device, int)
        or isinstance(workers_per_device, bool)
        or workers_per_device < 1
    ):
        raise ValueError("runtime.workers_per_device must be a positive integer")

    input_config = _mapping(raw.get("input"), "input")
    output_config = _mapping(raw.get("output"), "output")
    input_path = _resolve_path(input_config.get("path"), "input.path")
    trace_path = _resolve_path(
        output_config.get("trace_path"), "output.trace_path"
    )
    audio_dir_value = output_config.get("audio_dir")
    audio_dir = (
        _resolve_path(audio_dir_value, "output.audio_dir")
        if audio_dir_value is not None
        else trace_path.parent / "audio"
    )

    inference_config = _mapping(raw.get("inference", {}), "inference")
    target_framerate = inference_config.get("target_framerate_hz")
    if target_framerate is not None:
        target_framerate = float(target_framerate)
        if target_framerate <= 0:
            raise ValueError("inference.target_framerate_hz must be positive")
    sample_rate = int(
        inference_config.get("output_sample_rate", DEFAULT_OUTPUT_SAMPLE_RATE)
    )
    if sample_rate <= 0:
        raise ValueError("inference.output_sample_rate must be positive")

    checkpoint = inference_config.get("checkpoint")
    if checkpoint is not None:
        checkpoint = str(checkpoint)
    transcribe_model_path = inference_config.get("transcribe_model_path")
    if transcribe_model_path is not None:
        transcribe_model_path = str(transcribe_model_path)
    fail_fast = runtime.get("fail_fast", True)
    if not isinstance(fail_fast, bool):
        raise ValueError("runtime.fail_fast must be a boolean")
    error_path_value = output_config.get("error_path")
    error_path = (
        _resolve_path(error_path_value, "output.error_path")
        if error_path_value is not None
        else trace_path.with_name(f"{trace_path.stem}.errors.jsonl")
    )

    return CliConfig(
        engine_config=engine_config,
        devices=devices,
        workers_per_device=workers_per_device,
        input_path=input_path,
        trace_path=trace_path,
        audio_dir=audio_dir,
        checkpoint=checkpoint,
        target_framerate_hz=target_framerate,
        transcribe_model_path=transcribe_model_path,
        output_sample_rate=sample_rate,
        fail_fast=fail_fast,
        error_path=error_path,
    )


def run(config_path: Path) -> tuple[int, int]:
    """Execute configured inference and return successful and failed counts."""
    config = load_config(config_path)
    if not config.input_path.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {config.input_path}")

    config.trace_path.parent.mkdir(parents=True, exist_ok=True)
    config.audio_dir.mkdir(parents=True, exist_ok=True)
    if not config.fail_fast:
        config.error_path.parent.mkdir(parents=True, exist_ok=True)

    options = {
        "output_dir": str(config.audio_dir),
        "checkpoint": config.checkpoint,
        "target_framerate_hz": config.target_framerate_hz,
        "transcribe_model_path": config.transcribe_model_path,
        "output_sample_rate": config.output_sample_rate,
    }
    requests = _load_requests(config.input_path)
    worker_devices = tuple(
        device
        for device in config.devices
        for _ in range(config.workers_per_device)
    )

    print(
        f"Inference devices: {', '.join(config.devices)} "
        f"({len(worker_devices)} model replica(s), "
        f"{config.workers_per_device} per device)"
    )
    print(f"Input: {config.input_path}")
    print(f"Trace output: {config.trace_path}")

    if len(worker_devices) == 1:
        engine = _build_engine(config.engine_config, worker_devices[0])
        results = (_infer_one(engine, options, item) for item in requests)
        return _write_results(results, config)

    context = multiprocessing.get_context("spawn")
    device_queue = context.Queue()
    for device in worker_devices:
        device_queue.put(device)
    try:
        with ProcessPoolExecutor(
            max_workers=len(worker_devices),
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(config.engine_config, device_queue, options),
        ) as executor:
            results = _map_bounded(
                executor,
                _worker_infer,
                requests,
                max_pending=len(worker_devices) * 2,
            )
            return _write_results(results, config)
    finally:
        device_queue.close()
        device_queue.join_thread()


def _map_bounded(
    executor: Any,
    function: Any,
    items: Iterable[Tuple[int, dict[str, Any]]],
    *,
    max_pending: int,
) -> Iterator[dict[str, Any]]:
    """Submit a bounded number of jobs while yielding results in input order."""
    iterator = iter(items)
    pending = deque()
    for _ in range(max_pending):
        try:
            pending.append(executor.submit(function, next(iterator)))
        except StopIteration:
            break

    while pending:
        yield pending.popleft().result()
        try:
            pending.append(executor.submit(function, next(iterator)))
        except StopIteration:
            pass


def _build_engine(engine_config: Mapping[str, Any], device: str) -> Any:
    # Set the process-local CUDA default before importing the legacy engine. Its
    # Hugging Face loader uses ``device_map="cuda"``, which otherwise always
    # places independently launched single-device jobs on physical cuda:0.
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.set_device(device)

    # Keep the legacy module as the engine implementation only. Batch orchestration
    # belongs to this package and does not modify its legacy CLI.
    from src.inference_flexislm import (
        FlexiSLMInferenceConfig,
        FlexiSLMInference,
    )

    config = FlexiSLMInferenceConfig(**dict(engine_config))
    return FlexiSLMInference(config, device=device)


def _initialize_worker(
    engine_config: Mapping[str, Any],
    device_queue: Any,
    options: Mapping[str, Any],
) -> None:
    global _WORKER_ENGINE, _WORKER_OPTIONS

    # Each spawned process consumes exactly one device, ensuring that model
    # replicas never share a GPU accidentally. Set the process-local default
    # before importing the legacy engine because its HF loader uses
    # ``device_map="cuda"`` during construction.
    device = device_queue.get()
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.set_device(device)
    _WORKER_ENGINE = _build_engine(engine_config, device)
    _WORKER_OPTIONS = dict(options)


def _worker_infer(item: Tuple[int, dict[str, Any]]) -> dict[str, Any]:
    if _WORKER_ENGINE is None:
        raise RuntimeError("Inference worker was not initialized")
    return _infer_one(_WORKER_ENGINE, _WORKER_OPTIONS, item)


def _infer_one(
    engine: Any,
    options: Mapping[str, Any],
    item: Tuple[int, dict[str, Any]],
) -> dict[str, Any]:
    sequence, request = item
    try:
        trace = infer(engine, request, **dict(options))
        return {"sequence": sequence, "trace": trace, "error": None}
    except Exception as error:
        return {
            "sequence": sequence,
            "trace": None,
            "error": {
                "index": request.get("index"),
                "task": request.get("task"),
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }


def _load_requests(path: Path) -> Iterator[Tuple[int, dict[str, Any]]]:
    seen_indices: set[int] = set()
    with path.open(encoding="utf-8") as file:
        for sequence, line in enumerate(file):
            if not line.strip():
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {sequence + 1}: {error}"
                ) from error
            if not isinstance(request, dict):
                raise ValueError(f"{path} line {sequence + 1} must be a JSON object")
            request.setdefault("index", sequence)
            index = request["index"]
            if not isinstance(index, int) or isinstance(index, bool):
                raise ValueError(f"{path} line {sequence + 1} index must be an integer")
            if index in seen_indices:
                raise ValueError(f"{path} contains duplicate index {index}")
            seen_indices.add(index)
            yield sequence, request


def _write_results(
    results: Iterable[dict[str, Any]], config: CliConfig
) -> tuple[int, int]:
    successful = 0
    failed = 0
    error_file = None
    try:
        if not config.fail_fast:
            error_file = config.error_path.open("w", encoding="utf-8")
        with config.trace_path.open("w", encoding="utf-8") as trace_file:
            for result in results:
                error = result["error"]
                if error is not None:
                    failed += 1
                    if config.fail_fast:
                        raise RuntimeError(
                            f"Inference failed for index {error['index']}: "
                            f"{error['type']}: {error['message']}"
                        )
                    assert error_file is not None
                    error_file.write(json.dumps(error, ensure_ascii=False) + "\n")
                    error_file.flush()
                    continue

                trace_file.write(
                    json.dumps(result["trace"], ensure_ascii=False) + "\n"
                )
                trace_file.flush()
                successful += 1
    finally:
        if error_file is not None:
            error_file.close()

    print(f"Inference complete: {successful} succeeded, {failed} failed")
    if failed:
        print(f"Errors: {config.error_path}")
    return successful, failed


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a YAML mapping")
    return dict(value)


def _resolve_path(value: Any, name: str) -> Path:
    """Resolve a config path; relative paths are anchored at the repo root."""
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (_REPO_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FlexiSLM inference from a YAML configuration."
    )
    parser.add_argument("config", type=Path, help="Path to inference YAML")
    args = parser.parse_args()
    _, failed = run(args.config)
    if failed:
        raise SystemExit(1)
