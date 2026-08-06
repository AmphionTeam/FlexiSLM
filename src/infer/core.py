"""Task dispatch and trace construction for FlexiSLM inference."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping


DEFAULT_ASR_PROMPT = "Please transcribe the audio."
DEFAULT_OUTPUT_SAMPLE_RATE = 24_000


def infer(
    engine: Any,
    request: Mapping[str, Any],
    *,
    output_dir: str | os.PathLike[str] = "outputs/infer",
    checkpoint: str | None = None,
    target_framerate_hz: float | None = None,
    output_sample_rate: int = DEFAULT_OUTPUT_SAMPLE_RATE,
) -> dict[str, Any]:
    """Infer one request and return a record in the unified trace schema."""
    if not isinstance(request, Mapping):
        raise TypeError("request must be a mapping")

    task = request.get("task")
    if task == "tts":
        return _infer_tts(
            engine,
            request,
            output_dir=Path(output_dir),
            checkpoint=checkpoint,
            target_framerate_hz=target_framerate_hz,
            output_sample_rate=output_sample_rate,
        )
    if task == "asr":
        return _infer_asr(
            engine,
            request,
            checkpoint=checkpoint,
            target_framerate_hz=target_framerate_hz,
        )
    raise ValueError(f"Unsupported inference task: {task!r}; expected 'tts' or 'asr'")


def _infer_tts(
    engine: Any,
    request: Mapping[str, Any],
    *,
    output_dir: Path,
    checkpoint: str | None,
    target_framerate_hz: float | None,
    output_sample_rate: int,
) -> dict[str, Any]:
    parts = _request_parts(request)
    text = _required_string(parts.input, "text", task="tts")
    framerate = _target_framerate(engine, parts.metadata, target_framerate_hz)

    result = engine.generate_tts(
        sentence=text,
        framerate=framerate,
        system_prompt=parts.input.get("model_prompt"),
        flow_matching_prompt_audio_path=_existing_audio_path(
            parts.input.get("prompt_audio_path")
        ),
    )
    if not isinstance(result, Mapping):
        raise TypeError("generate_tts() must return a mapping")

    audio = result.get("audio")
    if audio is None:
        audio = _decode_tts_audio(engine, result)
    if audio is None:
        raise ValueError("generate_tts() did not return decodable audio")

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / _wav_name(parts.index, parts.metadata.get("sample_id"))
    duration = _save_wav(audio_path, audio, output_sample_rate)

    return _trace(
        index=parts.index,
        task="tts",
        input_data=parts.input,
        output_text=str(result.get("text") or ""),
        output_audio_path=str(audio_path.resolve()),
        reference_text=parts.evaluation.get("reference_text"),
        prediction_text=None,
        group=parts.evaluation.get("group"),
        metadata=parts.metadata,
        checkpoint=checkpoint,
        target_framerate_hz=framerate,
        audio_duration_seconds=duration,
        engine=engine,
    )


def _infer_asr(
    engine: Any,
    request: Mapping[str, Any],
    *,
    checkpoint: str | None,
    target_framerate_hz: float | None,
) -> dict[str, Any]:
    parts = _request_parts(request)
    audio_path = _required_string(parts.input, "audio_path", task="asr")
    framerate = _target_framerate(engine, parts.metadata, target_framerate_hz)
    model_prompt = parts.input.get("model_prompt") or DEFAULT_ASR_PROMPT

    result = engine.generate_from_audio(
        audio_path=audio_path,
        text_query=model_prompt,
        framerate=framerate,
        output_text_only=True,
    )
    if isinstance(result, Mapping):
        prediction = str(result.get("text") or "")
    elif isinstance(result, str):
        prediction = result
    else:
        raise TypeError("generate_from_audio() must return a mapping or string")

    return _trace(
        index=parts.index,
        task="asr",
        input_data=parts.input,
        output_text=prediction,
        output_audio_path=None,
        reference_text=parts.evaluation.get("reference_text"),
        prediction_text=prediction,
        group=parts.evaluation.get("group"),
        metadata=parts.metadata,
        checkpoint=checkpoint,
        target_framerate_hz=framerate,
        audio_duration_seconds=_audio_duration(Path(audio_path)),
        engine=engine,
    )


class _RequestParts:
    def __init__(self, request: Mapping[str, Any]) -> None:
        index = request.get("index", 0)
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("request['index'] must be an integer")
        self.index = index
        self.input = _mapping(request.get("input"), "request['input']")
        self.evaluation = _mapping(
            request.get("evaluation", {}), "request['evaluation']"
        )
        self.metadata = _mapping(request.get("metadata", {}), "request['metadata']")


def _request_parts(request: Mapping[str, Any]) -> _RequestParts:
    return _RequestParts(request)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _required_string(data: Mapping[str, Any], field: str, *, task: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{task} request input.{field} must be a non-empty string")
    return value


def _existing_audio_path(value: Any) -> str | None:
    """Return an existing per-request prompt, or allow the engine default to win."""
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return str(path) if path.is_file() else None


def _target_framerate(
    engine: Any,
    metadata: Mapping[str, Any],
    explicit_value: float | None,
) -> float | None:
    value = explicit_value
    if value is None:
        value = metadata.get("target_framerate_hz")
    if value is None:
        value = getattr(getattr(engine, "config", None), "default_framerate", None)
    return float(value) if value is not None else None


def _checkpoint(engine: Any, metadata: Mapping[str, Any], explicit: str | None) -> Any:
    if explicit is not None:
        return explicit
    if metadata.get("checkpoint") is not None:
        return metadata["checkpoint"]
    return getattr(getattr(engine, "config", None), "model_path", None)


def _trace(
    *,
    index: int,
    task: str,
    input_data: Mapping[str, Any],
    output_text: str,
    output_audio_path: str | None,
    reference_text: Any,
    prediction_text: str | None,
    group: Any,
    metadata: Mapping[str, Any],
    checkpoint: str | None,
    target_framerate_hz: float | None,
    audio_duration_seconds: float | None,
    engine: Any,
) -> dict[str, Any]:
    return {
        "index": index,
        "task": task,
        "input": {
            "text": input_data.get("text"),
            "audio_path": input_data.get("audio_path"),
            "model_prompt": input_data.get("model_prompt"),
        },
        "output": {"text": output_text, "audio_path": output_audio_path},
        "evaluation": {
            "reference_text": reference_text,
            "prediction_text": prediction_text,
            "group": group,
        },
        "metadata": {
            "sample_id": metadata.get("sample_id"),
            "checkpoint": _checkpoint(engine, metadata, checkpoint),
            "target_framerate_hz": target_framerate_hz,
            "audio_duration_seconds": audio_duration_seconds,
        },
    }


def _wav_name(index: int, sample_id: Any) -> str:
    stem = str(sample_id) if sample_id not in (None, "") else str(index)
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or str(index)
    return f"{index}_{safe_stem}.wav"


def _save_wav(path: Path, audio: Any, sample_rate: int) -> float:
    if sample_rate <= 0:
        raise ValueError("output_sample_rate must be positive")
    import torch
    import torchaudio

    waveform = torch.as_tensor(audio).detach().float().cpu().squeeze()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError(f"Unexpected generated audio shape: {tuple(waveform.shape)}")
    torchaudio.save(str(path), waveform, sample_rate)
    return waveform.shape[-1] / sample_rate


def _decode_tts_audio(engine: Any, result: Mapping[str, Any]) -> Any:
    audio_ids = result.get("audio_ids")
    if audio_ids is None:
        return None
    import torch

    device = getattr(engine, "device", None)
    audio_ids = torch.as_tensor(audio_ids, device=device)
    length_ids = result.get("length_ids")
    length_ids = (
        torch.as_tensor(length_ids, device=device) if length_ids is not None else None
    )
    config = getattr(engine, "config", None)
    if getattr(config, "use_flow_matching_decoder", False):
        prompt_path = result.get("flow_matching_prompt_audio_path") or getattr(
            config, "flow_matching_prompt_audio_path", None
        )
        default_prompt = getattr(config, "flow_matching_prompt_audio_path", None)
        prompt_audio = (
            getattr(engine, "prompt_audio_cache", None)
            if prompt_path == default_prompt
            else None
        )
        return engine._decode_audio_tokens(
            audio_tokens=audio_ids,
            length_ids=length_ids,
            prompt_audio=prompt_audio,
            prompt_audio_path=prompt_path,
            framerate=result.get("framerate"),
        )
    return engine._decode_audio_tokens_flexicodec(
        audio_tokens=audio_ids, length_ids=length_ids
    )


def _audio_duration(path: Path) -> float | None:
    try:
        import soundfile

        info = soundfile.info(str(path))
        return float(info.frames / info.samplerate)
    except (ImportError, OSError, RuntimeError, ValueError, ZeroDivisionError):
        return None
