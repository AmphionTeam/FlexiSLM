"""Task dispatch and trace construction for FlexiSLM inference."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping


DEFAULT_ASR_PROMPT = "Please transcribe the audio."
DEFAULT_OUTPUT_SAMPLE_RATE = 24_000

# Lazy per-(model_path, device) Whisper transcriber cache for s2s traces.
_TRANSCRIBER_CACHE: dict[tuple[str, str], Any] = {}


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
    if task == "audio_qa":
        return _infer_audio_qa(
            engine,
            request,
            checkpoint=checkpoint,
            target_framerate_hz=target_framerate_hz,
        )
    if task == "s2s":
        return _infer_s2s(
            engine,
            request,
            output_dir=Path(output_dir),
            checkpoint=checkpoint,
            target_framerate_hz=target_framerate_hz,
            output_sample_rate=output_sample_rate,
        )
    raise ValueError(
        f"Unsupported inference task: {task!r}; "
        "expected 'tts', 'asr', 'audio_qa' or 's2s'"
    )


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
        subset=parts.evaluation.get("subset"),
        group=parts.evaluation.get("group"),
        metadata=parts.metadata,
        checkpoint=checkpoint,
        target_framerate_hz=framerate,
        audio_duration_seconds=_audio_duration(Path(audio_path)),
        engine=engine,
    )


def _infer_audio_qa(
    engine: Any,
    request: Mapping[str, Any],
    *,
    checkpoint: str | None,
    target_framerate_hz: float | None,
) -> dict[str, Any]:
    """Audio question answering: audio + text query -> text answer only."""
    parts = _request_parts(request)
    audio_path = _required_string(parts.input, "audio_path", task="audio_qa")
    framerate = _target_framerate(engine, parts.metadata, target_framerate_hz)
    # audio-only by default (matches legacy VoiceBench s2t): empty text query,
    # the user turn contains only the audio token.
    model_prompt = parts.input.get("model_prompt") or ""

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
        task="audio_qa",
        input_data=parts.input,
        output_text=prediction,
        output_audio_path=None,
        reference_text=parts.evaluation.get("reference_text")
        or parts.evaluation.get("answer"),
        prediction_text=prediction,
        subset=parts.evaluation.get("subset"),
        group=parts.evaluation.get("group"),
        answer=parts.evaluation.get("answer"),
        audio_content=parts.evaluation.get("audio_content"),
        instruction=parts.evaluation.get("instruction"),
        instruction_kwargs=parts.evaluation.get("instruction_kwargs"),
        metadata=parts.metadata,
        checkpoint=checkpoint,
        target_framerate_hz=framerate,
        audio_duration_seconds=_audio_duration(Path(audio_path)),
        engine=engine,
    )


def _infer_s2s(
    engine: Any,
    request: Mapping[str, Any],
    *,
    output_dir: Path,
    checkpoint: str | None,
    target_framerate_hz: float | None,
    output_sample_rate: int,
) -> dict[str, Any]:
    """Speech-to-speech: audio + text query -> audio output, transcribed for eval."""
    parts = _request_parts(request)
    audio_path = _required_string(parts.input, "audio_path", task="s2s")
    framerate = _target_framerate(engine, parts.metadata, target_framerate_hz)
    # audio-only by default (matches legacy s2s): empty text query, the user
    # turn contains only the audio token.
    model_prompt = parts.input.get("model_prompt") or ""

    result = engine.generate_from_audio(
        audio_path=audio_path,
        text_query=model_prompt,
        framerate=framerate,
        output_text_only=False,
    )
    if not isinstance(result, Mapping):
        raise TypeError("generate_from_audio() must return a mapping for s2s")
    direct_text = str(result.get("text") or "")

    audio = result.get("audio")
    if audio is None:
        audio = _decode_tts_audio(engine, result)
    if audio is None:
        raise ValueError("generate_from_audio() did not return decodable audio")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path_out = output_dir / _wav_name(parts.index, parts.metadata.get("sample_id"))
    duration = _save_wav(audio_path_out, audio, output_sample_rate)

    transcribe_model_path = parts.input.get("transcribe_model_path") or parts.metadata.get(
        "transcribe_model_path"
    )
    if not transcribe_model_path:
        raise ValueError(
            "s2s request requires input.transcribe_model_path or "
            "metadata.transcribe_model_path (e.g. whisper-large-v3)"
        )
    transcribe_device = parts.metadata.get("transcribe_device") or getattr(
        engine, "device", "cuda:0"
    )
    transcription = _transcribe_audio(
        str(audio_path_out.resolve()),
        model_path=str(transcribe_model_path),
        device=str(transcribe_device),
    )

    return _trace(
        index=parts.index,
        task="s2s",
        input_data=parts.input,
        output_text=direct_text,
        output_audio_path=str(audio_path_out.resolve()),
        reference_text=parts.evaluation.get("reference_text")
        or parts.evaluation.get("answer"),
        prediction_text=transcription,
        subset=parts.evaluation.get("subset"),
        group=parts.evaluation.get("group"),
        answer=parts.evaluation.get("answer"),
        audio_content=parts.evaluation.get("audio_content"),
        instruction=parts.evaluation.get("instruction"),
        instruction_kwargs=parts.evaluation.get("instruction_kwargs"),
        metadata=parts.metadata,
        checkpoint=checkpoint,
        target_framerate_hz=framerate,
        audio_duration_seconds=duration,
        engine=engine,
    )


def _transcribe_audio(audio_path: str, *, model_path: str, device: str) -> str:
    """Transcribe one WAV with Whisper-Large-V3 (transformers), cached per worker."""
    key = (model_path, device)
    if key not in _TRANSCRIBER_CACHE:
        import numpy as np
        import torch
        from transformers import (
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )

        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
        model = (
            WhisperForConditionalGeneration.from_pretrained(
                model_path, local_files_only=True, torch_dtype=dtype
            )
            .to(device)
            .eval()
        )

        @torch.inference_mode()
        def _transcribe_one(path: str) -> str:
            import soundfile as sf
            from scipy.signal import resample_poly

            audio, source_rate = sf.read(path, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if source_rate != 16_000:
                divisor = np.gcd(source_rate, 16_000)
                audio = resample_poly(
                    audio, 16_000 // divisor, source_rate // divisor
                )
            inputs = processor(
                audio,
                sampling_rate=16_000,
                return_tensors="pt",
                return_attention_mask=True,
            )
            features = inputs.input_features.to(device, dtype=model.dtype)
            attention_mask = inputs.attention_mask.to(device)
            generated = model.generate(
                features,
                attention_mask=attention_mask,
                language="english",
                task="transcribe",
            )
            return processor.batch_decode(
                generated, skip_special_tokens=True
            )[0].strip()

        _TRANSCRIBER_CACHE[key] = _transcribe_one
    return _TRANSCRIBER_CACHE[key](audio_path)


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
    subset: Any,
    group: Any,
    answer: Any = None,
    audio_content: Any = None,
    instruction: Any = None,
    instruction_kwargs: Any = None,
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
            "subset": subset,
            "group": group,
            "answer": answer,
            "audio_content": audio_content,
            "instruction": instruction,
            "instruction_kwargs": instruction_kwargs,
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
