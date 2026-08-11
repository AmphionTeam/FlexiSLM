"""ASR transcription of generated TTS audio.

WER is computed by the pinned Kimi-Audio-Evalkit ``FlexiSLM-WER`` benchmark;
this module only transcribes generated WAVs (with resume/details support) and
returns ``{index: hypothesis}`` for the eval adapter.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .asr import _load_trace_records


def _load_audio(path: Path, sample_rate: int = 16_000) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, source_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if source_rate != sample_rate:
        divisor = np.gcd(source_rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, source_rate // divisor)
    return np.asarray(audio, dtype=np.float32)


def _batches(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _build_transcriber(backend: str, model_path: str, device: str):
    import torch

    if backend == "hubert_ctc":
        from transformers import HubertForCTC, Wav2Vec2Processor

        processor = Wav2Vec2Processor.from_pretrained(model_path, local_files_only=True)
        model = HubertForCTC.from_pretrained(model_path, local_files_only=True)
        model = model.to(device).eval()

        @torch.inference_mode()
        def transcribe(audio_batch: list[np.ndarray], language: str) -> list[str]:
            del language
            inputs = processor(
                audio_batch,
                sampling_rate=16_000,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(device)
            attention_mask = getattr(inputs, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            logits = model(input_values, attention_mask=attention_mask).logits
            token_ids = logits.argmax(dim=-1)
            return [text.strip() for text in processor.batch_decode(token_ids)]

        return transcribe

    if backend == "whisper":
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device).eval()

        @torch.inference_mode()
        def transcribe(audio_batch: list[np.ndarray], language: str) -> list[str]:
            inputs = processor(
                audio_batch,
                sampling_rate=16_000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
            features = inputs.input_features.to(device, dtype=model.dtype)
            attention_mask = getattr(inputs, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            language_name = "english" if language == "en" else language
            generated = model.generate(
                features,
                attention_mask=attention_mask,
                language=language_name,
                task="transcribe",
            )
            return [
                text.strip()
                for text in processor.batch_decode(generated, skip_special_tokens=True)
            ]

        return transcribe

    raise ValueError(f"Unsupported TTS ASR backend: {backend!r}")


def _load_resumable_details(
    details_path: Path, backend: str, model_path: str
) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not details_path.is_file():
        return completed
    with details_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {details_path} at line {line_number}: {error}"
                ) from error
            if (
                row.get("asr_backend") == backend
                and row.get("asr_model_path") == model_path
                and isinstance(row.get("index"), int)
                and isinstance(row.get("asr_hypothesis"), str)
            ):
                completed[int(row["index"])] = row
    return completed


def transcribe_tts_trace(
    *,
    trace_file: Path,
    backend: str,
    model_path: str,
    device: str,
    batch_size: int,
    details_path: Path,
    resume: bool = True,
    language: str = "en",
) -> dict[int, str]:
    """Transcribe generated TTS WAVs with resume; return ``{index: hypothesis}``."""
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    records = _load_trace_records(trace_file)
    for record in records:
        if record["task"] != "tts":
            raise ValueError(f"Trace index {record['index']} is not a TTS record")
        audio_path = record["output"].get("audio_path")
        if not isinstance(audio_path, str) or not Path(audio_path).is_file():
            raise FileNotFoundError(
                f"Trace index {record['index']} has no existing output audio: {audio_path}"
            )

    details_path.parent.mkdir(parents=True, exist_ok=True)
    completed = (
        _load_resumable_details(details_path, backend, model_path) if resume else {}
    )
    pending = [record for record in records if int(record["index"]) not in completed]
    print(
        f"TTS ASR preflight: {len(records)} records, {len(completed)} resumed, "
        f"{len(pending)} pending",
        flush=True,
    )

    if pending:
        transcribe = _build_transcriber(backend, model_path, device)
        file_mode = "a" if completed else "w"
        with details_path.open(file_mode, encoding="utf-8") as details_file:
            processed = len(completed)
            for batch in _batches(pending, batch_size):
                audio_batch = [
                    _load_audio(Path(record["output"]["audio_path"])) for record in batch
                ]
                hypotheses = transcribe(audio_batch, language)
                for record, hypothesis in zip(batch, hypotheses):
                    row = {
                        "index": int(record["index"]),
                        "sample_id": record["metadata"].get("sample_id"),
                        "audio_path": record["output"]["audio_path"],
                        "reference_text": record["evaluation"]["reference_text"],
                        "asr_hypothesis": hypothesis,
                        "asr_backend": backend,
                        "asr_model_path": model_path,
                    }
                    details_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    details_file.flush()
                    completed[row["index"]] = row
                processed += len(batch)
                print(f"TTS ASR progress: {processed}/{len(records)}", flush=True)
        del transcribe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {int(row["index"]): row["asr_hypothesis"] for row in completed.values()}
