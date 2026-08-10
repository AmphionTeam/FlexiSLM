"""Physical WebDataset layout adapters.

Adapters are intentionally independent from WebDataset itself: their input is the
mapping produced by ``group_by_keys``. This keeps layout behavior deterministic
and straightforward to unit test.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from .types import AudioAsset, AudioBinding, CanonicalSample

AUDIO_EXTENSIONS = ("wav", "flac", "mp3", "m4a", "ogg")
AUDIO_TAG = "<|audio|>"
_ROLE_MAP = {"human": "user", "gpt": "assistant", "assistant_content": "assistant"}


class LayoutError(ValueError):
    """Base class for physical-layout validation errors."""


class UnknownLayoutError(LayoutError):
    pass


class AmbiguousLayoutError(LayoutError):
    pass


class MissingMemberError(LayoutError):
    pass


class DuplicateMemberError(LayoutError):
    pass


class MalformedJsonError(LayoutError):
    pass


class MetadataDecodeError(LayoutError):
    """Metadata is valid JSON but does not satisfy the message schema."""


class AudioBindingMismatchError(LayoutError):
    pass


@dataclass(frozen=True)
class MatchResult:
    matched: bool
    details: Mapping[str, Any] = field(default_factory=dict)


class ErrorReporter(Protocol):
    def record_task(self, sample: Mapping[str, Any], task: str, error: Exception) -> None:
        ...


@dataclass
class ListErrorReporter:
    errors: List[Tuple[str, str, Exception]] = field(default_factory=list)

    def record_task(self, sample: Mapping[str, Any], task: str, error: Exception) -> None:
        self.errors.append((str(sample.get("__key__", "")), task, error))


@dataclass
class AdapterContext:
    source_name: str = "webdataset"
    tasks: Sequence[str] = ("asr", "tts")
    task_policy: str = "all"
    task_weights: Mapping[str, float] = field(
        default_factory=lambda: {"asr": 1.0, "tts": 1.0}
    )
    seed: int = 0
    epoch: int = 0
    duplicate_member_policy: str = "error"
    audio_extension_preference: Sequence[str] = AUDIO_EXTENSIONS
    physical_sample_atomic: bool = False
    error_reporter: ErrorReporter = field(default_factory=ListErrorReporter)
    text_keys: Sequence[str] = ("text", "transcript")
    asr_prompt: str = "Transcribe the following audio:"
    tts_prompt_template: str = "Read the following text out loud: {text}"


class PhysicalLayoutAdapter(Protocol):
    name: str

    def match(self, sample: Mapping[str, Any]) -> MatchResult:
        ...

    def expand(
        self, sample: Mapping[str, Any], context: AdapterContext
    ) -> Iterable[CanonicalSample]:
        ...


def _audio_candidates(sample: Mapping[str, Any], prefix: str) -> List[str]:
    pattern = re.compile(
        r"^" + re.escape(prefix) + r"\.(" + "|".join(AUDIO_EXTENSIONS) + r")$",
        re.IGNORECASE,
    )
    return [key for key in sample if pattern.fullmatch(key)]


def _single_audio(sample: Mapping[str, Any], prefix: str, context: AdapterContext) -> str:
    candidates = _audio_candidates(sample, prefix)
    if not candidates:
        raise MissingMemberError(
            f"{_location(sample)}: missing {prefix}.<audio_ext> member"
        )
    if len(candidates) == 1:
        return candidates[0]
    if context.duplicate_member_policy != "prefer":
        raise DuplicateMemberError(
            f"{_location(sample)}: multiple {prefix} audio members: {sorted(candidates)}"
        )
    rank = {ext.lower(): i for i, ext in enumerate(context.audio_extension_preference)}
    return min(candidates, key=lambda key: (rank.get(key.rsplit(".", 1)[-1].lower(), 10**6), key))


def _location(sample: Mapping[str, Any]) -> str:
    return f"{sample.get('__url__', '<unknown-shard>')}::{sample.get('__key__', '<unknown-key>')}"


def _parse_json(value: Any, sample: Mapping[str, Any], member: str) -> Dict[str, Any]:
    try:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        parsed = json.loads(value) if isinstance(value, str) else value
        if not isinstance(parsed, dict):
            raise TypeError("metadata root must be an object")
        return parsed
    except Exception as exc:
        raise MalformedJsonError(f"{_location(sample)}: invalid {member}: {exc}") from exc


def _fallback_messages(metadata: Mapping[str, Any], task: str, context: AdapterContext):
    text = next((metadata.get(key) for key in context.text_keys if metadata.get(key)), None)
    if text is None:
        raise MetadataDecodeError(f"metadata has neither messages nor any of {tuple(context.text_keys)}")
    text = str(text)
    if task == "asr":
        return [
            {"role": "user", "content": f"{context.asr_prompt}{AUDIO_TAG}"},
            {"role": "assistant", "content": text},
        ]
    if task == "tts":
        return [
            {"role": "user", "content": context.tts_prompt_template.format(text=text)},
            {"role": "assistant", "content": f"{text}{AUDIO_TAG}"},
        ]
    raise MetadataDecodeError("s2s metadata must contain messages")


def _normalize_messages(
    metadata: Mapping[str, Any], task: str, context: AdapterContext
) -> List[Dict[str, str]]:
    raw = metadata.get("messages")
    if not raw:
        raw = _fallback_messages(metadata, task, context)
    if not isinstance(raw, list):
        raise MetadataDecodeError("messages must be a list")
    messages: List[Dict[str, str]] = []
    for index, message in enumerate(raw):
        if not isinstance(message, Mapping):
            raise MetadataDecodeError(f"messages[{index}] must be an object")
        role = _ROLE_MAP.get(str(message.get("role", "")), str(message.get("role", "")))
        if role not in ("system", "user", "assistant"):
            raise MetadataDecodeError(f"messages[{index}] has unsupported role {role!r}")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise MetadataDecodeError(f"messages[{index}].content must be a string")
        messages.append({"role": role, "content": content})
    return messages


def _bindings(messages: Sequence[Mapping[str, str]], task: str, asset_ids: Mapping[str, str]):
    expected = {
        "s2s": {"user": 1, "assistant": 1},
        "asr": {"user": 1, "assistant": 0},
        "tts": {"user": 0, "assistant": 1},
    }[task]
    found = {"user": [], "assistant": []}
    for message_index, message in enumerate(messages):
        role = message["role"]
        if role in found:
            for tag_index in range(message["content"].count(AUDIO_TAG)):
                found[role].append((message_index, tag_index))
    counts = {role: len(items) for role, items in found.items()}
    if counts != expected:
        raise AudioBindingMismatchError(
            f"{task} expects audio tags {expected}, found {counts}"
        )
    result = []
    for role in ("user", "assistant"):
        for message_index, tag_index in found[role]:
            result.append(
                AudioBinding(
                    role=role,
                    asset_id=asset_ids[role],
                    message_index=message_index,
                    tag_index_in_message=tag_index,
                )
            )
    return result


def _ids(sample: Mapping[str, Any], context: AdapterContext, task: str):
    key = str(sample.get("__key__", ""))
    url = str(sample.get("__url__", ""))
    physical_uid = f"{context.source_name}:{url}:{key}"
    return key, url, physical_uid, f"{physical_uid}:{task}"


def _asset(asset_id: str, member: str, data: Any, metadata: Mapping[str, Any]) -> AudioAsset:
    if not isinstance(data, bytes):
        raise LayoutError(f"audio member {member} must contain bytes, got {type(data).__name__}")
    duration = None
    durations = metadata.get("audio_durations")
    audios = metadata.get("audios")
    if isinstance(durations, list) and isinstance(audios, list) and len(durations) == len(audios):
        basename = member
        for path, candidate_duration in zip(audios, durations):
            if os.path.basename(str(path)) == basename and candidate_duration is not None:
                duration = float(candidate_duration)
                break
    return AudioAsset(asset_id, member, member.rsplit(".", 1)[-1].lower(), data, duration)


class S2SPairAdapter:
    name = "s2s_pair"

    def match(self, sample: Mapping[str, Any]) -> MatchResult:
        question = _audio_candidates(sample, "question")
        response = _audio_candidates(sample, "response")
        return MatchResult(
            "json" in sample and len(question) == 1 and len(response) == 1,
            {"question": question, "response": response, "has_json": "json" in sample},
        )

    def expand(self, sample: Mapping[str, Any], context: AdapterContext):
        if "json" not in sample:
            raise MissingMemberError(f"{_location(sample)}: missing json member")
        question_key = _single_audio(sample, "question", context)
        response_key = _single_audio(sample, "response", context)
        metadata = _parse_json(sample["json"], sample, "json")
        messages = _normalize_messages(metadata, "s2s", context)
        bindings = _bindings(messages, "s2s", {"user": "question", "assistant": "response"})
        key, url, physical_uid, uid = _ids(sample, context, "s2s")
        if metadata.get("audios"):
            expected = {question_key, response_key}
            supplied = {os.path.basename(str(path)) for path in metadata["audios"]}
            if supplied != expected:
                warnings.warn(
                    f"{_location(sample)}: JSON audios do not match tar members; tar bytes are authoritative",
                    UserWarning,
                )
        yield CanonicalSample(
            uid=uid,
            physical_uid=physical_uid,
            physical_key=key,
            source_name=context.source_name,
            shard_url=url,
            task="s2s",
            messages=messages,
            assets={
                "question": _asset("question", question_key, sample[question_key], metadata),
                "response": _asset("response", response_key, sample[response_key], metadata),
            },
            bindings=bindings,
            metadata=dict(metadata),
        )


class SharedAudioTasksAdapter:
    name = "shared_audio_tasks"

    def match(self, sample: Mapping[str, Any]) -> MatchResult:
        audio = _audio_candidates(sample, "audio")
        task_members = [member for member in ("asr.json", "tts.json") if member in sample]
        return MatchResult(
            len(audio) == 1 and bool(task_members),
            {"audio": audio, "task_members": task_members},
        )

    def expand(self, sample: Mapping[str, Any], context: AdapterContext):
        audio_key = _single_audio(sample, "audio", context)
        requested = [task for task in ("asr", "tts") if task in context.tasks]
        present = [task for task in requested if f"{task}.json" in sample]
        if not present:
            raise MissingMemberError(f"{_location(sample)}: missing requested asr.json/tts.json")
        prepared_by_task: Dict[str, Tuple[Dict[str, Any], List[Dict[str, str]], List[AudioBinding]]] = {}
        errors: List[Exception] = []
        for task in present:
            member = f"{task}.json"
            try:
                metadata = _parse_json(sample[member], sample, member)
                messages = _normalize_messages(metadata, task, context)
                bindings = _bindings(
                    messages, task, {"user": "audio", "assistant": "audio"}
                )
                prepared_by_task[task] = (metadata, messages, bindings)
            except Exception as exc:
                errors.append(exc)
                context.error_reporter.record_task(sample, task, exc)
        if context.physical_sample_atomic and errors:
            raise errors[0]
        selected = self._select_tasks(sample, list(prepared_by_task), context)
        assets_by_audio_metadata = {}
        for task in selected:
            metadata, messages, bindings = prepared_by_task[task]
            # ASR and TTS sidecars carry duration/token metadata independently.
            # Preserve it for early rejection, while sharing the immutable asset
            # when both task sidecars describe the same physical audio.
            candidate = _asset("audio", audio_key, sample[audio_key], metadata)
            asset_key = (candidate.duration, candidate.sample_rate, candidate.num_frames)
            asset = assets_by_audio_metadata.setdefault(asset_key, candidate)
            key, url, physical_uid, uid = _ids(sample, context, task)
            yield CanonicalSample(
                uid=uid,
                physical_uid=physical_uid,
                physical_key=key,
                source_name=context.source_name,
                shard_url=url,
                task=task,
                messages=messages,
                assets={"audio": asset},
                bindings=bindings,
                metadata=dict(metadata),
            )

    @staticmethod
    def _select_tasks(sample, available: List[str], context: AdapterContext) -> List[str]:
        if context.task_policy == "all":
            return available
        if context.task_policy == "subset":
            return [task for task in available if task in context.tasks]
        if context.task_policy != "sample_one":
            raise ValueError(f"unknown task_policy {context.task_policy!r}")
        if not available:
            return []
        weights = [max(0.0, float(context.task_weights.get(task, 1.0))) for task in available]
        total = sum(weights)
        if total <= 0:
            raise ValueError("sample_one requires at least one positive task weight")
        identity = "\0".join(
            (str(context.seed), str(context.epoch), str(sample.get("__url__", "")), str(sample.get("__key__", "")))
        )
        value = int.from_bytes(hashlib.blake2b(identity.encode(), digest_size=8).digest(), "big") / 2**64
        threshold = value * total
        cumulative = 0.0
        for task, weight in zip(available, weights):
            cumulative += weight
            if threshold < cumulative:
                return [task]
        return [available[-1]]


class PhysicalLayoutRegistry:
    def __init__(self, adapters: Optional[Iterable[PhysicalLayoutAdapter]] = None):
        adapters = adapters or (S2SPairAdapter(), SharedAudioTasksAdapter())
        self._adapters = {adapter.name: adapter for adapter in adapters}

    def resolve(self, sample: Mapping[str, Any], expected_layout: str = "auto"):
        if expected_layout != "auto":
            try:
                return self._adapters[expected_layout]
            except KeyError as exc:
                raise UnknownLayoutError(f"unknown configured layout {expected_layout!r}") from exc
        matches = [adapter for adapter in self._adapters.values() if adapter.match(sample).matched]
        if not matches:
            raise UnknownLayoutError(f"{_location(sample)}: no physical layout matched")
        if len(matches) > 1:
            raise AmbiguousLayoutError(
                f"{_location(sample)}: multiple physical layouts matched: {[a.name for a in matches]}"
            )
        return matches[0]

    def expand(self, sample, context: Optional[AdapterContext] = None, expected_layout="auto"):
        context = context or AdapterContext()
        return self.resolve(sample, expected_layout).expand(sample, context)
