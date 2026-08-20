"""Canonical logical sample types shared by WebDataset adapters and processors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

TaskType = Literal["s2s", "asr", "tts"]
AudioRole = Literal["user", "assistant"]


@dataclass(frozen=True)
class AudioAsset:
    asset_id: str
    member_key: str
    codec: str
    data: bytes
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    num_frames: Optional[int] = None


@dataclass(frozen=True)
class AudioBinding:
    role: AudioRole
    asset_id: str
    message_index: int
    tag_index_in_message: int = 0
    channel: int = 0


@dataclass
class LengthVector:
    user_text_tokens: int = 0
    assistant_text_tokens: int = 0
    user_audio_tokens: int = 0
    assistant_audio_tokens: int = 0
    codec_tokens: int = 0

    @property
    def total(self) -> int:
        return (
            self.user_text_tokens
            + self.assistant_text_tokens
            + self.user_audio_tokens
            + self.assistant_audio_tokens
            + self.codec_tokens
        )


@dataclass
class CanonicalSample:
    uid: str
    physical_uid: str
    physical_key: str
    source_name: str
    shard_url: str
    task: TaskType
    messages: List[Dict[str, str]]
    assets: Dict[str, AudioAsset]
    bindings: List[AudioBinding]
    lengths: Optional[LengthVector] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
