"""Helpers for aligning delayed audio metadata during autoregressive generation."""

from __future__ import annotations

from typing import Any


class DelayedAudioLengthBuffer:
    """Pair each audio token with the length emitted on the following step.

    FlexiSLM training shifts the length stream one position farther than the
    audio stream.  Consequently, the length sampled alongside audio token
    ``n + 1`` describes audio token ``n``.  The final audio token is flushed by
    the length sampled alongside the audio end marker.
    """

    def __init__(self) -> None:
        self.audio_ids: list[Any] = []
        self.length_ids: list[Any] = []
        self.pending_audio_id: Any | None = None
        self.dropped_pairs = 0

    def push(
        self,
        audio_id: Any,
        length_id: Any,
        *,
        is_audio_code: bool,
        is_valid_length: bool = True,
    ) -> None:
        """Consume one synchronous model step and emit the previous pair."""
        if self.pending_audio_id is not None:
            if is_valid_length:
                self.audio_ids.append(self.pending_audio_id)
                self.length_ids.append(length_id)
            else:
                self.dropped_pairs += 1

        self.pending_audio_id = audio_id if is_audio_code else None

    def discard_pending(self) -> bool:
        """Discard an unflushable token after generation stops without EOS."""
        had_pending = self.pending_audio_id is not None
        self.pending_audio_id = None
        return had_pending
