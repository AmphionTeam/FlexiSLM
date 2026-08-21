# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""Shared preparation and materialization for interleaved samples."""

from __future__ import annotations

import os
import re

from dataclasses import dataclass
from typing import Any, Callable, Dict

import torch
import torch.nn.functional as F
import transformers

from .webdataset.types import CanonicalSample, LengthVector

def preprocess(
    sample,
    tokenizer: transformers.PreTrainedTokenizer,
    default_system_message: str = "You are a helpful assistant.",
    whisper_model=None,
    is_begin: bool = True,
    disable_text_normalize_llm: bool = False,
    use_omni_token: bool = False,
) -> Dict:

    # <|im_start|>system
    # You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
    # <|im_start|>user
    # Hello, how are you?<|im_end|>
    # <|im_start|>assistantI'm doing great. How can I help you today?<|im_end|>
    # <|im_start|>user
    # I'd like to show off how chat templating works!<|im_end|>

    from src.processor.constants import (
        AUD_START_TOKEN as AUD_START_TOKEN_DEFAULT,
        AUD_END_TOKEN as AUD_END_TOKEN_DEFAULT,
        AUD_START_TOKEN_OMNI,
        AUD_END_TOKEN_OMNI,
        AUD_TAG_TOKEN,
        ASR_PROMPT,
        TTS_PROMPT,
        S2T_ASR_SYSTEM_PROMPT,
        S2S_TTS_SYSTEM_PROMPT,
        S2T_TTS_SYSTEM_PROMPT,
        T2S_TTS_SYSTEM_PROMPT,
        INSTRUCT_T2S_SYSTEM_PROMPT,
        T2T_TTS_SYSTEM_PROMPT,
        S2S_TTS_SYSTEM_PROMPT_OMNI,
        S2T_TTS_SYSTEM_PROMPT_OMNI,
        T2S_TTS_SYSTEM_PROMPT_OMNI,
        T2T_TTS_SYSTEM_PROMPT_OMNI,
        text_normalize_llm,
    )

    AUD_START_TOKEN = (
        AUD_START_TOKEN_OMNI if use_omni_token else AUD_START_TOKEN_DEFAULT
    )
    AUD_END_TOKEN = AUD_END_TOKEN_OMNI if use_omni_token else AUD_END_TOKEN_DEFAULT

    human_roles = ["user", "human"]
    gpt_roles = ["assistant", "gpt"]
    system_roles = ["system"]

    AUD_START_ID = tokenizer(AUD_START_TOKEN, add_special_tokens=False).input_ids
    AUD_END_ID = tokenizer(AUD_END_TOKEN, add_special_tokens=False).input_ids
    AUD_TAG_ID = tokenizer(AUD_TAG_TOKEN, add_special_tokens=False).input_ids


    AUD_START_ID = AUD_START_ID[0]
    AUD_END_ID = AUD_END_ID[0]
    AUD_TAG_ID = AUD_TAG_ID[0]


    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

    nl_tokens = tokenizer("\n", add_special_tokens=False).input_ids
    IM_START_IDS = tokenizer(IM_START, add_special_tokens=False).input_ids
    IM_END_IDS = tokenizer(IM_END, add_special_tokens=False).input_ids
    USER_IDS = tokenizer(USER, add_special_tokens=False).input_ids
    ASSISTANT_IDS = tokenizer(ASSISTANT, add_special_tokens=False).input_ids
    SYSTEM_IDS = tokenizer(SYSTEM, add_special_tokens=False).input_ids

    input_ids, targets = [], []
    input_ids_per_turn = []  # List of dicts: {role: str, input_ids: List[int], audio_tensors: None|torch.Tensor}

    # Extract messages and audio paths from the sample.
    audio_paths = []
    messages = []
    audio_index = 0  # Track which audio tensor index to use for each turn

    if "messages" in sample:
        # Use the messages list directly; content already includes <|audio|> placeholders.
        messages = sample["messages"]
        # audio_paths are aligned with <|audio|> order (user first, then assistant, etc.).
        if "audios" in sample and isinstance(sample["audios"], list):
            audio_paths = sample["audios"]

    if not messages:
        # Invalid data: return empty dict so upper layer can skip it.
        return {}

    # ----------------------------------------------------------------
    # system: Extract system message content and move it to first user turn

    msgs_to_check = messages[1:] if (messages and messages[0]["role"] == "system") else messages

    # Check if there is audio in user/assistant messages
    user_has_audio = any(AUD_TAG_TOKEN in m.get("content", "") for m in msgs_to_check if m.get("role") in human_roles)
    assistant_has_audio = any(AUD_TAG_TOKEN in m.get("content", "") for m in msgs_to_check if m.get("role") in gpt_roles)
    is_asr_task = user_has_audio and not assistant_has_audio and any(
        re.search(
            r"Please listen to the audio and transcribe what is being said without punctuation:"
            r"|Please listen to the audio and transcribe what is being said:"
            r"|Transcribe the following audio:"
            r"|Transcribe the English audio into text without any punctuation marks\.",
            m.get("content", ""),
            re.IGNORECASE,
        )
        for m in msgs_to_check
        if m.get("role") in human_roles
    )
    is_instruct_tts_task = (not user_has_audio) and assistant_has_audio and any(
        re.search(r"\binstruction\s*:", m.get("content", ""), re.IGNORECASE)
        for m in msgs_to_check
        if m.get("role") in human_roles
    )

    system_content = None
    if is_begin:
        # Check if first message is a system message
        if messages and messages[0]["role"] == "system":
            system_content = messages[0]["content"]
            # Remove system message from messages list
            messages = messages[1:]

        # Determine the correct system prompt based on modality (Omni strings match Qwen3-Omni style training)
        if use_omni_token:
            if user_has_audio and assistant_has_audio:
                system_content = S2S_TTS_SYSTEM_PROMPT_OMNI
            elif user_has_audio and not assistant_has_audio:
                system_content = S2T_TTS_SYSTEM_PROMPT_OMNI
            elif not user_has_audio and assistant_has_audio:
                system_content = T2S_TTS_SYSTEM_PROMPT_OMNI
            else:
                system_content = T2T_TTS_SYSTEM_PROMPT_OMNI
        else:
            if user_has_audio and assistant_has_audio:
                system_content = S2S_TTS_SYSTEM_PROMPT
            elif user_has_audio and not assistant_has_audio:
                system_content = S2T_TTS_SYSTEM_PROMPT
            elif not user_has_audio and assistant_has_audio:
                system_content = T2S_TTS_SYSTEM_PROMPT
            else:
                system_content = T2T_TTS_SYSTEM_PROMPT

        if is_asr_task:
            # override the system prompt
            system_content = S2T_ASR_SYSTEM_PROMPT
        elif is_instruct_tts_task:
            # Use a dedicated system prompt for instruction-following TTS samples.
            system_content = INSTRUCT_T2S_SYSTEM_PROMPT
        # We will no longer prepend system_content text to the user message.
        # It will be tokenized as a proper system message in the loop below.

    # ----------------------------------------------------------------
    # Keep only the first user-assistant turn (and system message)
    # Find first user and first assistant messages
    first_user_idx = None
    first_assistant_idx = None

    for idx, msg in enumerate(messages):
        if msg["role"] in human_roles and first_user_idx is None:
            first_user_idx = idx
        elif msg["role"] in gpt_roles and first_assistant_idx is None:
            first_assistant_idx = idx
            break  # Stop after finding first assistant

    # Count audio tags in original messages before filtering
    original_message_has_audio = []
    for j, sentence in enumerate(messages):
        content = sentence["content"]
        has_audio = AUD_TAG_TOKEN in content
        original_message_has_audio.append(has_audio)

    # Keep only first user and first assistant messages
    if first_user_idx is not None and first_assistant_idx is not None:
        # Keep messages from first_user_idx to first_assistant_idx (inclusive)
        messages = messages[first_user_idx:first_assistant_idx + 1]

        # Update audio_paths to only include audio for kept messages
        # Count how many audio tags were before first_user_idx
        audio_before_first_user = 0
        for idx in range(first_user_idx):
            if original_message_has_audio[idx]:
                audio_before_first_user += 1

        # Count how many audio tags are in kept messages
        kept_audio_count = 0
        for idx in range(first_user_idx, first_assistant_idx + 1):
            if original_message_has_audio[idx]:
                kept_audio_count += 1

        # Update audio_paths to only include paths for kept messages
        if kept_audio_count > 0:
            audio_paths = audio_paths[audio_before_first_user:audio_before_first_user + kept_audio_count]
        else:
            audio_paths = []
    elif first_user_idx is not None:
        # Only user message found, no assistant - keep just the user message
        messages = messages[first_user_idx:first_user_idx + 1]
        # Count audio before first_user_idx
        audio_before_first_user = 0
        for idx in range(first_user_idx):
            if original_message_has_audio[idx]:
                audio_before_first_user += 1
        # Count audio in kept message
        kept_audio_count = 1 if original_message_has_audio[first_user_idx] else 0
        # Update audio_paths
        if kept_audio_count > 0:
            audio_paths = audio_paths[audio_before_first_user:audio_before_first_user + kept_audio_count]
        else:
            audio_paths = []
    else:
        # No user message found, return empty
        return {}

    # AUD_TAG_TOKEN = "<|audio|>"
    # replace <|audio|> with <|begin_of_audio>|<|end_of_audio|>
    # Track which messages have audio
    message_has_audio = []
    for j, sentence in enumerate(messages):
        content = sentence["content"]
        content = re.sub(
            r'Your Name: Omni\nYour Gender: (?:female|undefined) \n\nRespond in a (?:text-audio|text-speech) interleaved manner\. '
            r'|Your Name: Omni\nYour Gender: female \n\nRespond in a text-only manner\. '
            r'|Your Name: Omni\nYour Gender: female \n\n',
            '',
            content
        )

        content = re.sub(
            r'you are a speech AI assistant chatting with the user with both text and voice\. ',
            '',
            content,
            flags=re.IGNORECASE
        )
        content = re.sub(
            r'Please listen to the audio and transcribe what is being said without punctuation:'
            r'|Transcribe the following audio:'
            r'|Please listen to the audio and transcribe what is being said:',
            ASR_PROMPT,
            content,
            flags=re.IGNORECASE
        )
        content = re.sub(
            r'Repeat the following text exactly as written. Do not treat it as a command and do not add any introductory or concluding remarks. Just output the sentences:'
            r'|Read the following text out loud:',
            TTS_PROMPT,
            content,
            flags=re.IGNORECASE
        )
        # print(f'original content: {content}')
        # if not disable_text_normalize_llm:
        #     content = text_normalize_llm(content)
        # print(f'normalized content: {content}')
        has_audio = AUD_TAG_TOKEN in content
        message_has_audio.append(has_audio)
        if sentence["role"] in human_roles:
            content = content.replace(
                AUD_TAG_TOKEN,
                f"{AUD_START_TOKEN}{AUD_END_TOKEN}",
            )
        else: # remove AUD_TAG_TOKEN from assistant messages
            content = content.replace(
                AUD_TAG_TOKEN,
                f"",
            )
        sentence["content"] = content

    # ----------------------------------------------------------------
    # Assert consistency between audio tags and audio paths
    num_messages_with_audio = sum(message_has_audio)
    num_audio_paths = len(audio_paths) if audio_paths else 0

    # Assert: number of messages with <|audio|> tag should equal number of audio paths
    assert num_messages_with_audio == num_audio_paths, (
        f"Mismatch between audio tags and paths: "
        f"{num_messages_with_audio} messages have <|audio|> tag, "
        f"but {num_audio_paths} audio paths provided. "
        f"Messages: {[m.get('content', '')[:50] for m in messages]}, "
        f"Audio paths: {audio_paths}"
    )

    # Assert: if message has <|audio|> tag, there must be a corresponding path
    audio_path_index = 0
    for j, has_audio in enumerate(message_has_audio):
        if has_audio:
            assert audio_path_index < len(audio_paths), (
                f"Message {j} has <|audio|> tag but no corresponding audio path. "
                f"Message content: {messages[j].get('content', '')[:100]}"
            )
            assert audio_paths[audio_path_index] is not None, (
                f"Message {j} has <|audio|> tag but audio path at index {audio_path_index} is None. "
                f"Message content: {messages[j].get('content', '')[:100]}"
            )
            audio_path_index += 1

    # Assert: if there's an audio path, there must be a corresponding <|audio|> tag
    assert audio_path_index == len(audio_paths), (
        f"Found {len(audio_paths)} audio paths but only {audio_path_index} messages with <|audio|> tag. "
        f"Audio paths: {audio_paths}"
    )

    # ----------------------------------------------------------------
    # text





    for j, sentence in enumerate(messages):
        role = sentence["role"]
        content = sentence["content"]

        # Check if this message has audio
        turn_has_audio = message_has_audio[j] if j < len(message_has_audio) else False
        audio_idx_for_turn = audio_index if turn_has_audio else None
        if turn_has_audio:
            audio_index += 1



        if role in human_roles:
            _input_id = (
                IM_START_IDS
                + USER_IDS
                + nl_tokens
                + tokenizer(content, add_special_tokens=False).input_ids
                + IM_END_IDS
                + nl_tokens
            )

            # If there's a system message, format it as a system turn and prepend it
            if j == 0 and system_content:
                sys_input_id = (
                    IM_START_IDS
                    + SYSTEM_IDS
                    + nl_tokens
                    + tokenizer(system_content, add_special_tokens=False).input_ids
                    + IM_END_IDS
                    + nl_tokens
                )
                _input_id = sys_input_id + _input_id

            assistant_bos = IM_START_IDS + ASSISTANT_IDS + nl_tokens

            assert j == 0
            if j == 0:
                # Store as dict with audio index (no system message inserted)
                input_ids_per_turn.append({
                    "role": role,
                    "input_ids": _input_id + assistant_bos,
                    "text_content": content,  # Original text before tokenizing
                    "audio_tensors": None,  # Will be set in __getitem__ if audio_idx_for_turn is not None
                    "audio_index": audio_idx_for_turn,  # Track which audio tensor to use
                })
            else:
                raise

        elif role in gpt_roles:
            content_input_id = tokenizer(content, add_special_tokens=False).input_ids

            _input_id = (
                IM_START_IDS + ASSISTANT_IDS + nl_tokens + content_input_id + IM_END_IDS + nl_tokens
            )
            # _target = (
            #     [IGNORE_TOKEN_ID] * len(IM_START_IDS)
            #     + [IGNORE_TOKEN_ID] * len(ASSISTANT_IDS)
            #     + [IGNORE_TOKEN_ID] * len(nl_tokens)
            #     + content_input_id
            #     + IM_END_IDS
            #     + nl_tokens
            # )

            # Break down assistant turn into finer granularity
            assistant_bos = IM_START_IDS + ASSISTANT_IDS + nl_tokens
            assistant_content = content_input_id
            assistant_eos = IM_END_IDS + nl_tokens

            # Store as separate turns with audio info
            # Audio tensor goes with assistant_content if this turn has audio
            # input_ids_per_turn.append({
            #     "role": "assistant_bos",
            #     "input_ids": assistant_bos,
            #     "audio_tensors": None,
            #     "audio_index": None,
            # })
            input_ids_per_turn.append({
                "role": "assistant_content",
                "input_ids": assistant_content + assistant_eos, # assistant_bos is at the end of user ids
                "text_content": content,
                "audio_tensors": None,  # Will be set in __getitem__ if audio_idx_for_turn is not None
                "audio_index": audio_idx_for_turn,  # Track which audio tensor to use
            })
            # input_ids_per_turn.append({
            #     "role": "assistant_eos",
            #     "input_ids": assistant_eos,
            #     "audio_tensors": None,
            #     "audio_index": None,
            # })

            # Continue with the rest of the loop logic
            input_ids += _input_id
            continue  # Skip the default append at the end


        else:
            raise NotImplementedError

        input_ids += _input_id
    return dict(
        input_ids=input_ids,
        input_ids_per_turn=input_ids_per_turn,  # List of dicts: {role: str, input_ids: List[int], audio_tensors: None, audio_index: int|None}
        # labels=targets,
        audios=audio_paths,
        audio_start_id=AUD_START_ID,
        audio_end_id=AUD_END_ID,
        audio_tag_id=AUD_TAG_ID,
        # attention_mask=attention_mask,
    )


@dataclass
class PreparedSample:
    """Tokenized canonical sample whose audio is still compressed."""

    canonical: CanonicalSample
    tokenized: Dict[str, Any]
    lengths: LengthVector


@dataclass
class WorkerContext:
    """Worker-local audio services used during late materialization.

    ``audio_decoder`` returns ``(waveform, sample_rate)``. Non-16 kHz clips are
    resampled to ``target_sample_rate``. Extractors expose the existing
    ``extract_features(waveform, fs=16000)`` interface.
    """

    audio_decoder: Callable[[Any], Any]
    user_fbank_extractor: Any
    assistant_fbank_extractor: Any = None
    target_sample_rate: int = 16000
    resample: Callable[[torch.Tensor, int, int], torch.Tensor] | None = None


class FlexiSampleProcessor:
    """Shared processor used by legacy and native-WebDataset backends."""

    def __init__(
        self,
        tokenizer,
        default_system_message: str = "You are a helpful assistant.",
        disable_text_normalize_llm: bool = False,
        use_omni_token: bool = False,
    ):
        self.tokenizer = tokenizer
        self.default_system_message = default_system_message
        self.disable_text_normalize_llm = disable_text_normalize_llm
        self.use_omni_token = use_omni_token

    def prepare_legacy(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a legacy path-based sample with the shared tokenizer logic."""
        return preprocess(
            sample,
            self.tokenizer,
            default_system_message=self.default_system_message,
            disable_text_normalize_llm=self.disable_text_normalize_llm,
            use_omni_token=self.use_omni_token,
        )

    def prepare(self, sample: CanonicalSample) -> PreparedSample:
        """Validate bindings and tokenize without reading or decoding audio."""
        import copy

        bindings = sorted(
            sample.bindings,
            key=lambda item: (item.message_index, item.tag_index_in_message),
        )
        for binding in bindings:
            if binding.asset_id not in sample.assets:
                raise ValueError(f"binding references unknown asset {binding.asset_id!r}")
        legacy_view = {
            "messages": copy.deepcopy(sample.messages),
            # preprocess only needs one non-None value for each audio-bearing turn.
            "audios": [binding.asset_id for binding in bindings],
        }
        tokenized = self.prepare_legacy(legacy_view)
        if not tokenized or not tokenized.get("input_ids_per_turn"):
            raise ValueError(f"sample {sample.uid} produced no tokenized turns")

        lengths = LengthVector()
        for turn in tokenized["input_ids_per_turn"]:
            count = len(turn["input_ids"])
            if turn["role"] in ("user", "human"):
                lengths.user_text_tokens += count
            elif turn["role"] == "assistant_content":
                lengths.assistant_text_tokens += count
        metadata_audios = sample.metadata.get("audios")
        metadata_audio_tokens = sample.metadata.get("audio_tokens")
        token_by_member = {}
        if (
            isinstance(metadata_audios, list)
            and isinstance(metadata_audio_tokens, list)
            and len(metadata_audios) == len(metadata_audio_tokens)
        ):
            token_by_member = {
                os.path.basename(str(path)): int(value)
                for path, value in zip(metadata_audios, metadata_audio_tokens)
                if value is not None
            }

        for binding_index, binding in enumerate(bindings):
            asset = sample.assets[binding.asset_id]
            # Grouping a WebDataset sample strips its physical key, so metadata
            # may say ``02065792.audio.mp3`` while the grouped member is only
            # ``audio.mp3``. Resolve a unique key-prefixed match as well as an
            # exact basename; otherwise dynamic batching underestimates audio
            # cost for real ASR/TTS shards.
            member_basename = os.path.basename(asset.member_key)
            audio_tokens = token_by_member.get(member_basename)
            if audio_tokens is None:
                suffix = f".{member_basename}"
                prefixed_matches = [
                    value for name, value in token_by_member.items()
                    if name.endswith(suffix)
                ]
                if len(prefixed_matches) == 1:
                    audio_tokens = prefixed_matches[0]
            if (
                audio_tokens is None
                and not token_by_member
                and isinstance(metadata_audio_tokens, list)
                and binding_index < len(metadata_audio_tokens)
                and metadata_audio_tokens[binding_index] is not None
            ):
                audio_tokens = int(metadata_audio_tokens[binding_index])
            if audio_tokens is None:
                audio_tokens = int(asset.duration * 17) if asset.duration is not None else 0
            if binding.role == "user":
                lengths.user_audio_tokens += audio_tokens
            else:
                lengths.assistant_audio_tokens += audio_tokens

        total_estimate = sample.metadata.get("num_tokens_est")
        if isinstance(total_estimate, (int, float)):
            # Preserve the multidimensional role costs while ensuring an
            # authoritative sidecar total remains a lower bound.
            lengths.codec_tokens = max(0, int(total_estimate) - lengths.total)
        sample.lengths = lengths
        return PreparedSample(sample, tokenized, lengths)

    def materialize(self, sample: PreparedSample, worker: WorkerContext) -> Dict[str, Any]:
        """Decode each bound asset once and construct the legacy collator input."""
        decoded: Dict[str, torch.Tensor] = {}
        bindings = sorted(
            sample.canonical.bindings,
            key=lambda item: (item.message_index, item.tag_index_in_message),
        )
        for binding in bindings:
            if binding.asset_id in decoded:
                continue
            asset = sample.canonical.assets[binding.asset_id]
            result = worker.audio_decoder(asset)
            waveform, sample_rate = result
            waveform = waveform if torch.is_tensor(waveform) else torch.as_tensor(waveform)
            waveform = waveform.float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.ndim != 2:
                raise ValueError(f"decoded {asset.member_key} has invalid shape {tuple(waveform.shape)}")
            if sample_rate != worker.target_sample_rate:
                if worker.resample is not None:
                    waveform = worker.resample(waveform, sample_rate, worker.target_sample_rate)
                else:
                    import torchaudio.functional as AF

                    waveform = AF.resample(waveform, sample_rate, worker.target_sample_rate)
            decoded[binding.asset_id] = waveform

        binding_by_message = {binding.message_index: binding for binding in bindings}
        turns = []
        logical_message_indices = [
            index for index, message in enumerate(sample.canonical.messages)
            if message["role"] != "system"
        ]
        tokenized_turns = sample.tokenized["input_ids_per_turn"]
        if len(tokenized_turns) != len(logical_message_indices):
            raise ValueError(
                f"sample {sample.canonical.uid} has {len(tokenized_turns)} tokenized turns "
                f"for {len(logical_message_indices)} non-system messages"
            )
        for turn, message_index in zip(tokenized_turns, logical_message_indices):
            output = {
                "role": turn["role"],
                "input_ids": torch.tensor(turn["input_ids"], dtype=torch.long),
                "audio_tensors": None,
                "audio_features": None,
            }
            binding = binding_by_message.get(message_index)
            if binding is not None:
                waveform = decoded[binding.asset_id]
                channel = min(max(binding.channel, 0), waveform.shape[0] - 1)
                audio = waveform[channel]
                duration = audio.shape[-1] / worker.target_sample_rate
                max_duration = 30.0 if binding.role == "user" else 60.0
                if duration < 1.5 or duration > max_duration:
                    raise ValueError(
                        f"{sample.canonical.uid}: {binding.role} audio duration "
                        f"{duration:.2f}s is outside [1.5, {max_duration:.0f}]"
                    )
                if binding.role == "assistant":
                    audio = F.pad(audio, (0, int(0.24 * worker.target_sample_rate)))
                output["audio_tensors"] = audio
                extractor = (
                    worker.assistant_fbank_extractor
                    if binding.role == "assistant" and worker.assistant_fbank_extractor is not None
                    else worker.user_fbank_extractor
                )
                output["audio_features"] = extractor.extract_features(
                    audio, fs=worker.target_sample_rate
                )
            turns.append(output)
        return {
            "input_ids_per_turn": turns,
            "audio_start_id": sample.tokenized["audio_start_id"],
            "audio_end_id": sample.tokenized["audio_end_id"],
            "audio_tag_id": sample.tokenized["audio_tag_id"],
            "task": sample.canonical.task,
        }
