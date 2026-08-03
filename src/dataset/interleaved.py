# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
import io
import json
import logging
import math
import os
import pdb
import random
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from collections import defaultdict
from typing import Dict, List, Optional, Sequence
import re
import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from src.processor.constants import (
    DEFAULT_TTS_SYSTEM_PROMPT,
    T2T_TTS_SYSTEM_PROMPT_OMNI,
)
import torch.nn.functional as F

from .base import BaseDataset

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

from src.third_party.flexicodec.flexicodec.feature_extractors import FBankGen

from transformers.models.whisper import WhisperFeatureExtractor


def _parse_wds_uri(uri: str):
    if not isinstance(uri, str) or not uri.startswith("wds://"):
        return None
    payload = uri[len("wds://") :]
    channel = None
    if "#ch=" in payload:
        payload, ch_text = payload.rsplit("#ch=", 1)
        try:
            channel = int(ch_text)
        except Exception:
            channel = None
    if "::" not in payload:
        return None
    tar_path, member = payload.split("::", 1)
    if not tar_path or not member:
        return None
    return tar_path, member, channel


class Qwen3FbankExtractor:
    """
    Wrapper around HF WhisperFeatureExtractor (Qwen ASR config) with extract_features(speech, fs) -> (T, 128).
    Use this so dataset code can call extract_features() and get (T, feature_size) tensor matching Qwen ASR.
    """
    def __init__(self):
        # Match Qwen ASR processor's WhisperFeatureExtractor config (feature_size=128, hop_length=160, n_fft=400, 16kHz)
        self._extractor = WhisperFeatureExtractor(
            feature_size=128,
            sampling_rate=16000,
            hop_length=160,
            n_fft=400,
            chunk_length=30,
            padding_value=0.0,
            dither=0.0,
            return_attention_mask=True,
        )
        self.sr = 16000

    def extract_features(self, speech, fs=None):
        """Input: 1D [samples] or 2D [channels, samples]. Output: (T, 128) float32."""
        if torch.is_tensor(speech):
            speech = speech.cpu().numpy()
        else:
            speech = np.asarray(speech, dtype=np.float32)
        if speech.ndim == 2:
            speech = speech.squeeze()
        out = self._extractor(
            speech,
            sampling_rate=self.sr,
            return_tensors="pt",
            padding="do_not_pad",
            truncation=False,
        )
        # input_features: (1, 128, T) -> squeeze to (128, T) -> T to (T, 128)
        feats = out["input_features"]
        if feats.dim() == 3:
            feats = feats.squeeze(0)
        if feats.shape[0] == 128:
            feats = feats.T
        return feats.float()

class Qwen25OFbankExtractor:
    """
    Wrapper around HF WhisperFeatureExtractor (Qwen2.5-omni config) with extract_features(speech, fs) -> (T, 128).
    Use this so dataset code can call extract_features() and get (T, feature_size) tensor matching Qwen2.5-omni.
    """
    def __init__(self):
        # Match Qwen2.5-omni processor's WhisperFeatureExtractor config (feature_size=128, hop_length=160, n_fft=400, 16kHz)
        self._extractor = WhisperFeatureExtractor(
            feature_size=128,
            sampling_rate=16000,
            hop_length=160,
            n_fft=400,
            chunk_length=300, # changed from 30 to 300 to align qwen2.5-omni
            padding_value=0.0,
            dither=0.0,
            return_attention_mask=True,
        )    
        self.sr = 16000

    def extract_features(self, speech, fs=None):
        """Input: 1D [samples] or 2D [channels, samples]. Output: (T, 128) float32."""
        if torch.is_tensor(speech):
            speech = speech.cpu().numpy()
        else:
            speech = np.asarray(speech, dtype=np.float32)
        if speech.ndim == 2:
            speech = speech.squeeze()
        out = self._extractor(
            speech, 
            sampling_rate=self.sr, 
            return_tensors="pt", 
            padding="max_length", 
            return_attention_mask=True
        )
        feats = out["input_features"]
        feature_attention_mask = out["attention_mask"]

        # remove padding, ref: https://github.com/huggingface/transformers/blob/8426e7e63d49d9c3b5f0c09d43e792a59c75c62c/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L1769
        feats = feats.permute(0, 2, 1)[feature_attention_mask.bool()] 

        return feats.float()


def contains_code(text: str) -> bool:
    """
    Detect if text contains code snippets.
    
    Args:
        text: The text content to check
        
    Returns:
        True if code is detected, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
    
    # Common code patterns
    code_patterns = [
        # C/C++ patterns
        r'#include\s*<[^>]+>',  # #include <iostream>
        r'#include\s*"[^"]+"',   # #include "header.h"
        r'using\s+namespace\s+\w+;',  # using namespace std;
        r'int\s+main\s*\(',      # int main(
        r'cout\s*<<',            # cout <<
        r'cin\s*>>',             # cin >>
        r'std::',                # std::cout
        
        # Java patterns
        r'public\s+class\s+\w+',  # public class MyClass
        r'public\s+static\s+void\s+main',  # public static void main
        r'System\.out\.print',    # System.out.print
        r'@Override',             # @Override annotation
        r'package\s+\w+',         # package com.example
        
        # Python patterns (but less strict, as Python can be conversational)
        r'def\s+\w+\s*\([^)]*\)\s*:',  # def function():
        r'import\s+\w+',         # import os
        r'from\s+\w+\s+import',  # from os import
        r'class\s+\w+\s*\(',     # class MyClass(
        r'if\s+__name__\s*==\s*["\']__main__["\']',  # if __name__ == "__main__"
        
        # General code patterns
        r'function\s+\w+\s*\(',  # function name(
        r'const\s+\w+\s*=',      # const x =
        r'let\s+\w+\s*=',        # let x =
        r'var\s+\w+\s*=',        # var x =
        r'return\s+[^;]+;',      # return value;
        
        # Code block indicators
        r'\{[^}]*\}[^}]*\{',     # Multiple curly braces (likely code)
        r';\s*\n\s*\w+\s*\(',    # Semicolon followed by function call
        r'}\s*;',                # Closing brace with semicolon
    ]
    
    # Check for code patterns
    for pattern in code_patterns:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return True
    
    # Check for high density of code-like characters
    # Code typically has more semicolons, curly braces, and parentheses
    code_chars = text.count(';') + text.count('{') + text.count('}') + text.count('()')
    text_length = len(text)
    
    # If text is short and has many code characters, likely code
    if text_length > 0:
        code_char_ratio = code_chars / text_length
        # Threshold: if more than 2% of characters are code-like, likely code
        if code_char_ratio > 0.02 and code_chars >= 3:
            # Additional check: must have at least one of these patterns
            if ('{' in text and '}' in text) or (';' in text and '(' in text):
                return True
    
    # Check for code-like structure: multiple lines with indentation and code patterns
    lines = text.split('\n')
    code_like_lines = 0
    for line in lines:
        stripped = line.strip()
        # Skip empty lines
        if not stripped:
            continue
        # Check if line looks like code (has common code keywords or patterns)
        if (re.search(r'\b(int|void|return|class|def|function|const|let|var)\b', stripped, re.IGNORECASE) or
            (stripped.endswith(';') and '{' in stripped) or
            (stripped.endswith('}') and ';' in stripped)):
            code_like_lines += 1
    
    # If more than 3 lines look like code, likely a code block
    if code_like_lines >= 3:
        return True
    
    return False


def is_primarily_chinese(text: str, threshold: float = 0.15) -> bool:
    """
    Check if text is primarily in Chinese (CJK characters).

    Args:
        text: The text content to check
        threshold: Ratio of CJK chars to total non-space chars above which to consider Chinese

    Returns:
        True if text is primarily Chinese, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
    # Remove <|audio|> and similar tokens for the check
    text = re.sub(r"<\|[^|]+\|>", "", text).strip()
    if not text:
        return False
    cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff" or "\u3000" <= c <= "\u303f")
    total = sum(1 for c in text if not c.isspace())
    if total == 0:
        return False
    return (cjk_count / total) >= threshold

def is_tts_sample(sample: Dict) -> bool:
    """
    Detect TTS-style samples so question marks in text-to-speech targets are kept.

    TTS data in this pipeline usually appears as either:
    - a user text prompt asking to read/repeat text, or
    - text input with assistant audio output (T2S).
    """
    if not isinstance(sample, dict):
        return False

    messages = sample.get("messages", [])
    if not messages:
        return False


    human_roles = {"user", "human"}
    gpt_roles = {"assistant", "gpt"}
    audio_tag = "<|audio|>"

    user_contents = [
        message.get("content", "")
        for message in messages
        if message.get("role", "") in human_roles
    ]
    assistant_contents = [
        message.get("content", "")
        for message in messages
        if message.get("role", "") in gpt_roles
    ]

    user_has_audio = any(audio_tag in content for content in user_contents)
    assistant_has_audio = any(audio_tag in content for content in assistant_contents)
    has_tts_prompt = any(
        re.search(
            r"Repeat the following text exactly as written"
            r"|Read the following text out loud"
            r"|Read the following text",
            content,
            re.IGNORECASE,
        )
        for content in user_contents
    )

    return has_tts_prompt or (assistant_has_audio and not user_has_audio)


def should_filter_sample(sample: Dict) -> bool:
    """
    Check if a sample should be filtered out (e.g., contains code).
    
    Args:
        sample: Dictionary containing 'messages' and optionally 'audios'
        
    Returns:
        True if sample should be filtered out, False otherwise
    """
    if not isinstance(sample, dict):
        return False

    # Duplex policy filter (keep in sync with data generation):
    # - normal_conversation: turn_count must be 1
    # - user_interruption: turn_count must be 2 or 3
    # - backchanneling: turn_count must be 2
    # - assistant_waiting: always filtered out
    case_dir = str(sample.get("duplex_case_dir", "")).strip().lower()
    strategy = str(sample.get("duplex_strategy", "")).strip().lower()
    turn_count_raw = sample.get("duplex_turn_count", None)
    try:
        turn_count = int(turn_count_raw) if turn_count_raw is not None else None
    except Exception:
        turn_count = None

    if case_dir == "assistant_waiting" or strategy == "assistant_waiting":
        return True
    if case_dir == "normal_conversation":
        if turn_count is not None and turn_count != 1:
            return True
    if case_dir == "user_interruption":
        if turn_count is not None and turn_count not in (2, 3):
            return True
    if case_dir == "backchanneling":
        if turn_count is not None and turn_count != 2:
            return True
    
    messages = sample.get("messages", [])
    if not messages:
        return False

    # tts_sample = is_tts_sample(sample)
    
    # Check all assistant messages for code, Chinese, and unwanted question responses
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        # Only check assistant messages
        if role in ["assistant", "gpt"]:
            # if "?" in content and not tts_sample:
            #     return True
            if contains_code(content):
                return True
            if is_primarily_chinese(content):
                return True

    return False

class Qwen2Dataset(BaseDataset):
    """
    Dataset for Qwen2 training with S2S (Speech-to-Speech) support.
    
    Features:
    - Supports target audio for FlexiCodec encoding
    """
    def __init__(
        self,
        *args,
        target_audio_dir: str = None,
        use_qwen3_feature: bool = None,
        use_whisper_fetaure: bool = None,
        use_qwen25o_feature: bool = None,
        use_omni_token: bool = None,
        disable_text_normalize_llm: bool = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        # Prefer constructor flag; else keep BaseDataset default from YAML.
        if disable_text_normalize_llm is not None:
            self.disable_text_normalize_llm = bool(disable_text_normalize_llm)

        self.length = len(self.raw_data)
        
        # S2S configuration
        self.target_audio_dir = target_audio_dir

        # Root directory for audio files. Prefer explicit config value if present;
        # fall back to the constructor argument.
        # You can set this in your YAML as:
        #   audio_root: path/to/audio_root
        self.audio_root = self.cfg.get("audio_root", target_audio_dir)
        
        # Feature extractor: SenseVoice (80 mel) or Qwen3 Whisper-style (128 mel)
        # Prefer use_qwen3_feature from constructor (training script), then dataset YAML (use_qwen3_feature or num_mel_bins: 128).
        use_qwen3 = (
            use_qwen3_feature
            if use_qwen3_feature is not None
            else self.cfg.get("use_qwen3_feature", False)
        )
        use_whisper = (
            use_whisper_fetaure
            if use_whisper_fetaure is not None
            else self.cfg.get("use_whisper_fetaure", False)
        )
        use_qwen25o = (
            use_qwen25o_feature
            if use_qwen25o_feature is not None
            else self.cfg.get("use_qwen25o_feature", False)
        )
        self._use_qwen3_feature = bool(use_qwen3)
        self._use_whisper_fetaure = bool(use_whisper)
        self._use_qwen25o_feature = bool(use_qwen25o)
        self._use_128_mel_feature = self._use_qwen3_feature or self._use_whisper_fetaure or self._use_qwen25o_feature
        use_omni = (
            use_omni_token
            if use_omni_token is not None
            else self.cfg.get("use_omni_token", False)
        )
        self._use_omni_token = bool(use_omni)
        self.default_system_message = (
            T2T_TTS_SYSTEM_PROMPT_OMNI
            if self._use_omni_token
            else DEFAULT_TTS_SYSTEM_PROMPT
        )
        num_mel_bins = self.cfg.get("num_mel_bins", 128 if self._use_128_mel_feature else 80)
        if self._use_128_mel_feature or num_mel_bins == 128:
            self.fbank_extractor = Qwen3FbankExtractor()
            # SenseVoice (80 mel) is still required for the codec path (assistant/target audio).
            self.sensevoice_fbank_extractor = FBankGen(sr=16000)
            logging.getLogger(__name__).info(
                "[Qwen2Dataset] Using 128-bin Whisper-style features for user audio and FBankGen (SenseVoice, 80 mel) for assistant/codec."
            )
        elif use_qwen25o:
            self.fbank_extractor = Qwen25OFbankExtractor()
            # SenseVoice (80 mel) is still required for the codec path (assistant/target audio).
            self.sensevoice_fbank_extractor = FBankGen(sr=16000)
            logging.getLogger(__name__).info(
                "[Qwen2Dataset] Using Qwen25OFbankExtractor (128 mel) for user audio and FBankGen (SenseVoice, 80 mel) for assistant/codec."
            )
        else:
            self.fbank_extractor = FBankGen(sr=16000)
            self.sensevoice_fbank_extractor = None
            logging.getLogger(__name__).info(
                "[Qwen2Dataset] Using FBankGen (SenseVoice, 80 mel) for user audio features."
            )
        
        # Counter for filtered samples (e.g., code-containing samples)
        self.filtered_samples = 0
        self._wds_tar_cache = {}

    def _load_audio_from_wds_uri(self, uri: str):
        parsed = _parse_wds_uri(uri)
        if parsed is None:
            raise ValueError(f"Invalid wds uri: {uri}")
        tar_path, member, channel = parsed
        tar_obj = self._wds_tar_cache.get(tar_path)
        if tar_obj is None:
            tar_obj = tarfile.open(tar_path, mode="r")
            self._wds_tar_cache[tar_path] = tar_obj
        fp = tar_obj.extractfile(member)
        if fp is None:
            raise FileNotFoundError(f"Member {member} not found in tar {tar_path}")
        wav, sr = torchaudio.load(io.BytesIO(fp.read()))
        if channel is not None and wav.dim() == 2 and wav.shape[0] > 1:
            ch = max(0, min(int(channel), int(wav.shape[0]) - 1))
            wav = wav[ch : ch + 1, :]
        return wav, sr

    def _load_audio_any(self, path_or_uri: str):
        if isinstance(path_or_uri, str) and path_or_uri.startswith("wds://"):
            return self._load_audio_from_wds_uri(path_or_uri)
        return torchaudio.load(path_or_uri)
    @property
    def lengths(self):
        """Token count estimate per sample for TokenBudgetBatchSampler.

        Priority order (fastest → slowest fallback):

        1. ``num_tokens_est`` column – single pre-computed int per sample written
           by ``src/dataset/precompute_audio_durations.py``; zero Python computation.
        2. ``audio_tokens`` + ``messages`` columns – sum pre-computed per-audio
           int token counts, add char-based text estimate.
        3. ``audio_durations`` + ``messages`` columns – derive audio tokens via
           ``int(dur * 17)``, add char-based text estimate.
        4. ``audios`` + ``messages`` columns – heuristic ``n_audios * 170`` for
           audio, char-based for text.

        All reads use bulk Arrow column access (entire column at once).
        """
        if not hasattr(self, '_cached_lengths'):
            col_names = self.raw_data.column_names

            # --- fastest path: fully pre-computed total ---
            if 'num_tokens_est' in col_names:
                self._cached_lengths = list(self.raw_data['num_tokens_est'])
                return self._cached_lengths

            # --- text estimate via bulk column read ---
            if 'messages' in col_names:
                all_messages = self.raw_data['messages']  # list of list-of-dicts
                text_est = [
                    sum(len(m.get('content', '')) for m in msgs) // 4
                    for msgs in all_messages
                ]
            else:
                text_est = [0] * len(self.raw_data)

            # --- audio estimate via bulk column read ---
            if 'audio_tokens' in col_names:
                all_audio_tokens = self.raw_data['audio_tokens']  # list of list[int|None]
                audio_est = [
                    sum(t for t in toks if t is not None) if toks else 0
                    for toks in all_audio_tokens
                ]
            elif 'audio_durations' in col_names:
                all_durations = self.raw_data['audio_durations']  # list of list[float|None]
                audio_est = [
                    int(sum(d for d in durs if d is not None) * 17)
                    if durs is not None else 0
                    for durs in all_durations
                ]
            elif 'audios' in col_names:
                all_audios = self.raw_data['audios']
                audio_est = [
                    len(a) * 170 if a is not None else 0
                    for a in all_audios
                ]
            else:
                audio_est = [0] * len(self.raw_data)

            self._cached_lengths = [t + a for t, a in zip(text_est, audio_est)]
        return self._cached_lengths

    def __getitem__(self, index):
        if isinstance(index, slice):
            # Return a new Dataset or list (list is recommended).
            return [self[i] for i in range(*index.indices(len(self)))]  # Call this instance's getitem(int).

        while True:
            try:
                sample = self.raw_data[index]
                if sample is None:
                    index = self.get_next_index(index)
                    continue

                # Check if sample should be filtered (e.g., contains code or Chinese) before preprocessing
                if should_filter_sample(sample):
                    self.filtered_samples += 1
                    if self.filtered_samples % 100 == 0:
                        print(f"[Qwen2Dataset] Filtered {self.filtered_samples} samples (code, Chinese, or question mark)")
                    index = self.get_next_index(index)
                    continue

                try:
                    ret = preprocess(
                        sample,
                        self.tokenizer,
                        default_system_message=self.default_system_message,
                        disable_text_normalize_llm=getattr(
                            self, "disable_text_normalize_llm", False
                        ),
                        use_omni_token=getattr(self, "_use_omni_token", False),
                        # whisper_model=self.whisper_model,
                    )
                except Exception as e:
                    print(e)
                    exit(-1)

                if not ret or not ret.get('input_ids'):
                    print("ret is None, get next index")
                    index = self.get_next_index(index)
                    continue

                cur_length = len(ret["input_ids"])
                
                # Skip if total tokens for the full user+assistant sequence (question + response, incl. chat template) exceed 600
                if cur_length > 500:
                    # print(
                    #     f"Total tokens (question+response) {cur_length} exceed 500, skipping sample"
                    # )
                    index = self.get_next_index(index)
                    continue

                if cur_length > self.max_padding_length:
                    print("too many tokens, get next index!")
                    index = self.get_next_index(index)
                    continue
                self.unjoint_samples += 1


                to_ret = {}
                # Add target audio (for FlexiCodec)
                raw_audio_paths = ret["audios"]
                assert isinstance(raw_audio_paths, list), f"audios must be a list, got {type(raw_audio_paths)}"
                # Support text-only datasets (empty audio array)
                is_text_only = len(raw_audio_paths) == 0

                # Join relative paths with self.audio_root if needed
                audio_paths = []
                if not is_text_only:
                    for p in raw_audio_paths:
                        if p is None:
                            continue
                        if isinstance(p, str) and p.startswith("wds://"):
                            audio_paths.append(p)
                        elif self.audio_root and not os.path.isabs(p):
                            audio_paths.append(os.path.join(self.audio_root, p))
                        else:
                            audio_paths.append(p)

                    # -------- File existence check: if missing, skip this sample and pick another --------
                    missing_paths = []
                    for p in audio_paths:
                        if isinstance(p, str) and p.startswith("wds://"):
                            continue
                        if p and not os.path.isfile(p):
                            missing_paths.append(p)

                    if missing_paths:
                        print(f"[Qwen2Dataset] Missing audio files for index {index}: {missing_paths}")
                        # Randomly select the next sample and continue.
                        index = self.get_next_index(index)
                        continue

                    # Load audio tensors and resample to 16kHz
                    audio_tensors = []
                    audio_durations = []  # Track duration per audio file (by index)
                    total_audio_duration = 0.0  # Track total audio duration in seconds
                    load_success = True
                    for audio_idx, audio_path in enumerate(audio_paths):
                        try:
                            try:
                                wav, sr = self._load_audio_any(audio_path)
                            except UnicodeDecodeError:
                                if isinstance(audio_path, str) and audio_path.startswith("wds://"):
                                    raise
                                with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                                    subprocess.run(
                                        ["ffmpeg", "-y", "-i", audio_path, "-f", "wav", tmp.name],
                                        check=True,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                    )
                                    wav, sr = torchaudio.load(tmp.name)
                            if sr != 16000:
                                resampler = T.Resample(sr, 16000)
                                wav = resampler(wav)
                            
                            # Check duration: user_audio <= 20s, assistant audio <= 40s
                            # If more than one audio path: first is user, rest are assistant.
                            # If only one audio path: treat it as assistant (cannot assume user).
                            duration = wav.shape[-1] / 16000.0  # Duration in seconds
                            is_user_audio = len(audio_paths) > 1 and audio_idx == 0
                            max_duration = 30.0 if is_user_audio else 60.0
                            if duration < 1.5 or duration > max_duration:
                                audio_type = "user" if is_user_audio else "assistant"
                                print(f"Warning: {audio_type} audio {audio_path} duration ({duration:.2f}s) is less than 1.5s or greater than {max_duration:.0f}s, skipping sample")
                                index = self.get_next_index(index)
                                load_success = False
                                break
                            
                            total_audio_duration += duration
                            audio_durations.append(duration)  # Store duration per audio file
                            audio_tensors.append(wav)
                        except Exception as e:
                            print(f"Warning: Failed to load audio {audio_path}: {e}")
                            # Skip this sample if audio loading fails
                            index = self.get_next_index(index)
                            load_success = False
                            break
                    
                    if not load_success:
                        continue
                    
                    # Check if assistant speech tokens < assistant text tokens
                    # Find assistant turns and calculate their speech/text tokens
                    assistant_speech_duration = 0.0
                    assistant_text_tokens = 0
                    for turn_dict in ret["input_ids_per_turn"]:
                        if turn_dict.get("role") == "assistant_content":
                            # Calculate assistant text tokens from text_content
                            text_content = turn_dict.get("text_content", "")
                            if len(text_content) > 3000:
                                print(f"Warning: Total tokens ({total_tokens}) exceeds 2048 (text: {cur_length}, audio: {audio_tokens} from {total_audio_duration:.2f}s), skipping sample")
                                index = self.get_next_index(index)
                                continue

                            if text_content:
                                assistant_text_ids = self.tokenizer(text_content, add_special_tokens=False).input_ids
                                assistant_text_tokens += len(assistant_text_ids)
                            
                            # Calculate assistant speech duration from audio_index
                            audio_idx = turn_dict.get("audio_index")
                            if audio_idx is not None and audio_idx < len(audio_durations):
                                assistant_speech_duration += audio_durations[audio_idx]
                    
                    # Calculate assistant speech tokens (16 tokens per second)
                    assistant_speech_tokens = int(assistant_speech_duration * 9)
                    
                    # Skip if assistant speech tokens < assistant text tokens
                    if assistant_text_tokens > 0 and assistant_speech_tokens != 0 and assistant_speech_tokens < assistant_text_tokens:
                        print(f"Warning: Assistant speech tokens ({assistant_speech_tokens}) < text tokens ({assistant_text_tokens}) (speech duration: {assistant_speech_duration:.2f}s), skipping sample. Faulty audio paths: {audio_paths}")
                        index = self.get_next_index(index)
                        continue
                    
                    # Calculate total tokens: text tokens + audio tokens (17 tokens per second)
                    audio_tokens = int(total_audio_duration * 17)
                    total_tokens = cur_length + audio_tokens
                    
                    # Filter out samples where total tokens exceed 2048
                    if total_tokens > 2000:
                        print(f"Warning: Total tokens ({total_tokens}) exceeds 2048 (text: {cur_length}, audio: {audio_tokens} from {total_audio_duration:.2f}s), skipping sample")
                        index = self.get_next_index(index)
                        continue
                else:
                    # Text-only: no audio tensors
                    audio_tensors = []
                    # For text-only samples, check if text tokens exceed 2048
                    if cur_length > 1000:
                        print(f"Warning: Text-only sample has {cur_length} tokens (exceeds 2048), skipping sample")
                        index = self.get_next_index(index)
                        continue
                
                # Convert input_ids_per_turn to list of dicts with torch.Tensor input_ids
                # and map audio tensors to the correct turns
                input_ids_per_turn_processed = []
                for turn_dict in ret["input_ids_per_turn"]:
                    processed_turn = {
                        "role": turn_dict["role"],
                        "input_ids": torch.tensor(turn_dict["input_ids"], dtype=torch.long),
                        "audio_tensors": None,
                        "audio_features": None,
                    }
                    # Map audio tensor if this turn has audio
                    if turn_dict.get("audio_index") is not None:
                        audio_idx = turn_dict["audio_index"]
                        if audio_idx < len(audio_tensors):
                            audio_tensor = audio_tensors[audio_idx]
                            # Pad 0.24 seconds of silence after assistant response audio
                            if turn_dict.get("role") == "assistant_content":
                                pad_samples = int(0.24 * 16000)  # 0.24s at 16kHz
                                audio_tensor = F.pad(audio_tensor, (0, pad_samples), value=0.0)

                            processed_turn["audio_tensors"] = audio_tensor
                            # Extract fbank features from audio tensor
                            try:
                                # FBankGen.extract_features expects 1D tensor [samples] or 2D [batch, samples]
                                # torchaudio.load returns [channels, samples], so convert to 1D
                                if len(audio_tensor.shape) == 2:
                                    # If stereo/multi-channel, take first channel; if mono, squeeze channel dim
                                    if audio_tensor.shape[0] > 1:
                                        audio_for_features = audio_tensor[0]  # Take first channel: [samples]
                                    else:
                                        audio_for_features = audio_tensor.squeeze(0)  # Remove channel dim: [samples]
                                elif len(audio_tensor.shape) == 1:
                                    audio_for_features = audio_tensor  # Already 1D: [samples]
                                else:
                                    raise ValueError(f"Unexpected audio tensor shape: {audio_tensor.shape}")
                                
                                # When using Qwen3/Qwen2.5-Omni: user turns get Qwen-style (128-d) for encoder; assistant turns get SenseVoice (80-d) for codec.
                                role = turn_dict.get("role", "")
                                if self._use_128_mel_feature and role == "assistant_content":
                                    extractor = self.sensevoice_fbank_extractor
                                else:
                                    extractor = self.fbank_extractor
                                audio_features = extractor.extract_features(audio_for_features, fs=16000) # [T, H]
                                processed_turn["audio_features"] = audio_features
                            except Exception as e:
                                print(f"Warning: Failed to extract fbank features for audio_idx {audio_idx}: {e}")
                                # Continue without features if extraction fails
                    input_ids_per_turn_processed.append(processed_turn)
                
                to_ret["input_ids_per_turn"] = input_ids_per_turn_processed
                # to_ret["labels"] = torch.tensor(ret["labels"], dtype=torch.long)
                
                # Only set fields if all audio files loaded successfully
                # to_ret["audio_paths"] = audio_paths
                # to_ret["audio_tensors"] = audio_tensors  # Keep for backward compatibility
                to_ret["audio_start_id"] = ret["audio_start_id"]
                to_ret["audio_end_id"] = ret["audio_end_id"]
                to_ret["audio_tag_id"] = ret["audio_tag_id"]
                return to_ret

            except Exception as error:
                print(error)
                index = self.get_next_index(index)
                continue
    
    def get_next_index(self, current_index=None):
        # To maintain length grouping, search nearby instead of completely random
        if current_index is not None:
            # Try to get a sample with similar length by picking a nearby index
            offset = random.randint(1, 100)
            return (current_index + offset) % self.length
        return random.randint(0, self.length - 1)



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

def test_preprocess():
    """Test case for preprocess function with interleaved text-audio format."""
    from unittest.mock import MagicMock
    
    # Define token mappings for special tokens
    token_map = {
        "<|im_start|>": [101],
        "<|im_end|>": [102],
        "\n": [103],
        "user": [104],
        "assistant": [105],
        "system": [106],
        "<|begin_of_audio|>": [201],
        "<|end_of_audio|>": [202],
        "<|audio|>": [203],
    }
    
    def mock_tokenize(text, add_special_tokens=False):
        # Handle special tokens
        if text in token_map:
            return MagicMock(input_ids=token_map[text])
        # For other text, return a list of token IDs based on words
        tokens = []
        words = text.split()
        for word in words:
            # Generate a consistent token ID for each word
            token_id = hash(word) % 10000 + 1000
            tokens.append(token_id)
        return MagicMock(input_ids=tokens if tokens else [0])
    
    # Create a mock tokenizer that is callable
    tokenizer = MagicMock()
    tokenizer.side_effect = mock_tokenize
    
    # Test sample data
    sample = {
        "messages": [
            {
                "content": "Your Name: Omni\nYour Gender: female \n\nRespond in a text-audio interleaved manner.",
                "role": "system"
            },
            {
                "content": "<|audio|>",
                "role": "user"
            },
            {
                "content": "Hello! My name is Omni, and I'm an AI voice assistant designed to handle various language tasks like answering questions and providing advice. My cognitive abilities are powered by advanced machine learning algorithms and a cluster of GPUs, allowing me to understand and respond to queries in real-time.\n<|audio|>",
                "role": "assistant"
            }
        ],
        "audios": ["user/identity/625/01625.wav", "assistant/identity/625/01625.wav"]
    }
    
    # Call preprocess
    result = preprocess(
        sample,
        tokenizer,
        default_system_message="You are a helpful assistant.",
        is_begin=True,
    )
    

def test_code_filtering():
    """Test the code filtering function with examples."""
    # Test case 1: C++ code (should be filtered)
    sample1 = {
        "messages": [
            {"content": "Your Name: Omni\nYour Gender: female \n\nRespond in a text-audio interleaved manner.", "role": "system"},
            {"content": "<|audio|>", "role": "user"},
            {"content": "#include <iostream>\n#include <math.h>\n\nusing namespace std;\n\nint main()\n{\n    int num;\n\n    cout << \"Enter a number: \";\n    cin >> num;\n\n    double root = sqrt(num);\n\n    if (root == (int)root) {\n        cout << num << \" is a perfect square.\" << endl;\n    }\n    else {\n        cout << num << \" is not a perfect square.\" << endl;\n    }\n\n    return 0;\n}\n<|audio|>", "role": "assistant"}
        ],
        "audios": ["user/5/613832_1.wav", "assistant/5/613832_1.wav"]
    }
    
    # Test case 2: Java code (should be filtered)
    sample2 = {
        "messages": [
            {"content": "Your Name: Omni\nYour Gender: female \n\nRespond in a text-audio interleaved manner.", "role": "system"},
            {"content": "<|audio|>", "role": "user"},
            {"content": "public class InsertionSort {\n  public static void main(String[] args) {\n    int[] arr = {5, 2, 7, 1, 3, 9};\n    System.out.println(\"Before sorting:\");\n    display(arr);\n    insertionSort(arr);\n    System.out.println(\"After sorting:\");\n    display(arr);\n  }\n  \n  public static void insertionSort(int[] arr) {\n    int n = arr.length;\n    for(int i=1; i<n; i++) {\n      int key = arr[i];\n      int j = i-1;\n      \n      while(j>=0 && arr[j]>key) {\n        arr[j+1] = arr[j];\n        j--;\n      }\n      \n      arr[j+1] = key;\n    }\n  }\n  \n  public static void display(int[] arr) {\n    for(int i=0; i<arr.length; i++) {\n      System.out.print(arr[i] + \" \");\n    }\n    System.out.println();\n  }\n}\n<|audio|>", "role": "assistant"}
        ],
        "audios": ["user/5/613935_1.wav", "assistant/5/613935_1.wav"]
    }
    
    # Test case 3: Normal conversation (should NOT be filtered)
    sample3 = {
        "messages": [
            {"content": "Your Name: Omni\nYour Gender: female \n\nRespond in a text-audio interleaved manner.", "role": "system"},
            {"content": "<|audio|>", "role": "user"},
            {"content": "Hello! My name is Omni, and I'm an AI voice assistant designed to handle various language tasks like answering questions and providing advice.\n<|audio|>", "role": "assistant"}
        ],
        "audios": ["user/5/613832_1.wav", "assistant/5/613832_1.wav"]
    }
    
    print("Testing code filtering function...")
    print(f"Sample 1 (C++ code): Should filter = {should_filter_sample(sample1)} (Expected: True)")
    print(f"Sample 2 (Java code): Should filter = {should_filter_sample(sample2)} (Expected: True)")
    print(f"Sample 3 (Normal text): Should filter = {should_filter_sample(sample3)} (Expected: False)")
    
    # Test individual code detection
    cpp_code = sample1["messages"][2]["content"]
    java_code = sample2["messages"][2]["content"]
    normal_text = sample3["messages"][2]["content"]
    
    print(f"\nDirect code detection:")
    print(f"C++ code contains_code: {contains_code(cpp_code)} (Expected: True)")
    print(f"Java code contains_code: {contains_code(java_code)} (Expected: True)")
    print(f"Normal text contains_code: {contains_code(normal_text)} (Expected: False)")


if __name__ == "__main__":
    test_code_filtering()
    # test_preprocess()
