# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""Batch collation for materialized interleaved samples."""

import torch

from src.task_types import TASK_TO_ID


class InterleavedDataCollator:
    def __init__(self, tokenizer, use_omni_token: bool = False):
        self.tokenizer = tokenizer
        self.use_omni_token = use_omni_token

    def __call__(self, batch):
        tokenizer = self.tokenizer  # Keep the original tokenizer reference behavior.

        # Group by user and assistant messages (assuming one user turn and one assistant turn per sample)
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return {}

        # Get BOS and EOS token IDs
        bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id

        # Fallback: if BOS/EOS not available, use pad_token_id
        if bos_token_id is None:
            bos_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        if eos_token_id is None:
            eos_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # Collect user and assistant turns separately
        semantic_task_ids = torch.tensor(
            [TASK_TO_ID.get(sample.get("task"), -1) for sample in batch],
            dtype=torch.long,
        )
        user_input_ids = []
        user_audio_tensors = []
        user_turns = []  # Store user turns for later feature extraction
        user_has_audio_list = []
        assistant_input_ids = []
        assistant_audio_tensors = []
        assistant_turns = []  # Store assistant turns for later feature extraction
        assistant_has_audio_list = []

        for batch_idx, sample in enumerate(batch):
            turns = sample["input_ids_per_turn"]

            # Find user and assistant turns
            user_turn = None
            assistant_turn = None
            if len(turns) > 2:
                turns = turns[:2]
                # print(f"Warning: Sample {batch_idx} has more than 2 turns, only using the first 2 turns")
            for turn_dict in turns:
                role = turn_dict.get("role", "")
                assert role in ["user", "human", "assistant_content"], f"Invalid role: {role}"
                # Match user/human roles
                if role in ["user", "human"]:
                    user_turn = turn_dict
                # Match assistant_content (the actual assistant response with audio)
                elif role == "assistant_content":
                    assistant_turn = turn_dict

            # Process user turn
            if user_turn is not None:
                user_input_ids.append(torch.tensor(user_turn["input_ids"], dtype=torch.long))
                user_turns.append(user_turn)  # Store turn for later feature extraction

                audio_tensor = user_turn.get("audio_tensors")
                has_audio = audio_tensor is not None
                user_has_audio_list.append(has_audio)

                if has_audio:
                    # Convert audio_tensor to 1D if needed: [channels, samples] -> [samples]
                    if len(audio_tensor.shape) == 2:
                        if audio_tensor.shape[0] > 1:
                            audio_tensor = audio_tensor[0]  # Take first channel
                        else:
                            audio_tensor = audio_tensor.squeeze(0)
                    user_audio_tensors.append(audio_tensor)
                else:
                    user_audio_tensors.append(None)
            else:
                # No user turn found - this shouldn't happen but handle gracefully
                raise ValueError(f"Sample {batch_idx} has no user turn")

            # Process assistant turn
            if assistant_turn is not None:
                assistant_input_ids.append(torch.tensor(assistant_turn["input_ids"], dtype=torch.long))
                assistant_turns.append(assistant_turn)  # Store turn for later feature extraction

                audio_tensor = assistant_turn.get("audio_tensors")
                has_audio = audio_tensor is not None
                assistant_has_audio_list.append(has_audio)

                if has_audio:
                    # Convert audio_tensor to 1D if needed: [channels, samples] -> [samples]
                    if len(audio_tensor.shape) == 2:
                        if audio_tensor.shape[0] > 1:
                            audio_tensor = audio_tensor[0]  # Take first channel
                        else:
                            audio_tensor = audio_tensor.squeeze(0)
                    assistant_audio_tensors.append(audio_tensor)
                else:
                    assistant_audio_tensors.append(None)
            else:
                # No assistant turn found - this shouldn't happen but handle gracefully
                raise ValueError(f"Sample {batch_idx} has no assistant turn")
        from src.processor.constants import (
            AUD_START_TOKEN,
            AUD_END_TOKEN,
            AUD_START_TOKEN_OMNI,
            AUD_END_TOKEN_OMNI,
            AUD_TAG_TOKEN,
        )

        if self.use_omni_token:
            aud_s, aud_e = AUD_START_TOKEN_OMNI, AUD_END_TOKEN_OMNI
        else:
            aud_s, aud_e = AUD_START_TOKEN, AUD_END_TOKEN
        # Get the audio special token ids from the tokenizer to ensure consistency
        audio_start_id = tokenizer(aud_s, add_special_tokens=False).input_ids[0]
        audio_end_id = tokenizer(aud_e, add_special_tokens=False).input_ids[0]
        audio_tag_id = tokenizer(AUD_TAG_TOKEN, add_special_tokens=False).input_ids[0]

        for i, x in enumerate(batch[1:], start=1):
            if x.get("audio_start_id") != audio_start_id or x.get("audio_end_id") != audio_end_id or x.get("audio_tag_id") != audio_tag_id:
                raise ValueError(f"Inconsistent audio special token ids within batch at index {i}")

        # Pad user input_ids with LEFT padding using BOS token
        max_user_len = max(ids.shape[0] for ids in user_input_ids)
        user_padded_input_ids = []
        user_attention_masks = []

        for ids in user_input_ids:
            seq_len = ids.shape[0]
            pad_len = max_user_len - seq_len
            if pad_len > 0:
                # Left pad with BOS token
                pad_tokens = torch.full((pad_len,), bos_token_id, dtype=torch.long)
                padded_ids = torch.cat([pad_tokens, ids], dim=0)
            else:
                padded_ids = ids
            user_padded_input_ids.append(padded_ids)

            # Create attention mask: 0 for padding (BOS tokens), 1 for real tokens
            attn_mask = torch.zeros(max_user_len, dtype=torch.bool)
            attn_mask[pad_len:] = True  # Real tokens
            user_attention_masks.append(attn_mask)

        user_padded_input_ids = torch.stack(user_padded_input_ids, dim=0)  # [B, L_user_max]
        user_attention_mask = torch.stack(user_attention_masks, dim=0)  # [B, L_user_max]

        # Pad assistant input_ids with RIGHT padding using EOS token
        max_assistant_len = max(ids.shape[0] for ids in assistant_input_ids)
        assistant_padded_input_ids = []
        assistant_attention_masks = []

        for ids in assistant_input_ids:
            seq_len = ids.shape[0]
            pad_len = max_assistant_len - seq_len
            if pad_len > 0:
                # Right pad with EOS token
                pad_tokens = torch.full((pad_len,), eos_token_id, dtype=torch.long)
                padded_ids = torch.cat([ids, pad_tokens], dim=0)
            else:
                padded_ids = ids
            assistant_padded_input_ids.append(padded_ids)

            # Create attention mask: 1 for real tokens, 0 for padding (EOS tokens)
            attn_mask = torch.ones(max_assistant_len, dtype=torch.bool)
            attn_mask[seq_len:] = False  # Padding tokens
            assistant_attention_masks.append(attn_mask)

        assistant_padded_input_ids = torch.stack(assistant_padded_input_ids, dim=0)  # [B, L_assistant_max]
        assistant_attention_mask = torch.stack(assistant_attention_masks, dim=0)  # [B, L_assistant_max]

        user_audio_tensors_list = [t for t in user_audio_tensors if t is not None]
        user_has_audio = torch.tensor(user_has_audio_list, dtype=torch.bool)  # [B]

        # Get audio_features directly from turn dictionaries
        user_audio_features_list = []
        user_audio_features_lengths = []
        for i, turn in enumerate(user_turns):
            features = turn.get("audio_features")
            if features is not None:
                assert len(features.shape) == 2, f"Features shape should be [T, D], but got {features.shape}"
                user_audio_features_list.append(features)
                user_audio_features_lengths.append(features.shape[0])
            else:
                # No audio features for this sample
                user_audio_features_list.append(None)
                user_audio_features_lengths.append(0)

        # Pad fbank features to the same length
        valid_features = [f for f in user_audio_features_list if f is not None]
        if len(valid_features) > 0:
            max_feat_len = max(f.shape[0] for f in valid_features)
            feat_dim = valid_features[0].shape[1] if len(valid_features[0].shape) > 1 else 1

            padded_user_audio_features = []
            for features in user_audio_features_list:
                if features is not None:
                    feat_len = features.shape[0]
                    pad_len = max_feat_len - feat_len
                    if pad_len > 0:
                        # Pad along time dimension: [T_feat, D] -> [T_feat_max, D]
                        if features.dim() == 2:
                            features = torch.nn.functional.pad(features, (0, 0, 0, pad_len), mode='constant', value=0.0)
                        else:
                            features = torch.nn.functional.pad(features, (0, pad_len), mode='constant', value=0.0)
                    padded_user_audio_features.append(features)
                else:
                    # Create zero features for samples without audio
                    if feat_dim > 1:
                        padded_user_audio_features.append(torch.zeros(max_feat_len, feat_dim, dtype=torch.float32))
                    else:
                        padded_user_audio_features.append(torch.zeros(max_feat_len, dtype=torch.float32))

            user_padded_audio_features = torch.stack(padded_user_audio_features, dim=0)  # [B, T_feat_max, D]
            user_audio_features_lens = torch.tensor(user_audio_features_lengths, dtype=torch.long)  # [B]
        else:
            user_padded_audio_features = None
            user_audio_features_lens = None

        if len(user_audio_tensors_list) > 0:

            # Pad user audio_tensors to the same length
            max_user_audio_len = max(t.shape[0] for t in user_audio_tensors_list)
            padded_user_audio_tensors = []
            user_audio_lengths = []

            for i, audio_tensor in enumerate(user_audio_tensors):
                if audio_tensor is not None:
                    actual_len = audio_tensor.shape[0]
                    pad_len = max_user_audio_len - actual_len
                    if pad_len > 0:
                        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_len), mode='constant', value=0.0)
                    padded_user_audio_tensors.append(audio_tensor)
                    user_audio_lengths.append(actual_len)
                else:
                    padded_user_audio_tensors.append(torch.zeros(max_user_audio_len, dtype=torch.float32))
                    user_audio_lengths.append(0)

            user_padded_audio_tensors = torch.stack(padded_user_audio_tensors, dim=0)  # [B, T_user_audio_max]
            user_audio_tensors_lens = torch.tensor(user_audio_lengths, dtype=torch.long)  # [B]

        else:
            user_padded_audio_tensors = None
            user_padded_audio_features = None
            user_audio_tensors_lens = None
            user_audio_features_lens = None

        # Get audio_features directly from turn dictionaries for assistant audio
        assistant_audio_tensors_list = [t for t in assistant_audio_tensors if t is not None]
        assistant_has_audio = torch.tensor(assistant_has_audio_list, dtype=torch.bool)  # [B]

        assistant_audio_features_list = []
        assistant_audio_features_lengths = []
        for i, turn in enumerate(assistant_turns):
            features = turn.get("audio_features")
            if features is not None:
                assert len(features.shape) == 2, f"Features shape should be [T, D], but got {features.shape}"
                assistant_audio_features_list.append(features)
                assistant_audio_features_lengths.append(features.shape[0])
            else:
                # No audio features for this sample
                assistant_audio_features_list.append(None)
                assistant_audio_features_lengths.append(0)

        # Pad fbank features to the same length
        valid_features = [f for f in assistant_audio_features_list if f is not None]
        if len(valid_features) > 0:
            max_feat_len = max(f.shape[0] for f in valid_features)
            feat_dim = valid_features[0].shape[1] if len(valid_features[0].shape) > 1 else 1

            padded_assistant_audio_features = []
            for features in assistant_audio_features_list:
                if features is not None:
                    feat_len = features.shape[0]
                    pad_len = max_feat_len - feat_len
                    if pad_len > 0:
                        # Pad along time dimension: [T_feat, D] -> [T_feat_max, D]
                        if features.dim() == 2:
                            features = torch.nn.functional.pad(features, (0, 0, 0, pad_len), mode='constant', value=0.0)
                        else:
                            features = torch.nn.functional.pad(features, (0, pad_len), mode='constant', value=0.0)
                    padded_assistant_audio_features.append(features)
                else:
                    # Create zero features for samples without audio
                    if feat_dim > 1:
                        padded_assistant_audio_features.append(torch.zeros(max_feat_len, feat_dim, dtype=torch.float32))
                    else:
                        padded_assistant_audio_features.append(torch.zeros(max_feat_len, dtype=torch.float32))

            assistant_padded_audio_features = torch.stack(padded_assistant_audio_features, dim=0)  # [B, T_feat_max, D]
            assistant_audio_features_lens = torch.tensor(assistant_audio_features_lengths, dtype=torch.long)  # [B]
        else:
            assistant_padded_audio_features = None
            assistant_audio_features_lens = None

        if len(assistant_audio_tensors_list) > 0:

            # Pad assistant audio_tensors to the same length
            max_assistant_audio_len = max(t.shape[0] for t in assistant_audio_tensors_list)
            padded_assistant_audio_tensors = []
            assistant_audio_lengths = []

            for i, audio_tensor in enumerate(assistant_audio_tensors):
                if audio_tensor is not None:
                    actual_len = audio_tensor.shape[0]
                    pad_len = max_assistant_audio_len - actual_len
                    if pad_len > 0:
                        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_len), mode='constant', value=0.0)
                    padded_assistant_audio_tensors.append(audio_tensor)
                    assistant_audio_lengths.append(actual_len)
                else:
                    padded_assistant_audio_tensors.append(torch.zeros(max_assistant_audio_len, dtype=torch.float32))
                    assistant_audio_lengths.append(0)

            assistant_padded_audio_tensors = torch.stack(padded_assistant_audio_tensors, dim=0)  # [B, T_assistant_audio_max]
            assistant_audio_tensors_lens = torch.tensor(assistant_audio_lengths, dtype=torch.long)  # [B]

        else:
            assistant_padded_audio_tensors = None
            assistant_padded_audio_features = None
            assistant_audio_tensors_lens = None
            assistant_audio_features_lens = None

        return {
            "user_input_ids": user_padded_input_ids,
            "user_attention_mask": user_attention_mask,
            "user_audio_tensors": user_padded_audio_tensors,
            "user_audio_tensors_lens": user_audio_tensors_lens,
            "user_audio_features": user_padded_audio_features,
            "user_audio_features_lens": user_audio_features_lens,
            "user_has_audio": user_has_audio,
            "assistant_input_ids": assistant_padded_input_ids,
            "assistant_attention_mask": assistant_attention_mask,
            "assistant_audio_tensors": assistant_padded_audio_tensors,
            "assistant_audio_tensors_lens": assistant_audio_tensors_lens,
            "assistant_audio_features": assistant_padded_audio_features,
            "assistant_audio_features_lens": assistant_audio_features_lens,
            "assistant_has_audio": assistant_has_audio,
            "semantic_task_ids": semantic_task_ids,
            "audio_start_id": audio_start_id,
            "audio_end_id": audio_end_id,
            "audio_tag_id": audio_tag_id,
        }
