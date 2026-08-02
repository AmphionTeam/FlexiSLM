# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
import itertools
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import transformers
from torch.utils.data import default_collate
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_TOKEN_ID,
        )

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "images" in instances[0]:
            images = [instance["images"] for instance in instances]
            batch["images"] = torch.cat(images, dim=0)

        if "doclm_images" in instances[0]:
            images = [instance["doclm_images"] for instance in instances]
            batch["doclm_images"] = torch.cat(images, dim=0)

        if "image_paths" in instances[0]:
            image_paths = [instance["image_paths"] for instance in instances]
            batch["image_paths"] = image_paths

        if "pixel_values" in instances[0]:
            pixel_values = torch.cat([instance["pixel_values"] for instance in instances])
            batch["pixel_values"] = pixel_values

        if "image_flags" in instances[0]:
            image_flags = torch.cat([instance["image_flags"] for instance in instances])
            batch["image_flags"] = image_flags

        return batch


def collate_fn_deepspeed_old(batch):

    keys = list(set().union(*[set(x.keys()) for x in batch]))
    tmp_batch = [{} for _ in range(len(batch))]
    if "actual_seq_len" in batch[0]:
        actual_seq_len = [x["actual_seq_len"] for x in batch]
    else:
        actual_seq_len = None

    for k in keys:
        if "images" in k or k == "image_indices":
            for x, y in zip(tmp_batch, batch):
                if k in y:
                    x[k] = y.pop(k)
                    # print("x[image_indices]", x["image_indices"].size())

    new_batch = default_collate(batch)

    for k in keys:
        if "images" in k or k == "image_indices":
            cat_dim = 0 if k != "image_indices" else 1
            if k == "image_indices":
                cnt = 0
                for sample in tmp_batch:
                    if k in sample:
                        sample[k][0] = cnt
                    cnt += 1
            new_batch[k] = torch.cat([x[k] for x in tmp_batch if k in x], dim=cat_dim)
    # print("new_batch[image_indices]", new_batch["image_indices"].size())

    if actual_seq_len is not None:
        seq_len = actual_seq_len[0][-1]
        actual_seq_len = [elem + i * seq_len for i, elem in enumerate(actual_seq_len)]
        new_batch["actual_seq_len"] = torch.cat(actual_seq_len)

    return new_batch


def collate_fn_deepspeed(batch):
    # print(f"[collate_fn] received batch with {len(batch)} samples")
    # for i, sample in enumerate(batch):
    #     print(f"[collate_fn] sample {i} keys: {list(sample.keys())}")

    def is_valid_sample(x):
        return (
            x is not None and
            isinstance(x.get("input_ids"), torch.Tensor) and
            x["input_ids"].numel() > 0
        )

    batch = [x for x in batch if is_valid_sample(x)]

    if len(batch) == 0:
        print("[collate_fn] Empty batch after filtering — returning dummy.")
        dummy = {
            "input_ids": torch.zeros((1, 1), dtype=torch.long),
            "labels": torch.full((1, 1), -100, dtype=torch.long),
        }
        return default_collate([dummy])
    

    keys = list(set().union(*[set(x.keys()) for x in batch]))
    # print(f"{keys=}")

    tmp_batch = [{} for _ in range(len(batch))]
    if "actual_seq_len" in batch[0]:
        actual_seq_len = [x["actual_seq_len"] for x in batch]
    else:
        actual_seq_len = None

    if "images" in batch[0].keys():
        for new_x, x in zip(tmp_batch, batch):
            new_x["images"] = x.pop("images")
            new_x["image_indices"] = x.pop("image_indices")

    if "audios" in batch[0].keys():
        for new_x, x in zip(tmp_batch, batch):
            new_x["audios"] = x.pop("audios")
            new_x["audio_indices"] = x.pop("audio_indices")

    new_batch = default_collate(batch)


    
    if "images" in tmp_batch[0].keys():
        new_batch["images"] = torch.cat([x["images"] for x in tmp_batch], dim=0)

        for sample_idx, sample in enumerate(tmp_batch):
            for j in range(len(sample["image_indices"])):
                sample["image_indices"][j][0, :, :] = sample_idx

        new_batch["image_indices"] = torch.cat([x["image_indices"] for x in tmp_batch], dim=1)

    if "audios" in tmp_batch[0].keys():
        new_batch["audios"] = list(itertools.chain.from_iterable([x["audios"] for x in tmp_batch]))
        # print(f"{[x.size() for x in sample['audios']]}")

        for sample_idx, sample in enumerate(tmp_batch):
            for j in range(len(sample["audio_indices"])):
                sample["audio_indices"][j][0, :, :] = sample_idx

        new_batch["audio_indices"] = list(
            itertools.chain.from_iterable([x["audio_indices"] for x in tmp_batch])
        )
        # print(f"{[x.size() for x in sample['audio_indices']]}")

    if actual_seq_len is not None:
        seq_len = actual_seq_len[0][-1]
        actual_seq_len = [elem + i * seq_len for i, elem in enumerate(actual_seq_len)]
        new_batch["actual_seq_len"] = torch.cat(actual_seq_len)

    return new_batch




from transformers import DataCollatorWithPadding

def get_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt", padding=True)