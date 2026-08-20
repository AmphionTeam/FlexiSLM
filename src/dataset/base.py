# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
import contextlib
import glob
import json
import logging
import os
import pdb
import re
import subprocess
import sys
import traceback
import uuid

import numpy as np
import torch
import torchaudio
import yaml

from torchvision import transforms
from torchvision.transforms import InterpolationMode

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _feature_is_null_type(feat):
    """Check if a HF Feature is a null/all-null type (incompatible with concrete types for concatenation)."""
    from datasets import Features, Sequence, Value
    if isinstance(feat, Value) and feat.dtype == "null":
        return True
    if isinstance(feat, Sequence):
        return _feature_is_null_type(feat.feature)
    return False


_KNOWN_COLUMN_DEFAULTS = None


def _get_default_feature_for_column(column_name):
    """Return a sensible non-null Feature type for columns that are all-null on both sides."""
    from datasets import Sequence, Value

    global _KNOWN_COLUMN_DEFAULTS
    if _KNOWN_COLUMN_DEFAULTS is None:
        _KNOWN_COLUMN_DEFAULTS = {
            "audios": Sequence(Value("string")),
            "audio_durations": Sequence(Value("float64")),
            "audio_tokens": Sequence(Value("int64")),
            "num_tokens_est": Value("int64"),
        }
    return _KNOWN_COLUMN_DEFAULTS.get(column_name, Value("string"))


def _resolve_feature_type(column_name, left_feat, right_feat):
    """Given two features for the same column, pick the non-null concrete type when one side is all-null."""
    from datasets import Value

    if left_feat is None:
        return right_feat if right_feat is not None else _get_default_feature_for_column(column_name)
    if right_feat is None:
        return left_feat if left_feat is not None else _get_default_feature_for_column(column_name)

    left_is_null = _feature_is_null_type(left_feat)
    right_is_null = _feature_is_null_type(right_feat)

    if left_is_null and not right_is_null:
        return right_feat
    if right_is_null and not left_is_null:
        return left_feat
    if left_is_null and right_is_null:
        return _get_default_feature_for_column(column_name)
    return left_feat


def _concatenate_datasets_with_optional_columns(left, right):
    """Concatenate datasets after filling source-specific optional columns and aligning feature types."""
    from datasets import Features, Sequence, Value, concatenate_datasets

    columns = list(left.column_names)
    for column in right.column_names:
        if column not in columns:
            columns.append(column)

    for column in columns:
        if column not in left.column_names:
            left = left.add_column(column, [None] * len(left))
        if column not in right.column_names:
            right = right.add_column(column, [None] * len(right))

    left = left.select_columns(columns)
    right = right.select_columns(columns)

    aligned_features = {}
    needs_cast = False
    for column in columns:
        left_feat = left.features[column]
        right_feat = right.features[column]
        target_feat = _resolve_feature_type(column, left_feat, right_feat)
        aligned_features[column] = target_feat
        if _feature_is_null_type(left_feat) or _feature_is_null_type(right_feat):
            if target_feat != left_feat or target_feat != right_feat:
                needs_cast = True

    if needs_cast:
        null_columns = [
            col for col in columns
            if _feature_is_null_type(left.features[col]) or _feature_is_null_type(right.features[col])
        ]
        logger.info(
            f"Casting null-type columns to concrete features for concatenation: {null_columns}"
        )
        left = left.cast(Features(aligned_features))
        right = right.cast(Features(aligned_features))

    return concatenate_datasets([left, right])


def resolve_jsonl_path_with_durations(
    data_path: str, data_info: dict, cfg: dict, training_args=None
) -> str:
    """
    When ``training_args.max_tokens_per_batch`` is set (see ``arguments.py``), prefer
    ``data.with_durations.jsonl`` over ``data.jsonl`` when it exists. If only the base
    JSONL exists, start ``local/precompute_audio_durations.py`` once in the background
    (rank 0 only under distributed) and load the base JSONL; the sidecar is picked up on
    a later run when present.

    If ``max_tokens_per_batch`` is not set, return ``data_path`` unchanged (no duration
    sidecar resolution or precompute).
    """
    max_tb = (
        getattr(training_args, "max_tokens_per_batch", None)
        if training_args is not None
        else None
    )
    if max_tb is None:
        return data_path

    if not os.path.isfile(data_path):
        return data_path
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    if base_name.endswith(".with_durations"):
        return data_path

    with_path = _jsonl_with_durations_path(data_path)
    if os.path.isfile(with_path):
        logger.info(f"Using pre-computed durations file: {with_path}")
        return with_path

    script = _precompute_audio_durations_script_path()
    if not os.path.isfile(script):
        logger.warning(
            f"Precompute script not found at {script}; loading without durations sidecar: {data_path}"
        )
        return data_path

    global_audio_root = cfg.get("audio_root", None)
    audio_root = data_info.get("audio_root", global_audio_root)
    workers = int(cfg.get("precompute_audio_durations_workers", 16))

    def _run_precompute():
        cmd = [
            sys.executable,
            script,
            "--input",
            data_path,
            "--workers",
            str(workers),
        ]
        # if audio_root:
        #     cmd.extend(["--audio-root", audio_root])
        log_path = f"{os.path.splitext(data_path)[0]}.precompute_durations.log"
        logger.info(f"Starting pre-compute audio durations in background: {' '.join(cmd)} (log: {log_path})")
        log_f = None
        try:
            log_f = open(log_path, "a", encoding="utf-8")
        except OSError as e:
            logger.warning(f"Could not open {log_path} for precompute logs: {e}; discarding child output")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f if log_f is not None else subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            if log_f is not None:
                log_f.close()
        logger.info(f"precompute_audio_durations started pid={proc.pid}")

    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        if torch.distributed.get_rank() == 0:
            try:
                _run_precompute()
            except Exception as e:
                logger.error(
                    f"precompute_audio_durations failed for {data_path}: {e}; loading original JSONL"
                )
                traceback.print_exc()
        torch.distributed.barrier()
    else:
        try:
            _run_precompute()
        except Exception as e:
            logger.error(
                f"precompute_audio_durations failed for {data_path}: {e}; loading original JSONL"
            )
            traceback.print_exc()

    if os.path.isfile(with_path):
        logger.info(f"Using pre-computed durations file: {with_path}")
        return with_path
    return data_path


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg_path,
        tokenizer,
        max_padding_length=1024,
        variable_length=False,
        output_dir="",
        add_task_symbol=True,
        training_args=None,
        shift_token=False,
        create_position_ids=True,
        create_attention_mask=True,
        create_attention_mask_2d=False,
        create_loss_mask=False,
        max_num_frame=8,
        max_fps=1,
        reset_position_ids=False,
        reset_attention_mask=False,
        seed=42,
        cross_dataset_joint=False,
        dataset_joint=True,
        use_megatron=True,
        # whisper_model=None,
    ):
        super(BaseDataset, self).__init__()

        self.cfg_path = cfg_path
        with open(self.cfg_path, "r", encoding="utf8") as cfg_file:
            cfg_data = cfg_file.read()

        self.cfg = yaml.load(cfg_data, Loader=yaml.CLoader)
        logger.info(f"cfg {self.cfg}")

        # Default for interleaved preprocess; Qwen2Dataset may override from CLI/YAML.
        # Set here so unpickled workers / older checkpoints always have the attribute.
        self.disable_text_normalize_llm = bool(
            self.cfg.get("disable_text_normalize", False)
        )

        self.tokenizer = tokenizer
        self.max_padding_length = max_padding_length
        self.variable_length = variable_length
        self.output_dir = output_dir
        self.training_args = training_args
        self.shift_token = shift_token
        self.create_position_ids = create_position_ids
        self.create_attention_mask = create_attention_mask
        self.create_attention_mask_2d = create_attention_mask_2d
        self.create_loss_mask = create_loss_mask
        self.max_num_frame = max_num_frame
        self.max_fps = max_fps
        self.reset_position_ids = reset_position_ids
        self.reset_attention_mask = reset_attention_mask
        # self.whisper_model = whisper_model
        self.seed = seed
        self.cross_dataset_joint = cross_dataset_joint
        self.dataset_joint = dataset_joint

        self.do_dataset_format = self.cfg.get("do_dataset_format", False)
        self.do_dataset_cast = self.cfg.get("do_dataset_cast", False)
        self.xlsx_sample_num = self.cfg.get("xlsx_sample_num", 5)

        if use_megatron:
            self.load_data()
        else:
            with main_process_first(local=True, desc="Loading data"):
                self.load_data()

        self.processed_samples = 0
        self.unjoint_samples = 0
        self.joint_samples = 0
        self.skip_samples = 0

    def load_data(self):
        from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

        raw_data = None

        sampled_data = {}
        source_idx = 0
        for data_name, data_info in self.cfg["dataset"].items():
            data_ratio = data_info.get("ratio", 1)
            data_num = data_info.get("num", 999999999)

            if data_ratio == 0:
                continue

            if data_num == 0:
                continue

            for data_idx, data_path in enumerate(data_info["data_paths"]):
                if isinstance(data_path, str):
                    data_path = os.path.expanduser(os.path.expandvars(data_path))

                # Allow HF hub names (e.g. "yuantuo666/qwen3omni_gends_428k_0227") or parquet paths
                is_hf_hub = (
                    isinstance(data_path, str)
                    and "/" in data_path
                    and not data_path.startswith("/")
                    and not data_path.startswith(".")
                    and not os.path.isfile(data_path)
                    and not os.path.isdir(data_path)
                )
                is_parquet = isinstance(data_path, str) and (
                    data_path.endswith(".parquet")
                    or (os.path.isfile(data_path) and "parquet" in data_path.lower())
                    or os.path.isdir(data_path)  # local HF dataset dir (parquet, arrow, etc.)
                )

                if not is_hf_hub and not is_parquet and not os.path.isfile(data_path) and not os.path.isdir(data_path):
                    logger.warning(f"Data file not found {data_path}")
                    continue

                if is_hf_hub or is_parquet:
                    this_data = load_parquet_qwen3omni(
                        data_path,
                        data_info,
                        data_name,
                        self.output_dir,
                    )
                else:
                    jsonl_path = resolve_jsonl_path_with_durations(
                        data_path, data_info, self.cfg, training_args=self.training_args
                    )
                    this_data = load_json(jsonl_path, self.output_dir)
                # this_data = load_data_one(data_path, self.outout_dir)
                if this_data is None:
                    logger.warning(f"Failed to load {data_path}")
                    continue
                # print(f"this_data {this_data}")

                column_names = list(this_data.features)
                if "id" in column_names:
                    this_data = this_data.remove_columns("id")
                
                remove_column_names = [c for c in column_names if c.startswith("_")]
                logger.info(f"Removing debug columns from dataset: {remove_column_names}")
                this_data = this_data.remove_columns(remove_column_names)

                # Keep both a stable numeric source id and human-readable source
                # metadata for downstream dataset-specific masking and diagnostics.
                sources = [source_idx] * len(this_data)
                source_names = [data_name] * len(this_data)
                source_paths = [str(data_path)] * len(this_data)
                source_idx += 1
                this_data = this_data.add_column("source", sources)
                this_data = this_data.add_column("source_name", source_names)
                this_data = this_data.add_column("source_path", source_paths)

                # OPTIMIZATION: Don't shuffle here if we are going to shuffle at the end anyway.
                # Shuffling creates an index mapping which is heavy.
                # this_data = this_data.shuffle(seed=self.seed)

                data_ratio = float(data_ratio)
                total_num = len(this_data)
                used_num = min(int(total_num * data_ratio), data_num)
                logger.info(f"total_num {total_num}")
                logger.info(f"data_ratio {data_ratio}")
                logger.info(f"data_num {data_num}")
                logger.info(f"used_num {used_num}")

                # OPTIMIZATION: Use range directly instead of list comprehension for large ranges
                # indices = [x % total_num for x in range(used_num)]
                # this_data = this_data.select(indices)
                
                # If we are just taking the first N, use slice which is faster
                if used_num < total_num:
                    this_data = this_data.select(range(used_num))
                elif used_num > total_num:
                    # If we need to repeat data, we must use indices
                    indices = [x % total_num for x in range(used_num)]
                    this_data = this_data.select(indices)

                if raw_data is None:
                    raw_data = this_data
                else:
                    logger.info(f"concatenate_datasets {raw_data} {this_data}")
                    raw_data = _concatenate_datasets_with_optional_columns(
                        raw_data,
                        this_data,
                    )

                sampled_data[data_path] = {}
                sampled_data[data_path]["data"] = this_data.select(
                    range(min(self.xlsx_sample_num, used_num))
                )
                sampled_data[data_path]["total_num"] = total_num
                sampled_data[data_path]["used_num"] = used_num

                logger.info(f"this_data {this_data}")
                logger.info(f"raw_data {raw_data}")
                logger.info(f"Successful load {data_path}")

        _max_tokens_per_batch = getattr(self.training_args, "max_tokens_per_batch", None) if self.training_args is not None else None
        if _max_tokens_per_batch is None:
            # Shuffle the final concatenated dataset
            raw_data = raw_data.shuffle(seed=self.seed)

            # -------------------------------------------------------------------------
            # CRITICAL FIX FOR PICKLE ERROR:
            # -------------------------------------------------------------------------
            # .shuffle() and .select() create a "Subset" or "Index-Mapped" dataset.
            # This keeps the original data + a massive list of indices (12M integers) in RAM.
            # Pickling this list takes forever.
            #
            # .flatten_indices() forces HF Datasets to write the shuffled data to a
            # new Arrow file on disk (memory-mapped) and removes the index list.
            # The resulting object is tiny and pickles instantly.
            logger.info("Flattening indices to optimize multiprocessing pickling... (This may take a moment but prevents hangs)")
            raw_data = raw_data.flatten_indices(num_proc=96)
            # -------------------------------------------------------------------------
        else:
            logger.info("max_tokens_per_batch is set; skipping shuffle and flatten_indices (will be done later by the token-budget sampler)")

        self.raw_data = raw_data

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            output_xlsx = os.path.basename(self.cfg_path).replace("yaml", "xlsx")
            output_xlsx = os.path.join(self.output_dir, output_xlsx)
            logger.info(f"output_xlsx {output_xlsx}")

        logger.info(f"raw_data {raw_data}")

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(f"raw_data {raw_data[:10]}")
            logger.info(f"raw_data {raw_data[-10:]}")

    def __len__(self):
        return len(self.raw_data)

def format_function_general(examples):
    messages = [x for x in examples["messages"]]

    return {
        "messages": messages,
    }

def load_json_A(data_file):
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

    with open(data_file, "r") as f:
        raw_data = json.load(f)
    this_data = Dataset.from_list(raw_data)
    return this_data

def load_json_B(data_file):
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

    # keep_in_memory=False ensures we use memory mapping (Arrow)
    this_data = load_dataset("json", data_files=data_file, keep_in_memory=False)
    return this_data["train"]

def load_json_C(data_file):
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

    # Optimization: Use a generator instead of reading all lines into RAM list
    def gen():
        with open(data_file, "r") as f:
            for line in f:
                d = json.loads(line)
                item = {}
                if "conversations" in d:
                    item["conversations"] = d["conversations"]
                if "messages" in d:
                    item["messages"] = d["messages"]
                if "audios" in d:
                    item["audios"] = d["audios"]
                if "audio_durations" in d:
                    item["audio_durations"] = d["audio_durations"]
                if "id" in d:
                    item["id"] = d["id"]
                if item:
                    yield item
    
    # from_generator is much more memory efficient than from_list for large files
    this_data = Dataset.from_generator(gen)
    return this_data



def _get_audio_path_or_bytes(generated_wav):
    """
    Extract audio as (path_or_bytes, is_path) from HF Audio feature.
    Returns (path, True) if we have a file path, (bytes, False) if raw bytes, (None, None) if invalid.
    HF decoded format: {"path": str, "array": np.ndarray, "sampling_rate": int}
    HF raw format: {"bytes": bytes, "path": str}
    """
    if generated_wav is None:
        return None, None
    if isinstance(generated_wav, bytes):
        return generated_wav, False
    if isinstance(generated_wav, dict):
        if "bytes" in generated_wav and generated_wav["bytes"]:
            return generated_wav["bytes"], False
        if "path" in generated_wav and generated_wav["path"]:
            path = generated_wav["path"]
            if path and os.path.isfile(path):
                return path, True
        if "array" in generated_wav and generated_wav["array"] is not None:
            return generated_wav, "array"
    return None, None


def _convert_qwen3omni_row(row, cache_dir):
    """
    Convert qwen3omni parquet row to amphion format: {messages, audios}.
    Features: uuid, generated_text, generated_wav, speaker, src, prompt_text
    """
    uid = str(row.get("uuid", uuid.uuid4()))
    prompt_text = row.get("prompt_text") or ""
    generated_text = row.get("generated_text") or ""
    generated_wav = row.get("generated_wav")

    # Resolve audio path
    wav_path = None
    if generated_wav is not None:
        val, kind = _get_audio_path_or_bytes(generated_wav)
        if kind is True and val:
            wav_path = val
        elif kind is False and val:
            os.makedirs(cache_dir, exist_ok=True)
            wav_path = os.path.join(cache_dir, f"{uid}.wav")
            if not os.path.isfile(wav_path):
                with open(wav_path, "wb") as f:
                    f.write(val)
        elif kind == "array" and val:
            os.makedirs(cache_dir, exist_ok=True)
            wav_path = os.path.join(cache_dir, f"{uid}.wav")
            if not os.path.isfile(wav_path):
                arr = val["array"]
                sr = val.get("sampling_rate", 16000)
                if hasattr(arr, "__array__"):
                    arr = np.asarray(arr)
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                torchaudio.save(wav_path, torch.from_numpy(arr).float(), sr)
        elif isinstance(generated_wav, str) and os.path.isfile(generated_wav):
            wav_path = generated_wav

    if not wav_path or not generated_text.strip():
        return None

    messages = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": generated_text.rstrip() + "\n<|audio|>"},
    ]
    return {"messages": messages, "audios": [wav_path]}


def load_parquet_qwen3omni(data_path, data_info, data_name, output_dir):
    """
    Load qwen3omni-style parquet/HF dataset and convert to amphion format.
    Supports: HF hub name (e.g. "yuantuo666/qwen3omni_gends_428k_0227") or local .parquet path.
    """
    from datasets import Dataset, load_dataset

    cache_dir = data_info.get("parquet_audio_cache_dir")
    if not cache_dir:
        cache_dir = os.path.join(output_dir, "parquet_audio_cache", data_name.replace(" ", "_"))

    logger.info(f"Loading parquet/HF dataset: {data_path}, audio cache: {cache_dir}")

    try:
        if os.path.isdir(data_path):
            # Local dataset dir: try load_dataset first (handles parquet, arrow, etc.)
            try:
                ds = load_dataset(data_path, split="train", keep_in_memory=False)
            except Exception:
                parquet_files = glob.glob(os.path.join(data_path, "**/*.parquet"), recursive=True)
                if not parquet_files:
                    parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
                if not parquet_files:
                    raise ValueError(f"No parquet files and load_dataset failed for {data_path}")
                ds = load_dataset("parquet", data_files=parquet_files, split="train", keep_in_memory=False)
        elif data_path.endswith(".parquet") or (os.path.isfile(data_path) and "parquet" in data_path.lower()):
            ds = load_dataset("parquet", data_files=data_path, split="train", keep_in_memory=False)
        else:
            ds = load_dataset(data_path, split="train", keep_in_memory=False)
    except Exception as e:
        logger.warning(f"Failed to load parquet/HF dataset {data_path}: {e}")
        return None

    # Ensure required columns exist
    cols = ds.column_names
    if "generated_text" not in cols or "generated_wav" not in cols:
        logger.warning(f"Dataset {data_path} missing generated_text/generated_wav columns: {cols}")
        return None

    def convert_and_filter(example):
        out = _convert_qwen3omni_row(example, cache_dir)
        return out if out else {"messages": [], "audios": []}

    ds = ds.map(
        convert_and_filter,
        remove_columns=cols,
        num_proc=1,
        desc="Convert qwen3omni to amphion format",
    )

    # Filter out rows that failed conversion (empty messages)
    def keep_valid(example):
        return len(example.get("messages", [])) > 0

    ds = ds.filter(keep_valid, num_proc=1, desc="Filter invalid rows")
    logger.info(f"Loaded {len(ds)} samples from {data_path}")
    return ds


def load_json(data_file, output_dir):
    # Try B first (Fastest, Arrow), then C (Generator), then A (Full RAM)
    for func in [load_json_B, load_json_C, load_json_A]:
        try:
            this_data = func(data_file)
            return this_data
        except Exception as error:
            # Only log errors if we are sure it failed (optional)
            # print(f"Method {func.__name__} failed for {data_file}: {error}")
            continue
            
    # If all failed, log the error
    with open(os.path.join(output_dir, "data_error.log"), "a") as f:
        print("-" * 100, file=f)
        print(f"All load methods failed for {data_file}", file=f)
    return None

def load_data_one(data_file, output_dir):
    if data_file.endswith("json") or data_file.endswith("jsonl"):
        return load_json(data_file, output_dir)
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

    this_data = load_dataset(data_file, keep_in_memory=False)
    return this_data["train"]

@contextlib.contextmanager
def main_process_first(local=True, desc="work"):

    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        if local:
            rank = int(os.environ["LOCAL_RANK"])
        else:
            rank = torch.distributed.get_rank()
        is_main_process = rank == 0

        try:
            if not is_main_process:
                torch.distributed.barrier()
            yield
        finally:
            if is_main_process:
                torch.distributed.barrier()
    else:
        yield
