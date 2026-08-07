# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""
# Skip optimizer signal: zero loss (zero grads) when text loss spikes after warmup.
_TEXT_LOSS_SKIP_THRESHOLD = 6.0
_TEXT_LOSS_SKIP_AFTER_STEP = 30000

import json
import os
import random
import logging
from functools import partial
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Any, Union
from tqdm import tqdm
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import SaveStrategy, seed_worker
from transformers.utils import is_datasets_available, is_sagemaker_mp_enabled

# Import datasets module
try:
    import datasets
except ImportError:
    datasets = None

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

# Import swanlab for logging
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    logger.warning("swanlab not available. Custom metrics will not be logged.")


class TokenBudgetBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        lengths,
        max_tokens,
        max_batch_size=None,
        shuffle=True,
        seed=42,
        drop_last=False,
        rank=0,
        world_size=1,
        pad_to_world_size=False,
    ):
        self.lengths = list(lengths)
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.pad_to_world_size = pad_to_world_size
        self.epoch = 0

        self._batches = self._build_batches()      # all batches, same on every rank
        self._order = list(range(len(self._batches)))

    def _build_batches(self):
        rng = random.Random(self.seed + int(self.epoch))

        print(f'Before sort indices')
        indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        print(f'After sort indices')

        if self.shuffle:
            window = 500000
            for i in tqdm(range(0, len(indices), window), desc="TokenBudgetSampler Shuffling indices"):
                w = indices[i:i + window]
                rng.shuffle(w)
                indices[i:i + window] = w

        batches = []
        batch, batch_tokens = [], 0
        for idx in indices:
            tok = self.lengths[idx]
            if batch and (
                (batch_tokens + tok > self.max_tokens
                or (
                    self.max_batch_size is not None
                    and len(batch) >= self.max_batch_size
                )) and len(batch) >= self.max_batch_size//5
            ):
                batches.append(batch)
                batch, batch_tokens = [], 0
            batch.append(idx)
            batch_tokens += tok
        if batch and not self.drop_last:
            batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def _rank_batches(self):
        """Slice batch order for one rank, optionally padding deterministic eval."""
        total = len(self._order)
        if self.pad_to_world_size and total:
            usable = ((total + self.world_size - 1) // self.world_size) * self.world_size
            padding = usable - total
            order = self._order + [self._order[i % total] for i in range(padding)]
        else:
            usable = (total // self.world_size) * self.world_size
            order = self._order[:usable]
        return order[self.rank::self.world_size]

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.shuffle and self.epoch > 0:
            # Rebuild batches with a new epoch seed so samples are globally reshuffled,
            # while still grouped by similar length inside each batch.
            self._batches = self._build_batches()
            rng = random.Random(self.seed + int(epoch))
            self._order = list(range(len(self._batches)))
            rng.shuffle(self._order)

    def __iter__(self):
        for i in self._rank_batches():
            yield self._batches[i]

    def __len__(self):
        return len(self._rank_batches())


class SkipFinalCheckpointCallback(TrainerCallback):
    """Keep the final model in output_dir without duplicating it in a checkpoint."""

    @staticmethod
    def _skip_final_save(args, state, control, strategy):
        if (
            args.save_strategy == strategy
            and state.global_step >= state.max_steps
        ):
            control.should_save = False
        return control

    def on_step_end(self, args, state, control, **kwargs):
        return self._skip_final_save(args, state, control, SaveStrategy.STEPS)

    def on_epoch_end(self, args, state, control, **kwargs):
        return self._skip_final_save(args, state, control, SaveStrategy.EPOCH)


class ATrainer(Trainer):
    _NATIVE_DATALOADER_STATE = "native_dataloader_state_rank{rank}.json"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._native_resume_checkpoint = None
        self._native_train_loader = None
        # Run after DefaultFlowCallback so its forced final checkpoint is disabled.
        self.add_callback(SkipFinalCheckpointCallback)

    def _native_dataloader_state_path(self, checkpoint_dir):
        return os.path.join(
            checkpoint_dir,
            self._NATIVE_DATALOADER_STATE.format(rank=self.args.process_index),
        )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        result = super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        if getattr(self.train_dataset, "is_native_webdataset", False):
            state_path = self._native_dataloader_state_path(resume_from_checkpoint)
            if os.path.isfile(state_path):
                self._native_resume_checkpoint = resume_from_checkpoint
                # The stateful loader performs deterministic replay itself. Letting
                # Trainer also call skip_first_batches would advance the stream twice.
                self.args.ignore_data_skip = True
                logger.info("Will restore native WebDataset state from %s", state_path)
            else:
                logger.warning(
                    "Native WebDataset state is absent from %s; falling back to "
                    "Trainer batch skipping",
                    resume_from_checkpoint,
                )
        return result

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        loader = self._native_train_loader
        state_fn = getattr(loader, "state_dict", None)
        if not callable(state_fn):
            return
        run_dir = self._get_output_dir(trial=trial)
        checkpoint_dir = os.path.join(run_dir, f"checkpoint-{self.state.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        state_path = self._native_dataloader_state_path(checkpoint_dir)
        temporary_path = f"{state_path}.tmp.{os.getpid()}"
        with open(temporary_path, "w", encoding="utf-8") as stream:
            json.dump(state_fn(), stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, state_path)

    def create_optimizer(self):
        talker_lr = getattr(self.args, "talker_learning_rate", None)
        audio_encoder_lr = getattr(self.args, "audio_encoder_learning_rate", None)
        combine_proj_lr = getattr(self.args, "combine_proj_learning_rate", None)
        if (
            talker_lr is None
            and audio_encoder_lr is None
            and combine_proj_lr is None
        ):
            return super().create_optimizer()

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is not None:
            return self.optimizer

        decay_parameters = self.get_decay_parameter_names(opt_model)
        optimizer_grouped_parameters = []

        def is_talker_param(name):
            return (
                "talker" in name
                or "input_merging_transformer" in name
            )

        def is_audio_encoder_param(name):
            # Names of the (deep-copied / lazily-loaded) audio encoders unfrozen
            # when finetune_speech_encoder=True.
            return (
                "sensevoice_finetune_copy" in name
                or "_qwen3_encoder" in name
                or "_whisper_encoder" in name
                or "_qwen25o_encoder" in name
            )

        def get_param_group(name):
            # Audio encoder takes precedence so its LR isn't overridden by
            # generic substring matches (e.g. some encoder params may live
            # nested under modules whose names also contain other tokens).
            if audio_encoder_lr is not None and is_audio_encoder_param(name):
                return "audio_encoder"
            if combine_proj_lr is not None and "combined_embed_proj" in name:
                return "combine_proj"
            if talker_lr is not None and is_talker_param(name):
                return "talker"
            return "base"

        group_specs = [
            ("base_decay", "base", True, self.args.learning_rate, self.args.weight_decay),
            ("base_no_decay", "base", False, self.args.learning_rate, 0.0),
            ("talker_decay", "talker", True, talker_lr, self.args.weight_decay),
            ("talker_no_decay", "talker", False, talker_lr, 0.0),
            ("audio_encoder_decay", "audio_encoder", True, audio_encoder_lr, self.args.weight_decay),
            ("audio_encoder_no_decay", "audio_encoder", False, audio_encoder_lr, 0.0),
            ("combine_proj_decay", "combine_proj", True, combine_proj_lr, self.args.weight_decay),
            ("combine_proj_no_decay", "combine_proj", False, combine_proj_lr, 0.0),
        ]

        for group_name, param_group, use_decay, lr, weight_decay in group_specs:
            if lr is None:
                continue
            params = [
                p
                for n, p in opt_model.named_parameters()
                if p.requires_grad
                and get_param_group(n) == param_group
                and (n in decay_parameters) == use_decay
            ]
            if params:
                optimizer_grouped_parameters.append(
                    {
                        "params": params,
                        "weight_decay": weight_decay,
                        "lr": lr,
                        "name": group_name,
                    }
                )

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        logger.info(
            "Created optimizer with talker lr=%s, audio_encoder lr=%s, combine_proj lr=%s, "
            "and base lr=%s across %d parameter groups.",
            talker_lr,
            audio_encoder_lr,
            combine_proj_lr,
            self.args.learning_rate,
            len(optimizer_grouped_parameters),
        )
        return self.optimizer

    def _get_dataset_lengths(self, dataset):
        if hasattr(dataset, "lengths"):
            return dataset.lengths
        if (
            isinstance(dataset, torch.utils.data.Subset)
            and hasattr(dataset.dataset, "lengths")
        ):
            base_lengths = dataset.dataset.lengths
            return [base_lengths[i] for i in dataset.indices]
        return None
    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        if self.args.group_by_length and hasattr(train_dataset, "lengths"):
            from transformers.trainer_pt_utils import LengthGroupedSampler
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=train_dataset,
                lengths=train_dataset.lengths,
            )
        return super()._get_train_sampler(train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset
        if getattr(train_dataset, "is_native_webdataset", False):
            # The dataset already partitions shards by rank and emits complete
            # dynamic batches. Accelerate must not dispatch or split them again.
            dataloader_config = getattr(self.accelerator, "dataloader_config", None)
            if dataloader_config is not None:
                dataloader_config.dispatch_batches = False
                dataloader_config.split_batches = False
                dataloader_config.even_batches = False
            loader = train_dataset.build_loader(
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                persistent_workers=self.args.dataloader_persistent_workers,
                prefetch_factor=getattr(self.args, "dataloader_prefetch_factor", None),
            )
            if self._native_resume_checkpoint is not None:
                state_path = self._native_dataloader_state_path(
                    self._native_resume_checkpoint
                )
                with open(state_path, "r", encoding="utf-8") as stream:
                    loader.load_state_dict(json.load(stream))
                logger.info(
                    "Restored native WebDataset cursor from %s; replay will occur "
                    "from the start of the saved epoch",
                    state_path,
                )
                self._native_resume_checkpoint = None
            self._native_train_loader = loader
            return loader

        max_tokens = getattr(self.args, "max_tokens_per_batch", None)
        if max_tokens is None:
            return super().get_train_dataloader()

        max_batch_size = None
        if getattr(self.args, "per_device_train_batch_size", None) is not None:
            max_batch_size = max(1, 3 * self.args.per_device_train_batch_size)

        if not hasattr(train_dataset, "lengths"):
            raise ValueError("Dataset has no 'lengths' attribute; "
                             "falling back to default dataloader")


        batch_sampler = TokenBudgetBatchSampler(
            lengths=train_dataset.lengths,
            max_tokens=max_tokens,
            max_batch_size=max_batch_size,
            shuffle=True,
            seed=self.args.seed,
            drop_last=self.args.dataloader_drop_last,
            rank=self.args.process_index,
            world_size=self.args.world_size,
        )


        dataloader_params = {
            "batch_sampler": batch_sampler,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "worker_init_fn": partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            ),
        }
        if self.args.dataloader_persistent_workers and self.args.dataloader_num_workers > 0:
            dataloader_params["persistent_workers"] = True

        dataloader = DataLoader(train_dataset, **dataloader_params)

        # Sample a small subset of batches for stats to avoid iterating all N batches at startup.
        import random as _random
        _all_batches = batch_sampler._batches
        _sample = _all_batches[:500] if len(_all_batches) > 500 else _all_batches
        batch_sizes = [len(b) for b in _sample]
        logger.info(
            f"TokenBudgetBatchSampler: max_tokens={max_tokens}, "
            f"max_batch_size={max_batch_size}, "
            f"{len(batch_sampler)} batches/epoch, "
            f"batch_size (sampled) min={min(batch_sizes)} avg={sum(batch_sizes)/len(batch_sizes):.1f} "
            f"max={max(batch_sizes)}"
        )
        return dataloader
        # return self.accelerator.prepare(dataloader)
    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """
        Build the eval dataloader without Hugging Face's RemoveColumnsCollator.

        InterleavedDataCollator needs fields such as `input_ids_per_turn` that
        are not standard model-forward columns for wrapped/PEFT models. The
        default Trainer eval dataloader can strip them before collation.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self._eval_dataloaders[dataloader_key]

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        max_tokens = getattr(self.args, "max_tokens_per_batch", None)
        if max_tokens is not None:
            max_batch_size = None
            if getattr(self.args, "per_device_train_batch_size", None) is not None:
                max_batch_size = max(1, 3 * self.args.per_device_train_batch_size)

            eval_lengths = self._get_dataset_lengths(eval_dataset)
            if eval_lengths is None:
                raise ValueError(
                    "Eval dataset has no 'lengths' attribute; "
                    "cannot use the training token-budget sampler"
                )

            batch_sampler = TokenBudgetBatchSampler(
                lengths=eval_lengths,
                max_tokens=max_tokens,
                max_batch_size=max_batch_size,
                shuffle=False,
                seed=self.args.seed,
                # Evaluation must retain the final partial batch so indexed
                # cardinality does not depend on the training drop-last policy.
                drop_last=False,
                rank=self.args.process_index,
                world_size=self.args.world_size,
                pad_to_world_size=True,
            )

            dataloader_params = {
                "batch_sampler": batch_sampler,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "worker_init_fn": partial(
                    seed_worker,
                    num_workers=self.args.dataloader_num_workers,
                    rank=self.args.process_index,
                ),
            }
            if self.args.dataloader_persistent_workers and self.args.dataloader_num_workers > 0:
                dataloader_params["persistent_workers"] = True

            dataloader = DataLoader(eval_dataset, **dataloader_params)

            _sample = batch_sampler._batches[:500] if len(batch_sampler._batches) > 500 else batch_sampler._batches
            batch_sizes = [len(b) for b in _sample]
            logger.info(
                f"Eval TokenBudgetBatchSampler: max_tokens={max_tokens}, "
                f"max_batch_size={max_batch_size}, "
                f"{len(batch_sampler)} batches/eval, "
                f"batch_size (sampled) min={min(batch_sizes)} avg={sum(batch_sizes)/len(batch_sizes):.1f} "
                f"max={max(batch_sizes)}"
            )

            if self.args.dataloader_persistent_workers:
                if hasattr(self, "_eval_dataloaders"):
                    self._eval_dataloaders[dataloader_key] = dataloader
                else:
                    self._eval_dataloaders = {dataloader_key: dataloader}

            return dataloader

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if self.args.dataloader_persistent_workers and self.args.dataloader_num_workers > 0:
            dataloader_params["persistent_workers"] = True
        if self.args.dataloader_num_workers > 0:
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            sampler = self._get_eval_sampler(eval_dataset)
            if sampler is not None:
                dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = False

        dataloader = self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}

        return dataloader

    def _unwrap_model(self, model):
        if getattr(self, "accelerator", None) is not None:
            try:
                return self.accelerator.unwrap_model(model)
            except Exception:
                pass
        m = model
        if hasattr(m, "module"):
            m = m.module
        return m

    def _get_local_batch_size(self, inputs):
        uid = inputs.get("user_input_ids")
        if uid is not None and torch.is_tensor(uid) and uid.dim() > 0:
            return int(uid.shape[0])

        for value in inputs.values():
            if torch.is_tensor(value) and value.dim() > 0:
                return int(value.shape[0])
        return None

    def _update_total_samples_seen(self, inputs):
        if self.args.process_index != 0:
            return None, None

        local_batch_size = self._get_local_batch_size(inputs)
        if local_batch_size is None:
            return None, None

        world_size = max(1, int(getattr(self.args, "world_size", 1) or 1))
        global_batch_size = local_batch_size * world_size

        self._latest_local_batch_size = local_batch_size
        self._latest_global_batch_size = global_batch_size
        self._total_samples_seen = int(getattr(self, "_total_samples_seen", 0)) + global_batch_size
        return local_batch_size, global_batch_size

    def log(self, logs, start_time=None):
        logs = dict(logs)
        dataset = getattr(self, "train_dataset", None)
        snapshot_fn = getattr(dataset, "metrics_snapshot", None)
        if callable(snapshot_fn):
            for name, value in snapshot_fn().items():
                if name.endswith("_sum") or name == "unpadded_cost_sum":
                    continue
                logs[f"webdataset/{name}"] = value

        if self.args.process_index == 0:
            total_samples_seen = getattr(self, "_total_samples_seen", None)
            if total_samples_seen is not None:
                logs["total_samples_seen"] = total_samples_seen

            global_batch_size = getattr(self, "_latest_global_batch_size", None)
            if global_batch_size is not None:
                logs["global_effective_batch_size"] = global_batch_size

        return super().log(logs, start_time=start_time)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to extract and log text_loss and length_loss.
        """
        local_batch_size = None
        global_batch_size = None
        is_training = model.training
        if self.args.process_index == 0 and is_training:
            local_batch_size, global_batch_size = self._update_total_samples_seen(inputs)

        outputs = model(**inputs)
        
        # Extract the main loss
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
        else:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        
        if (
            loss is not None
            and getattr(self.state, "global_step", 0) > _TEXT_LOSS_SKIP_AFTER_STEP
        ):
            if isinstance(outputs, dict):
                text_for_skip = outputs.get("text_ce_loss")
                if text_for_skip is None:
                    text_for_skip = outputs.get("text_token_loss")
            else:
                text_for_skip = getattr(outputs, "text_ce_loss", None)
                if text_for_skip is None:
                    text_for_skip = getattr(outputs, "text_token_loss", None)
            if text_for_skip is not None and torch.is_tensor(text_for_skip):
                if bool((text_for_skip.detach() > _TEXT_LOSS_SKIP_THRESHOLD).item()):
                    print(f"Trigger text loss skip threshold: {text_for_skip.item()}, loss: {loss.item()}")
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
                    loss = loss * 0.0

        # Log metrics to swanlab if available (only on main process)
        if SWANLAB_AVAILABLE and self.args.process_index == 0:
            metrics_to_log = {}
            base = self._unwrap_model(model)
            w = float(getattr(getattr(base, "config", None), "text_loss_weight", 1.0) or 1.0)

            if isinstance(outputs, dict):
                text_ce = outputs.get("text_ce_loss")
                text_tok_existing = outputs.get("text_token_loss")
            else:
                text_ce = getattr(outputs, "text_ce_loss", None)
                text_tok_existing = getattr(outputs, "text_token_loss", None)

            metric_prefix = "" if is_training else "eval/"

            if text_ce is not None and torch.is_tensor(text_ce):
                metrics_to_log[f"{metric_prefix}text_ce_loss"] = text_ce.item()
                metrics_to_log[f"{metric_prefix}text_token_loss"] = text_ce.item() * w
            elif text_tok_existing is not None and torch.is_tensor(text_tok_existing):
                metrics_to_log[f"{metric_prefix}text_token_loss"] = text_tok_existing.item()
            if hasattr(outputs, "length_loss") and outputs.length_loss is not None:
                metrics_to_log[f"{metric_prefix}length_loss"] = outputs.length_loss.item()
            
            if hasattr(outputs, "audio_token_loss") and outputs.audio_token_loss is not None:
                metrics_to_log[f"{metric_prefix}audio_token_loss"] = outputs.audio_token_loss.item()

            # Depth transformer (acoustic) losses
            if hasattr(outputs, "acoustic_loss") and outputs.acoustic_loss is not None:
                metrics_to_log[f"{metric_prefix}acoustic_loss"] = outputs.acoustic_loss.item()
            if hasattr(outputs, "acoustic_ce_loss") and outputs.acoustic_ce_loss is not None:
                metrics_to_log[f"{metric_prefix}acoustic_ce_loss"] = outputs.acoustic_ce_loss.item()
            if hasattr(outputs, "acoustic_per_codebook_loss") and outputs.acoustic_per_codebook_loss is not None:
                per_q = outputs.acoustic_per_codebook_loss
                if torch.is_tensor(per_q):
                    for q_idx, val in enumerate(per_q.tolist()):
                        metrics_to_log[f"{metric_prefix}acoustic_loss_q{q_idx}"] = val
            
            if hasattr(outputs, "loss_text_only_data") and outputs.loss_text_only_data is not None:
                metrics_to_log[f"{metric_prefix}loss_text_only_data"] = outputs.loss_text_only_data.item()
            
            if hasattr(outputs, "loss_audio_dialog_data") and outputs.loss_audio_dialog_data is not None:
                metrics_to_log[f"{metric_prefix}loss_audio_dialog_data"] = outputs.loss_audio_dialog_data.item()

            if is_training and local_batch_size is not None:
                metrics_to_log["effective_batch_size"] = local_batch_size
                metrics_to_log["global_effective_batch_size"] = global_batch_size
                metrics_to_log["total_samples_seen"] = getattr(self, "_total_samples_seen", 0)

            # Log metrics only when SwanLab is enabled for this run.
            report_to = getattr(self.args, "report_to", []) if hasattr(self, "args") else []
            if isinstance(report_to, str):
                report_to = [report_to]
            use_swanlab = SWANLAB_AVAILABLE and any(
                str(x).strip().lower() == "swanlab" for x in report_to
            )

            if metrics_to_log and use_swanlab:
                # Get step if available, otherwise swanlab will track automatically
                step = None
                if hasattr(self, "state") and hasattr(self.state, "global_step"):
                    step = self.state.global_step
                swanlab.log(metrics_to_log, step=step)
        
        return (loss, outputs) if return_outputs else loss
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Run eval with prediction_loss_only=False so custom eval metrics are emitted.

        Hugging Face Trainer passes prediction_loss_only=True when compute_metrics
        is None. This trainer logs model-specific metrics from compute_loss, so we
        temporarily install a no-op compute_metrics and force args.prediction_loss_only
        off for evaluation.
        """
        original_compute_metrics = self.compute_metrics
        original_prediction_loss_only = getattr(self.args, "prediction_loss_only", False)

        if original_compute_metrics is None:
            self.compute_metrics = lambda eval_pred: {}
        self.args.prediction_loss_only = False

        try:
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = original_compute_metrics
            self.args.prediction_loss_only = original_prediction_loss_only

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        """
        Force loss computation for eval batches that do not contain `labels`.

        The interleaved collator provides model-specific inputs instead of a
        standard `labels` field. The base Trainer otherwise skips loss
        computation during eval and only reports throughput metrics.
        """

        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, return_outputs=False)

        if loss is None:
            return (None, None, None)
        # Keep logits/labels as None to avoid retaining very large model outputs
        # while still running with prediction_loss_only=False and collecting loss.

        return (loss.detach().mean(), None, None)
    def on_epoch_begin(self, args, state, control, **kwargs):
        dl = self.get_train_dataloader()
        bs = getattr(dl, "batch_sampler", None) or getattr(
            getattr(dl, "batch_sampler", None), "batch_sampler", None
        )
        if hasattr(bs, "set_epoch"):
            bs.set_epoch(int(state.epoch))
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        if inputs is None or len(inputs) == 0:
            raise RuntimeError("inputs is None or empty")

        loss = super().training_step(model, inputs)

        if loss is None:
            raise RuntimeError("training_step returned None")

        if torch.is_tensor(loss):
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        return loss

    
    # def save_model(self, output_dir=None, _internal_call=False):
    #     output_dir = output_dir or self.args.output_dir
        
    #     # In distributed training, only the main process saves the model.
    #     if self.args.local_rank != -1 and self.args.local_rank != 0:
    #         logger.info(f"Process {self.args.local_rank}: skipping model save (only main process saves)")
    #         return
        
    #     # Check whether this is a LoRA model.
    #     is_peft_model = False
    #     try:
    #         # Method 1: check PEFT model attributes.
    #         if hasattr(self.model, 'is_peft_model'):
    #             is_peft_model = self.model.is_peft_model
    #         # Method 2: check model type.
    #         elif hasattr(self.model, '__class__') and 'PeftModel' in str(self.model.__class__):
    #             is_peft_model = True
    #         # Method 3: check whether PEFT-related config exists.
    #         elif hasattr(self.model, 'peft_config') and self.model.peft_config is not None:
    #             is_peft_model = True
    #     except Exception:
    #         pass
        
    #     if is_peft_model:
    #         # Save LoRA model.
    #         self.model.save_pretrained(output_dir, safe_serialization=True)
    #         logger.info(f"LoRA model saved to {output_dir}")
            
    #     else:
    #         # Save full model.
    #         self.model.save_pretrained(output_dir, safe_serialization=True)
    #         logger.info(f"Full model saved to {output_dir}")
        
    #     # Save tokenizer.
    #     if self.tokenizer is not None:
    #         self.tokenizer.save_pretrained(output_dir)
    #         logger.info(f"Tokenizer saved to {output_dir}")
    
    def save_metrics(self, split, metrics, combined=True):
        """Override save_metrics to ensure only the main process saves metrics."""
        # In distributed training, only the main process saves metrics.
        if self.args.local_rank != -1 and self.args.local_rank != 0:
            logger.info(f"Process {self.args.local_rank}: skipping metrics save (only main process saves)")
            return
        
        # Call parent implementation.
        super().save_metrics(split, metrics, combined)
    
    def save_state(self):
        """Override save_state to ensure only the main process saves state."""
        # In distributed training, only the main process saves state.
        if self.args.local_rank != -1 and self.args.local_rank != 0:
            logger.info(f"Process {self.args.local_rank}: skipping state save (only main process saves)")
            return
        
        # Call parent implementation.
        super().save_state()
    
    def log_metrics(self, split, metrics, epoch=None, step=None):
        """Override log_metrics to ensure only the main process logs metrics."""
        # In distributed training, only the main process logs metrics.
        if self.args.local_rank != -1 and self.args.local_rank != 0:
            logger.info(f"Process {self.args.local_rank}: skipping metrics logging (only main process logs)")
            return
        
        # Call parent implementation.
        super().log_metrics(split, metrics)
