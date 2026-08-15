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
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import EvalLoopOutput, SaveStrategy, seed_worker
from transformers.utils import is_sagemaker_mp_enabled

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
    ):
        self.lengths = list(lengths)
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
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
        """Slice complete batch groups evenly across training ranks."""
        total = len(self._order)
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


class TrainingProfilerCallback(TrainerCallback):
    """Capture a short CPU/CUDA/NCCL trace when explicitly enabled.

    Set ``FLEXISLM_PROFILE_DIR`` to enable profiling. The wait, warmup, and
    active step counts default to 10/5/20 and can be changed with matching
    ``FLEXISLM_PROFILE_*_STEPS`` environment variables. Each distributed rank
    writes a separate TensorBoard-compatible trace directory.
    """

    def __init__(self):
        self.profiler = None
        self.profile_steps = 0
        self.total_profile_steps = 0

    def on_train_begin(self, args, state, control, **kwargs):
        root = os.environ.get("FLEXISLM_PROFILE_DIR")
        if not root:
            return control
        wait = int(os.environ.get("FLEXISLM_PROFILE_WAIT_STEPS", "10"))
        warmup = int(os.environ.get("FLEXISLM_PROFILE_WARMUP_STEPS", "5"))
        active = int(os.environ.get("FLEXISLM_PROFILE_ACTIVE_STEPS", "20"))
        if min(wait, warmup, active) < 0 or active == 0:
            raise ValueError(
                "FlexiSLM profiler step counts must be non-negative and active > 0"
            )
        trace_dir = os.path.join(root, f"rank{args.process_index}")
        os.makedirs(trace_dir, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        self.profiler.__enter__()
        self.profile_steps = 0
        self.total_profile_steps = wait + warmup + active
        logger.info(
            "Enabled training profiler for rank %d: wait=%d warmup=%d active=%d output=%s",
            args.process_index,
            wait,
            warmup,
            active,
            trace_dir,
        )
        return control

    def _close(self):
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.profiler = None

    def on_step_end(self, args, state, control, **kwargs):
        if self.profiler is not None:
            self.profiler.step()
            self.profile_steps += 1
            if self.profile_steps >= self.total_profile_steps:
                self._close()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._close()
        return control


class ATrainer(Trainer):
    _NATIVE_DATALOADER_STATE = "native_dataloader_state_rank{rank}.json"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This trainer performs its own task/token normalization.
        self.model_accepts_loss_kwargs = False
        self._native_resume_checkpoint = None
        self._native_train_loader = None
        # Run after DefaultFlowCallback so its forced final checkpoint is disabled.
        self.add_callback(SkipFinalCheckpointCallback)
        self.add_callback(TrainingProfilerCallback)

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

    def _get_named_param_group_lr(self, group_name):
        optimizer = getattr(self, "optimizer", None)
        if optimizer is None:
            return None
        for param_group in optimizer.param_groups:
            name = param_group.get("name")
            if name == group_name or (
                isinstance(name, str) and name.startswith(f"{group_name}_")
            ):
                lr = param_group.get("lr")
                if lr is None:
                    return None
                if torch.is_tensor(lr):
                    return float(lr.detach().item())
                return float(lr)
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

    def _disable_accelerate_batch_dispatch(self):
        dataloader_config = getattr(self.accelerator, "dataloader_config", None)
        if dataloader_config is not None:
            dataloader_config.dispatch_batches = False
            dataloader_config.split_batches = False
            dataloader_config.even_batches = False

    def _build_native_webloader(self, dataset):
        self._disable_accelerate_batch_dispatch()
        return dataset.build_loader(
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=getattr(self.args, "dataloader_prefetch_factor", None),
        )

    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset
        if getattr(train_dataset, "is_native_webdataset", False):
            # The dataset already partitions shards by rank and emits complete
            # dynamic batches. Accelerate must not dispatch or split them again.
            loader = self._build_native_webloader(train_dataset)
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

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if isinstance(eval_dataset, str):
            eval_dataset = self.eval_dataset[eval_dataset]
        if getattr(eval_dataset, "is_native_webdataset", False):
            set_epoch = getattr(eval_dataset, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(0)
            return self._build_native_webloader(eval_dataset)
        return super().get_eval_dataloader(eval_dataset)

    _EVAL_STAT_FIELDS = (
        "text_loss_sum",
        "text_token_count",
        "audio_loss_sum",
        "audio_token_count",
        "length_loss_sum",
        "length_token_count",
        "auxiliary_loss_sum",
        "samples",
        "batches",
        "finite_batches",
    )

    def _init_eval_stats(self, device):
        return {
            name: torch.zeros((), device=device, dtype=torch.float64)
            for name in self._EVAL_STAT_FIELDS
        }

    def _accumulate_eval_stats(self, stats, outputs, inputs, loss):
        stats["batches"] += 1.0
        sample_count = self._get_local_batch_size(inputs)
        if sample_count is not None:
            stats["samples"] += float(sample_count)
        if loss is None or not bool(torch.isfinite(loss.detach()).all().item()):
            return
        stats["finite_batches"] += 1.0
        for name in (
            "text_loss_sum",
            "text_token_count",
            "audio_loss_sum",
            "audio_token_count",
            "length_loss_sum",
            "length_token_count",
        ):
            value = self._output_value(outputs, name)
            if value is not None and torch.is_tensor(value):
                stats[name] += value.detach().to(dtype=torch.float64).reshape(())
        auxiliary = self._output_value(outputs, "auxiliary_loss")
        if auxiliary is not None and torch.is_tensor(auxiliary):
            stats["auxiliary_loss_sum"] += auxiliary.detach().to(
                dtype=torch.float64
            ).reshape(())

    def _finalize_eval_stats(self, stats, metric_key_prefix):
        names = list(self._EVAL_STAT_FIELDS)
        stacked = torch.stack([stats[name] for name in names])
        stacked = self._distributed_sum(stacked)
        reduced = {
            name: float(value.item()) for name, value in zip(names, stacked)
        }

        def mean(sum_name, count_name):
            count = reduced[count_name]
            if count <= 0:
                return None
            return reduced[sum_name] / count

        text_token_loss = mean("text_loss_sum", "text_token_count")
        audio_token_loss = mean("audio_loss_sum", "audio_token_count")
        length_loss = mean("length_loss_sum", "length_token_count")
        config = getattr(self._unwrap_model(self.model), "config", None)
        text_weight = float(getattr(config, "text_loss_weight", 1.0) or 1.0)
        if getattr(config, "freeze_talker", False):
            combined = text_token_loss
        elif getattr(config, "only_train_talker", False):
            parts = [part for part in (audio_token_loss, length_loss) if part is not None]
            combined = sum(parts) if parts else None
        else:
            parts = [
                part
                for part in (text_token_loss, audio_token_loss, length_loss)
                if part is not None
            ]
            combined = sum(parts) if parts else None
        finite_batches = reduced["finite_batches"]
        if combined is not None and finite_batches > 0 and reduced["auxiliary_loss_sum"]:
            combined = combined + reduced["auxiliary_loss_sum"] / finite_batches

        metrics = {}
        if combined is not None:
            metrics["loss"] = combined
        if text_token_loss is not None:
            metrics["text_token_loss"] = text_token_loss
            metrics["text_ce_loss"] = (
                text_token_loss / text_weight if text_weight else text_token_loss
            )
        if audio_token_loss is not None:
            metrics["audio_token_loss"] = audio_token_loss
        if length_loss is not None:
            metrics["length_loss"] = length_loss
        metrics["samples"] = reduced["samples"]
        metrics["batches"] = reduced["batches"]
        metrics["text_token_count"] = reduced["text_token_count"]
        metrics["audio_token_count"] = reduced["audio_token_count"]
        metrics["length_token_count"] = reduced["length_token_count"]
        prefixed = {
            key
            if key.startswith(f"{metric_key_prefix}_")
            else f"{metric_key_prefix}_{key}": value
            for key, value in metrics.items()
        }
        return prefixed, int(reduced["samples"])

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        eval_dataset = getattr(dataloader, "dataset", None)
        if not getattr(eval_dataset, "is_native_webdataset", False):
            return super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        args = self.args
        model = self.model
        if hasattr(model, "eval") and callable(model.eval):
            model.eval()
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples: Unknown")
        logger.info("  Native WebDataset validation: token-weighted loss reduction")
        self.callback_handler.eval_dataloader = dataloader

        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = args.device
        stats = self._init_eval_stats(device)
        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True
                    )
            self._accumulate_eval_stats(stats, outputs, inputs, loss)
            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control
            )

        metrics, num_samples = self._finalize_eval_stats(stats, metric_key_prefix)
        logger.info(
            "Native WebDataset %s finished: samples=%d batches=%.0f loss=%s",
            description,
            num_samples,
            metrics.get(f"{metric_key_prefix}_batches", 0.0),
            metrics.get(f"{metric_key_prefix}_loss"),
        )
        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=num_samples,
        )

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

    @staticmethod
    def _output_value(outputs, name):
        if isinstance(outputs, dict):
            return outputs.get(name)
        return getattr(outputs, name, None)

    @staticmethod
    def _distributed_sum(value):
        total = value.detach().clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        return total

    @staticmethod
    def _zero_loss_from_model(model, reference):
        zero_loss = None
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.numel() > 0:
                term = parameter.reshape(-1)[0] * 0.0
                zero_loss = term if zero_loss is None else zero_loss + term
        if zero_loss is None:
            zero_loss = reference.new_zeros((), requires_grad=True)
        return zero_loss

    def _normalize_model_loss(
        self, model, outputs, fallback_loss, *, reduce_across_processes=True
    ):
        component_names = (
            ("text_loss_sum", "text_token_count"),
            ("audio_loss_sum", "audio_token_count"),
            ("length_loss_sum", "length_token_count"),
        )
        local_sums = []
        local_counts = []
        for loss_name, count_name in component_names:
            local_sum = self._output_value(outputs, loss_name)
            local_count = self._output_value(outputs, count_name)
            if local_sum is None or local_count is None:
                return fallback_loss
            local_sums.append(local_sum)
            local_counts.append(local_count.float().reshape(()))

        if reduce_across_processes:
            global_counts = self._distributed_sum(torch.stack(local_counts))
            world_size = (
                torch.distributed.get_world_size()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 1
            )
        else:
            global_counts = torch.stack(local_counts)
            world_size = 1
        components = [
            local_sum * world_size / global_count
            if bool(global_count.gt(0).item())
            else local_sum
            for local_sum, global_count in zip(local_sums, global_counts)
        ]
        text_loss, audio_loss, length_loss = components
        config = getattr(self._unwrap_model(model), "config", None)
        if getattr(config, "freeze_talker", False):
            loss = text_loss
        elif getattr(config, "only_train_talker", False):
            loss = audio_loss + length_loss
        else:
            loss = text_loss + audio_loss + length_loss

        auxiliary_loss = self._output_value(outputs, "auxiliary_loss")
        if auxiliary_loss is not None:
            loss = loss + auxiliary_loss
        return loss

    def _get_local_batch_tokens(self, inputs):
        cost = inputs.get("batch_cost") if isinstance(inputs, dict) else None
        if cost is not None:
            if torch.is_tensor(cost):
                return float(cost.detach().float().reshape(-1)[0].item())
            return float(cost)

        tokens = 0.0
        for key in ("user_input_ids", "assistant_input_ids"):
            value = inputs.get(key) if isinstance(inputs, dict) else None
            if value is not None and torch.is_tensor(value) and value.dim() >= 2:
                tokens += float(value.shape[0] * value.shape[1])
        if tokens > 0:
            return tokens
        value = inputs.get("input_ids") if isinstance(inputs, dict) else None
        if value is not None and torch.is_tensor(value) and value.dim() >= 2:
            return float(value.shape[0] * value.shape[1])
        return None

    def _update_total_samples_seen(self, inputs):
        local_batch_size = self._get_local_batch_size(inputs)
        if local_batch_size is None:
            return None, None, None, None

        device = next(
            (
                value.device
                for key, value in inputs.items()
                if key != "batch_cost" and torch.is_tensor(value)
            ),
            self.args.device,
        )
        global_batch = torch.tensor(local_batch_size, device=device, dtype=torch.long)
        global_batch = self._distributed_sum(global_batch)
        global_batch_size = int(global_batch.item())

        local_batch_tokens = self._get_local_batch_tokens(inputs)
        global_batch_tokens = None
        if local_batch_tokens is not None:
            global_tokens = torch.tensor(
                local_batch_tokens, device=device, dtype=torch.float64
            )
            global_tokens = self._distributed_sum(global_tokens)
            global_batch_tokens = float(global_tokens.item())

        if isinstance(inputs, dict):
            inputs.pop("batch_cost", None)

        if self.args.process_index == 0:
            self._latest_local_batch_size = local_batch_size
            self._latest_global_batch_size = global_batch_size
            self._latest_local_batch_tokens = local_batch_tokens
            self._latest_global_batch_tokens = global_batch_tokens
            self._total_samples_seen = (
                int(getattr(self, "_total_samples_seen", 0)) + global_batch_size
            )
        return local_batch_size, global_batch_size, local_batch_tokens, global_batch_tokens

    def log(self, logs, start_time=None):
        logs = dict(logs)
        is_eval_log = any(str(key).startswith("eval_") for key in logs)
        if not is_eval_log:
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
            if global_batch_size is not None and not is_eval_log:
                logs["global_effective_batch_size"] = global_batch_size

            global_batch_tokens = getattr(self, "_latest_global_batch_tokens", None)
            if global_batch_tokens is not None and not is_eval_log:
                logs["effective_batch_tokens"] = getattr(
                    self, "_latest_local_batch_tokens", global_batch_tokens
                )
                logs["global_effective_batch_tokens"] = global_batch_tokens

            if not is_eval_log:
                talker_lr = self._get_named_param_group_lr("talker")
                if talker_lr is not None:
                    logs["talker_learning_rate"] = talker_lr

        return super().log(logs, start_time=start_time)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to extract and log text_loss and length_loss.
        """
        is_training_step = bool(model.training)
        if is_training_step:
            (
                local_batch_size,
                global_batch_size,
                local_batch_tokens,
                global_batch_tokens,
            ) = self._update_total_samples_seen(inputs)
        else:
            local_batch_size, global_batch_size = None, None
            local_batch_tokens, global_batch_tokens = None, None

        outputs = model(**inputs)

        # Extract and globally normalize the model's summed task losses.
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
        else:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        if loss is None:
            raise RuntimeError("model did not return a loss")
        loss = self._normalize_model_loss(
            model, outputs, loss, reduce_across_processes=is_training_step
        )

        if is_training_step:
            text_for_skip = self._output_value(outputs, "text_ce_loss")
            if text_for_skip is None:
                text_for_skip = self._output_value(outputs, "text_token_loss")
            local_nonfinite = not bool(torch.isfinite(loss.detach()).all().item())
            local_spike = False
            if (
                text_for_skip is not None
                and torch.is_tensor(text_for_skip)
                and getattr(self.state, "global_step", 0) > _TEXT_LOSS_SKIP_AFTER_STEP
            ):
                local_spike = bool(
                    torch.isfinite(text_for_skip.detach()).all().item()
                    and (text_for_skip.detach() > _TEXT_LOSS_SKIP_THRESHOLD).any().item()
                )

            skip_flags = torch.tensor(
                [int(local_nonfinite), int(local_spike)],
                device=loss.device,
                dtype=torch.int32,
            )
            skip_flags = self._distributed_sum(skip_flags)
            if bool(skip_flags.any().item()):
                if int(skip_flags[0].item()) > 0:
                    self._nonfinite_loss_count = int(
                        getattr(self, "_nonfinite_loss_count", 0)
                    ) + 1
                if self.args.process_index == 0:
                    logger.warning(
                        "Skipping micro-step %s before backward: "
                        "nonfinite_ranks=%d, spike_ranks=%d",
                        getattr(self.state, "global_step", 0),
                        int(skip_flags[0].item()),
                        int(skip_flags[1].item()),
                    )
                loss = self._zero_loss_from_model(model, loss)

        # Log metrics to swanlab if available (only on main process)
        if is_training_step and SWANLAB_AVAILABLE and self.args.process_index == 0:
            metrics_to_log = {}
            base = self._unwrap_model(model)
            w = float(getattr(getattr(base, "config", None), "text_loss_weight", 1.0) or 1.0)

            if isinstance(outputs, dict):
                text_ce = outputs.get("text_ce_loss")
                text_tok_existing = outputs.get("text_token_loss")
            else:
                text_ce = getattr(outputs, "text_ce_loss", None)
                text_tok_existing = getattr(outputs, "text_token_loss", None)

            if text_ce is not None and torch.is_tensor(text_ce):
                metrics_to_log[f"text_ce_loss"] = text_ce.item()
                metrics_to_log[f"text_token_loss"] = text_ce.item() * w
            elif text_tok_existing is not None and torch.is_tensor(text_tok_existing):
                metrics_to_log[f"text_token_loss"] = text_tok_existing.item()
            if hasattr(outputs, "length_loss") and outputs.length_loss is not None:
                metrics_to_log[f"length_loss"] = outputs.length_loss.item()
            
            if hasattr(outputs, "audio_token_loss") and outputs.audio_token_loss is not None:
                metrics_to_log[f"audio_token_loss"] = outputs.audio_token_loss.item()

            # Depth transformer (acoustic) losses
            if hasattr(outputs, "acoustic_loss") and outputs.acoustic_loss is not None:
                metrics_to_log[f"acoustic_loss"] = outputs.acoustic_loss.item()
            if hasattr(outputs, "acoustic_ce_loss") and outputs.acoustic_ce_loss is not None:
                metrics_to_log[f"acoustic_ce_loss"] = outputs.acoustic_ce_loss.item()
            if hasattr(outputs, "acoustic_per_codebook_loss") and outputs.acoustic_per_codebook_loss is not None:
                per_q = outputs.acoustic_per_codebook_loss
                if torch.is_tensor(per_q):
                    for q_idx, val in enumerate(per_q.tolist()):
                        metrics_to_log[f"acoustic_loss_q{q_idx}"] = val
            
            if hasattr(outputs, "loss_text_only_data") and outputs.loss_text_only_data is not None:
                metrics_to_log[f"loss_text_only_data"] = outputs.loss_text_only_data.item()
            
            if hasattr(outputs, "loss_audio_dialog_data") and outputs.loss_audio_dialog_data is not None:
                metrics_to_log[f"loss_audio_dialog_data"] = outputs.loss_audio_dialog_data.item()

            if local_batch_size is not None:
                metrics_to_log["effective_batch_size"] = local_batch_size
                metrics_to_log["global_effective_batch_size"] = global_batch_size
                metrics_to_log["total_samples_seen"] = getattr(self, "_total_samples_seen", 0)
            if local_batch_tokens is not None:
                metrics_to_log["effective_batch_tokens"] = local_batch_tokens
                metrics_to_log["global_effective_batch_tokens"] = global_batch_tokens

            # Log metrics only when SwanLab is enabled for this run.
            report_to = getattr(self.args, "report_to", []) if hasattr(self, "args") else []
            if isinstance(report_to, str):
                report_to = [report_to]
            use_swanlab = SWANLAB_AVAILABLE and any(
                str(x).strip().lower() == "swanlab" for x in report_to
            )

            if metrics_to_log and use_swanlab:
                # Gradient accumulation can call compute_loss multiple times before
                # global_step advances. SwanLab accepts the first value for a key at
                # a step and warns for every duplicate, so emit custom metrics only
                # once per training step.
                step = None
                if hasattr(self, "state") and hasattr(self.state, "global_step"):
                    step = self.state.global_step
                metric_log_id = step
                if getattr(self, "_last_swanlab_metric_log_id", None) != metric_log_id:
                    swanlab.log(metrics_to_log, step=step)
                    self._last_swanlab_metric_log_id = metric_log_id
        
        return (loss, outputs) if return_outputs else loss
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

        loss = super().training_step(model, inputs, num_items_in_batch)

        if loss is None:
            raise RuntimeError("training_step returned None")

        return loss

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
