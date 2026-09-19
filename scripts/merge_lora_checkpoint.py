#!/usr/bin/env python3
"""Merge PEFT LoRA weights into a full-parameter FlexiSLM checkpoint.

Stage 2 saves PEFT key names such as
``model.base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight``.
Stage 3 trains without LoRA, so those adapters must be merged into the base
Thinker weights first.

Example:
  python scripts/merge_lora_checkpoint.py \\
    --input /path/to/checkpoint-150000 \\
    --output /path/to/checkpoint-150000_merged_lora
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from accelerate import init_empty_weights
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.modeling_flexislm import ParallelS2SConfig, ParallelS2SForCausalLM


def _load_state_dict_from_checkpoint(checkpoint: str) -> dict[str, torch.Tensor]:
    checkpoint_path = Path(checkpoint)
    st_index = checkpoint_path / "model.safetensors.index.json"
    st_single = checkpoint_path / "model.safetensors"
    bin_index = checkpoint_path / "pytorch_model.bin.index.json"
    bin_single = checkpoint_path / "pytorch_model.bin"

    def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
        from safetensors import safe_open

        selected: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                selected[key] = handle.get_tensor(key)
        return selected

    if st_index.is_file():
        with st_index.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map", index)
        state_dict: dict[str, torch.Tensor] = {}
        for shard_file in sorted(set(weight_map.values())):
            shard_path = checkpoint_path / shard_file
            if not shard_path.is_file():
                raise FileNotFoundError(f"Shard not found: {shard_path}")
            state_dict.update(load_safetensors(shard_path))
        logger.info(
            "Loaded {} sharded safetensors weights from {} shard(s)",
            len(state_dict),
            len(set(weight_map.values())),
        )
        return state_dict

    if st_single.is_file():
        return load_safetensors(st_single)

    if bin_index.is_file():
        with bin_index.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map", index)
        state_dict = {}
        for shard_file in sorted(set(weight_map.values())):
            shard_path = checkpoint_path / shard_file
            if not shard_path.is_file():
                raise FileNotFoundError(f"Shard not found: {shard_path}")
            state_dict.update(torch.load(shard_path, map_location="cpu"))
        return state_dict

    if bin_single.is_file():
        return torch.load(bin_single, map_location="cpu")

    raise FileNotFoundError(
        f"No supported model weights found in {checkpoint_path} "
        "(expected model.safetensors or sharded safetensors/pytorch checkpoints)"
    )


def _infer_lora_config(
    state_dict: dict[str, torch.Tensor],
    saved_config: ParallelS2SConfig,
) -> LoraConfig:
    lora_a_keys = [key for key in state_dict if ".lora_A." in key]
    if not lora_a_keys:
        raise ValueError(
            "Checkpoint config has use_lora=True but no LoRA tensors were found."
        )

    target_modules = sorted(
        {
            key.split(".lora_A.")[0].rsplit(".", 1)[-1]
            for key in lora_a_keys
        }
    )
    lora_rank = int(state_dict[lora_a_keys[0]].shape[0])
    lora_alpha = int(getattr(saved_config, "lora_alpha", lora_rank * 2) or lora_rank * 2)
    logger.info(
        "Detected LoRA: rank={}, alpha={}, target_modules={}",
        lora_rank,
        lora_alpha,
        target_modules,
    )
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
        bias="none",
        modules_to_save=None,
    )


def _materialize_meta_parameters(model: ParallelS2SForCausalLM) -> None:
    text_slice_raw = getattr(model, "_combined_embed_proj_text_slice", 0)
    text_slice = int(text_slice_raw or 0)
    for mod_name, module in model.named_modules():
        for pname, param in list(module._parameters.items()):
            if param is None or not param.is_meta:
                continue
            full_name = f"{mod_name}.{pname}" if mod_name else pname
            logger.info("Materializing missing meta parameter: {}", full_name)
            new_tensor = torch.empty(param.shape, dtype=param.dtype, device="cpu")
            with torch.no_grad():
                if full_name == "combined_embed_proj.weight" and text_slice > 0:
                    new_tensor.zero_()
                    new_tensor[:, :text_slice].copy_(torch.eye(text_slice, dtype=param.dtype))
                else:
                    torch.nn.init.normal_(new_tensor, mean=0.0, std=0.02)
            module._parameters[pname] = torch.nn.Parameter(
                new_tensor, requires_grad=param.requires_grad
            )

        for bname, buf in list(module._buffers.items()):
            if buf is None or not buf.is_meta:
                continue
            full_name = f"{mod_name}.{bname}" if mod_name else bname
            logger.info("Materializing missing meta buffer: {}", full_name)
            module._buffers[bname] = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu")


def merged_checkpoint_is_ready(checkpoint: str | Path) -> bool:
    checkpoint_path = Path(checkpoint)
    if not (checkpoint_path / "config.json").is_file():
        return False
    has_weights = (
        (checkpoint_path / "model.safetensors").is_file()
        or (checkpoint_path / "model.safetensors.index.json").is_file()
    )
    if not has_weights:
        return False
    with (checkpoint_path / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("use_lora") is not False:
        return False
    index_path = checkpoint_path / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as handle:
            keys = json.load(handle).get("weight_map", {})
        wrapped = [
            key
            for key in keys
            if "lora_" in key or ".base_layer." in key or ".base_model.model." in key
        ]
        if wrapped:
            return False
    return True


def merge_lora_checkpoint(
    input_dir: str,
    output_dir: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input checkpoint directory not found: {input_path}")

    saved_config = ParallelS2SConfig.from_pretrained(str(input_path))
    if not getattr(saved_config, "use_lora", False):
        logger.info(
            "Input config already has use_lora=False; using {} as the merged checkpoint",
            input_path,
        )
        if output_path.resolve() != input_path.resolve():
            if output_path.exists():
                shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            keep = {
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
                "added_tokens.json",
                "chat_template.jinja",
                "model.safetensors",
                "model.safetensors.index.json",
            }
            for path in input_path.iterdir():
                if path.name in keep or (
                    path.name.startswith("model-") and path.name.endswith(".safetensors")
                ):
                    shutil.copy2(path, output_path / path.name)
        return

    logger.info("Loading checkpoint weights from {}", input_path)
    state_dict = _load_state_dict_from_checkpoint(str(input_path))
    lora_keys = [key for key in state_dict if "lora_" in key]
    logger.info("Found {} LoRA tensors in the checkpoint", len(lora_keys))

    lora_config = _infer_lora_config(state_dict, saved_config)

    logger.info("Building model skeleton and applying LoRA wrapper")
    with init_empty_weights():
        model = ParallelS2SForCausalLM._from_config(saved_config, torch_dtype=torch_dtype)

    model.model = get_peft_model(model.model, lora_config)
    load_result = model.load_state_dict(state_dict, strict=False, assign=True)
    logger.info(
        "Loaded checkpoint into LoRA model (missing={}, unexpected={})",
        len(load_result.missing_keys),
        len(load_result.unexpected_keys),
    )

    _materialize_meta_parameters(model)

    logger.info("Merging LoRA adapters into base weights")
    if not hasattr(model.model, "merge_and_unload"):
        raise RuntimeError("Expected PEFT-wrapped model.model with merge_and_unload()")
    model.model = model.model.merge_and_unload()
    model.config.use_lora = False

    merged_state = model.state_dict()
    remaining_lora = [key for key in merged_state if "lora_" in key]
    if remaining_lora:
        raise RuntimeError(
            f"Merge left {len(remaining_lora)} LoRA tensors in the state dict"
        )
    peft_prefix = [key for key in merged_state if key.startswith("model.base_model.")]
    if peft_prefix:
        raise RuntimeError(
            f"Merge left {len(peft_prefix)} PEFT-prefixed tensors in the state dict"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Saving merged checkpoint to {}", output_path)
    model.save_pretrained(str(output_path), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(str(input_path))
    tokenizer.save_pretrained(str(output_path))

    for filename in ("generation_config.json", "chat_template.jinja"):
        src = input_path / filename
        if src.is_file():
            shutil.copy2(src, output_path / filename)

    logger.info("Done. Saved {} merged tensors to {}", len(merged_state), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Stage 2 LoRA weights into base weights for Stage 3 training."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Stage 2 checkpoint directory containing LoRA weights.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to write the merged full-parameter checkpoint.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Torch dtype used while building and saving the merged model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    merge_lora_checkpoint(args.input, args.output, torch_dtype=dtype)
    if not merged_checkpoint_is_ready(args.output):
        raise SystemExit(
            f"Merged checkpoint is missing weights or still contains LoRA: {args.output}"
        )


if __name__ == "__main__":
    main()
