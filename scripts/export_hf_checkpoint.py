#!/usr/bin/env python3
"""Export a training checkpoint for Hugging Face upload.

Strips Qwen2.5-Omni encoder tensors (``_qwen25o_encoder.*``). Those weights are
loaded at inference from the separate ``FlexiSLM/Qwen2_5-Omni-Audio_Encoder``
repo and are never applied from the FlexiSLM checkpoint (``strict=False`` /
unexpected keys). Uploading them is redundant (~1.3GB BF16).

Also skips trainer/optimizer resume artifacts.

Example:
  python \\
    scripts/export_hf_checkpoint.py \\
    /path/to/checkpoint-75000 \\
    /path/to/hf_export_dir
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import safetensors.torch as st

# Loaded separately via qwen25o_encoder_path; never applied from this checkpoint.
STRIP_PREFIXES = ("_qwen25o_encoder.",)

# Training resume only; not needed for Hub inference repos.
SKIP_NAMES = {
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
}
SKIP_PREFIXES = ("rng_state_", "native_dataloader_state_")

INFERENCE_COPY_NAMES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
}


def _should_skip_sidecar(name: str) -> bool:
    if name in SKIP_NAMES:
        return True
    return any(name.startswith(p) for p in SKIP_PREFIXES)


def strip_and_export(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst = dst.resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {src}")

    weight_path = src / "model.safetensors"
    index_path = src / "model.safetensors.index.json"
    if index_path.is_file():
        raise NotImplementedError(
            "Sharded checkpoints are not supported yet; consolidate to model.safetensors first."
        )
    if not weight_path.is_file():
        raise FileNotFoundError(f"missing model.safetensors in {src}")

    dst.mkdir(parents=True, exist_ok=True)

    state = st.load_file(str(weight_path), device="cpu")
    kept = {}
    stripped = []
    for key, tensor in state.items():
        if any(key.startswith(p) for p in STRIP_PREFIXES):
            stripped.append(key)
            continue
        kept[key] = tensor

    out_weights = dst / "model.safetensors"
    st.save_file(kept, str(out_weights))

    copied = []
    for name in sorted(INFERENCE_COPY_NAMES):
        path = src / name
        if path.is_file():
            shutil.copy2(path, dst / name)
            copied.append(name)

    print(f"source: {src}")
    print(f"dest:   {dst}")
    print(f"kept tensors:     {len(kept)}")
    print(f"stripped tensors: {len(stripped)} ({', '.join(STRIP_PREFIXES)})")
    if stripped:
        print(f"  first stripped: {stripped[0]}")
        print(f"  last stripped:  {stripped[-1]}")
    print(f"copied sidecars:  {copied}")
    print(f"output size:      {out_weights.stat().st_size / 1e9:.3f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path, help="Training checkpoint directory")
    parser.add_argument("output_dir", type=Path, help="HF-ready export directory")
    args = parser.parse_args()
    strip_and_export(args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()
