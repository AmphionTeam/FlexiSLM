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
import json
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


def strip_and_export(src: Path, dst: Path, *, weights_only: bool = False) -> None:
    src = src.resolve()
    dst = dst.resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {src}")

    weight_path = src / "model.safetensors"
    index_path = src / "model.safetensors.index.json"
    if not index_path.is_file() and not weight_path.is_file():
        raise FileNotFoundError(f"missing model.safetensors in {src}")

    dst.mkdir(parents=True, exist_ok=True)

    source_weights = [weight_path]
    index = None
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        source_weights = [
            src / name for name in sorted(set(index["weight_map"].values()))
        ]

    stripped = []
    total_size = 0
    for source_weight in source_weights:
        state = st.load_file(str(source_weight), device="cpu")
        kept = {}
        for key, tensor in state.items():
            if any(key.startswith(p) for p in STRIP_PREFIXES):
                stripped.append(key)
                continue
            kept[key] = tensor
            total_size += tensor.nbytes
        st.save_file(kept, str(dst / source_weight.name))

    if index is not None:
        index["weight_map"] = {
            key: filename
            for key, filename in index["weight_map"].items()
            if key not in stripped
        }
        index.setdefault("metadata", {})["total_size"] = total_size
        (dst / index_path.name).write_text(json.dumps(index, indent=2) + "\n")

    copied = []
    if not weights_only:
        for name in sorted(INFERENCE_COPY_NAMES):
            path = src / name
            if path.is_file():
                shutil.copy2(path, dst / name)
                copied.append(name)

    print(f"source: {src}")
    print(f"dest:   {dst}")
    print(f"kept tensors:     {len(index['weight_map']) if index else len(kept)}")
    print(f"stripped tensors: {len(stripped)} ({', '.join(STRIP_PREFIXES)})")
    if stripped:
        print(f"  first stripped: {stripped[0]}")
        print(f"  last stripped:  {stripped[-1]}")
    print(f"copied sidecars:  {copied}")
    print(
        f"output size:      "
        f"{sum(path.stat().st_size for path in dst.glob('*.safetensors')) / 1e9:.3f} GB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path, help="Training checkpoint directory")
    parser.add_argument("output_dir", type=Path, help="HF-ready export directory")
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Export only safetensors weight files and their shard index.",
    )
    args = parser.parse_args()
    strip_and_export(args.checkpoint_dir, args.output_dir, weights_only=args.weights_only)


if __name__ == "__main__":
    main()
