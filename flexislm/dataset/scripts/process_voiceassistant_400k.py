#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""Process VoiceAssistant-400K JSONL and normalize audio paths."""
import argparse
import json
import os

DEFAULT_INPUT = os.environ.get("VOICEASSISTANT_INPUT_JSONL", "data/VoiceAssistant-400K/data.jsonl")
DEFAULT_AUDIO_ROOT = os.environ.get("VOICEASSISTANT_AUDIO_ROOT", "data/VoiceAssistant-400K/audio")
DEFAULT_OUTPUT = os.environ.get(
    "VOICEASSISTANT_OUTPUT_JSONL",
    "flexislm/dataset/voiceassistant_400k_processed.jsonl",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input VoiceAssistant JSONL path")
    parser.add_argument("--audio-root", default=DEFAULT_AUDIO_ROOT, help="Audio root for relative paths")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    count = 0
    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            messages = obj.get("messages", [])
            audios = obj.get("audios", [])
            # Remove system message (first message if role is system)
            if messages and messages[0].get("role") == "system":
                messages = messages[1:]
            # Absolute audio paths
            audios_abs = []
            for p in audios:
                path = p.replace("\\/", "/") if isinstance(p, str) else p
                if path and not os.path.isabs(path):
                    path = os.path.join(args.audio_root, path)
                audios_abs.append(path)
            out = {"messages": messages, "audios": audios_abs}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} samples to {args.output}")

if __name__ == "__main__":
    main()
