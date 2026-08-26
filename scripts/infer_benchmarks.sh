#!/usr/bin/env bash
# Run VoiceBench/OpenAudioBench s2s and LibriSpeech ASR jobs.
#
# CONFIG is required (repo-relative or absolute):
#   CONFIG=config/infer_benchmarks_12_5hz.yaml bash scripts/infer_benchmarks.sh
#   CONFIG=config/infer_benchmarks_6_25hz.yaml bash scripts/infer_benchmarks.sh
#
# GPUs: uses every device in CUDA_VISIBLE_DEVICES when set, otherwise all
# nvidia-smi GPUs. One job per GPU per wave; runtime.devices is set by this
# launcher (do not put devices in the YAML).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "${CONFIG:-}" ]]; then
    echo "Error: set CONFIG to an inference manifest, e.g." >&2
    echo "  CONFIG=config/infer_benchmarks_12_5hz.yaml bash scripts/infer_benchmarks.sh" >&2
    echo "  CONFIG=config/infer_benchmarks_6_25hz.yaml bash scripts/infer_benchmarks.sh" >&2
    exit 1
fi
if [[ "$CONFIG" != /* ]]; then
    CONFIG="$REPO_ROOT/$CONFIG"
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "Error: inference config not found: $CONFIG" >&2
    exit 1
fi

MODEL_ROOT=${MODEL_ROOT:-$REPO_ROOT/models}
LOG_ROOT=${LOG_ROOT:-$REPO_ROOT/logs/evaluation/inference/benchmarks}
TRANSCRIBE_MODEL_PATH=${TRANSCRIBE_MODEL_PATH:-$MODEL_ROOT/whisper-large-v3}
export REPO_ROOT MODEL_ROOT LOG_ROOT TRANSCRIBE_MODEL_PATH

if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    IFS=',' read -ra VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES// /}"
    GPU_COUNT=0
    for gpu in "${VISIBLE_GPUS[@]}"; do
        if [[ -n "$gpu" && "$gpu" != -1 ]]; then
            GPU_COUNT=$((GPU_COUNT + 1))
        fi
    done
else
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
fi
if (( GPU_COUNT == 0 )); then
    echo "Error: benchmark inference requires a CUDA GPU." >&2
    exit 1
fi

RUNTIME_DEVICES=()
for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
    RUNTIME_DEVICES+=("cuda:$gpu")
done
printf 'Inference devices: %s\n' "${RUNTIME_DEVICES[*]}"
printf 'Inference config: %s\n' "$CONFIG"
printf 'Model root: %s\n' "$MODEL_ROOT"
printf 'Whisper transcribe model: %s\n' "$TRANSCRIBE_MODEL_PATH"

TMP_CONFIG_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_CONFIG_DIR"' EXIT

python - "$CONFIG" "$TMP_CONFIG_DIR" "${RUNTIME_DEVICES[@]}" <<'PY'
import os
import sys
from pathlib import Path

import yaml

manifest_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
devices = sys.argv[3:]
manifest = yaml.safe_load(manifest_path.read_text())


def expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value


for index, job in enumerate(manifest["jobs"]):
    config = expand(job["config"])
    config.setdefault("runtime", {})["devices"] = [devices[index % len(devices)]]
    output = output_dir / f"{index:02d}_{job['name']}.yaml"
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Prepared {job['name']} on {config['runtime']['devices'][0]}")
PY

cd "$REPO_ROOT"
mapfile -t JOB_CONFIGS < <(find "$TMP_CONFIG_DIR" -maxdepth 1 -name '*.yaml' -type f | sort)
overall_status=0
for ((wave_start = 0; wave_start < ${#JOB_CONFIGS[@]}; wave_start += GPU_COUNT)); do
    pids=()
    wave_end=$((wave_start + GPU_COUNT))
    if (( wave_end > ${#JOB_CONFIGS[@]} )); then
        wave_end=${#JOB_CONFIGS[@]}
    fi
    for ((i = wave_start; i < wave_end; i++)); do
        echo "Launching $(basename "${JOB_CONFIGS[$i]}" .yaml)"
        python -m src.infer "${JOB_CONFIGS[$i]}" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || overall_status=$?
    done
done
exit "$overall_status"
