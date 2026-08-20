#!/usr/bin/env bash
# Legacy-aligned evaluation for FlexiSLM Stage2, driven by a plan YAML.
#
#   config/yutong/eval_plan_stage2_legacy_aligned.yaml  (what to evaluate)
#
# Pipeline:
#   1. build requests + infer manifest + eval config (local/build_eval_plan.py)
#   2. split the manifest into per-job configs (one device per job)
#   3. finish inference for every job in device-sized waves
#   4. evaluate every completed trace via `python -m src.eval`
#      (kimi-audio-evalkit env)
#
# Usage:
#   bash scripts/yutong/run_stage2_legacy_aligned.sh [options]
#
# With no options, this runs the full inference-then-evaluation pipeline.
# Options:
#   --limit N               Cap samples per request file (smoke testing).
#   --workers-per-device N  Model replicas per GPU/CPU (default: 1).
#
# WORKERS_PER_DEVICE may also set the replica count. Keeping the default at one
# avoids loading multiple Stage2 + Whisper replicas onto the same GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONDA_ROOT=${CONDA_ROOT:-/F00120260003/flexislm_project/miniconda3}
CONDA_ENV_PATH=${CONDA_ENV_PATH:-$CONDA_ROOT/envs/fslm}
EVALKIT_ENV=${EVALKIT_ENV:-$CONDA_ROOT/envs/kimi-audio-evalkit}
INFER_PYTHON=${INFER_PYTHON:-$CONDA_ENV_PATH/bin/python}
EVAL_PYTHON=${EVAL_PYTHON:-$EVALKIT_ENV/bin/python}
PLAN=${PLAN:-$REPO_ROOT/config/yutong/eval_plan_stage2_legacy_aligned.yaml}

LIMIT=""
WORKERS_PER_DEVICE=${WORKERS_PER_DEVICE:-1}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit)
            [[ $# -ge 2 ]] || { echo "--limit requires a value" >&2; exit 2; }
            LIMIT="$2"
            shift 2
            ;;
        --workers-per-device)
            [[ $# -ge 2 ]] || {
                echo "--workers-per-device requires a value" >&2
                exit 2
            }
            WORKERS_PER_DEVICE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done
if [[ ! "$WORKERS_PER_DEVICE" =~ ^[1-9][0-9]*$ ]]; then
    echo "workers per device must be a positive integer" >&2
    exit 2
fi
if [[ -n "$LIMIT" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "--limit must be a positive integer" >&2
    exit 2
fi

if [[ ! -x "$INFER_PYTHON" ]]; then
    echo "Inference Python is not executable: $INFER_PYTHON" >&2
    exit 2
fi
if [[ ! -x "$EVAL_PYTHON" ]]; then
    echo "Evalkit Python is not executable: $EVAL_PYTHON" >&2
    exit 2
fi

# Invoke each environment's interpreter directly. This is more reliable for
# detached runs than `conda activate`, which can block while evaluating hooks.
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME=${HF_HOME:-/F00120260003/flexislm_project/yutong/models/huggingface}

GPU_COUNT=0
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    IFS=',' read -ra VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES// /}"
    for gpu in "${VISIBLE_GPUS[@]}"; do
        if [[ -n "$gpu" && "$gpu" != -1 ]]; then
            GPU_COUNT=$((GPU_COUNT + 1))
        fi
    done
else
    mapfile -t VISIBLE_GPUS < <(
        nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true
    )
    GPU_COUNT=${#VISIBLE_GPUS[@]}
fi
RUNTIME_DEVICES=()
if (( GPU_COUNT > 0 )); then
    for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
        RUNTIME_DEVICES+=("cuda:$gpu")
    done
else
    RUNTIME_DEVICES+=("cpu")
fi
printf 'Inference devices detected by launcher: %s\n' "${RUNTIME_DEVICES[*]}"
printf 'Model replicas per device: %s\n' "$WORKERS_PER_DEVICE"

# --- Step 1: build plan (requests + infer manifest + eval config) ------------
cd "$REPO_ROOT"
LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
    LIMIT_ARGS=(--limit "$LIMIT")
fi
"$INFER_PYTHON" local/build_eval_plan.py \
    --plan "$PLAN" \
    "${LIMIT_ARGS[@]}"

OUTPUT_ROOT=$("$INFER_PYTHON" - "$PLAN" <<'PY'
import sys
from pathlib import Path

import yaml

plan = yaml.safe_load(Path(sys.argv[1]).read_text())
print(Path(plan["output_root"]).expanduser().resolve())
PY
)
MANIFEST="$OUTPUT_ROOT/infer_manifest.yaml"
EVAL_CONFIG="$OUTPUT_ROOT/eval_config.yaml"
printf 'Output root: %s\n' "$OUTPUT_ROOT"

# --- Step 2: split the manifest into per-job configs (one device each) -------
TMP_CONFIG_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_CONFIG_DIR"' EXIT
"$INFER_PYTHON" - "$MANIFEST" "$TMP_CONFIG_DIR" "$WORKERS_PER_DEVICE" \
    "${RUNTIME_DEVICES[@]}" <<'PY'
import sys
from pathlib import Path

import yaml

manifest_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
workers_per_device = int(sys.argv[3])
devices = sys.argv[4:]
if not devices:
    raise ValueError("The launcher did not provide any inference devices")
manifest = yaml.safe_load(manifest_path.read_text())
# Assign devices in the same (lexicographic) order the wave loop below runs
# jobs, so each wave spreads one job per GPU instead of colliding several
# jobs onto the same device.
jobs = sorted(manifest.get("jobs", []), key=lambda job: job["name"])
for index, job in enumerate(jobs):
    runtime = job["config"].setdefault("runtime", {})
    runtime["devices"] = [devices[index % len(devices)]]
    runtime["workers_per_device"] = workers_per_device
    output = output_dir / f"{job['name']}.yaml"
    output.write_text(yaml.safe_dump(job["config"], sort_keys=False))
    print(
        f"Prepared job {job['name']} on {devices[index % len(devices)]} "
        f"with {workers_per_device} worker(s): {output}"
    )
PY

# --- Step 3: run inference in device-sized waves ------------------------------
cd "$REPO_ROOT"
WAVE_SIZE=${#RUNTIME_DEVICES[@]}
JOB_NAMES=()
for job_yaml in "$TMP_CONFIG_DIR"/*.yaml; do
    JOB_NAMES+=("$(basename "$job_yaml" .yaml)")
done
overall_status=0
wave_start=0
while (( wave_start < ${#JOB_NAMES[@]} )); do
    pids=()
    wave_end=$((wave_start + WAVE_SIZE))
    if (( wave_end > ${#JOB_NAMES[@]} )); then
        wave_end=${#JOB_NAMES[@]}
    fi
    echo "Launching wave: ${JOB_NAMES[*]:wave_start:wave_end - wave_start}"
    for ((i = wave_start; i < wave_end; i++)); do
        "$INFER_PYTHON" -m src.infer "$TMP_CONFIG_DIR/${JOB_NAMES[$i]}.yaml" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || overall_status=$?
    done
    wave_start=$wave_end
done
if (( overall_status != 0 )); then
    echo "Inference finished with non-zero status: $overall_status" >&2
    exit "$overall_status"
fi

# --- Step 4: evaluate ----------------------------------------------------------
export PATH="$EVALKIT_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$EVALKIT_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME=${HF_HOME:-/F00120260003/flexislm_project/yutong/models/huggingface}
cd "$REPO_ROOT"
"$EVAL_PYTHON" -m src.eval "$EVAL_CONFIG"
