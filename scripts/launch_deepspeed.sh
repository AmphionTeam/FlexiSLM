#!/usr/bin/env bash
# Launch an arbitrary training command with DeepSpeed, using all hostfile GPUs by default.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEEPSPEED_HOSTFILE="${DEEPSPEED_HOSTFILE:-${HOSTFILE:-/opt/kube/hostfile}}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python 2>/dev/null || true)}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-${PYTHON_BIN:+$(dirname "$PYTHON_BIN")/deepspeed}}"
DEEPSPEED_TIMEOUT_SECONDS="${DEEPSPEED_TIMEOUT_SECONDS:-604800}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEEPSPEED_LOG_DIR="${DEEPSPEED_LOG_DIR:-${OUTPUT_DIR_BASE:-$REPO_ROOT/outputs}/launcher_logs/$TIMESTAMP}"
DEEPSPEED_LOG_PATH="${DEEPSPEED_LOG_PATH:-$DEEPSPEED_LOG_DIR/deepspeed.log}"
DEEPSPEED_NUM_NODES="${DEEPSPEED_NUM_NODES:-}"
DEEPSPEED_NUM_GPUS="${DEEPSPEED_NUM_GPUS:-}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

[ "$#" -gt 0 ] || fail "usage: $0 <python-script> [arguments...]"
[ -x "$PYTHON_BIN" ] || fail "Python is not executable; set PYTHON_BIN"
if [ ! -x "$DEEPSPEED_BIN" ]; then
    DEEPSPEED_BIN="$(command -v deepspeed 2>/dev/null || true)"
fi
[ -n "$DEEPSPEED_BIN" ] && [ -x "$DEEPSPEED_BIN" ] \
    || fail "deepspeed executable was not found; set DEEPSPEED_BIN"
command -v timeout >/dev/null 2>&1 || fail "GNU timeout is required"
[[ "$DEEPSPEED_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] \
    || fail "DEEPSPEED_TIMEOUT_SECONDS must be a positive integer"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
mkdir -p "$DEEPSPEED_LOG_DIR"
cd "$REPO_ROOT"

LAUNCHER_ARGS=()
HOST_LINES=()
if [ -r "$DEEPSPEED_HOSTFILE" ]; then
    mapfile -t HOST_LINES < <(awk 'NF && $1 !~ /^#/ {print}' "$DEEPSPEED_HOSTFILE")
    [ "${#HOST_LINES[@]}" -gt 0 ] || fail "hostfile has no usable entries: $DEEPSPEED_HOSTFILE"

    total_slots=0
    for line in "${HOST_LINES[@]}"; do
        host="$(awk '{print $1}' <<<"$line")"
        slots="$(awk '{for (i=2; i<=NF; i++) if ($i ~ /^slots=[0-9]+$/) {sub(/^slots=/, "", $i); print $i; exit}}' <<<"$line")"
        [ -n "$slots" ] && [ "$slots" -gt 0 ] \
            || fail "hostfile entry has no positive slots=N value: $line"
        total_slots=$((total_slots + slots))
    done

    if [ -n "$DEEPSPEED_NUM_NODES" ]; then
        [[ "$DEEPSPEED_NUM_NODES" =~ ^[1-9][0-9]*$ ]] \
            || fail "DEEPSPEED_NUM_NODES must be a positive integer"
        [ "$DEEPSPEED_NUM_NODES" -le "${#HOST_LINES[@]}" ] \
            || fail "requested $DEEPSPEED_NUM_NODES nodes, but hostfile has ${#HOST_LINES[@]}"
        LAUNCHER_ARGS+=("--num_nodes=$DEEPSPEED_NUM_NODES")
    fi
    if [ -n "$DEEPSPEED_NUM_GPUS" ]; then
        [[ "$DEEPSPEED_NUM_GPUS" =~ ^[1-9][0-9]*$ ]] \
            || fail "DEEPSPEED_NUM_GPUS must be a positive integer"
        selected_nodes="${DEEPSPEED_NUM_NODES:-${#HOST_LINES[@]}}"
        for ((index = 0; index < selected_nodes; index++)); do
            slots="$(awk '{for (i=2; i<=NF; i++) if ($i ~ /^slots=[0-9]+$/) {sub(/^slots=/, "", $i); print $i; exit}}' <<<"${HOST_LINES[$index]}")"
            [ "$slots" -ge "$DEEPSPEED_NUM_GPUS" ] \
                || fail "hostfile entry has $slots slots, but $DEEPSPEED_NUM_GPUS GPUs were requested: ${HOST_LINES[$index]}"
        done
        LAUNCHER_ARGS+=("--num_gpus=$DEEPSPEED_NUM_GPUS")
    fi
    LAUNCHER_ARGS=("--hostfile=$DEEPSPEED_HOSTFILE" "${LAUNCHER_ARGS[@]}")
    echo "DeepSpeed hostfile: $DEEPSPEED_HOSTFILE (${#HOST_LINES[@]} nodes, $total_slots total slots)"
else
    [ -z "$DEEPSPEED_NUM_NODES" ] || [ "$DEEPSPEED_NUM_NODES" = 1 ] \
        || fail "multi-node DeepSpeed requires a readable hostfile: $DEEPSPEED_HOSTFILE"
    if [ -z "$DEEPSPEED_NUM_GPUS" ]; then
        if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
            IFS=',' read -r -a visible_devices <<<"${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
            DEEPSPEED_NUM_GPUS="${#visible_devices[@]}"
        else
            DEEPSPEED_NUM_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | xargs)"
        fi
    fi
    [ "$DEEPSPEED_NUM_GPUS" -gt 0 ] || fail "no local GPUs detected"
    LAUNCHER_ARGS=(--num_nodes=1 "--num_gpus=$DEEPSPEED_NUM_GPUS")
    echo "No readable hostfile found; using $DEEPSPEED_NUM_GPUS local GPUs"
fi

echo "DeepSpeed log: $DEEPSPEED_LOG_PATH"
COMMAND=("$DEEPSPEED_BIN" "${LAUNCHER_ARGS[@]}" "$@")
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

# Launch one --no_ssh coordinator per host so recipe-specific environment
# variables are propagated consistently without relying on pdsh configuration.
selected_nodes="${DEEPSPEED_NUM_NODES:-${#HOST_LINES[@]}}"
if [ "${#HOST_LINES[@]}" -le 1 ]; then
    set +e
    timeout --signal=TERM --kill-after=30s "${DEEPSPEED_TIMEOUT_SECONDS}s" \
        "${COMMAND[@]}" 2>&1 | tee "$DEEPSPEED_LOG_PATH"
    status=${PIPESTATUS[0]}
    set -e
else
    command -v ssh >/dev/null 2>&1 || fail "multi-node launch requires ssh"
    command -v scp >/dev/null 2>&1 || fail "multi-node launch requires scp"
    [ "$DEEPSPEED_TIMEOUT_SECONDS" -gt 60 ] \
        || fail "DEEPSPEED_TIMEOUT_SECONDS must be greater than 60 for SSH launch"

    remote_timeout=$((DEEPSPEED_TIMEOUT_SECONDS - 30))
    master_addr="$(awk '{print $1}' <<<"${HOST_LINES[0]}")"
    master_port="${MASTER_PORT:-29500}"
    remote_hostfile="/tmp/flexislm_deepspeed_${TIMESTAMP}.hostfile"
    echo "Launching $selected_nodes nodes with DeepSpeed --no_ssh"
    echo "Rendezvous: $master_addr:$master_port"

    export_names="${DEEPSPEED_EXPORT_ENV_VARS:-ASR_TTS_DATA_ROOT ASR_TTS_TRAIN_SHARDS S2S_DATA_ROOT S2S_TRAIN_SHARDS WDS_QUARANTINE_PATH SKIP_FINAL_SAVE_MODEL SWANLAB_API_KEY SWANLAB_PROJECT SWANLAB_MODE SWANLAB_LOG_DIR SWANLAB_SAVE_DIR PYTORCH_CUDA_ALLOC_CONF FLEXISLM_PROFILE_DIR FLEXISLM_PROFILE_WAIT_STEPS FLEXISLM_PROFILE_WARMUP_STEPS FLEXISLM_PROFILE_ACTIVE_STEPS HF_HOME TRANSFORMERS_CACHE}"
    pids=()
    node_logs=()
    for ((node_rank = 0; node_rank < selected_nodes; node_rank++)); do
        host="$(awk '{print $1}' <<<"${HOST_LINES[$node_rank]}")"
        node_log="$DEEPSPEED_LOG_DIR/node${node_rank}.log"
        node_logs+=("$node_log")

        timeout --signal=TERM --kill-after=5s 30s \
            scp -q -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no \
            "$DEEPSPEED_HOSTFILE" "$host:$remote_hostfile" \
            || fail "failed to copy hostfile to $host"

        remote_env=(
            env
            "PATH=$PATH"
            "PYTHONPATH=$PYTHONPATH"
            "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
            "OMP_NUM_THREADS=$OMP_NUM_THREADS"
            "NCCL_DEBUG=$NCCL_DEBUG"
            "TORCH_NCCL_ASYNC_ERROR_HANDLING=$TORCH_NCCL_ASYNC_ERROR_HANDLING"
        )
        for env_name in $export_names; do
            if [[ -v "$env_name" ]]; then
                remote_env+=("$env_name=${!env_name}")
            fi
        done
        remote_args=(
            "${remote_env[@]}"
            timeout --signal=TERM --kill-after=30s "${remote_timeout}s"
            "$DEEPSPEED_BIN"
            "--hostfile=$remote_hostfile"
        )
        [ -z "$DEEPSPEED_NUM_NODES" ] || remote_args+=("--num_nodes=$DEEPSPEED_NUM_NODES")
        [ -z "$DEEPSPEED_NUM_GPUS" ] || remote_args+=("--num_gpus=$DEEPSPEED_NUM_GPUS")
        remote_args+=(
            --no_ssh
            "--node_rank=$node_rank"
            "--master_addr=$master_addr"
            "--master_port=$master_port"
            "$@"
        )

        printf -v quoted_repo '%q' "$REPO_ROOT"
        printf -v quoted_hostfile '%q' "$remote_hostfile"
        printf -v quoted_args ' %q' "${remote_args[@]}"
        remote_setup=""
        if [[ -v SWANLAB_SAVE_DIR ]]; then
            printf -v quoted_swanlab_save_dir '%q' "$SWANLAB_SAVE_DIR"
            remote_setup="mkdir -p $quoted_swanlab_save_dir && "
        fi
        remote_command="set -e; trap 'rm -f $quoted_hostfile' EXIT; ${remote_setup}cd $quoted_repo &&$quoted_args"
        echo "Starting node_rank=$node_rank on $host (log: $node_log)"
        timeout --signal=TERM --kill-after=30s "${DEEPSPEED_TIMEOUT_SECONDS}s" \
            ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no \
            "$host" "$remote_command" >"$node_log" 2>&1 &
        pids+=("$!")
    done

    statuses=()
    set +e
    for pid in "${pids[@]}"; do
        wait "$pid"
        statuses+=("$?")
    done
    set -e

    cat "${node_logs[@]}" | tee "$DEEPSPEED_LOG_PATH"
    status=0
    for node_status in "${statuses[@]}"; do
        if [ "$node_status" -ne 0 ]; then
            status="$node_status"
            break
        fi
    done
fi

if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
    fail "DeepSpeed job timed out after ${DEEPSPEED_TIMEOUT_SECONDS}s; inspect $DEEPSPEED_LOG_PATH"
fi
[ "$status" -eq 0 ] || fail "DeepSpeed job failed with exit code $status; inspect $DEEPSPEED_LOG_PATH"
echo "DeepSpeed job completed. Log: $DEEPSPEED_LOG_PATH"
