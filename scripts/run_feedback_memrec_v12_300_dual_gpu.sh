#!/usr/bin/env bash
set -uo pipefail

PROJECT=/root/autodl-tmp/MemRec
PYTHON=/root/miniconda3/envs/memrec/bin/python
LOCK_FILE="$PROJECT/docs/feedback_memrec_v12_300_sha256.txt"
STATUS="$PROJECT/outputs/feedback_memrec_v12_300_status.log"
GPU_LOG="$PROJECT/outputs/feedback_memrec_v12_300_gpu_usage.csv"
PID_FILE="$PROJECT/outputs/feedback_memrec_v12_300_queue.pid"

cd "$PROJECT" || exit 1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

stamp() {
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$STATUS"
}

health() {
    local port=$1
    curl -fsS --max-time 10 -H 'Authorization: Bearer local-vllm' \
        "http://127.0.0.1:${port}/v1/models" >/dev/null
}

check_lock() {
    sha256sum --check --strict "$LOCK_FILE" >> "$STATUS" 2>&1
}

monitor_gpu() {
    printf 'timestamp_utc,index,name,memory_total_mib,memory_used_mib,utilization_gpu_percent\n' > "$GPU_LOG"
    while true; do
        local now index values
        now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        for index in 0 1; do
            values=$(nvidia-smi -i "$index" \
                --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
                --format=csv,noheader,nounits)
            printf '%s,%s\n' "$now" "$values" >> "$GPU_LOG"
        done
        sleep 30
    done
}

run_one() {
    local label=$1 config=$2 out=$3 variant=$4 gpu=$5 port=$6
    local pid rc current_warmup current_test milestone
    local last_warmup=0 last_test=0

    if [[ -e "$out" ]]; then
        stamp "FAILED $label: output already exists: $out"
        return 1
    fi
    if ! check_lock; then
        stamp "FAILED $label: locked file hash mismatch"
        return 1
    fi
    if ! health "$port"; then
        stamp "FAILED $label: vLLM port $port unhealthy"
        return 1
    fi

    mkdir -p "$out"
    stamp "START $label gpu=$gpu endpoint=http://127.0.0.1:${port}/v1 config=$config output=$out"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/run_train.py \
        --model memrec_agent \
        --dataset instructrec-books \
        --config "$config" \
        --device cuda:0 \
        --output_dir "$out" > "$out/run.log" 2>&1 &
    pid=$!
    stamp "PID $label $pid"

    while kill -0 "$pid" 2>/dev/null; do
        current_warmup=0
        current_test=0
        [[ -f "$out/warmup_events.jsonl" ]] && current_warmup=$(wc -l < "$out/warmup_events.jsonl")
        [[ -f "$out/per_user_metrics.jsonl" ]] && current_test=$(wc -l < "$out/per_user_metrics.jsonl")
        for milestone in 50 100 150 200 250 300; do
            if (( current_warmup >= milestone && last_warmup < milestone )); then
                stamp "PROGRESS $label warmup=$milestone/300"
                last_warmup=$milestone
            fi
            if (( current_test >= milestone && last_test < milestone )); then
                stamp "PROGRESS $label test=$milestone/300"
                last_test=$milestone
            fi
        done
        sleep 30
    done

    wait "$pid"
    rc=$?
    if [[ "$rc" -ne 0 ]]; then
        stamp "FAILED $label: exit=$rc; no retry"
        return 1
    fi
    stamp "EXIT $label code=0"
    if ! "$PYTHON" scripts/validate_feedback_memrec_v12_300_run.py \
        "$out" "$variant" >> "$STATUS" 2>&1; then
        stamp "FAILED $label: acceptance check failed; no retry"
        return 1
    fi
    stamp "ACCEPTED $label"
    return 0
}

gpu0_queue() {
    if ! run_one corrected \
        configs/feedback_memrec_books_v12_300_corrected.yaml \
        outputs/feedback_memrec_v12_continuous_300_corrected \
        corrected 0 8000; then
        stamp "GPU0 QUEUE STOPPED before Full"
        return 1
    fi
    if ! run_one full_v12 \
        configs/feedback_memrec_books_v12_300_full.yaml \
        outputs/feedback_memrec_v12_continuous_300_full \
        full 0 8000; then
        stamp "GPU0 QUEUE FAILED during Full"
        return 1
    fi
    stamp "GPU0 QUEUE COMPLETE"
}

gpu1_queue() {
    if ! run_one closed_loop_read \
        configs/feedback_memrec_books_v12_300_read.yaml \
        outputs/feedback_memrec_v12_continuous_300_read \
        read 1 8001; then
        stamp "GPU1 QUEUE FAILED"
        return 1
    fi
    stamp "GPU1 QUEUE COMPLETE"
}

: > "$STATUS"
printf '%s\n' "$$" > "$PID_FILE"
stamp "DUAL-GPU QUEUE STARTING controller=$$"
if ! check_lock || ! health 8000 || ! health 8001; then
    stamp "QUEUE PRECHECK FAILED"
    exit 1
fi

monitor_gpu &
MONITOR_PID=$!
gpu0_queue &
GPU0_QUEUE_PID=$!
gpu1_queue &
GPU1_QUEUE_PID=$!
stamp "QUEUE PIDS gpu0=$GPU0_QUEUE_PID gpu1=$GPU1_QUEUE_PID monitor=$MONITOR_PID"

wait "$GPU0_QUEUE_PID"
GPU0_RC=$?
wait "$GPU1_QUEUE_PID"
GPU1_RC=$?
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true

if [[ "$GPU0_RC" -eq 0 && "$GPU1_RC" -eq 0 ]]; then
    stamp "DUAL-GPU QUEUE COMPLETE"
    exit 0
fi
stamp "DUAL-GPU QUEUE FAILED gpu0_rc=$GPU0_RC gpu1_rc=$GPU1_RC"
exit 1

