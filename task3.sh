#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-models/Qwen3-0.6B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
OUTPUT_DIR="${OUTPUT_DIR:-results/lab3/task3}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
REQUEST_RATE="${REQUEST_RATE:-8}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-1024}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-256}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
REPEATS="${REPEATS:-2}"
VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

mkdir -p "${OUTPUT_DIR}/raw" "${OUTPUT_DIR}/kv_traces"

wait_for_server() {
  local timeout="${1:-180}"
  local elapsed=0
  until curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; do
    sleep 2
    elapsed=$((elapsed + 2))
    if [[ "${elapsed}" -ge "${timeout}" ]]; then
      echo "vLLM server did not become ready within ${timeout}s" >&2
      return 1
    fi
  done
}

cleanup_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_server EXIT

run_strategy() {
  local strategy="$1"
  local scheduler_cls="$2"

  for repeat_id in $(seq 1 "${REPEATS}"); do
    local server_log="${OUTPUT_DIR}/server_${strategy}_repeat_${repeat_id}.log"
    local trace_file="${OUTPUT_DIR}/kv_traces/${strategy}_repeat_${repeat_id}.jsonl"
    local result_file="task3_${strategy}_repeat_${repeat_id}.json"

    echo "=== Starting strategy=${strategy}, repeat=${repeat_id} ==="
    local scheduler_args=()
    if [[ -n "${scheduler_cls}" ]]; then
      scheduler_args+=(--scheduler-cls "${scheduler_cls}")
    fi

    PYTHONPATH="${PWD}:${PYTHONPATH:-}" \
    MLS_LAB3_KV_TRACE="${trace_file}" \
    VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL}" \
      vllm serve "${MODEL_PATH}" \
        --trust-remote-code \
        --host "${HOST}" \
        --port "${PORT}" \
        --dtype auto \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --enable-chunked-prefill \
        "${scheduler_args[@]}" \
        >"${server_log}" 2>&1 &
    SERVER_PID=$!
    wait_for_server 240

    echo "=== Benchmark strategy=${strategy}, repeat=${repeat_id} ==="
    vllm bench serve \
      --base-url "http://${HOST}:${PORT}" \
      --model "${MODEL_PATH}" \
      --tokenizer "${MODEL_PATH}" \
      --trust-remote-code \
      --dataset-name random \
      --num-prompts "${NUM_PROMPTS}" \
      --random-input-len "${RANDOM_INPUT_LEN}" \
      --random-output-len "${RANDOM_OUTPUT_LEN}" \
      --random-range-ratio "${RANDOM_RANGE_RATIO}" \
      --seed 42 \
      --request-rate "${REQUEST_RATE}" \
      --percentile-metrics ttft,itl \
      --metric-percentiles 50,99 \
      --ignore-eos \
      --save-result \
      --result-dir "${OUTPUT_DIR}/raw" \
      --result-filename "${result_file}" \
      --metadata task=task3 strategy="${strategy}" repeat_id="${repeat_id}" scheduler_cls="${scheduler_cls:-default}" \
      --disable-tqdm || true

    cleanup_server
    unset SERVER_PID
    sleep 5
  done
}

run_strategy "default" ""
run_strategy "prefill_first" "my_scheduler.StepExclusivePrefillFirstScheduler"
run_strategy "sjf" "my_scheduler.SJFScheduler"

"${PYTHON:-python3}" scripts/lab3_parse_results.py --task task3 --input-dir "${OUTPUT_DIR}/raw" --kv-trace-dir "${OUTPUT_DIR}/kv_traces" --output-dir "${OUTPUT_DIR}"
