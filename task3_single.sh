#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-models/Qwen3-0.6B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
OUTPUT_DIR="${OUTPUT_DIR:-results/lab3/task3}"
STRATEGY="${STRATEGY:-default}"
SCHEDULER_CLS="${SCHEDULER_CLS:-}"
REPEAT_ID="${REPEAT_ID:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
REQUEST_RATE="${REQUEST_RATE:-8}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-1024}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-256}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

mkdir -p "${OUTPUT_DIR}/raw" "${OUTPUT_DIR}/kv_traces"

wait_for_server() {
  local timeout="${1:-240}"
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

scheduler_args=()
if [[ -n "${SCHEDULER_CLS}" ]]; then
  scheduler_args+=(--scheduler-cls "${SCHEDULER_CLS}")
fi

server_log="${OUTPUT_DIR}/server_${STRATEGY}_repeat_${REPEAT_ID}.log"
trace_file="${OUTPUT_DIR}/kv_traces/${STRATEGY}_repeat_${REPEAT_ID}.jsonl"
result_file="task3_${STRATEGY}_repeat_${REPEAT_ID}.json"

PYTHONPATH="${PWD}:${PYTHONPATH:-}" \
MLS_LAB3_KV_TRACE="${trace_file}" \
VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL}" \
  vllm serve "${MODEL_PATH}" \
    --trust-remote-code \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --enable-chunked-prefill \
    --enforce-eager \
    "${scheduler_args[@]}" \
    >"${server_log}" 2>&1 &
SERVER_PID=$!
wait_for_server 300

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
  --metadata task=task3 strategy="${STRATEGY}" repeat_id="${REPEAT_ID}" scheduler_cls="${SCHEDULER_CLS:-default}" \
  --disable-tqdm || true
