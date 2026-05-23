#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-models/Qwen3-0.6B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
OUTPUT_DIR="${OUTPUT_DIR:-results/lab3/task2}"
MAX_NUM_BATCHED_TOKENS_LIST="${MAX_NUM_BATCHED_TOKENS_LIST:-512 1024 2048 4096 8192}"
REPEATS="${REPEATS:-2}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
REQUEST_RATE="${REQUEST_RATE:-16}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-1024}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-256}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

mkdir -p "${OUTPUT_DIR}/raw"

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

for token_budget in ${MAX_NUM_BATCHED_TOKENS_LIST}; do
  server_log="${OUTPUT_DIR}/server_max_tokens_${token_budget}.log"
  echo "=== Starting vLLM: max_num_batched_tokens=${token_budget} ==="
  PYTHONPATH="${PWD}:${PYTHONPATH:-}" VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL}" \
    vllm serve "${MODEL_PATH}" \
      --trust-remote-code \
      --host "${HOST}" \
      --port "${PORT}" \
      --dtype auto \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${token_budget}" \
      --enable-chunked-prefill \
      >"${server_log}" 2>&1 &
  SERVER_PID=$!
  wait_for_server 240

  for repeat_id in $(seq 1 "${REPEATS}"); do
    result_file="task2_max_tokens_${token_budget}_repeat_${repeat_id}.json"
    echo "=== Benchmark max_num_batched_tokens=${token_budget}, repeat=${repeat_id} ==="
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
      --metadata task=task2 max_num_batched_tokens="${token_budget}" repeat_id="${repeat_id}" \
      --disable-tqdm || true
  done

  cleanup_server
  unset SERVER_PID
  sleep 5
done

"${PYTHON:-python3}" scripts/lab3_parse_results.py --task task2 --input-dir "${OUTPUT_DIR}/raw" --output-dir "${OUTPUT_DIR}"
