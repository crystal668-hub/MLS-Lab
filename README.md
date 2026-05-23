# MLS Lab 1 / Lab 2 / Lab 3: Qwen3 Local Deployment, Benchmark, and vLLM Scheduling

This directory is used to complete USTC 2026 MLS Lab 1, Lab 2, and Lab 3.

Lab page:
- https://ttylerzh.github.io/ustc-ml-system-2026/labs/lab1.html
- https://ttylerzh.github.io/ustc-ml-system-2026/labs/lab2.html
- https://ttylerzh.github.io/ustc-ml-system-2026/labs/lab3.html

The current default model is `Qwen/Qwen3-0.6B`.

Files:
- `infer.py`: basic `generate()` inference and manual `prefill/decode` timing
- `benchmark.py`: batch benchmark for the three required experiment groups
- `matmul_benchmark.py`: LAB2 BF16 GEMM sweep for empirical Roofline points
- `roofline_analysis.py`: LAB2 theoretical/measured Roofline point generation
- `hook_profiler.py`: LAB2 layer-level timing with forward hooks + CUDA events
- `torch_profiler_run.py`: LAB2 PyTorch profiler traces for Perfetto/Chrome tracing
- `kv_cache_analysis.py`: LAB2 KV cache sizing and service-capacity estimation
- `lab2_utils.py`: shared formulas, model-spec parsing, and hook utilities
- `task1_vllm_vs_hf.py`: LAB3 transformers serial/static batching vs vLLM offline comparison
- `task2.sh`: LAB3 vLLM `max_num_batched_tokens` benchmark sweep
- `my_scheduler.py`: LAB3 custom vLLM V1 scheduler implementations
- `task3.sh`: LAB3 default scheduler vs custom scheduler benchmark
- `task3_single.sh`: LAB3 single-scheduler benchmark helper for long Docker runs
- `scripts/lab3_parse_results.py`: LAB3 vLLM benchmark result parser and plot generator
- `lab3_report.md`: LAB3 Markdown report
- `requirements.txt`: Python dependencies
- `results/`: generated raw data, tables, and figures

## 1. Recommended Environment

Use an isolated environment and a CUDA-enabled PyTorch build.

```powershell
conda create -n mls-lab1 python=3.12 -y
conda activate mls-lab1

python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

For Lab 3, vLLM must be installed in a Linux CUDA environment. Native Windows
does not provide a supported vLLM runtime. Use Linux, WSL2 with CUDA passthrough,
or a remote Linux GPU server for `task2.sh` and `task3.sh`.

Check CUDA and Transformers:

```powershell
python -c "import torch, transformers; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(transformers.__version__)"
nvidia-smi
```

For LAB2, make sure the Python interpreter is the CUDA-enabled environment used in Lab 1.
If `torch.cuda.is_available()` is `False`, `matmul_benchmark.py`, `hook_profiler.py`,
and `torch_profiler_run.py` will not produce valid submission data.

## 2. Model Choice

For a 6GB GPU, I use `Qwen/Qwen3-0.6B`.

When I use `*-FP8` on this machine, I see errors like:
- `FP8 quantized models is only supported on GPUs with compute capability >= 8.9`
- `ModuleNotFoundError: No module named 'triton'`

That indicates my GPU or runtime does not support the Transformers FP8 loading path. `RTX 3050` is compute capability `8.6`, so switching to the non-FP8 model is the correct fix.

Download the model locally:

```powershell
huggingface-cli download Qwen/Qwen3-0.6B --local-dir .\models\Qwen3-0.6B
```

The scripts default to Qwen3 `non-thinking` mode for stable local benchmarking. If needed, you can enable thinking mode with `--enable-thinking`.

## 3. Basic Inference

Run `generate()` for Step 2:

```powershell
python infer.py --model-path .\models\Qwen3-0.6B --mode generate --prompt "Please explain how a large language model works in about 150 words."
```

Optional thinking mode:

```powershell
python infer.py --model-path .\models\Qwen3-0.6B --mode generate --enable-thinking --temperature 0.6 --top-p 0.95
```

Monitor GPU memory in another terminal:

```powershell
nvidia-smi -l 1
```

## 4. Manual Prefill / Decode Timing

`manual` mode bypasses `generate()` and calls `model.forward(...)` directly:

```powershell
python infer.py --model-path .\models\Qwen3-0.6B --mode manual --max-new-tokens 128
```

Use exact token lengths for benchmark validation:

```powershell
python infer.py --model-path .\models\Qwen3-0.6B --mode manual --input-len 128 --batch-size 1 --max-new-tokens 128
```

Reported metrics:
- `TTFT (ms)`
- `TBT (ms)`
- `Prefill tok/s`
- `Decode tok/s`
- `Throughput tok/s`
- `Peak memory (GB)`
- `KV cache prefill/final/delta (GB)`

## 5. Run the Three Benchmark Groups

`benchmark.py` runs:
- `input_len = 64 / 256 / 512 / 1024`, fixed `output_len = 128`, `batch = 1`
- `output_len = 64 / 256 / 512 / 1024`, fixed `input_len = 128`, `batch = 1`
- `batch_size = 1 / 4 / 8 / 16`, fixed `input_len = 128`, `output_len = 128`

Run:

```powershell
python benchmark.py --model-path .\models\Qwen3-0.6B --output-dir results
```

Adjust repeats if needed:

```powershell
python benchmark.py --model-path .\models\Qwen3-0.6B --warmup-runs 1 --repeats 3
```

Outputs:
- `results/raw/benchmark_raw.csv`
- `results/tables/table_input_length_prefill.csv`
- `results/tables/table_output_length_decode.csv`
- `results/tables/table_batch_size_scaling.csv`
- `results/figures/*.png`

If `batch=8` or `batch=16` runs OOM, the script records the failure and continues.

## 6. Suggested Report Structure

1. Environment
   Include GPU, VRAM, driver/CUDA, PyTorch, Transformers, and model choice.

2. Results
   Include Step 2 terminal output, `nvidia-smi` screenshots, and the three result tables.

3. Analysis
   Explain:
   - why input length mainly affects TTFT / prefill
   - why output length mainly affects TBT / decode and KV cache
   - how batch size affects throughput and memory
   - which settings were skipped due to memory limits

## 7. Submission Checklist

- `infer.py` runs in both `generate` and `manual` modes
- `benchmark.py` exports CSV results
- `README.md` is reproducible
- the PDF includes screenshots, tables, and analysis
- the archive name follows `studentid_name_lab1.zip`

## 8. LAB2 Workflow

Recommended order:

1. Verify CUDA environment.
2. Run the BF16 GEMM sweep for Task 1/2.
3. Run hook-based layer profiling for Qwen3.
4. Generate Roofline plots and KV cache tables.
5. Export PyTorch traces for Perfetto screenshots.

### 8.1 Task 1/2: BF16 GEMM Benchmark

Run:

```powershell
python matmul_benchmark.py --device cuda:0
```

Default settings:
- `M = N = 1024`
- `K = 8,16,32,64,128,256,512,1024,2048,4096,8192`
- `warmup = 20`
- `measure = 100`
- `dtype = bfloat16`

Outputs:
- `results/lab2/matmul/matmul_benchmark.csv`
- `results/lab2/matmul/matmul_tflops_vs_k.png`
- `results/lab2/matmul/matmul_roofline.png`

### 8.2 Task 3: Qwen3 Layer Profiling + Roofline

Hook-based timing:

```powershell
python hook_profiler.py --model-path .\models\Qwen3-0.6B --device cuda:0 --input-len 512 --decode-steps 128
```

Outputs:
- `results/lab2/hook_profiler/raw_hook_events.csv`
- `results/lab2/hook_profiler/layer_breakdown.csv`
- `results/lab2/hook_profiler/module_summary.csv`
- `results/lab2/hook_profiler/prefill_layer_breakdown.png`
- `results/lab2/hook_profiler/decode_layer_breakdown.png`
- `results/lab2/hook_profiler/global_time_share.png`

Generate Roofline plot:

```powershell
python roofline_analysis.py --model-path .\models\Qwen3-0.6B
```

Outputs:
- `results/lab2/roofline/theoretical_points.csv`
- `results/lab2/roofline/measured_points.csv` if `hook_profiler.py` has been run
- `results/lab2/roofline/roofline_qwen3.png`

### 8.3 Task 4: KV Cache Analysis

Theory-only tables:

```powershell
python kv_cache_analysis.py --model-path .\models\Qwen3-0.6B
```

Optional measured KV cache sizes:

```powershell
python kv_cache_analysis.py --model-path .\models\Qwen3-0.6B --device cuda:0 --measure
```

Outputs:
- `results/lab2/kv_cache/kv_cache_theory.csv`
- `results/lab2/kv_cache/kv_cache_measurements.csv` when `--measure` is enabled
- `results/lab2/kv_cache/capacity_estimates.csv`
- `results/lab2/kv_cache/lab1_kv_crosscheck.csv`

### 8.4 Task 5: PyTorch Profiler Traces

Run:

```powershell
python torch_profiler_run.py --model-path .\models\Qwen3-0.6B --device cuda:0 --input-len 512 --decode-steps 16
```

Outputs:
- `results/lab2/torch_profiler/prefill_trace.json`
- `results/lab2/torch_profiler/decode_trace.json`
- `results/lab2/torch_profiler/prefill_events.csv`
- `results/lab2/torch_profiler/decode_events.csv`
- `results/lab2/torch_profiler/*_top_kernels.txt`

Open the trace JSON files with [Perfetto](https://ui.perfetto.dev/) or Chrome tracing.

## 9. LAB2 Notes

- `roofline_analysis.py` uses `Qwen3-0.6B` real config values from `config.json`:
  `L=28`, `d=1024`, `d_ff=3072`, `n_q=16`, `n_kv=8`, `head_dim=128`.
- `kv_cache_analysis.py` uses the exact KV cache formula
  `2 * layers * n_kv * head_dim * 2 bytes = 112 KiB/token`.
- The default Roofline assumptions are `54.2 TFLOPS` BF16 peak and `168 GB/s`
  memory bandwidth, matching the Lab 2 write-up used in this repo.

## 10. LAB3 Workflow

Lab 3 evaluates vLLM's scheduling and batching behavior. The default model is
still `models/Qwen3-0.6B`, which is small enough for a 6GB RTX 3050 when
`max_model_len`, `max_num_seqs`, and `max_num_batched_tokens` are kept modest.

### 10.1 Environment

On Linux or WSL2:

```bash
conda create -n mls-lab3 python=3.12 -y
conda activate mls-lab3
python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch, vllm; print(torch.__version__, torch.cuda.is_available()); print(vllm.__version__)"
```

### 10.2 Task 1: transformers vs vLLM Offline

Full run:

```bash
python task1_vllm_vs_hf.py \
  --model-path models/Qwen3-0.6B \
  --device cuda:0 \
  --dtype bfloat16 \
  --hf-batch-size 64 \
  --ignore-eos
```

Outputs:
- `results/lab3/task1/workload.json`
- `results/lab3/task1/workload.csv`
- `results/lab3/task1/task1_results.json`
- `results/lab3/task1/task1_results.csv`

For a fast workload-generation smoke test:

```bash
python task1_vllm_vs_hf.py --model-path models/Qwen3-0.6B --run metadata-only
```

### 10.3 Task 2: vLLM Token Budget Sweep

Run on Linux or WSL2:

```bash
bash task2.sh
```

Useful overrides for a 6GB GPU:

```bash
MAX_NUM_BATCHED_TOKENS_LIST="512 1024 2048" \
NUM_PROMPTS=100 \
REQUEST_RATE=4 \
MAX_MODEL_LEN=2048 \
MAX_NUM_SEQS=32 \
bash task2.sh
```

Outputs:
- `results/lab3/task2/raw/*.json`
- `results/lab3/task2/task2_raw_summary.csv`
- `results/lab3/task2/task2_median_summary.csv`
- `results/lab3/task2/figures/task2_a1_token_budget_vs_ttft_itl.png`
- `results/lab3/task2/figures/task2_a2_token_budget_vs_throughput.png`

### 10.4 Task 3: Custom Scheduler Comparison

Run on Linux or WSL2:

```bash
bash task3.sh
```

The script compares:
- vLLM default scheduler
- `my_scheduler.StepExclusivePrefillFirstScheduler`
- `my_scheduler.SJFScheduler`

Outputs:
- `results/lab3/task3/raw/*.json`
- `results/lab3/task3/kv_traces/*.jsonl`
- `results/lab3/task3/task3_raw_summary.csv`
- `results/lab3/task3/task3_median_summary.csv`
- `results/lab3/task3/figures/task3_ttft_cdf.png`
- `results/lab3/task3/figures/task3_itl_cdf.png`

### 10.5 Report

`lab3_report.md` contains the generated Lab 3 tables, figure references, Docker
runtime notes, and scheduler analysis.
