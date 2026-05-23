import argparse
import csv
import gc
import json
import math
import os
import multiprocessing as mp
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = "models/Qwen3-0.6B"
DEFAULT_OUTPUT_DIR = Path("results") / "lab3" / "task1"
DEFAULT_NUM_PROMPTS = 64
DEFAULT_OUTPUT_LEN = 128
DEFAULT_MIN_INPUT_LEN = 128
DEFAULT_MAX_INPUT_LEN = 512
DEFAULT_SEED = 42


@dataclass
class Workload:
    prompts: list[str]
    input_lens: list[int]
    output_len: int


@dataclass
class MethodResult:
    method: str
    status: str
    total_time_s: float | None
    generated_tokens: int
    throughput_tok_s: float | None
    peak_kv_cache_mb: float | None
    peak_reserved_memory_gb: float | None
    batch_size: int | None
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Task 1 for MLS Lab 3: compare naive transformers serial "
            "generation, static-padding batching, and vLLM offline inference."
        )
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS)
    parser.add_argument("--min-input-len", type=int, default=DEFAULT_MIN_INPUT_LEN)
    parser.add_argument("--max-input-len", type=int, default=DEFAULT_MAX_INPUT_LEN)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_OUTPUT_LEN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto", help="auto, cuda:0, cpu, ...")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype for transformers and vLLM when supported.",
    )
    parser.add_argument(
        "--hf-batch-size",
        type=int,
        default=64,
        help="Initial static-padding batch size to try for transformers.",
    )
    parser.add_argument(
        "--no-auto-batch-shrink",
        action="store_true",
        help="Use --hf-batch-size directly instead of probing larger failing batches.",
    )
    parser.add_argument(
        "--run",
        default="all",
        choices=["all", "hf-only", "vllm-only", "metadata-only"],
        help="Select which inference backends to run.",
    )
    parser.add_argument(
        "--hf-method",
        default="both",
        choices=["both", "serial", "batch", "batch-probe"],
        help="When running HF backends, run both methods or only one method.",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge new Task 1 rows with existing task1_results.json in output-dir.",
    )
    parser.add_argument(
        "--time-limit-s",
        type=float,
        default=None,
        help="Optional per-method wall-clock time limit. Completed work is still saved.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.9,
    )
    parser.add_argument("--vllm-max-model-len", type=int, default=2048)
    parser.add_argument(
        "--vllm-scheduler-cls",
        default="my_scheduler.PeakKVTrackingScheduler",
        help=(
            "Scheduler class used by vLLM offline inference. The default records "
            "scheduler-side KV token peaks to MLS_LAB3_KV_TRACE."
        ),
    )
    parser.add_argument(
        "--vllm-kv-trace-file",
        default=None,
        help="Optional JSONL trace path for vLLM scheduler-side KV usage.",
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help="Pass enforce_eager=True to vLLM. Useful on small GPUs.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Force exactly max_new_tokens where the backend supports it.",
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dtype_from_arg(dtype: str) -> torch.dtype | str:
    if dtype == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype]


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_reserved(device) / (1024**3)


def cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def append_partial_row(output_dir: Path, row: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "task1_partial_rows.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_hf_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype | str,
):
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    kwargs["dtype"] = dtype if dtype != "auto" else "auto"
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.to(device)
    model.eval()
    return model


def kv_cache_per_token_bytes(model_path: str, dtype_bytes: int = 2) -> int:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    return (
        2
        * int(config.num_hidden_layers)
        * int(config.num_key_value_heads)
        * int(head_dim)
        * dtype_bytes
    )


def build_text_with_token_len(tokenizer, target_len: int, rng: random.Random) -> str:
    words = [
        "benchmark",
        "prefill",
        "decode",
        "scheduler",
        "paged",
        "attention",
        "throughput",
        "latency",
        "cache",
        "request",
        "batching",
        "system",
        "tokens",
        "memory",
        "kernel",
        "queue",
    ]
    pieces: list[str] = []
    token_ids: list[int] = []
    while len(token_ids) < target_len:
        pieces.extend(rng.sample(words, k=len(words)))
        text = " ".join(pieces)
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    token_ids = token_ids[:target_len]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def build_workload(
    tokenizer,
    *,
    num_prompts: int,
    min_input_len: int,
    max_input_len: int,
    output_len: int,
    seed: int,
) -> Workload:
    rng = random.Random(seed)
    prompts: list[str] = []
    lens: list[int] = []
    for _ in range(num_prompts):
        target_len = rng.randint(min_input_len, max_input_len)
        prompt = build_text_with_token_len(tokenizer, target_len, rng)
        actual_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        prompts.append(prompt)
        lens.append(actual_len)
    return Workload(prompts=prompts, input_lens=lens, output_len=output_len)


def save_workload(output_dir: Path, workload: Workload) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"prompt_id": idx, "input_len": input_len, "prompt": prompt}
        for idx, (prompt, input_len) in enumerate(zip(workload.prompts, workload.input_lens))
    ]
    with (output_dir / "workload.json").open("w", encoding="utf-8") as f:
        json.dump({"output_len": workload.output_len, "rows": rows}, f, ensure_ascii=False, indent=2)
    with (output_dir / "workload.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "input_len", "prompt"])
        writer.writeheader()
        writer.writerows(rows)


def generation_kwargs(tokenizer, max_new_tokens: int, ignore_eos: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "top_k": None,
        "top_p": None,
        "temperature": None,
    }
    if ignore_eos:
        kwargs["eos_token_id"] = None
    else:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    return kwargs


def run_hf_serial(
    *,
    model,
    tokenizer,
    workload: Workload,
    device: torch.device,
    ignore_eos: bool,
    per_token_kv_bytes: int,
    output_dir: Path,
    time_limit_s: float | None,
) -> MethodResult:
    reset_peak_memory(device)
    total_generated = 0
    maybe_sync(device)
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            for prompt_id, prompt in enumerate(workload.prompts):
                if time_limit_s is not None and time.perf_counter() - start >= time_limit_s:
                    maybe_sync(device)
                    elapsed = time.perf_counter() - start
                    return MethodResult(
                        method="transformers serial",
                        status="partial_timeout",
                        total_time_s=elapsed,
                        generated_tokens=total_generated,
                        throughput_tok_s=total_generated / elapsed if elapsed > 0 else None,
                        peak_kv_cache_mb=None,
                        peak_reserved_memory_gb=peak_memory_gb(device),
                        batch_size=1,
                        notes=(
                            f"Stopped after {prompt_id}/{len(workload.prompts)} requests "
                            f"because --time-limit-s={time_limit_s} was reached."
                        ),
                    )
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
                output_ids = model.generate(
                    **inputs,
                    **generation_kwargs(tokenizer, workload.output_len, ignore_eos),
                )
                generated = int(output_ids.shape[1] - inputs["input_ids"].shape[1])
                total_generated += generated
                append_partial_row(
                    output_dir,
                    {
                        "method": "transformers serial",
                        "prompt_id": prompt_id,
                        "input_len": workload.input_lens[prompt_id],
                        "generated_tokens": generated,
                        "elapsed_s": time.perf_counter() - start,
                    },
                )
                del inputs, output_ids
                cleanup(device)
        maybe_sync(device)
        elapsed = time.perf_counter() - start
        peak_tokens = max(length + workload.output_len for length in workload.input_lens)
        return MethodResult(
            method="transformers serial",
            status="ok",
            total_time_s=elapsed,
            generated_tokens=total_generated,
            throughput_tok_s=total_generated / elapsed if elapsed > 0 else None,
            peak_kv_cache_mb=peak_tokens * per_token_kv_bytes / (1024**2),
            peak_reserved_memory_gb=peak_memory_gb(device),
            batch_size=1,
            notes=f"{len(workload.prompts)} requests are submitted one by one.",
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            cleanup(device)
            return MethodResult(
                method="transformers serial",
                status="oom",
                total_time_s=None,
                generated_tokens=total_generated,
                throughput_tok_s=None,
                peak_kv_cache_mb=None,
                peak_reserved_memory_gb=peak_memory_gb(device),
                batch_size=1,
                notes=str(exc),
            )
        raise


def try_hf_batch(
    *,
    model,
    tokenizer,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int,
    ignore_eos: bool,
) -> int:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    ).to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            **generation_kwargs(tokenizer, max_new_tokens, ignore_eos),
        )
    generated = int((output_ids.shape[1] - inputs["input_ids"].shape[1]) * len(prompts))
    del inputs, output_ids
    cleanup(device)
    return generated


def find_max_hf_batch_size(
    *,
    model,
    tokenizer,
    workload: Workload,
    device: torch.device,
    initial_batch_size: int,
    ignore_eos: bool,
) -> tuple[int, str]:
    max_try = min(initial_batch_size, len(workload.prompts))
    batch_size = max_try
    last_error = ""
    while batch_size >= 1:
        try:
            try_hf_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=workload.prompts[:batch_size],
                device=device,
                max_new_tokens=workload.output_len,
                ignore_eos=ignore_eos,
            )
            return batch_size, last_error
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            last_error = str(exc)
            cleanup(device)
            batch_size //= 2
    return 0, last_error


def run_hf_static_batch(
    *,
    model,
    tokenizer,
    workload: Workload,
    device: torch.device,
    initial_batch_size: int,
    ignore_eos: bool,
    per_token_kv_bytes: int,
    output_dir: Path,
    time_limit_s: float | None,
    auto_batch_shrink: bool,
) -> MethodResult:
    if auto_batch_shrink:
        batch_size, last_error = find_max_hf_batch_size(
            model=model,
            tokenizer=tokenizer,
            workload=workload,
            device=device,
            initial_batch_size=initial_batch_size,
            ignore_eos=ignore_eos,
        )
    else:
        batch_size, last_error = min(initial_batch_size, len(workload.prompts)), ""
    if batch_size < 1:
        return MethodResult(
            method="transformers static batching",
            status="oom",
            total_time_s=None,
            generated_tokens=0,
            throughput_tok_s=None,
            peak_kv_cache_mb=None,
            peak_reserved_memory_gb=peak_memory_gb(device),
            batch_size=None,
            notes=f"No feasible batch size found. Last OOM: {last_error}",
        )

    reset_peak_memory(device)
    total_generated = 0
    max_padded_total_len = 0
    maybe_sync(device)
    start = time.perf_counter()
    for start_idx in range(0, len(workload.prompts), batch_size):
        if time_limit_s is not None and time.perf_counter() - start >= time_limit_s:
            maybe_sync(device)
            elapsed = time.perf_counter() - start
            return MethodResult(
                method="transformers static batching",
                status="partial_timeout",
                total_time_s=elapsed,
                generated_tokens=total_generated,
                throughput_tok_s=total_generated / elapsed if elapsed > 0 else None,
                peak_kv_cache_mb=(
                    max_padded_total_len * per_token_kv_bytes / (1024**2)
                    if max_padded_total_len
                    else None
                ),
                peak_reserved_memory_gb=peak_memory_gb(device),
                batch_size=batch_size,
                notes=(
                    f"Stopped after {start_idx}/{len(workload.prompts)} requests "
                    f"because --time-limit-s={time_limit_s} was reached."
                ),
            )
        end_idx = min(start_idx + batch_size, len(workload.prompts))
        chunk = workload.prompts[start_idx:end_idx]
        chunk_lens = workload.input_lens[start_idx:end_idx]
        generated = try_hf_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=chunk,
            device=device,
            max_new_tokens=workload.output_len,
            ignore_eos=ignore_eos,
        )
        total_generated += generated
        append_partial_row(
            output_dir,
            {
                "method": "transformers static batching",
                "prompt_start": start_idx,
                "prompt_end": end_idx,
                "batch_size": len(chunk),
                "max_input_len": max(chunk_lens),
                "generated_tokens": generated,
                "elapsed_s": time.perf_counter() - start,
            },
        )
        max_padded_total_len = max(
            max_padded_total_len,
            len(chunk) * (max(chunk_lens) + workload.output_len),
        )
    maybe_sync(device)
    elapsed = time.perf_counter() - start
    notes = f"{len(workload.prompts)} requests submitted as one static-padded batch."
    if batch_size < len(workload.prompts):
        notes = (
            f"Batch=64 did not fit or was not requested; ran chunks with "
            f"max feasible batch={batch_size}."
        )
    return MethodResult(
        method="transformers static batching",
        status="ok",
        total_time_s=elapsed,
        generated_tokens=total_generated,
        throughput_tok_s=total_generated / elapsed if elapsed > 0 else None,
        peak_kv_cache_mb=max_padded_total_len * per_token_kv_bytes / (1024**2),
        peak_reserved_memory_gb=peak_memory_gb(device),
        batch_size=batch_size,
        notes=notes,
    )


def run_hf_batch_probe(
    *,
    model,
    tokenizer,
    workload: Workload,
    device: torch.device,
    batch_size: int,
    ignore_eos: bool,
    per_token_kv_bytes: int,
) -> MethodResult:
    prompts = workload.prompts[:batch_size]
    chunk_lens = workload.input_lens[:batch_size]
    reset_peak_memory(device)
    maybe_sync(device)
    start = time.perf_counter()
    try:
        generated = try_hf_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            device=device,
            max_new_tokens=workload.output_len,
            ignore_eos=ignore_eos,
        )
        maybe_sync(device)
        elapsed = time.perf_counter() - start
        padded_tokens = len(prompts) * (max(chunk_lens) + workload.output_len)
        return MethodResult(
            method=f"transformers batch probe b{batch_size}",
            status="ok",
            total_time_s=elapsed,
            generated_tokens=generated,
            throughput_tok_s=generated / elapsed if elapsed > 0 else None,
            peak_kv_cache_mb=padded_tokens * per_token_kv_bytes / (1024**2),
            peak_reserved_memory_gb=peak_memory_gb(device),
            batch_size=batch_size,
            notes=(
                f"One static-padded batch of {batch_size} prompts completed "
                f"with output_len={workload.output_len}."
            ),
        )
    except RuntimeError as exc:
        status = "oom" if "out of memory" in str(exc).lower() else "runtime_error"
        cleanup(device)
        return MethodResult(
            method=f"transformers batch probe b{batch_size}",
            status=status,
            total_time_s=None,
            generated_tokens=0,
            throughput_tok_s=None,
            peak_kv_cache_mb=None,
            peak_reserved_memory_gb=peak_memory_gb(device),
            batch_size=batch_size,
            notes=str(exc),
        )


def run_vllm_offline(
    *,
    model_path: str,
    workload: Workload,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    enforce_eager: bool,
    ignore_eos: bool,
    per_token_kv_bytes: int,
    scheduler_cls: str | None,
    kv_trace_file: Path | None,
) -> MethodResult:
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        return MethodResult(
            method="vLLM offline",
            status="skipped",
            total_time_s=None,
            generated_tokens=0,
            throughput_tok_s=None,
            peak_kv_cache_mb=None,
            peak_reserved_memory_gb=None,
            batch_size=len(workload.prompts),
            notes=f"vLLM import failed: {exc}",
        )

    llm_kwargs: dict[str, Any] = {
        "model": model_path,
        "trust_remote_code": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
    }
    if scheduler_cls:
        llm_kwargs["scheduler_cls"] = scheduler_cls
    if dtype != "auto":
        llm_kwargs["dtype"] = dtype
    if enforce_eager:
        llm_kwargs["enforce_eager"] = True

    sampling_kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": workload.output_len,
    }
    if ignore_eos:
        sampling_kwargs["ignore_eos"] = True
    sampling = SamplingParams(**sampling_kwargs)

    start = time.perf_counter()
    old_trace_env = os.environ.get("MLS_LAB3_KV_TRACE")
    if kv_trace_file is not None:
        kv_trace_file.parent.mkdir(parents=True, exist_ok=True)
        if kv_trace_file.exists():
            kv_trace_file.unlink()
        os.environ["MLS_LAB3_KV_TRACE"] = str(kv_trace_file)
    try:
        llm = LLM(**llm_kwargs)
        outputs = llm.generate(workload.prompts, sampling)
        elapsed = time.perf_counter() - start
        generated_tokens = 0
        for output in outputs:
            if getattr(output, "outputs", None):
                token_ids = getattr(output.outputs[0], "token_ids", None)
                if token_ids is not None:
                    generated_tokens += len(token_ids)
                else:
                    generated_tokens += workload.output_len
        if generated_tokens == 0:
            generated_tokens = len(workload.prompts) * workload.output_len
        trace_peak_tokens = read_scheduler_peak_tokens(kv_trace_file)
        peak_tokens = trace_peak_tokens or sum(
            length + workload.output_len for length in workload.input_lens
        )
        peak_note = (
            f"KV peak came from scheduler trace: {trace_peak_tokens:.0f} tokens."
            if trace_peak_tokens
            else "KV peak is formula-estimated because no scheduler trace was recorded."
        )
        return MethodResult(
            method="vLLM offline",
            status="ok",
            total_time_s=elapsed,
            generated_tokens=generated_tokens,
            throughput_tok_s=generated_tokens / elapsed if elapsed > 0 else None,
            peak_kv_cache_mb=peak_tokens * per_token_kv_bytes / (1024**2),
            peak_reserved_memory_gb=None,
            batch_size=len(workload.prompts),
            notes=peak_note,
        )
    except Exception as exc:
        return MethodResult(
            method="vLLM offline",
            status="error",
            total_time_s=None,
            generated_tokens=0,
            throughput_tok_s=None,
            peak_kv_cache_mb=None,
            peak_reserved_memory_gb=None,
            batch_size=len(workload.prompts),
            notes=str(exc),
        )
    finally:
        if old_trace_env is None:
            os.environ.pop("MLS_LAB3_KV_TRACE", None)
        else:
            os.environ["MLS_LAB3_KV_TRACE"] = old_trace_env


def read_scheduler_peak_tokens(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    peak = 0.0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        peak = max(peak, float(record.get("peak_running_computed_tokens", 0.0)))
    return peak or None


def save_results(output_dir: Path, results: list[MethodResult], metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "results": [asdict(result) for result in results]}
    with (output_dir / "task1_results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with (output_dir / "task1_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = list(asdict(results[0]).keys()) if results else list(MethodResult.__annotations__)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def load_existing_results(output_dir: Path) -> tuple[dict[str, Any] | None, list[MethodResult]]:
    path = output_dir / "task1_results.json"
    if not path.exists():
        return None, []
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = []
    for item in payload.get("results", []):
        rows.append(MethodResult(**item))
    return payload.get("metadata"), rows


def merge_results(existing: list[MethodResult], new: list[MethodResult]) -> list[MethodResult]:
    merged = {row.method: row for row in existing}
    for row in new:
        merged[row.method] = row
    preferred_order = [
        "transformers serial",
        "transformers static batching",
        "vLLM offline",
    ]
    return sorted(
        merged.values(),
        key=lambda row: (
            preferred_order.index(row.method)
            if row.method in preferred_order
            else len(preferred_order),
            row.method,
        ),
    )


def print_summary(results: list[MethodResult]) -> None:
    print("\nTask 1 summary")
    print("=" * 80)
    for result in results:
        time_text = "n/a" if result.total_time_s is None else f"{result.total_time_s:.3f}s"
        tps_text = (
            "n/a"
            if result.throughput_tok_s is None
            else f"{result.throughput_tok_s:.2f} tok/s"
        )
        kv_text = (
            "n/a"
            if result.peak_kv_cache_mb is None
            else f"{result.peak_kv_cache_mb:.2f} MB"
        )
        print(
            f"{result.method:32s} status={result.status:8s} "
            f"time={time_text:>10s} throughput={tps_text:>16s} "
            f"peak_kv={kv_text:>12s} batch={result.batch_size}"
        )
        if result.notes:
            print(f"  notes: {result.notes}")


def main() -> None:
    args = parse_args()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    output_dir = Path(args.output_dir)
    # vLLM starts CUDA work in its own engine process. Do not touch
    # torch.cuda in the parent process before constructing LLM(), otherwise
    # Linux fork-based engine startup can fail with CUDA re-initialization
    # errors.
    device = torch.device("cpu") if args.run == "vllm-only" else choose_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda" and args.run != "vllm-only":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = load_tokenizer(args.model_path)
    workload = build_workload(
        tokenizer,
        num_prompts=args.num_prompts,
        min_input_len=args.min_input_len,
        max_input_len=args.max_input_len,
        output_len=args.max_new_tokens,
        seed=args.seed,
    )
    save_workload(output_dir, workload)
    per_token_kv_bytes = kv_cache_per_token_bytes(args.model_path)

    metadata = {
        "model_path": args.model_path,
        "num_prompts": len(workload.prompts),
        "output_len": workload.output_len,
        "input_len_min": min(workload.input_lens),
        "input_len_median": statistics.median(workload.input_lens),
        "input_len_max": max(workload.input_lens),
        "per_token_kv_bytes": per_token_kv_bytes,
        "per_token_kv_kib": per_token_kv_bytes / 1024,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": None if args.run == "vllm-only" else torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "dtype": args.dtype,
    }

    existing_metadata = None
    existing_results: list[MethodResult] = []
    if args.merge_existing:
        existing_metadata, existing_results = load_existing_results(output_dir)

    results: list[MethodResult] = []
    hf_model = None
    if args.run in {"all", "hf-only"}:
        print(f"Loading transformers model on {device}...")
        hf_model = load_hf_model(args.model_path, device, dtype_from_arg(args.dtype))
        if args.hf_method in {"both", "serial"}:
            results.append(
                run_hf_serial(
                    model=hf_model,
                    tokenizer=tokenizer,
                    workload=workload,
                    device=device,
                    ignore_eos=args.ignore_eos,
                    per_token_kv_bytes=per_token_kv_bytes,
                    output_dir=output_dir,
                    time_limit_s=args.time_limit_s,
                )
            )
        if args.hf_method in {"both", "batch"}:
            results.append(
                run_hf_static_batch(
                    model=hf_model,
                    tokenizer=tokenizer,
                    workload=workload,
                    device=device,
                    initial_batch_size=args.hf_batch_size,
                    ignore_eos=args.ignore_eos,
                    per_token_kv_bytes=per_token_kv_bytes,
                    output_dir=output_dir,
                    time_limit_s=args.time_limit_s,
                    auto_batch_shrink=not args.no_auto_batch_shrink,
                )
            )
        if args.hf_method == "batch-probe":
            results.append(
                run_hf_batch_probe(
                    model=hf_model,
                    tokenizer=tokenizer,
                    workload=workload,
                    device=device,
                    batch_size=args.hf_batch_size,
                    ignore_eos=args.ignore_eos,
                    per_token_kv_bytes=per_token_kv_bytes,
                )
            )
        del hf_model
        cleanup(device)

    if args.run in {"all", "vllm-only"}:
        kv_trace_file = (
            Path(args.vllm_kv_trace_file)
            if args.vllm_kv_trace_file
            else output_dir / "vllm_kv_trace.jsonl"
        )
        results.append(
            run_vllm_offline(
                model_path=args.model_path,
                workload=workload,
                dtype=args.dtype,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_len=args.vllm_max_model_len,
                enforce_eager=args.vllm_enforce_eager,
                ignore_eos=args.ignore_eos,
                per_token_kv_bytes=per_token_kv_bytes,
                scheduler_cls=args.vllm_scheduler_cls,
                kv_trace_file=kv_trace_file,
            )
        )

    if args.run == "metadata-only":
        peak_tokens_serial = max(length + workload.output_len for length in workload.input_lens)
        peak_tokens_static = len(workload.prompts) * (
            max(workload.input_lens) + workload.output_len
        )
        peak_tokens_vllm = sum(length + workload.output_len for length in workload.input_lens)
        results.extend(
            [
                MethodResult(
                    method="transformers serial",
                    status="not_run",
                    total_time_s=None,
                    generated_tokens=0,
                    throughput_tok_s=None,
                    peak_kv_cache_mb=peak_tokens_serial * per_token_kv_bytes / (1024**2),
                    peak_reserved_memory_gb=None,
                    batch_size=1,
                    notes="Formula-only metadata mode.",
                ),
                MethodResult(
                    method="transformers static batching",
                    status="not_run",
                    total_time_s=None,
                    generated_tokens=0,
                    throughput_tok_s=None,
                    peak_kv_cache_mb=peak_tokens_static * per_token_kv_bytes / (1024**2),
                    peak_reserved_memory_gb=None,
                    batch_size=len(workload.prompts),
                    notes="Formula-only metadata mode with full static padding.",
                ),
                MethodResult(
                    method="vLLM offline",
                    status="not_run",
                    total_time_s=None,
                    generated_tokens=0,
                    throughput_tok_s=None,
                    peak_kv_cache_mb=peak_tokens_vllm * per_token_kv_bytes / (1024**2),
                    peak_reserved_memory_gb=None,
                    batch_size=len(workload.prompts),
                    notes="Formula-only metadata mode without padding.",
                ),
            ]
        )

    final_results = merge_results(existing_results, results) if args.merge_existing else results
    final_metadata = metadata
    if existing_metadata:
        final_metadata = {**existing_metadata, **metadata}
    save_results(output_dir, final_results, final_metadata)
    print_summary(final_results)
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
