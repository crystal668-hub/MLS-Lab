from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from hook_profiler import run_decode_steps, run_prefill
from infer import (
    DEFAULT_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    choose_device,
    load_tokenizer_and_model,
    prepare_batch_inputs,
    set_seed,
)
from lab2_utils import (
    CudaEventHookProfiler,
    DEFAULT_LAB2_OUTPUT_DIR,
    ensure_cuda,
    ensure_directory,
    export_environment_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect PyTorch profiler traces for Qwen3 prefill and decode."
    )
    parser.add_argument("--model-path", required=True, help="Local model path or HF repo id.")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cuda:0, ...")
    parser.add_argument(
        "--device-map",
        default="none",
        choices=["none", "auto"],
        help="Use auto dispatch only if the model cannot fit on a single device.",
    )
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_LAB2_OUTPUT_DIR / "torch_profiler"),
        help="Directory for trace/table outputs.",
    )
    return parser.parse_args()


def profiler_events_to_dataframe(prof: profile) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for event in prof.key_averages():
        device_time_total = getattr(event, "device_time_total", 0.0)
        self_device_time_total = getattr(event, "self_device_time_total", 0.0)
        device_memory_usage = getattr(event, "device_memory_usage", 0)
        self_device_memory_usage = getattr(event, "self_device_memory_usage", 0)
        rows.append(
            {
                "name": event.key,
                "count": event.count,
                "cpu_time_total_ms": event.cpu_time_total / 1000.0,
                "self_cpu_time_total_ms": event.self_cpu_time_total / 1000.0,
                "cuda_time_total_ms": device_time_total / 1000.0,
                "self_cuda_time_total_ms": self_device_time_total / 1000.0,
                "cpu_memory_usage": getattr(event, "cpu_memory_usage", 0),
                "cuda_memory_usage": device_memory_usage,
                "self_cuda_memory_usage": self_device_memory_usage,
            }
        )
    dataframe = pd.DataFrame(rows)
    if not dataframe.empty:
        dataframe = dataframe.sort_values("self_cuda_time_total_ms", ascending=False)
    return dataframe


def save_profiler_outputs(
    *,
    prof: profile,
    output_dir: Path,
    prefix: str,
    topk: int,
) -> None:
    trace_path = output_dir / f"{prefix}_trace.json"
    prof.export_chrome_trace(str(trace_path))

    table = prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=topk,
    )
    (output_dir / f"{prefix}_top_kernels.txt").write_text(table, encoding="utf-8")

    dataframe = profiler_events_to_dataframe(prof)
    dataframe.to_csv(output_dir / f"{prefix}_events.csv", index=False, encoding="utf-8-sig")
    export_top_cuda_kernel_table(
        trace_path=trace_path,
        output_csv=output_dir / f"{prefix}_top10_cuda_kernels.csv",
        top_n=10,
    )


def export_top_cuda_kernel_table(
    *,
    trace_path: Path,
    output_csv: Path,
    top_n: int,
) -> None:
    with trace_path.open("r", encoding="utf-8") as trace_file:
        trace_payload = json.load(trace_file)

    trace_events = trace_payload.get("traceEvents", [])
    rows: list[dict[str, float | int | str]] = []
    for event in trace_events:
        if event.get("ph") != "X" or event.get("cat") != "kernel":
            continue
        duration_us = float(event.get("dur", 0.0))
        if duration_us <= 0:
            continue
        rows.append({"kernel_name": event.get("name", ""), "duration_us": duration_us})

    if not rows:
        pd.DataFrame(
            columns=["rank", "kernel_name", "total_ms", "time_share_pct", "calls", "avg_us"]
        ).to_csv(output_csv, index=False, encoding="utf-8-sig")
        return

    kernel_df = pd.DataFrame(rows)
    summary_df = (
        kernel_df.groupby("kernel_name", as_index=False)
        .agg(total_us=("duration_us", "sum"), calls=("duration_us", "count"))
        .sort_values("total_us", ascending=False)
        .head(top_n)
    )
    total_kernel_us = kernel_df["duration_us"].sum()
    summary_df.insert(0, "rank", range(1, len(summary_df) + 1))
    summary_df["total_ms"] = summary_df["total_us"] / 1000.0
    summary_df["time_share_pct"] = summary_df["total_us"] / total_kernel_us * 100.0
    summary_df["avg_us"] = summary_df["total_us"] / summary_df["calls"]
    summary_df[
        ["rank", "kernel_name", "total_ms", "time_share_pct", "calls", "avg_us"]
    ].to_csv(output_csv, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    ensure_cuda(device)

    output_dir = ensure_directory(args.output_dir)
    export_environment_json(output_dir / "environment.json")

    tokenizer, model = load_tokenizer_and_model(
        args.model_path,
        device=device,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    inputs = prepare_batch_inputs(
        tokenizer=tokenizer,
        device=device,
        prompt=args.prompt,
        system_prompt=args.system_prompt,
        input_len=args.input_len,
        batch_size=args.batch_size,
        enable_thinking=args.enable_thinking,
    )

    hook_labels = CudaEventHookProfiler(enable_events=False, enable_record_functions=True)
    hook_labels.register(model)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    try:
        hook_labels.set_phase("prefill_trace")
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prefill_prof:
            with record_function("prefill_only"):
                run_prefill(model=model, inputs=inputs, device=device)
        save_profiler_outputs(
            prof=prefill_prof,
            output_dir=output_dir,
            prefix="prefill",
            topk=args.topk,
        )

        cache, next_token = run_prefill(model=model, inputs=inputs, device=device)
        hook_labels.set_phase("decode_trace")
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as decode_prof:
            with record_function("decode_first_steps"):
                run_decode_steps(
                    model=model,
                    cache=cache,
                    next_token=next_token,
                    attention_mask=inputs["attention_mask"],
                    decode_steps=args.decode_steps,
                    device=device,
                )
        save_profiler_outputs(
            prof=decode_prof,
            output_dir=output_dir,
            prefix="decode",
            topk=args.topk,
        )
    finally:
        hook_labels.remove()

    print(f"Saved prefill trace to: {output_dir / 'prefill_trace.json'}")
    print(f"Saved decode trace to: {output_dir / 'decode_trace.json'}")
    print(f"Open these JSON files in https://ui.perfetto.dev/ or Chrome tracing.")


if __name__ == "__main__":
    main()
