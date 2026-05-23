import argparse
import gc
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from infer import (
    DEFAULT_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    load_tokenizer_and_model,
    manual_prefill_decode,
    prepare_batch_inputs,
    choose_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the three required MLS Lab 1 benchmarks.")
    parser.add_argument("--model-path", required=True, help="Local path or HF repo id.")
    parser.add_argument("--output-dir", default="results", help="Directory for csv/png outputs.")
    parser.add_argument("--device", default="auto", help="auto, cuda:0, cpu, ...")
    parser.add_argument(
        "--device-map",
        default="none",
        choices=["none", "auto"],
        help="Use auto dispatch only if the model cannot fit on a single device.",
    )
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode in the chat template. Disabled by default for stable benchmarking.",
    )
    return parser.parse_args()


def ensure_output_dirs(base_dir: Path) -> dict[str, Path]:
    raw_dir = base_dir / "raw"
    tables_dir = base_dir / "tables"
    figures_dir = base_dir / "figures"
    for path in (base_dir, raw_dir, tables_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {"base": base_dir, "raw": raw_dir, "tables": tables_dir, "figures": figures_dir}


def cleanup_after_run(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_one_case(
    *,
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    system_prompt: str,
    input_len: int,
    output_len: int,
    batch_size: int,
    warmup_runs: int,
    repeats: int,
    group_name: str,
    variable_name: str,
    variable_value: int,
    enable_thinking: bool,
) -> list[dict]:
    rows: list[dict] = []
    inputs = prepare_batch_inputs(
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        system_prompt=system_prompt,
        input_len=input_len,
        batch_size=batch_size,
        enable_thinking=enable_thinking,
    )

    for _ in range(warmup_runs):
        try:
            manual_prefill_decode(
                model=model,
                tokenizer=tokenizer,
                inputs=inputs,
                requested_output_len=output_len,
                device=device,
            )
        except RuntimeError:
            cleanup_after_run(device)
            break
        cleanup_after_run(device)

    for repeat_id in range(1, repeats + 1):
        row = {
            "group": group_name,
            "variable": variable_name,
            "variable_value": variable_value,
            "input_len": input_len,
            "output_len": output_len,
            "batch_size": batch_size,
            "repeat_id": repeat_id,
            "status": "ok",
            "error": "",
        }
        try:
            _, _, metrics = manual_prefill_decode(
                model=model,
                tokenizer=tokenizer,
                inputs=inputs,
                requested_output_len=output_len,
                device=device,
            )
            row.update(asdict(metrics))
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                row["status"] = "oom"
            else:
                row["status"] = "runtime_error"
            row["error"] = str(exc)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows.append(row)
        cleanup_after_run(device)
    return rows


def aggregate_results(df: pd.DataFrame, variable_name: str) -> pd.DataFrame:
    success_df = df[df["status"] == "ok"].copy()
    agg_spec = {
        "input_len": ("input_len", "first"),
        "output_len": ("output_len", "first"),
        "batch_size": ("batch_size", "first"),
        "ttft_ms": ("ttft_ms", "median"),
        "tbt_ms": ("tbt_ms", "median"),
        "prefill_tok_s": ("prefill_tok_s", "median"),
        "decode_tok_s": ("decode_tok_s", "median"),
        "throughput_tok_s": ("throughput_tok_s", "median"),
        "peak_memory_gb": ("peak_memory_gb", "median"),
        "kv_cache_prefill_gb": ("kv_cache_prefill_gb", "median"),
        "kv_cache_final_gb": ("kv_cache_final_gb", "median"),
        "kv_cache_delta_gb": ("kv_cache_delta_gb", "median"),
        "generated_len": ("generated_len", "median"),
    }
    if variable_name in agg_spec:
        del agg_spec[variable_name]
    agg = (
        success_df.groupby([variable_name], as_index=False)
        .agg(**agg_spec)
        .sort_values(variable_name)
    )

    skipped = (
        df[df["status"] != "ok"]
        .groupby(variable_name, as_index=False)
        .agg(status=("status", "first"), error=("error", "first"))
    )
    if skipped.empty:
        agg["status"] = "ok"
        agg["error"] = ""
        return agg
    return agg.merge(skipped, on=variable_name, how="outer").sort_values(variable_name)


def save_plot(
    table_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: Path,
    y_label: str,
) -> None:
    plot_df = table_df[table_df["status"] == "ok"].copy() if "status" in table_df.columns else table_df
    if plot_df.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(plot_df[x_col], plot_df[y_col], marker="o")
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_label)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dirs = ensure_output_dirs(Path(args.output_dir))

    print(f"Using device: {device}")
    if device.type != "cuda":
        print(
            "Warning: benchmark.py is intended for CUDA. CPU runs are only suitable for dry-run validation."
        )

    tokenizer, model = load_tokenizer_and_model(
        args.model_path,
        device=device,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )

    all_rows: list[dict] = []

    experiments = [
        {
            "group_name": "input_length_prefill",
            "variable_name": "input_len",
            "values": [64, 256, 512, 1024],
            "fixed": {"output_len": 128, "batch_size": 1},
        },
        {
            "group_name": "output_length_decode",
            "variable_name": "output_len",
            "values": [64, 256, 512, 1024],
            "fixed": {"input_len": 128, "batch_size": 1},
        },
        {
            "group_name": "batch_size_scaling",
            "variable_name": "batch_size",
            "values": [1, 4, 8, 16],
            "fixed": {"input_len": 128, "output_len": 128},
        },
    ]

    for experiment in experiments:
        print(f"=== Running {experiment['group_name']} ===")
        for value in experiment["values"]:
            input_len = experiment["fixed"].get("input_len", value)
            output_len = experiment["fixed"].get("output_len", value)
            batch_size = experiment["fixed"].get("batch_size", value)
            print(
                f"Case: input_len={input_len}, output_len={output_len}, batch_size={batch_size}"
            )
            rows = run_one_case(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=args.prompt,
                system_prompt=args.system_prompt,
                input_len=input_len,
                output_len=output_len,
                batch_size=batch_size,
                warmup_runs=args.warmup_runs,
                repeats=args.repeats,
                group_name=experiment["group_name"],
                variable_name=experiment["variable_name"],
                variable_value=value,
                enable_thinking=args.enable_thinking,
            )
            all_rows.extend(rows)

    raw_df = pd.DataFrame(all_rows)
    raw_path = output_dirs["raw"] / "benchmark_raw.csv"
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    input_table = aggregate_results(
        raw_df[raw_df["group"] == "input_length_prefill"].copy(), "input_len"
    )
    output_table = aggregate_results(
        raw_df[raw_df["group"] == "output_length_decode"].copy(), "output_len"
    )
    batch_table = aggregate_results(
        raw_df[raw_df["group"] == "batch_size_scaling"].copy(), "batch_size"
    )

    input_table.to_csv(output_dirs["tables"] / "table_input_length_prefill.csv", index=False, encoding="utf-8-sig")
    output_table.to_csv(output_dirs["tables"] / "table_output_length_decode.csv", index=False, encoding="utf-8-sig")
    batch_table.to_csv(output_dirs["tables"] / "table_batch_size_scaling.csv", index=False, encoding="utf-8-sig")

    save_plot(
        input_table,
        x_col="input_len",
        y_col="ttft_ms",
        title="Input Length vs TTFT",
        output_path=output_dirs["figures"] / "input_length_vs_ttft.png",
        y_label="TTFT (ms)",
    )
    save_plot(
        output_table,
        x_col="output_len",
        y_col="tbt_ms",
        title="Output Length vs TBT",
        output_path=output_dirs["figures"] / "output_length_vs_tbt.png",
        y_label="TBT (ms)",
    )
    save_plot(
        batch_table,
        x_col="batch_size",
        y_col="throughput_tok_s",
        title="Batch Size vs Throughput",
        output_path=output_dirs["figures"] / "batch_size_vs_throughput.png",
        y_label="Throughput (tok/s)",
    )

    print(f"Saved raw results to: {raw_path}")
    print(f"Saved tables to: {output_dirs['tables']}")
    print(f"Saved figures to: {output_dirs['figures']}")


if __name__ == "__main__":
    main()
