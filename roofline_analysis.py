from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from lab2_utils import (
    DEFAULT_GPU_BANDWIDTH_GBPS,
    DEFAULT_GPU_PEAK_TFLOPS,
    DEFAULT_LAB2_OUTPUT_DIR,
    attention_decode_bytes,
    attention_decode_flops,
    attention_prefill_bytes,
    attention_prefill_flops,
    build_decode_context_lengths,
    ensure_directory,
    export_environment_json,
    ffn_bytes,
    ffn_flops,
    flops_bytes_to_point,
    load_model_spec,
    ridge_point,
    roofline_limit_tflops,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and plot LAB2 Roofline points.")
    parser.add_argument("--model-path", required=True, help="Local model path or HF repo id.")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=DEFAULT_GPU_PEAK_TFLOPS,
        help="Measured or estimated GPU BF16 peak TFLOPS.",
    )
    parser.add_argument(
        "--bandwidth-gbps",
        type=float,
        default=DEFAULT_GPU_BANDWIDTH_GBPS,
        help="Measured or estimated GPU memory bandwidth in GB/s.",
    )
    parser.add_argument(
        "--matmul-csv",
        default=str(DEFAULT_LAB2_OUTPUT_DIR / "matmul" / "matmul_benchmark.csv"),
        help="CSV from matmul_benchmark.py.",
    )
    parser.add_argument(
        "--hook-summary-csv",
        default=str(DEFAULT_LAB2_OUTPUT_DIR / "hook_profiler" / "module_summary.csv"),
        help="CSV from hook_profiler.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_LAB2_OUTPUT_DIR / "roofline"),
        help="Directory for LAB2 Roofline outputs.",
    )
    return parser.parse_args()


def build_theoretical_points(
    *,
    model_path: str,
    input_len: int,
    decode_steps: int,
    peak_tflops: float,
    bandwidth_gbps: float,
) -> pd.DataFrame:
    spec = load_model_spec(model_path)
    decode_contexts = build_decode_context_lengths(input_len, decode_steps)

    attn_decode_flops_total = sum(attention_decode_flops(spec, context_len) for context_len in decode_contexts)
    attn_decode_bytes_total = sum(attention_decode_bytes(spec, context_len) for context_len in decode_contexts)

    rows = [
        flops_bytes_to_point(
            module="ffn",
            phase="prefill",
            flops=ffn_flops(spec, input_len),
            bytes_moved=ffn_bytes(spec, input_len),
            peak_tflops=peak_tflops,
            bandwidth_gbps=bandwidth_gbps,
        ),
        flops_bytes_to_point(
            module="attention",
            phase="prefill",
            flops=attention_prefill_flops(spec, input_len),
            bytes_moved=attention_prefill_bytes(spec, input_len),
            peak_tflops=peak_tflops,
            bandwidth_gbps=bandwidth_gbps,
        ),
        flops_bytes_to_point(
            module="ffn",
            phase="decode",
            flops=ffn_flops(spec, 1),
            bytes_moved=ffn_bytes(spec, 1),
            peak_tflops=peak_tflops,
            bandwidth_gbps=bandwidth_gbps,
        ),
        flops_bytes_to_point(
            module="attention",
            phase="decode",
            flops=attn_decode_flops_total / decode_steps,
            bytes_moved=attn_decode_bytes_total / decode_steps,
            peak_tflops=peak_tflops,
            bandwidth_gbps=bandwidth_gbps,
        ),
    ]
    dataframe = pd.DataFrame(rows)
    dataframe["num_layers"] = spec.num_layers
    dataframe["input_len"] = input_len
    dataframe["decode_steps"] = decode_steps
    dataframe["query_proj_dim"] = spec.query_proj_dim
    dataframe["kv_proj_dim"] = spec.kv_proj_dim
    return dataframe


def build_measured_points(
    *,
    hook_summary_csv: Path,
    model_path: str,
    input_len: int,
    decode_steps: int,
    peak_tflops: float,
    bandwidth_gbps: float,
) -> pd.DataFrame:
    if not hook_summary_csv.exists():
        return pd.DataFrame()

    summary_df = pd.read_csv(hook_summary_csv)
    spec = load_model_spec(model_path)
    decode_contexts = build_decode_context_lengths(input_len, decode_steps)

    expected = {
        ("prefill", "ffn"): ffn_flops(spec, input_len) * spec.num_layers,
        ("prefill", "attention"): attention_prefill_flops(spec, input_len) * spec.num_layers,
        ("decode", "ffn"): ffn_flops(spec, 1) * spec.num_layers * decode_steps,
        ("decode", "attention"): sum(
            attention_decode_flops(spec, context_len) * spec.num_layers
            for context_len in decode_contexts
        ),
    }

    rows: list[dict[str, float | str]] = []
    for (phase, category), flops in expected.items():
        matched = summary_df[(summary_df["phase"] == phase) & (summary_df["category"] == category)]
        if matched.empty:
            continue

        total_ms = float(matched.iloc[0]["total_ms"])
        theoretical = build_theoretical_points(
            model_path=model_path,
            input_len=input_len,
            decode_steps=decode_steps,
            peak_tflops=peak_tflops,
            bandwidth_gbps=bandwidth_gbps,
        )
        theory_row = theoretical[
            (theoretical["phase"] == phase) & (theoretical["module"] == category)
        ].iloc[0]

        rows.append(
            {
                "phase": phase,
                "module": category,
                "flops": flops,
                "total_ms": total_ms,
                "achieved_tflops": flops / (total_ms / 1000.0) / 1e12 if total_ms > 0 else 0.0,
                "arithmetic_intensity": float(theory_row["arithmetic_intensity"]),
                "roofline_tflops": float(theory_row["roofline_tflops"]),
            }
        )
    return pd.DataFrame(rows)


def plot_roofline(
    *,
    output_dir: Path,
    theoretical_df: pd.DataFrame,
    measured_df: pd.DataFrame,
    matmul_df: pd.DataFrame,
    peak_tflops: float,
    bandwidth_gbps: float,
) -> None:
    ridge_ai = ridge_point(peak_tflops, bandwidth_gbps)
    roofline_x = [2**value for value in range(-4, 14)]
    roofline_y = [roofline_limit_tflops(ai, peak_tflops, bandwidth_gbps) for ai in roofline_x]

    plt.figure(figsize=(7.5, 5.5))
    plt.plot(roofline_x, roofline_y, linewidth=2.2, label="Theoretical Roofline")

    if not matmul_df.empty:
        plt.plot(
            matmul_df["arithmetic_intensity"],
            matmul_df["achieved_tflops"],
            marker="o",
            linestyle="--",
            color="tab:gray",
            label="Measured GEMM sweep",
        )

    if not theoretical_df.empty:
        for row in theoretical_df.itertuples():
            plt.scatter(
                row.arithmetic_intensity,
                row.roofline_tflops,
                marker="^",
                s=70,
                label=f"Theory {row.module}-{row.phase}",
            )

    if not measured_df.empty:
        for row in measured_df.itertuples():
            plt.scatter(
                row.arithmetic_intensity,
                row.achieved_tflops,
                marker="o",
                s=80,
                label=f"Measured {row.module}-{row.phase}",
            )

    plt.axvline(ridge_ai, color="tab:red", linestyle="--", linewidth=1.2, label=f"Ridge ≈ {ridge_ai:.1f}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Arithmetic Intensity (FLOP/Byte)")
    plt.ylabel("TFLOPS")
    plt.title("Qwen3-0.6B Roofline Analysis")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    handles, labels = plt.gca().get_legend_handles_labels()
    dedup: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        dedup.setdefault(label, handle)
    plt.legend(dedup.values(), dedup.keys(), fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "roofline_qwen3.png", dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    export_environment_json(output_dir / "environment.json")

    theoretical_df = build_theoretical_points(
        model_path=args.model_path,
        input_len=args.input_len,
        decode_steps=args.decode_steps,
        peak_tflops=args.peak_tflops,
        bandwidth_gbps=args.bandwidth_gbps,
    )
    theoretical_df.to_csv(output_dir / "theoretical_points.csv", index=False, encoding="utf-8-sig")

    measured_df = build_measured_points(
        hook_summary_csv=Path(args.hook_summary_csv),
        model_path=args.model_path,
        input_len=args.input_len,
        decode_steps=args.decode_steps,
        peak_tflops=args.peak_tflops,
        bandwidth_gbps=args.bandwidth_gbps,
    )
    if not measured_df.empty:
        measured_df.to_csv(output_dir / "measured_points.csv", index=False, encoding="utf-8-sig")

    matmul_csv = Path(args.matmul_csv)
    matmul_df = pd.read_csv(matmul_csv) if matmul_csv.exists() else pd.DataFrame()
    if not matmul_df.empty:
        matmul_df.to_csv(output_dir / "matmul_points.csv", index=False, encoding="utf-8-sig")

    plot_roofline(
        output_dir=output_dir,
        theoretical_df=theoretical_df,
        measured_df=measured_df,
        matmul_df=matmul_df,
        peak_tflops=args.peak_tflops,
        bandwidth_gbps=args.bandwidth_gbps,
    )
    print(f"Saved theoretical points to: {output_dir / 'theoretical_points.csv'}")
    if not measured_df.empty:
        print(f"Saved measured points to: {output_dir / 'measured_points.csv'}")
    print(f"Saved roofline plot to: {output_dir / 'roofline_qwen3.png'}")


if __name__ == "__main__":
    main()
