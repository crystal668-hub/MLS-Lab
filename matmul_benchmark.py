from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from infer import choose_device, maybe_synchronize, set_seed
from lab2_utils import (
    BF16_BYTES,
    DEFAULT_GPU_BANDWIDTH_GBPS,
    DEFAULT_GPU_PEAK_TFLOPS,
    DEFAULT_LAB2_OUTPUT_DIR,
    DEFAULT_MATMUL_K_VALUES,
    achieved_tflops,
    arithmetic_intensity,
    ensure_cuda,
    ensure_directory,
    export_environment_json,
    matmul_bytes,
    matmul_flops,
    parse_int_list,
    ridge_point,
    roofline_limit_tflops,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LAB2 BF16 GEMM benchmark for Roofline analysis."
    )
    parser.add_argument("--device", default="auto", help="auto, cuda:0, ...")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument(
        "--k-values",
        default=",".join(str(value) for value in DEFAULT_MATMUL_K_VALUES),
        help="Comma-separated K sweep.",
    )
    parser.add_argument("--warmup-runs", type=int, default=20)
    parser.add_argument("--measure-runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_LAB2_OUTPUT_DIR / "matmul"),
        help="Directory for CSV/PNG outputs.",
    )
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=DEFAULT_GPU_PEAK_TFLOPS,
        help="GPU BF16 peak throughput used only for the overlay roofline curve.",
    )
    parser.add_argument(
        "--bandwidth-gbps",
        type=float,
        default=DEFAULT_GPU_BANDWIDTH_GBPS,
        help="GPU memory bandwidth in GB/s used only for the overlay roofline curve.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow a CPU dry run for debugging. Results are not valid for submission.",
    )
    return parser.parse_args()


def benchmark_one_case(
    *,
    m: int,
    n: int,
    k: int,
    device: torch.device,
    warmup_runs: int,
    measure_runs: int,
) -> dict[str, float | int | str]:
    left = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    right = torch.randn((k, n), device=device, dtype=torch.bfloat16)

    for _ in range(warmup_runs):
        _ = torch.matmul(left, right)
    maybe_synchronize(device)

    timings_ms: list[float] = []
    if device.type == "cuda":
        for _ in range(measure_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _ = torch.matmul(left, right)
            end_event.record()
            torch.cuda.synchronize(device)
            timings_ms.append(start_event.elapsed_time(end_event))
    else:
        import time

        for _ in range(measure_runs):
            start_time = time.perf_counter()
            _ = torch.matmul(left, right)
            maybe_synchronize(device)
            timings_ms.append((time.perf_counter() - start_time) * 1000.0)

    median_ms = float(pd.Series(timings_ms).median())
    flops = matmul_flops(m, n, k)
    bytes_moved = matmul_bytes(m, n, k, dtype_bytes=BF16_BYTES)
    ai = arithmetic_intensity(flops, bytes_moved)
    return {
        "m": m,
        "n": n,
        "k": k,
        "dtype": "bfloat16",
        "warmup_runs": warmup_runs,
        "measure_runs": measure_runs,
        "median_ms": median_ms,
        "mean_ms": float(pd.Series(timings_ms).mean()),
        "std_ms": float(pd.Series(timings_ms).std(ddof=0)),
        "flops": flops,
        "bytes_moved": bytes_moved,
        "arithmetic_intensity": ai,
        "achieved_tflops": achieved_tflops(flops, median_ms),
    }


def plot_results(
    dataframe: pd.DataFrame,
    output_dir: Path,
    peak_tflops: float,
    bandwidth_gbps: float,
) -> None:
    ridge_ai = ridge_point(peak_tflops, bandwidth_gbps)

    plt.figure(figsize=(7, 4.5))
    plt.plot(dataframe["k"], dataframe["achieved_tflops"], marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("K")
    plt.ylabel("Achieved TFLOPS")
    plt.title("BF16 GEMM Throughput vs K")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "matmul_tflops_vs_k.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6.5, 5))
    ai_values = dataframe["arithmetic_intensity"].tolist()
    tflops_values = dataframe["achieved_tflops"].tolist()
    roofline_x = [2**value for value in range(-4, 14)]
    roofline_y = [roofline_limit_tflops(ai, peak_tflops, bandwidth_gbps) for ai in roofline_x]

    plt.plot(roofline_x, roofline_y, label="Theoretical Roofline", linewidth=2.0)
    plt.scatter(ai_values, tflops_values, color="tab:orange", s=50, label="Measured GEMM")
    for row in dataframe.itertuples():
        plt.annotate(f"K={row.k}", (row.arithmetic_intensity, row.achieved_tflops), fontsize=8)

    plt.axvline(ridge_ai, color="tab:red", linestyle="--", linewidth=1.2, label=f"Ridge ≈ {ridge_ai:.1f}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Arithmetic Intensity (FLOP/Byte)")
    plt.ylabel("Achieved TFLOPS")
    plt.title("BF16 GEMM Roofline Points")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "matmul_roofline.png", dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    ensure_cuda(device, allow_cpu=args.allow_cpu)

    output_dir = ensure_directory(args.output_dir)
    export_environment_json(output_dir / "environment.json")

    rows: list[dict[str, float | int | str]] = []
    for k in parse_int_list(args.k_values):
        print(f"Running BF16 GEMM: M={args.m}, N={args.n}, K={k}")
        rows.append(
            benchmark_one_case(
                m=args.m,
                n=args.n,
                k=k,
                device=device,
                warmup_runs=args.warmup_runs,
                measure_runs=args.measure_runs,
            )
        )

    dataframe = pd.DataFrame(rows).sort_values("k")
    dataframe.to_csv(output_dir / "matmul_benchmark.csv", index=False, encoding="utf-8-sig")
    plot_results(
        dataframe=dataframe,
        output_dir=output_dir,
        peak_tflops=args.peak_tflops,
        bandwidth_gbps=args.bandwidth_gbps,
    )
    print(f"Saved benchmark CSV to: {output_dir / 'matmul_benchmark.csv'}")
    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
