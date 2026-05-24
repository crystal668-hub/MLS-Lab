import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse MLS Lab 3 vLLM benchmark JSON files.")
    parser.add_argument("--task", choices=["task2", "task3"], required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kv-trace-dir", default=None)
    parser.add_argument(
        "--discard-fraction",
        type=float,
        default=0.10,
        help="Discard the first fraction of detailed request records as warmup.",
    )
    return parser.parse_args()


def load_json_records(input_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name.endswith(".pytorch.json"):
            continue
        text = path.read_text(encoding="utf-8-sig").strip()
        if not text:
            continue
        # vLLM writes one JSON object. With --append-result, it can be JSONL.
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            payload["_source_file"] = path.name
            records.append(payload)
    return records


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q / 100.0
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return values[low]
    frac = pos - low
    return values[low] * (1 - frac) + values[high] * frac


def first_existing(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def get_metadata_value(record: dict[str, Any], key: str) -> Any:
    if key in record:
        return record[key]
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get(key)
    return None


def flatten_record(record: dict[str, Any], discard_fraction: float) -> dict[str, Any]:
    raw_ttfts = [to_float(x) for x in record.get("ttfts", [])]
    raw_ttfts = [x for x in raw_ttfts if x is not None]
    itls_nested = record.get("itls", [])
    request_count = max(len(raw_ttfts), len(itls_nested) if isinstance(itls_nested, list) else 0)
    discard_count = int(request_count * discard_fraction)
    # vLLM detailed ttfts / itls are stored in seconds. Convert to ms for tables.
    ttfts = [x * 1000.0 for x in raw_ttfts[discard_count:]]
    itls: list[float] = []
    if isinstance(itls_nested, list):
        for item in itls_nested[discard_count:]:
            if isinstance(item, list):
                itls.extend(float(x) * 1000.0 for x in item if to_float(x) is not None)
            else:
                value = to_float(item)
                if value is not None:
                    itls.append(value * 1000.0)

    median_ttft_ms = percentile(ttfts, 50) if ttfts else to_float(record.get("median_ttft_ms"))
    p99_ttft_ms = percentile(ttfts, 99) if ttfts else to_float(record.get("p99_ttft_ms"))
    mean_ttft_ms = (
        statistics.mean(ttfts)
        if ttfts
        else to_float(record.get("mean_ttft_ms"))
    )
    median_itl_ms = percentile(itls, 50) if itls else to_float(record.get("median_itl_ms"))
    p99_itl_ms = percentile(itls, 99) if itls else to_float(record.get("p99_itl_ms"))
    mean_itl_ms = (
        statistics.mean(itls)
        if itls
        else to_float(record.get("mean_itl_ms"))
    )

    return {
        "source_file": record.get("_source_file"),
        "task": get_metadata_value(record, "task"),
        "strategy": get_metadata_value(record, "strategy"),
        "scheduler_cls": get_metadata_value(record, "scheduler_cls"),
        "repeat_id": to_float(get_metadata_value(record, "repeat_id")),
        "max_num_batched_tokens": to_float(
            get_metadata_value(record, "max_num_batched_tokens")
        ),
        "request_rate": first_existing(record, ["request_rate"]),
        "completed": to_float(first_existing(record, ["completed", "successful_requests"])),
        "total_input": to_float(first_existing(record, ["total_input", "total_input_tokens"])),
        "total_output": to_float(
            first_existing(record, ["total_output", "total_output_tokens"])
        ),
        "request_throughput": to_float(record.get("request_throughput")),
        "output_throughput": to_float(record.get("output_throughput")),
        "total_token_throughput": to_float(record.get("total_token_throughput")),
        "timeouts": to_float(
            first_existing(
                record,
                [
                    "timeouts",
                    "timeout_count",
                    "num_timeouts",
                    "failed",
                    "failed_requests",
                ],
            )
        )
        or 0.0,
        "median_ttft_ms": median_ttft_ms,
        "p99_ttft_ms": p99_ttft_ms,
        "mean_ttft_ms": mean_ttft_ms,
        "median_itl_ms": median_itl_ms,
        "p99_itl_ms": p99_itl_ms,
        "mean_itl_ms": mean_itl_ms,
        "median_tpot_ms": to_float(record.get("median_tpot_ms")),
        "mean_tpot_ms": to_float(record.get("mean_tpot_ms")),
        "ttfts": ttfts,
        "itls": itls,
    }


def median_aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = [
        "completed",
        "total_input",
        "total_output",
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
        "median_ttft_ms",
        "p99_ttft_ms",
        "mean_ttft_ms",
        "median_itl_ms",
        "p99_itl_ms",
        "mean_itl_ms",
        "median_tpot_ms",
        "mean_tpot_ms",
        "timeouts",
    ]
    available = [col for col in metric_cols if col in df.columns]
    return (
        df.groupby(group_cols, as_index=False)[available]
        .median(numeric_only=True)
        .sort_values(group_cols)
    )


def save_task2_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    x = "max_num_batched_tokens"
    if summary.empty or x not in summary:
        return

    plt.figure(figsize=(6, 4))
    plt.plot(summary[x], summary["median_ttft_ms"], marker="o", label="TTFT P50")
    if "p99_ttft_ms" in summary:
        plt.plot(summary[x], summary["p99_ttft_ms"], marker="s", label="TTFT P99")
    if "median_itl_ms" in summary:
        plt.plot(summary[x], summary["median_itl_ms"], marker="^", label="ITL P50")
    if "p99_itl_ms" in summary:
        plt.plot(summary[x], summary["p99_itl_ms"], marker="v", label="ITL P99")
    plt.xlabel("max_num_batched_tokens")
    plt.ylabel("Latency (ms)")
    plt.title("Task 2 A1: token budget vs TTFT / ITL")
    plt.xscale("log", base=2)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "task2_a1_token_budget_vs_ttft_itl.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    if "output_throughput" in summary:
        plt.plot(
            summary[x],
            summary["output_throughput"],
            marker="o",
            label="Output throughput",
        )
    plt.ylabel("Output throughput (tok/s)")
    plt.xlabel("max_num_batched_tokens")
    plt.title("Task 2 A2: token budget vs throughput")
    plt.xscale("log", base=2)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "task2_a2_token_budget_vs_throughput.png", dpi=200)
    plt.close()


def read_kv_peak_tokens(kv_trace_dir: Path | None) -> dict[str, float]:
    peaks: dict[str, list[float]] = {}
    if kv_trace_dir is None or not kv_trace_dir.exists():
        return {}
    for path in sorted(kv_trace_dir.glob("*.jsonl")):
        strategy = path.stem.split("_repeat_")[0]
        peak = 0.0
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            peak = max(peak, float(record.get("peak_running_computed_tokens", 0)))
        peaks.setdefault(strategy, []).append(peak)
    return {strategy: statistics.median(values) for strategy, values in peaks.items()}


def save_cdf(
    values_by_label: dict[str, list[float]],
    title: str,
    xlabel: str,
    path: Path,
    xlim_percentile: float | None = None,
) -> None:
    plt.figure(figsize=(6, 4))
    all_values: list[float] = []
    for label, values in values_by_label.items():
        values = sorted(x for x in values if x is not None)
        if not values:
            continue
        all_values.extend(values)
        y = [(idx + 1) / len(values) for idx in range(len(values))]
        plt.plot(values, y, label=label)
    if xlim_percentile is not None and all_values:
        upper = percentile(all_values, xlim_percentile)
        if upper is not None:
            plt.xlim(left=0, right=upper)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_task3_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ttft_by_strategy: dict[str, list[float]] = {}
    itl_by_strategy: dict[str, list[float]] = {}
    for row in rows:
        strategy = str(row.get("strategy") or "default")
        ttft_by_strategy.setdefault(strategy, []).extend(row.get("ttfts") or [])
        itl_by_strategy.setdefault(strategy, []).extend(row.get("itls") or [])
    save_cdf(
        ttft_by_strategy,
        "Task 3: TTFT CDF by scheduler",
        "TTFT (ms)",
        figures / "task3_ttft_cdf.png",
    )
    save_cdf(
        itl_by_strategy,
        "Task 3: ITL CDF by scheduler",
        "ITL (ms)",
        figures / "task3_itl_cdf.png",
        xlim_percentile=99.5,
    )


def parse_task2(
    records: list[dict[str, Any]], output_dir: Path, discard_fraction: float
) -> None:
    rows = [flatten_record(record, discard_fraction) for record in records]
    df = pd.DataFrame([{k: v for k, v in row.items() if k not in {"ttfts", "itls"}} for row in rows])
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "task2_raw_summary.csv", index=False, encoding="utf-8-sig")
    summary = median_aggregate(df, ["max_num_batched_tokens"])
    summary.to_csv(output_dir / "task2_median_summary.csv", index=False, encoding="utf-8-sig")
    save_task2_plots(summary, output_dir)


def parse_task3(
    records: list[dict[str, Any]],
    output_dir: Path,
    kv_trace_dir: Path | None,
    discard_fraction: float,
) -> None:
    rows = [flatten_record(record, discard_fraction) for record in records]
    df = pd.DataFrame([{k: v for k, v in row.items() if k not in {"ttfts", "itls"}} for row in rows])
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "task3_raw_summary.csv", index=False, encoding="utf-8-sig")
    summary = median_aggregate(df, ["strategy"])
    kv_peaks = read_kv_peak_tokens(kv_trace_dir)
    if kv_peaks:
        summary["peak_running_computed_tokens"] = summary["strategy"].map(kv_peaks)
    summary.to_csv(output_dir / "task3_median_summary.csv", index=False, encoding="utf-8-sig")
    save_task3_plots(rows, output_dir)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    records = load_json_records(input_dir)
    if not records:
        raise SystemExit(f"No benchmark JSON records found in {input_dir}")
    if args.task == "task2":
        parse_task2(records, output_dir, args.discard_fraction)
    else:
        parse_task3(
            records,
            output_dir,
            Path(args.kv_trace_dir) if args.kv_trace_dir else None,
            args.discard_fraction,
        )
    print(f"Parsed {len(records)} records into {output_dir}")


if __name__ == "__main__":
    main()
