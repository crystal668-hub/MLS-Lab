from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from infer import (
    DEFAULT_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    choose_device,
    estimate_tensor_bytes,
    load_tokenizer_and_model,
    maybe_synchronize,
    prepare_batch_inputs,
    set_seed,
)
from lab2_utils import (
    DEFAULT_LAB2_OUTPUT_DIR,
    bytes_to_gib,
    ensure_directory,
    export_environment_json,
    kv_cache_per_token_bytes,
    kv_cache_total_bytes,
    load_model_spec,
    parse_int_list,
    safe_floor_divide,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Qwen3 KV cache memory and service capacity.")
    parser.add_argument("--model-path", required=True, help="Local model path or HF repo id.")
    parser.add_argument(
        "--cases",
        default="1x512,1x1024,1x2048,1x4096,4x1024,8x1024",
        help="Comma-separated batch x seq cases, e.g. 1x512,4x1024.",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda:0, cpu, ...")
    parser.add_argument(
        "--device-map",
        default="none",
        choices=["none", "auto"],
        help="Use auto dispatch only if the model cannot fit on a single device.",
    )
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--measure",
        action="store_true",
        help="Run actual CUDA/CPU model forwards and measure past_key_values size.",
    )
    parser.add_argument(
        "--total-vram-gb",
        type=float,
        default=6.0,
        help="GPU VRAM used for capacity estimation.",
    )
    parser.add_argument(
        "--baseline-runtime-gb",
        type=float,
        default=1.176,
        help="Model + runtime baseline memory from LAB1 small run.",
    )
    parser.add_argument(
        "--safety-margin-gb",
        type=float,
        default=0.5,
        help="Reserved free memory margin.",
    )
    parser.add_argument(
        "--context-scenarios",
        default="4096,8192,32768,40960",
        help="Context lengths for capacity estimates.",
    )
    parser.add_argument(
        "--lab1-output-table",
        default="results/tables/table_output_length_decode.csv",
        help="Existing LAB1 output-length table used for KV cache cross-check.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_LAB2_OUTPUT_DIR / "kv_cache"),
        help="Directory for CSV/JSON outputs.",
    )
    return parser.parse_args()


def parse_cases(raw: str) -> list[tuple[int, int]]:
    cases: list[tuple[int, int]] = []
    for part in raw.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "x" not in part:
            raise ValueError(f"Invalid case {part!r}; expected format batchxseq.")
        batch_size_raw, seq_len_raw = part.split("x", 1)
        cases.append((int(batch_size_raw), int(seq_len_raw)))
    return cases


def build_theory_table(model_path: str, cases: list[tuple[int, int]]) -> pd.DataFrame:
    spec = load_model_spec(model_path)
    per_token_bytes = kv_cache_per_token_bytes(spec)
    rows: list[dict[str, float | int]] = []
    for batch_size, seq_len in cases:
        total_bytes = kv_cache_total_bytes(spec, batch_size=batch_size, seq_len=seq_len)
        rows.append(
            {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "kv_per_token_bytes": per_token_bytes,
                "kv_per_token_kib": per_token_bytes / 1024,
                "theory_bytes": total_bytes,
                "theory_gib": bytes_to_gib(total_bytes),
            }
        )
    return pd.DataFrame(rows)


def measure_kv_cache(
    *,
    model_path: str,
    cases: list[tuple[int, int]],
    device: torch.device,
    device_map: str,
    attn_implementation: str | None,
    prompt: str,
    system_prompt: str,
    enable_thinking: bool,
) -> pd.DataFrame:
    tokenizer, model = load_tokenizer_and_model(
        model_path,
        device=device,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    rows: list[dict[str, float | int | str]] = []
    for batch_size, seq_len in cases:
        row: dict[str, float | int | str] = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "status": "ok",
            "error": "",
        }
        try:
            inputs = prepare_batch_inputs(
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                system_prompt=system_prompt,
                input_len=seq_len,
                batch_size=batch_size,
                enable_thinking=enable_thinking,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=True,
                )
                maybe_synchronize(device)
            actual_bytes = estimate_tensor_bytes(outputs.past_key_values)
            row["actual_bytes"] = actual_bytes
            row["actual_gib"] = bytes_to_gib(actual_bytes)
            row["peak_memory_gib"] = (
                torch.cuda.max_memory_reserved(device) / (1024**3)
                if device.type == "cuda"
                else ""
            )
        except RuntimeError as exc:
            row["status"] = "oom" if "out of memory" in str(exc).lower() else "runtime_error"
            row["error"] = str(exc)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows.append(row)
    return pd.DataFrame(rows)


def build_capacity_table(
    *,
    model_path: str,
    total_vram_gb: float,
    baseline_runtime_gb: float,
    safety_margin_gb: float,
    context_lengths: list[int],
) -> pd.DataFrame:
    spec = load_model_spec(model_path)
    per_token_gib = bytes_to_gib(kv_cache_per_token_bytes(spec))
    available_kv_gb = max(total_vram_gb - baseline_runtime_gb - safety_margin_gb, 0.0)

    rows: list[dict[str, float | int | str]] = [
        {
            "scenario": "single_user_max_context",
            "context_len": safe_floor_divide(available_kv_gb, per_token_gib),
            "max_users": 1,
            "available_kv_gib": available_kv_gb,
            "kv_per_token_gib": per_token_gib,
        }
    ]
    for context_len in context_lengths:
        kv_per_user_gib = context_len * per_token_gib
        rows.append(
            {
                "scenario": f"max_users_at_{context_len}_tokens",
                "context_len": context_len,
                "max_users": safe_floor_divide(available_kv_gb, kv_per_user_gib),
                "available_kv_gib": available_kv_gb,
                "kv_per_user_gib": kv_per_user_gib,
            }
        )
    return pd.DataFrame(rows)


def build_lab1_crosscheck(model_path: str, lab1_output_table: Path) -> pd.DataFrame:
    if not lab1_output_table.exists():
        return pd.DataFrame()
    spec = load_model_spec(model_path)
    lab1_df = pd.read_csv(lab1_output_table)
    if "output_len" not in lab1_df.columns or "kv_cache_delta_gb" not in lab1_df.columns:
        return pd.DataFrame()
    rows: list[dict[str, float | int]] = []
    for row in lab1_df.itertuples():
        output_len = int(row.output_len)
        theory_gib = bytes_to_gib(kv_cache_total_bytes(spec, batch_size=1, seq_len=output_len))
        measured_gib = float(row.kv_cache_delta_gb)
        error_pct = abs(measured_gib - theory_gib) / theory_gib * 100.0 if theory_gib > 0 else 0.0
        rows.append(
            {
                "output_len": output_len,
                "theory_delta_gib": theory_gib,
                "lab1_measured_delta_gib": measured_gib,
                "error_pct": error_pct,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_directory(args.output_dir)
    export_environment_json(output_dir / "environment.json")

    cases = parse_cases(args.cases)
    theory_df = build_theory_table(args.model_path, cases)
    theory_df.to_csv(output_dir / "kv_cache_theory.csv", index=False, encoding="utf-8-sig")

    if args.measure:
        device = choose_device(args.device)
        measured_df = measure_kv_cache(
            model_path=args.model_path,
            cases=cases,
            device=device,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            enable_thinking=args.enable_thinking,
        )
        merged_df = theory_df.merge(measured_df, on=["batch_size", "seq_len"], how="left")
        merged_df["measurement_error_pct"] = (
            (merged_df["actual_gib"] - merged_df["theory_gib"]).abs()
            / merged_df["theory_gib"]
            * 100.0
        )
        merged_df.to_csv(output_dir / "kv_cache_measurements.csv", index=False, encoding="utf-8-sig")

    capacity_df = build_capacity_table(
        model_path=args.model_path,
        total_vram_gb=args.total_vram_gb,
        baseline_runtime_gb=args.baseline_runtime_gb,
        safety_margin_gb=args.safety_margin_gb,
        context_lengths=parse_int_list(args.context_scenarios),
    )
    capacity_df.to_csv(output_dir / "capacity_estimates.csv", index=False, encoding="utf-8-sig")

    crosscheck_df = build_lab1_crosscheck(
        model_path=args.model_path,
        lab1_output_table=Path(args.lab1_output_table),
    )
    if not crosscheck_df.empty:
        crosscheck_df.to_csv(output_dir / "lab1_kv_crosscheck.csv", index=False, encoding="utf-8-sig")

    print(f"Saved KV theory to: {output_dir / 'kv_cache_theory.csv'}")
    print(f"Saved capacity estimates to: {output_dir / 'capacity_estimates.csv'}")
    if args.measure:
        print(f"Saved measurements to: {output_dir / 'kv_cache_measurements.csv'}")
    if not crosscheck_df.empty:
        print(f"Saved LAB1 cross-check to: {output_dir / 'lab1_kv_crosscheck.csv'}")


if __name__ == "__main__":
    main()
