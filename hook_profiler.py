from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from infer import (
    DEFAULT_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    choose_device,
    load_tokenizer_and_model,
    maybe_synchronize,
    prepare_batch_inputs,
    set_seed,
)
from lab2_utils import (
    CudaEventHookProfiler,
    DEFAULT_LAB2_OUTPUT_DIR,
    arithmetic_intensity,
    attention_decode_bytes,
    attention_decode_flops,
    attention_prefill_bytes,
    attention_prefill_flops,
    build_decode_context_lengths,
    achieved_tflops,
    ensure_cuda,
    ensure_directory,
    export_environment_json,
    ffn_bytes,
    ffn_flops,
    load_model_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile Qwen3 layers with forward hooks and CUDA events."
    )
    parser.add_argument("--model-path", required=True, help="Local model path or HF repo id.")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=128)
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
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_LAB2_OUTPUT_DIR / "hook_profiler"),
        help="Directory for CSV/PNG outputs.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow a CPU dry run for debugging. Results are not valid for submission.",
    )
    return parser.parse_args()


def run_prefill(
    *,
    model,
    inputs: dict[str, torch.Tensor],
    device: torch.device,
):
    with torch.inference_mode():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=True,
        )
        maybe_synchronize(device)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return outputs.past_key_values, next_token


def run_decode_steps(
    *,
    model,
    cache,
    next_token: torch.Tensor,
    attention_mask: torch.Tensor,
    decode_steps: int,
    device: torch.device,
):
    running_attention_mask = attention_mask
    with torch.inference_mode():
        for _ in range(decode_steps):
            running_attention_mask = torch.cat(
                [
                    running_attention_mask,
                    torch.ones(
                        (running_attention_mask.shape[0], 1),
                        dtype=running_attention_mask.dtype,
                        device=running_attention_mask.device,
                    ),
                ],
                dim=1,
            )
            outputs = model(
                input_ids=next_token,
                attention_mask=running_attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
            cache = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        maybe_synchronize(device)
    return cache, next_token


def build_layer_breakdown(raw_df: pd.DataFrame, decode_steps: int) -> pd.DataFrame:
    summary = (
        raw_df.groupby(["phase", "layer_idx", "category"], as_index=False)
        .agg(total_ms=("elapsed_ms", "sum"), call_count=("elapsed_ms", "count"))
        .sort_values(["phase", "layer_idx", "category"])
    )

    rows: list[dict[str, float | int | str]] = []
    for (phase, layer_idx), group in summary.groupby(["phase", "layer_idx"]):
        values = {row.category: float(row.total_ms) for row in group.itertuples()}
        counts = {row.category: int(row.call_count) for row in group.itertuples()}
        divisor = decode_steps if phase == "decode" else 1
        total_ms = values.get("layer_total", 0.0) / divisor
        attention_ms = values.get("attention", 0.0) / divisor
        ffn_ms = values.get("ffn", 0.0) / divisor
        layernorm_ms = values.get("layernorm", 0.0) / divisor
        other_ms = max(total_ms - attention_ms - ffn_ms - layernorm_ms, 0.0)
        rows.append(
            {
                "phase": phase,
                "layer_idx": int(layer_idx),
                "layer_total_ms": total_ms,
                "attention_ms": attention_ms,
                "ffn_ms": ffn_ms,
                "layernorm_ms": layernorm_ms,
                "other_ms": other_ms,
                "layer_calls": counts.get("layer_total", 0),
            }
        )
    return pd.DataFrame(rows).sort_values(["phase", "layer_idx"])


def build_module_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_df.groupby(["phase", "category"], as_index=False)
        .agg(total_ms=("elapsed_ms", "sum"), mean_ms=("elapsed_ms", "mean"), call_count=("elapsed_ms", "count"))
        .sort_values(["phase", "category"])
    )


def attach_flops_summary(
    summary_df: pd.DataFrame,
    *,
    model_path: str,
    input_len: int,
    decode_steps: int,
    batch_size: int,
) -> pd.DataFrame:
    spec = load_model_spec(model_path)
    decode_contexts = build_decode_context_lengths(input_len, decode_steps)

    flops_lookup = {
        ("prefill", "ffn"): ffn_flops(spec, input_len) * spec.num_layers * batch_size,
        ("prefill", "attention"): attention_prefill_flops(spec, input_len) * spec.num_layers * batch_size,
        ("decode", "ffn"): ffn_flops(spec, 1) * spec.num_layers * decode_steps * batch_size,
        ("decode", "attention"): sum(
            attention_decode_flops(spec, context_len) * spec.num_layers * batch_size
            for context_len in decode_contexts
        ),
    }
    bytes_lookup = {
        ("prefill", "ffn"): ffn_bytes(spec, input_len) * spec.num_layers * batch_size,
        ("prefill", "attention"): attention_prefill_bytes(spec, input_len) * spec.num_layers * batch_size,
        ("decode", "ffn"): ffn_bytes(spec, 1) * spec.num_layers * decode_steps * batch_size,
        ("decode", "attention"): sum(
            attention_decode_bytes(spec, context_len) * spec.num_layers * batch_size
            for context_len in decode_contexts
        ),
    }

    rows: list[dict] = []
    for row in summary_df.to_dict(orient="records"):
        flops = flops_lookup.get((row["phase"], row["category"]))
        bytes_moved = bytes_lookup.get((row["phase"], row["category"]))
        row["flops"] = flops if flops is not None else ""
        row["bytes_moved"] = bytes_moved if bytes_moved is not None else ""
        row["arithmetic_intensity"] = (
            arithmetic_intensity(float(flops), float(bytes_moved))
            if flops is not None and bytes_moved is not None
            else ""
        )
        row["achieved_tflops"] = (
            achieved_tflops(float(flops), float(row["total_ms"])) if flops is not None else ""
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_layer_breakdown(layer_df: pd.DataFrame, output_dir: Path) -> None:
    for phase in ["prefill", "decode"]:
        phase_df = layer_df[layer_df["phase"] == phase].copy()
        if phase_df.empty:
            continue
        plt.figure(figsize=(8, 4.5))
        bottom = [0.0] * len(phase_df)
        x_values = phase_df["layer_idx"].tolist()
        for column, label in [
            ("attention_ms", "Attention"),
            ("ffn_ms", "FFN"),
            ("layernorm_ms", "LayerNorm"),
            ("other_ms", "Other"),
        ]:
            values = phase_df[column].tolist()
            plt.bar(x_values, values, bottom=bottom, label=label)
            bottom = [left + value for left, value in zip(bottom, values)]
        ylabel = "Time per pass (ms)" if phase == "prefill" else "Mean time per decode step (ms)"
        plt.xlabel("Layer index")
        plt.ylabel(ylabel)
        plt.title(f"Qwen3 {phase.capitalize()} Layer Breakdown")
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{phase}_layer_breakdown.png", dpi=220)
        plt.close()


def plot_global_share(layer_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for phase in ["prefill", "decode"]:
        phase_df = layer_df[layer_df["phase"] == phase]
        if phase_df.empty:
            continue
        totals = {
            "attention": float(phase_df["attention_ms"].sum()),
            "ffn": float(phase_df["ffn_ms"].sum()),
            "layernorm": float(phase_df["layernorm_ms"].sum()),
            "other": float(phase_df["other_ms"].sum()),
        }
        total = sum(totals.values())
        for category, value in totals.items():
            rows.append(
                {
                    "phase": phase,
                    "category": category,
                    "time_ms": value,
                    "share": value / total if total > 0 else 0.0,
                }
            )

    share_df = pd.DataFrame(rows)
    if share_df.empty:
        return share_df

    pivot_df = share_df.pivot(index="phase", columns="category", values="share").fillna(0.0)
    pivot_df.plot(kind="bar", stacked=True, figsize=(6, 4), colormap="tab20")
    plt.ylabel("Time Share")
    plt.title("Global Module Time Share")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "global_time_share.png", dpi=220)
    plt.close()
    return share_df


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    ensure_cuda(device, allow_cpu=args.allow_cpu)

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

    profiler = CudaEventHookProfiler(enable_events=device.type == "cuda")
    profiler.register(model)
    try:
        profiler.set_phase("prefill")
        cache, next_token = run_prefill(model=model, inputs=inputs, device=device)

        profiler.set_phase("decode")
        run_decode_steps(
            model=model,
            cache=cache,
            next_token=next_token,
            attention_mask=inputs["attention_mask"],
            decode_steps=args.decode_steps,
            device=device,
        )
    finally:
        profiler.remove()

    raw_df = pd.DataFrame(profiler.records())
    raw_df.to_csv(output_dir / "raw_hook_events.csv", index=False, encoding="utf-8-sig")

    layer_df = build_layer_breakdown(raw_df, decode_steps=args.decode_steps)
    layer_df.to_csv(output_dir / "layer_breakdown.csv", index=False, encoding="utf-8-sig")

    module_summary_df = build_module_summary(raw_df)
    module_summary_df = attach_flops_summary(
        module_summary_df,
        model_path=args.model_path,
        input_len=args.input_len,
        decode_steps=args.decode_steps,
        batch_size=args.batch_size,
    )
    module_summary_df.to_csv(output_dir / "module_summary.csv", index=False, encoding="utf-8-sig")

    share_df = plot_global_share(layer_df, output_dir)
    if not share_df.empty:
        share_df.to_csv(output_dir / "global_time_share.csv", index=False, encoding="utf-8-sig")
    plot_layer_breakdown(layer_df, output_dir)

    print(f"Saved hook events to: {output_dir / 'raw_hook_events.csv'}")
    print(f"Saved layer breakdown to: {output_dir / 'layer_breakdown.csv'}")
    print(f"Saved module summary to: {output_dir / 'module_summary.csv'}")


if __name__ == "__main__":
    main()
