import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_PROMPT = "请用简洁的中文解释什么是 prefill 和 decode。"
MARKER_TEXT = "<<<BENCHMARK_MARKER_91F1>>>"


@dataclass
class InferenceMetrics:
    batch_size: int
    input_len: int
    requested_output_len: int
    generated_len: int
    ttft_ms: float
    decode_total_ms: float
    tbt_ms: float
    prefill_tok_s: float
    decode_tok_s: float
    throughput_tok_s: float
    peak_memory_gb: float | None
    kv_cache_prefill_gb: float
    kv_cache_final_gb: float
    kv_cache_delta_gb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3 inference with generate() or manual prefill/decode timing."
    )
    parser.add_argument("--model-path", required=True, help="Local path or HF repo id.")
    parser.add_argument("--mode", choices=["generate", "manual"], default="generate")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode in the chat template. Default is disabled for stable local benchmarking.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="auto", help="auto, cuda:0, cpu, ...")
    parser.add_argument(
        "--device-map",
        default="none",
        choices=["none", "auto"],
        help="Use auto dispatch if the model does not fit on a single device.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional attention backend, such as sdpa or flash_attention_2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Only used when sampling is enabled.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="If set, build a synthetic prompt with an exact token length.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for synthetic benchmark-style prompts.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save metrics and generated text as JSON.",
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    peak_bytes = torch.cuda.max_memory_reserved(device)
    return peak_bytes / (1024**3)


def format_gb(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def find_subsequence(sequence: Sequence[int], pattern: Sequence[int]) -> int:
    for idx in range(0, len(sequence) - len(pattern) + 1):
        if list(sequence[idx : idx + len(pattern)]) == list(pattern):
            return idx
    raise ValueError("Marker token sequence not found in chat template output.")


def chat_template_to_ids(encoded: Any) -> list[int]:
    if torch.is_tensor(encoded):
        return encoded[0].tolist()
    if hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids
        if torch.is_tensor(input_ids):
            return input_ids[0].tolist()
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                return input_ids[0]
            return input_ids
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    if isinstance(encoded, list):
        if not encoded:
            return []
        first = encoded[0]
        if isinstance(first, int):
            return encoded
        if hasattr(first, "ids"):
            return list(first.ids)
        if isinstance(first, list):
            return first
    raise TypeError(f"Unsupported chat template return type: {type(encoded)!r}")


def build_chat_input_ids(
    tokenizer: AutoTokenizer,
    prompt: str,
    system_prompt: str,
    enable_thinking: bool = False,
) -> list[int]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return chat_template_to_ids(encoded)


def build_exact_length_input_ids(
    tokenizer: AutoTokenizer,
    target_len: int,
    system_prompt: str,
    enable_thinking: bool = False,
    filler_token_id: int | None = None,
) -> list[int]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": MARKER_TEXT},
    ]
    template_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    template_ids = chat_template_to_ids(template_ids)
    marker_ids = tokenizer(MARKER_TEXT, add_special_tokens=False)["input_ids"]
    marker_start = find_subsequence(template_ids, marker_ids)
    prefix_ids = template_ids[:marker_start]
    suffix_ids = template_ids[marker_start + len(marker_ids) :]

    if filler_token_id is None:
        candidate_ids = tokenizer(" benchmark", add_special_tokens=False)["input_ids"]
        if not candidate_ids:
            raise ValueError("Failed to derive a filler token from tokenizer.")
        filler_token_id = candidate_ids[0]

    base_len = len(prefix_ids) + len(suffix_ids)
    if base_len > target_len:
        raise ValueError(
            f"Target input length {target_len} is too small. Minimum is {base_len}."
        )
    filler_len = target_len - base_len
    filler_ids = [filler_token_id] * filler_len
    return prefix_ids + filler_ids + suffix_ids


def prepare_batch_inputs(
    tokenizer: AutoTokenizer,
    device: torch.device,
    prompt: str,
    system_prompt: str,
    input_len: int | None = None,
    batch_size: int = 1,
    enable_thinking: bool = False,
) -> dict[str, torch.Tensor]:
    if input_len is None:
        input_ids = build_chat_input_ids(
            tokenizer, prompt, system_prompt, enable_thinking=enable_thinking
        )
    else:
        input_ids = build_exact_length_input_ids(
            tokenizer,
            input_len,
            system_prompt,
            enable_thinking=enable_thinking,
        )
    batch_ids = torch.tensor([input_ids] * batch_size, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(batch_ids, dtype=torch.long, device=device)
    return {"input_ids": batch_ids, "attention_mask": attention_mask}


def estimate_tensor_bytes(value: Any, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    if torch.is_tensor(value):
        ptr = value.data_ptr()
        if ptr in seen:
            return 0
        seen.add(ptr)
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(estimate_tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(estimate_tensor_bytes(item, seen) for item in value)
    if hasattr(value, "to_legacy_cache"):
        try:
            return estimate_tensor_bytes(value.to_legacy_cache(), seen)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return estimate_tensor_bytes(vars(value), seen)
    return 0


def load_tokenizer_and_model(
    model_path: str,
    device: torch.device,
    device_map: str,
    attn_implementation: str | None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": "auto",
        "trust_remote_code": True,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if device_map == "auto":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if device_map == "none":
        model.to(device)
    model.eval()
    return tokenizer, model


def generate_text(
    model,
    tokenizer,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[torch.Tensor, str]:
    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
    generated_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return new_tokens, generated_text[0] if len(generated_text) == 1 else "\n".join(generated_text)


def manual_prefill_decode(
    model,
    tokenizer,
    inputs: dict[str, torch.Tensor],
    requested_output_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, str, InferenceMetrics]:
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    batch_size, input_len = input_ids.shape

    generated_tokens: list[torch.Tensor] = []
    decode_step_times: list[float] = []

    reset_peak_memory(device)
    with torch.inference_mode():
        maybe_synchronize(device)
        prefill_start = time.perf_counter()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        maybe_synchronize(device)
        prefill_time_s = time.perf_counter() - prefill_start

        cache = outputs.past_key_values
        kv_prefill_bytes = estimate_tensor_bytes(cache)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        running_attention_mask = attention_mask
        if requested_output_len > 0:
            generated_tokens.append(next_token)

        for _ in range(max(requested_output_len - 1, 0)):
            running_attention_mask = torch.cat(
                [
                    running_attention_mask,
                    torch.ones(
                        (batch_size, 1),
                        dtype=running_attention_mask.dtype,
                        device=running_attention_mask.device,
                    ),
                ],
                dim=1,
            )
            maybe_synchronize(device)
            step_start = time.perf_counter()
            outputs = model(
                input_ids=next_token,
                attention_mask=running_attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
            maybe_synchronize(device)
            decode_step_times.append(time.perf_counter() - step_start)
            cache = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token)

    generated_ids = (
        torch.cat(generated_tokens, dim=1)
        if generated_tokens
        else torch.empty((batch_size, 0), dtype=torch.long, device=input_ids.device)
    )
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    generated_len = generated_ids.shape[1]
    decode_total_s = sum(decode_step_times)
    decode_denominator = max(generated_len - 1, 1)
    kv_final_bytes = estimate_tensor_bytes(cache)
    total_time_s = prefill_time_s + decode_total_s

    metrics = InferenceMetrics(
        batch_size=batch_size,
        input_len=input_len,
        requested_output_len=requested_output_len,
        generated_len=generated_len,
        ttft_ms=prefill_time_s * 1000,
        decode_total_ms=decode_total_s * 1000,
        tbt_ms=(decode_total_s / decode_denominator) * 1000 if generated_len > 1 else 0.0,
        prefill_tok_s=(batch_size * input_len / prefill_time_s) if prefill_time_s > 0 else 0.0,
        decode_tok_s=(
            batch_size * max(generated_len - 1, 0) / decode_total_s
            if decode_total_s > 0
            else 0.0
        ),
        throughput_tok_s=(
            batch_size * generated_len / total_time_s if total_time_s > 0 else 0.0
        ),
        peak_memory_gb=get_peak_memory_gb(device),
        kv_cache_prefill_gb=kv_prefill_bytes / (1024**3),
        kv_cache_final_gb=kv_final_bytes / (1024**3),
        kv_cache_delta_gb=(kv_final_bytes - kv_prefill_bytes) / (1024**3),
    )
    rendered_text = generated_text[0] if len(generated_text) == 1 else "\n".join(generated_text)
    return generated_ids, rendered_text, metrics


def print_metrics(metrics: InferenceMetrics) -> None:
    print("=== Metrics ===")
    print(f"batch_size            : {metrics.batch_size}")
    print(f"input_len             : {metrics.input_len}")
    print(f"requested_output_len  : {metrics.requested_output_len}")
    print(f"generated_len         : {metrics.generated_len}")
    print(f"TTFT (ms)             : {metrics.ttft_ms:.2f}")
    print(f"Decode total (ms)     : {metrics.decode_total_ms:.2f}")
    print(f"TBT (ms)              : {metrics.tbt_ms:.2f}")
    print(f"Prefill tok/s         : {metrics.prefill_tok_s:.2f}")
    print(f"Decode tok/s          : {metrics.decode_tok_s:.2f}")
    print(f"Throughput tok/s      : {metrics.throughput_tok_s:.2f}")
    print(f"Peak memory (GB)      : {format_gb(metrics.peak_memory_gb)}")
    print(f"KV cache prefill (GB) : {metrics.kv_cache_prefill_gb:.4f}")
    print(f"KV cache final (GB)   : {metrics.kv_cache_final_gb:.4f}")
    print(f"KV cache delta (GB)   : {metrics.kv_cache_delta_gb:.4f}")


def maybe_save_json(
    path: str | None,
    metrics: InferenceMetrics | None,
    generated_text: str,
    args: argparse.Namespace,
) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "generated_text": generated_text,
        "metrics": asdict(metrics) if metrics else None,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    print(f"Using device: {device}")
    if args.mode == "manual" and device.type != "cuda":
        print(
            "Warning: manual benchmark mode is designed for CUDA. "
            "Current metrics on CPU are only for code validation."
        )

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

    print(f"Input token length: {inputs['input_ids'].shape[1]}")
    if args.mode == "generate":
        generated_ids, generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print("=== Generated Text ===")
        print(generated_text.strip())
        maybe_save_json(args.save_json, None, generated_text, args)
        print(f"Generated tokens: {generated_ids.shape[1]}")
        return

    _, generated_text, metrics = manual_prefill_decode(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        requested_output_len=args.max_new_tokens,
        device=device,
    )
    print("=== Generated Text ===")
    print(generated_text.strip())
    print_metrics(metrics)
    maybe_save_json(args.save_json, metrics, generated_text, args)


if __name__ == "__main__":
    main()
