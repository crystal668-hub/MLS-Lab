from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.profiler import record_function
from transformers import AutoConfig


BF16_BYTES = 2
DEFAULT_LAB2_OUTPUT_DIR = Path("results") / "lab2"
DEFAULT_MATMUL_K_VALUES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
DEFAULT_GPU_PEAK_TFLOPS = 54.2
DEFAULT_GPU_BANDWIDTH_GBPS = 168.0


@dataclass(frozen=True)
class QwenModelSpec:
    model_type: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    dtype_bytes: int = BF16_BYTES

    @property
    def query_proj_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_proj_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim


@dataclass(frozen=True)
class RooflinePoint:
    module: str
    phase: str
    flops: float
    bytes_moved: float
    arithmetic_intensity: float
    roofline_tflops: float
    region: str


@dataclass(frozen=True)
class EnvironmentInfo:
    torch_version: str
    transformers_version: str
    cuda_available: bool
    cuda_version: str | None
    device_name: str | None
    total_memory_gb: float | None


def ensure_directory(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_model_spec(model_path: str) -> QwenModelSpec:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return QwenModelSpec(
        model_type=config.model_type,
        num_layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
    )


def collect_environment_info() -> EnvironmentInfo:
    device_name: str | None = None
    total_memory_gb: float | None = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        device_name = props.name
        total_memory_gb = props.total_memory / (1024**3)

    transformers_module = __import__("transformers")
    return EnvironmentInfo(
        torch_version=torch.__version__,
        transformers_version=getattr(transformers_module, "__version__", "unknown"),
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
        device_name=device_name,
        total_memory_gb=total_memory_gb,
    )


def ensure_cuda(device: torch.device, allow_cpu: bool = False) -> None:
    if device.type == "cuda":
        return
    if allow_cpu:
        return
    raise RuntimeError(
        "LAB2 正式实验必须在 CUDA 环境下运行。当前 device="
        f"{device}; torch.cuda.is_available()={torch.cuda.is_available()}."
    )


def bytes_to_gib(value: float) -> float:
    return value / (1024**3)


def bytes_to_mib(value: float) -> float:
    return value / (1024**2)


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def matmul_flops(m: int, n: int, k: int) -> int:
    return 2 * m * n * k


def matmul_bytes(m: int, n: int, k: int, dtype_bytes: int = BF16_BYTES) -> int:
    return dtype_bytes * (m * k + k * n + m * n)


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    if bytes_moved <= 0:
        return 0.0
    return flops / bytes_moved


def achieved_tflops(flops: float, elapsed_ms: float) -> float:
    elapsed_s = elapsed_ms / 1000.0
    if elapsed_s <= 0:
        return 0.0
    return flops / elapsed_s / 1e12


def roofline_limit_tflops(
    arithmetic_intensity_value: float,
    peak_tflops: float,
    bandwidth_gbps: float,
) -> float:
    memory_bound_tflops = arithmetic_intensity_value * bandwidth_gbps / 1000.0
    return min(memory_bound_tflops, peak_tflops)


def region_from_ai(
    arithmetic_intensity_value: float,
    peak_tflops: float,
    bandwidth_gbps: float,
) -> str:
    ridge_point = peak_tflops * 1000.0 / bandwidth_gbps
    return "compute-bound" if arithmetic_intensity_value >= ridge_point else "memory-bound"


def ridge_point(peak_tflops: float, bandwidth_gbps: float) -> float:
    return peak_tflops * 1000.0 / bandwidth_gbps


def ffn_flops(spec: QwenModelSpec, tokens: int) -> int:
    return 6 * tokens * spec.hidden_size * spec.intermediate_size


def ffn_bytes(spec: QwenModelSpec, tokens: int) -> int:
    weight_bytes = spec.dtype_bytes * 3 * spec.hidden_size * spec.intermediate_size
    activation_bytes = spec.dtype_bytes * (
        2 * tokens * spec.hidden_size + 3 * tokens * spec.intermediate_size
    )
    return weight_bytes + activation_bytes


def attention_prefill_flops(spec: QwenModelSpec, tokens: int) -> int:
    q_dim = spec.query_proj_dim
    kv_dim = spec.kv_proj_dim
    return (
        2 * tokens * spec.hidden_size * (q_dim + 2 * kv_dim)
        + 2 * tokens * q_dim * spec.hidden_size
        + 4 * tokens * tokens * q_dim
    )


def attention_prefill_bytes(spec: QwenModelSpec, tokens: int) -> int:
    q_dim = spec.query_proj_dim
    kv_dim = spec.kv_proj_dim
    weight_bytes = spec.dtype_bytes * spec.hidden_size * (2 * q_dim + 2 * kv_dim)
    activation_bytes = spec.dtype_bytes * tokens * (
        2 * spec.hidden_size + q_dim + 4 * kv_dim
    )
    return weight_bytes + activation_bytes


def attention_decode_flops(spec: QwenModelSpec, context_len: int) -> int:
    q_dim = spec.query_proj_dim
    kv_dim = spec.kv_proj_dim
    return (
        2 * spec.hidden_size * (q_dim + 2 * kv_dim)
        + 2 * q_dim * spec.hidden_size
        + 4 * context_len * q_dim
    )


def attention_decode_bytes(spec: QwenModelSpec, context_len: int) -> int:
    q_dim = spec.query_proj_dim
    kv_dim = spec.kv_proj_dim
    weight_bytes = spec.dtype_bytes * spec.hidden_size * (2 * q_dim + 2 * kv_dim)
    activation_bytes = spec.dtype_bytes * (
        2 * spec.hidden_size + q_dim + 4 * kv_dim
    )
    kv_cache_read_bytes = spec.dtype_bytes * 2 * context_len * kv_dim
    return weight_bytes + activation_bytes + kv_cache_read_bytes


def roofline_point(
    module: str,
    phase: str,
    flops: float,
    bytes_moved: float,
    peak_tflops: float,
    bandwidth_gbps: float,
) -> RooflinePoint:
    ai = arithmetic_intensity(flops, bytes_moved)
    return RooflinePoint(
        module=module,
        phase=phase,
        flops=flops,
        bytes_moved=bytes_moved,
        arithmetic_intensity=ai,
        roofline_tflops=roofline_limit_tflops(ai, peak_tflops, bandwidth_gbps),
        region=region_from_ai(ai, peak_tflops, bandwidth_gbps),
    )


def kv_cache_per_token_bytes(spec: QwenModelSpec) -> int:
    return (
        2
        * spec.num_layers
        * spec.num_key_value_heads
        * spec.head_dim
        * spec.dtype_bytes
    )


def kv_cache_total_bytes(spec: QwenModelSpec, batch_size: int, seq_len: int) -> int:
    return batch_size * seq_len * kv_cache_per_token_bytes(spec)


def build_decode_context_lengths(input_len: int, decode_steps: int) -> list[int]:
    return [input_len + step for step in range(decode_steps)]


def flops_bytes_to_point(
    module: str,
    phase: str,
    flops: float,
    bytes_moved: float,
    peak_tflops: float,
    bandwidth_gbps: float,
) -> dict[str, float | str]:
    point = roofline_point(
        module=module,
        phase=phase,
        flops=flops,
        bytes_moved=bytes_moved,
        peak_tflops=peak_tflops,
        bandwidth_gbps=bandwidth_gbps,
    )
    return {
        "module": point.module,
        "phase": point.phase,
        "flops": point.flops,
        "bytes_moved": point.bytes_moved,
        "arithmetic_intensity": point.arithmetic_intensity,
        "roofline_tflops": point.roofline_tflops,
        "region": point.region,
    }


def export_environment_json(path: str | Path) -> None:
    save_json(path, asdict(collect_environment_info()))


def decoder_layers(model) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("Unable to locate decoder layers on this model.")


class CudaEventHookProfiler:
    def __init__(
        self,
        *,
        enable_events: bool = True,
        enable_record_functions: bool = True,
    ) -> None:
        self.enable_events = enable_events and torch.cuda.is_available()
        self.enable_record_functions = enable_record_functions
        self.phase = "idle"
        self._handles: list[Any] = []
        self._pending: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._raw_records: list[dict[str, Any]] = []
        self._call_counters: dict[tuple[str, str], int] = defaultdict(int)

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def register(self, model) -> None:
        for layer_idx, layer in enumerate(decoder_layers(model)):
            self._register_module(
                module=layer,
                label=f"layer_{layer_idx}.total",
                category="layer_total",
                layer_idx=layer_idx,
            )
            self._register_module(
                module=layer.self_attn,
                label=f"layer_{layer_idx}.self_attn",
                category="attention",
                layer_idx=layer_idx,
            )
            self._register_module(
                module=layer.mlp,
                label=f"layer_{layer_idx}.mlp",
                category="ffn",
                layer_idx=layer_idx,
            )
            self._register_module(
                module=layer.input_layernorm,
                label=f"layer_{layer_idx}.input_layernorm",
                category="layernorm",
                layer_idx=layer_idx,
            )
            self._register_module(
                module=layer.post_attention_layernorm,
                label=f"layer_{layer_idx}.post_attention_layernorm",
                category="layernorm",
                layer_idx=layer_idx,
            )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def finalize(self) -> None:
        if self.enable_events:
            torch.cuda.synchronize()
        for record in self._raw_records:
            if record.get("elapsed_ms") is not None:
                continue
            if self.enable_events:
                record["elapsed_ms"] = record["start_event"].elapsed_time(record["end_event"])
            else:
                record["elapsed_ms"] = (record["cpu_end"] - record["cpu_start"]) * 1000.0

    def records(self) -> list[dict[str, Any]]:
        self.finalize()
        return [
            {
                key: value
                for key, value in record.items()
                if key not in {"start_event", "end_event", "trace_context", "cpu_start", "cpu_end"}
            }
            for record in self._raw_records
        ]

    def _register_module(
        self,
        *,
        module,
        label: str,
        category: str,
        layer_idx: int,
    ) -> None:
        metadata = {"label": label, "category": category, "layer_idx": layer_idx}
        self._handles.append(module.register_forward_pre_hook(self._pre_hook(metadata)))
        self._handles.append(module.register_forward_hook(self._post_hook(metadata)))

    def _pre_hook(self, metadata: dict[str, Any]):
        def hook_fn(module, inputs):
            trace_context = None
            if self.enable_record_functions:
                trace_context = record_function(f"{self.phase}:{metadata['label']}")
                trace_context.__enter__()

            record = {
                "phase": self.phase,
                "module": metadata["label"],
                "category": metadata["category"],
                "layer_idx": metadata["layer_idx"],
                "start_event": None,
                "end_event": None,
                "trace_context": trace_context,
                "cpu_start": time.perf_counter(),
                "cpu_end": None,
                "elapsed_ms": None,
                "call_index": self._call_counters[(self.phase, metadata["label"])],
            }
            self._call_counters[(self.phase, metadata["label"])] += 1

            if self.enable_events:
                start_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                record["start_event"] = start_event

            self._pending[metadata["label"]].append(record)

        return hook_fn

    def _post_hook(self, metadata: dict[str, Any]):
        def hook_fn(module, inputs, output):
            pending_record = self._pending[metadata["label"]].pop()
            if self.enable_events:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                pending_record["end_event"] = end_event
            pending_record["cpu_end"] = time.perf_counter()

            trace_context = pending_record.pop("trace_context", None)
            if trace_context is not None:
                trace_context.__exit__(None, None, None)
            self._raw_records.append(pending_record)

        return hook_fn


def safe_floor_divide(numerator: float, denominator: float) -> int:
    if denominator <= 0:
        return 0
    return math.floor(numerator / denominator)
