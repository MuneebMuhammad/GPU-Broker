"""Deterministic, approximate dense-transformer estimator for a semester project.

Python 3.10+, standard library only. No network access, model download, or training.
All GB values are decimal (1 GB = 10**9 bytes). Checkpointing is always enabled.
Run this file to see an example using explicitly illustrative hourly prices.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
import math
import re
from typing import Mapping


# Aggregate effective throughput for the WHOLE instance, already including MFU.
# Order: BF16/FP16 Tensor Core, pure FP32 (no TF32), FP8 Tensor Core.
# Usable VRAM is calculated as 90% of the physical memory below.
GPU_CATALOG = (
    ("ml.g5.2xlarge", "A10G", 1, 24, 37.5, 9.4, None),
    ("ml.g6e.2xlarge", "L40S", 1, 48, 108.6, 27.5, 219.9),
    ("ml.g6e.12xlarge", "L40S", 4, 48, 362.1, 91.6, 733.0),
    ("ml.p4d.24xlarge", "A100", 8, 40, 998.4, 62.4, None),
    ("ml.p5.48xlarge", "H100", 8, 80, 3166.4, 214.4, 6332.8),
)

DEMO_HOURLY_PRICES_USD = {
    "ml.g5.2xlarge": 1.5,
    "ml.g6e.2xlarge": 2.24,
    "ml.g6e.12xlarge": 10.49,
    "ml.p4d.24xlarge": 25.25,
    "ml.p5.48xlarge": 63.30,
}

TRAINING_METHOD = {"full": "full_ft", "fullft": "full_ft", "fullfinetuning": "full_ft",
              "fullfinetune": "full_ft", "lora": "lora", "qlora": "qlora"}

PRECISION_ALIASES = {
        "bf16": "bf16_mixed", "fp16": "bf16_mixed", "bf16mixed": "bf16_mixed",
        "fp16mixed": "bf16_mixed", "bf16mixedfp32adam": "bf16_mixed",
        "fp32": "pure_fp32", "purefp32": "pure_fp32",
        "purebf16": "pure_bf16", "purefp16": "pure_bf16",
        "bf168bitadam": "bf16_8bit_adam", "bf16mixed8bitadam": "bf16_8bit_adam",
        "fp8": "fp8_mixed", "fp8mixed": "fp8_mixed",
}

FULL_FT_STATE_BYTES_PER_PARAMETER = {
    "bf16_mixed": 16,
    "pure_fp32": 16,
    "pure_bf16": 8,
    "bf16_8bit_adam": 10,
    "fp8_mixed": 16,
}

def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _positive(value, name: str, *, integer: bool = False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive {'integer' if integer else 'number'}.")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive number.") from exc
    if not math.isfinite(number) or number <= 0 or (integer and not number.is_integer()):
        raise ValueError(f"{name} must be a finite positive {'integer' if integer else 'number'}.")
    return int(number) if integer else number


def parse_parameter_count(value: str | int | float) -> int:
    """Accept 7B, 400M, 100K, 15,700, 1.5B, or a plain positive integer."""
    if isinstance(value, bool):
        raise ValueError("parameter_count must be a positive integer or K/M/B string.")
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMB]?)\s*", str(value).replace(",", ""), re.I)
    if not match:
        raise ValueError("Invalid parameter_count. Examples: '7B', '400M', '100K', 15700.")
    try:
        count = Decimal(match[1]) * {"": 1, "K": 10**3, "M": 10**6, "B": 10**9}[match[2].upper()]
    except InvalidOperation as exc:
        raise ValueError("Invalid parameter_count.") from exc
    if not count.is_finite() or count <= 0 or count != count.to_integral_value():
        raise ValueError("parameter_count must resolve to a positive whole number.")
    return int(count)


def estimate_training(
    model_name: str,
    parameter_count: str | int | float,
    dataset_size_gb: float,
    sequence_length: int,
    batch_size: int,
    training_method: str,
    precision: str = "BF16",
    number_of_layers: int | None = None,
    hidden_size: int | None = None,
    epochs: int = 1,
    *,
    lora_rank: int = 16,
    bytes_per_token: float = 4.0,
    tokens_per_epoch: int | None = None,
    shard_model_states: bool = True,
    extra_memory_gb_per_gpu: float = 0.0,
    qlora_time_multiplier: float = 1.3,
) -> dict:
    """Estimate all five configurations and choose the first feasible one.

    Selection order: ml.g5.2xlarge, ml.g6e.2xlarge, ml.g6e.12xlarge,
        ml.p4d.24xlarge, ml.p5.48xlarge. Select the first configuration with
        enough per-GPU memory and support for the requested precision.
        Time and cost are reported but do not affect configuration selection.

    batch_size: PER-GPU micro-batch size, before gradient accumulation. It stays
        fixed on every candidate, so global batch size can change with GPU count.
    dataset_size_gb: uncompressed training text bytes / 1e9; NOT archive size,
        image data, or automatically the on-disk size of Parquet/JSON files.
    epochs: defaults to 1. D = tokens_per_epoch * epochs. An exact token count
        overrides the bytes_per_token conversion when supplied.
    precision aliases:
        BF16 / FP16 / bf16_mixed -> 16 bytes/parameter for full fine-tuning;
        FP32 / pure_fp32 -> 16;
        pure_bf16 / pure_fp16 -> 8 (including LOW-PRECISION Adam states);
        bf16_8bit_adam / BF16 mixed + 8-bit Adam -> 10;
        FP8 / fp8_mixed -> 16, with BF16 activations/frozen weights.
        Eight bytes is the user's explicit low-precision optimizer assumption,
        NOT a universal memory requirement of BF16/FP16 training.
    LoRA/QLoRA: always 16 bytes per adapter parameter per the supplied formula.
        Quantized base storage in QLoRA is always 0.52 bytes/parameter. QLoRA
        accepts BF16/FP16 mixed only; it does NOT use FP8 compute throughput.
    sharding: when True, full-FT states or LoRA/QLoRA base weights are ideally
        sharded across all GPUs. Adapter states and activations are replicated.
        Requires a compatible distributed implementation, including FSDP-QLoRA
        for quantized sharding. False means ordinary replicated data parallelism.
    missing architecture: C = .12*N**(2/3) if either L or h is missing, exactly
        as requested (do NOT divide by 10 again). Missing h for P and S/h is
        approximated from N = 12*L*h**2; if L is absent too, assume L = h/128.
        These are crude dense-transformer heuristics, not model-name lookups.
    pricing: None uses DEMO prices, with explicit labels in the result. Pass all
        five actual instance-hour prices to estimate costs at your AWS rates.
    time: effective TFLOPS already includes GPU count and MFU. Do not multiply
        by either again. QLoRA divides throughput by qlora_time_multiplier.

    Returns a JSON-serializable dict, including per-GPU memory breakdown,
    elapsed training hours, GPU-hours, cost, feasibility, and recommendation.
    Model loading, data loading, checkpoint saves, and storage costs are omitted.
    Estimates are conditional on these heuristics; they are not fit guarantees.
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a nonempty string.")
    n = parse_parameter_count(parameter_count)
    size = _positive(dataset_size_gb, "dataset_size_gb")
    s = _positive(sequence_length, "sequence_length", integer=True)
    batch = _positive(batch_size, "batch_size", integer=True)
    epochs = _positive(epochs, "epochs", integer=True)
    rank = _positive(lora_rank, "lora_rank", integer=True)
    bpt = _positive(bytes_per_token, "bytes_per_token")
    slowdown = _positive(qlora_time_multiplier, "qlora_time_multiplier")
    layers = None if number_of_layers is None else _positive(number_of_layers, "number_of_layers", integer=True)
    supplied_h = None if hidden_size is None else _positive(hidden_size, "hidden_size", integer=True)
    if not isinstance(shard_model_states, bool):
        raise ValueError("shard_model_states must be True or False.")
    if isinstance(extra_memory_gb_per_gpu, bool):
        raise ValueError("extra_memory_gb_per_gpu must be finite and nonnegative.")
    extra = float(extra_memory_gb_per_gpu)
    if not math.isfinite(extra) or extra < 0:
        raise ValueError("extra_memory_gb_per_gpu must be finite and nonnegative.")

    method = TRAINING_METHOD.get(_key(training_method))
    if method is None:
        raise ValueError("training_method must be LoRA, QLoRA, or Full fine-tuning.")

    mode = PRECISION_ALIASES.get(_key(precision))
    if mode is None:
        raise ValueError("Unknown precision. Use BF16, FP32, pure_bf16, pure_fp16, bf16_8bit_adam, or FP8.")
    if method == "qlora" and mode != "bf16_mixed":
        raise ValueError("This QLoRA formula requires BF16/FP16 mixed compute; use precision='BF16'.")

    warnings = []
    h = supplied_h
    if h is None:
        h = math.sqrt(n / (12 * layers)) if layers is not None else (128 * n / 12) ** (1 / 3)
        warnings.append("Hidden size inferred.")
    if layers is not None and supplied_h is not None:
        c = 34 * layers * supplied_h / 10
        c_source = "34*L*h/10"
    else:
        c = 0.12 * n ** (2 / 3)
        c_source = "0.12*N**(2/3)"

    if tokens_per_epoch is None:
        epoch_tokens = size * 1e9 / bpt
        warnings.append("Token count estimated from uncompressed text size and bytes_per_token.")
    else:
        epoch_tokens = _positive(tokens_per_epoch, "tokens_per_epoch", integer=True)
    d = epoch_tokens * epochs
    t = batch * s
    activation_bytes = 4 if mode == "pure_fp32" else 2
    full_state_bytes = FULL_FT_STATE_BYTES_PER_PARAMETER[mode]
    adapter_params = 0 if method == "full_ft" else 1.6 * rank * n / h
    activation_gb = t * c * (activation_bytes / 2) / 1e9
    replicated_adapter_gb = 16 * adapter_params / 1e9
    base_bytes = full_state_bytes if method == "full_ft" else (0.52 if method == "qlora" else activation_bytes)
    flops = n * d * ((8 if method == "full_ft" else 6) + s / h)

    candidates = []
    for name, gpu, gpus, vram, bf16_tf, fp32_tf, fp8_tf in GPU_CATALOG:
        shards = gpus if shard_model_states else 1
        fixed_gb = base_bytes * n / shards / 1e9
        required_gb = fixed_gb + replicated_adapter_gb + activation_gb + extra
        usable_gb = 0.9 * vram
        effective_tf = fp32_tf if mode == "pure_fp32" else (fp8_tf if mode == "fp8_mixed" else bf16_tf)
        supported = effective_tf is not None
        fits = required_gb <= usable_gb
        feasible = supported and fits
        hours = None
        if feasible:
            hours = flops / (effective_tf * 1e12 * 3600)
            if method == "qlora":
                hours *= slowdown
        reasons = []
        if not supported:
            reasons.append("No native FP8 support in the supplied hardware table.")
        if not fits:
            reasons.append(f"Needs {required_gb:.3f} GB/GPU; only {usable_gb:.3f} GB/GPU usable.")
        candidates.append({
            "instance": name, "gpu": gpu, "gpu_count": gpus, "shards": shards,
            "usable_memory_gb_per_gpu": usable_gb, "usable_memory_gb_total": usable_gb * gpus,
            "fixed_base_or_full_states_gb_per_gpu": fixed_gb,
            "adapter_states_gb_per_gpu": replicated_adapter_gb,
            "activation_memory_gb_per_gpu": activation_gb,
            "extra_memory_gb_per_gpu": extra, "required_memory_gb_per_gpu": required_gb,
            "required_memory_gb_total": required_gb * gpus,
            "effective_tflops_per_instance": effective_tf,
            "feasible": feasible, "rejection_reasons": reasons,
            "training_hours": hours, "gpu_hours": None if hours is None else hours * gpus,
            "hourly_price_usd": DEMO_HOURLY_PRICES_USD[name],
            "estimated_cost_usd": None if hours is None else hours * DEMO_HOURLY_PRICES_USD[name],
        })
    best = next((row for row in candidates if row["feasible"]), None)
    return {
        "model_name": model_name, "parameter_count": n, "training_method": method,
        "precision": mode, "epochs": epochs, "checkpointing_enabled": True,
        "micro_batch_size_per_gpu": batch, "sequence_length": s,
        "tokens_per_epoch": epoch_tokens, "total_training_tokens": d,
        "hidden_size_used": h, "hidden_size_inferred": supplied_h is None,
        "activation_coefficient_c": c, "activation_coefficient_source": c_source,
        "estimated_adapter_parameters": adapter_params,
        "full_ft_state_bytes_per_parameter": full_state_bytes if method == "full_ft" else None,
        "estimated_flops": flops,
        "status": "estimated_configuration_found" if best else "no_feasible_configuration",
        "selection_rule": "First memory-feasible, precision-supported configuration in catalog order: ml.g5.2xlarge, ml.g6e.2xlarge, ml.g6e.12xlarge, ml.p4d.24xlarge, ml.p5.48xlarge. Time and cost do not affect selection.",
        "warnings": warnings,
    }


if __name__ == "__main__":
    result = estimate_training(
        model_name="Llama 3.1 8B", parameter_count="8B", dataset_size_gb=0.4,
        sequence_length=4096, batch_size=2, training_method="LoRA", precision="BF16",
        number_of_layers=32, hidden_size=4096, epochs=1,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
